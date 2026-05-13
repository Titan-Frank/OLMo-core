#!/usr/bin/env python3
"""
OLMo3-7B Stage 1 Pretraining Script (On-the-fly Tokenization Version).

This script reads jsonl.zst files directly and tokenizes on-the-fly,
with caching to .npy files in the work directory.

By inheriting from NumpyFSLDataset, we get:
- Distributed data sharding (each rank sees its own data)
- Deterministic shuffle
- State dict for checkpoint/resume (no data repetition)
- Full compatibility with NumpyFSLDataLoader
"""

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.numpy_dataset import NumpyFSLDataset, NumpyUIntTypes
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    MonkeyPatcherCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

DEFAULT_SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 8192 * 512  # ~4M tokens
LR = 3e-4
EOS_TOKEN_ID = 100257


def get_tokenizer_config(tokenizer_path: str) -> TokenizerConfig:
    """Get tokenizer config from local path or HF identifier."""
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    return TokenizerConfig(
        vocab_size=tokenizer.vocab_size,
        eos_token_id=tokenizer.eos_token_id or EOS_TOKEN_ID,
        pad_token_id=tokenizer.pad_token_id or 100277,
        identifier=tokenizer_path,
    )


def find_jsonl_zst_files(data_root: str) -> List[Path]:
    """Recursively find all jsonl.zst files under data_root."""
    return sorted(Path(data_root).rglob("*.jsonl.zst"))


def _npy_cache_path(jsonl_path: Path, cache_dir: Path) -> Path:
    """
    Generate a unique .npy cache path for a jsonl.zst file.
    Uses a short hash of the relative path to avoid name collisions
    across different source directories.
    """
    # e.g. common_crawl-adult_content-0017/shard_00000014.jsonl.zst
    # -> common_crawl-adult_content-0017_shard_00000014.npy
    # Use parent name as prefix to avoid collisions
    rel_parts = jsonl_path.name, jsonl_path.parent.name
    unique_key = "/".join(rel_parts)
    short_hash = hashlib.md5(unique_key.encode()).hexdigest()[:8]
    return cache_dir / f"{jsonl_path.parent.name}_{jsonl_path.stem.replace('.jsonl', '')}_{short_hash}.npy"


def tokenize_shard_to_npy(
    jsonl_path: Path,
    npy_path: Path,
    tokenizer_name: str,
    eos_token_id: int,
) -> int:
    """
    Tokenize a single jsonl.zst file and save to .npy.
    Returns the total number of tokens.
    """
    import zstandard

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    all_token_ids: List[int] = []
    total_docs = 0

    print(f"Tokenizing {jsonl_path.name}...")
    with open(jsonl_path, "rb") as f:
        dctx = zstandard.ZstdDecompressor()
        stream = dctx.stream_reader(f)
        text_stream = io.TextIOWrapper(stream, encoding="utf-8")

        for line in text_stream:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = doc.get("text", "")
            if not text or text == "[REMOVED]":
                continue

            token_ids = tokenizer.encode(text, add_special_tokens=False)
            all_token_ids.extend(token_ids)
            all_token_ids.append(eos_token_id)
            total_docs += 1

    arr = np.array(all_token_ids, dtype=np.uint32) if all_token_ids else np.array([], dtype=np.uint32)
    np.save(npy_path, arr)
    print(f"  {jsonl_path.name}: {total_docs} docs -> {len(arr)} tokens")
    return len(arr)


class LazyTokenizedDataset(NumpyFSLDataset):
    """
    A NumpyFSLDataset that lazily tokenizes jsonl.zst files.
    Tokenized files are cached as .npy in cache_dir.

    IMPORTANT: super().__init__() is deferred to prepare() to avoid
    creating placeholder files that cause 128 processes to race on
    the shared filesystem and mislead the parent class about dataset
    size. Before prepare() is called, only basic attributes are available.
    """

    def __init__(
        self,
        jsonl_paths: List[Path],
        cache_dir: Path,
        tokenizer_name: str,
        sequence_length: int,
        eos_token_id: int,
        pad_token_id: int,
        vocab_size: int,
        bos_token_id: Optional[int] = None,
    ):
        self.jsonl_paths = jsonl_paths
        self.cache_dir = cache_dir
        self.tokenizer_name = tokenizer_name
        self._sequence_length = sequence_length
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id
        self._vocab_size = vocab_size
        self._bos_token_id = bos_token_id

        cache_dir.mkdir(parents=True, exist_ok=True)

        # Map each jsonl -> npy path
        self._npy_paths: List[Path] = []
        for jsonl_path in jsonl_paths:
            npy_path = _npy_cache_path(jsonl_path, cache_dir)
            self._npy_paths.append(npy_path)

        # Set _array_paths so the parent's `paths` property works before prepare().
        # super().__init__() will overwrite this in prepare().
        self._array_paths = tuple(str(p) for p in self._npy_paths)
        self._initialized = False

    def prepare(self):
        """Tokenize missing files (one rank per node in parallel), then init parent."""
        import torch.distributed as dist
        from olmo_core.distributed.utils import get_fs_local_rank, get_world_size, get_rank

        fs_local_rank = get_fs_local_rank()

        if dist.is_initialized():
            rank = get_rank()
            world_size = get_world_size()
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 8))
            node_rank = rank // local_world_size
            num_nodes = world_size // local_world_size
        else:
            rank = 0
            world_size = 1
            node_rank = 0
            num_nodes = 1

        # Only local rank 0 on each node does tokenization (avoids tokenizer contention)
        if fs_local_rank == 0:
            my_files = [
                (jsonl_path, npy_path)
                for jsonl_path, npy_path in zip(self.jsonl_paths, self._npy_paths)
                if not (npy_path.exists() and npy_path.stat().st_size > 132)
            ]

            # Distribute files across nodes (not across all ranks)
            my_files = my_files[node_rank::num_nodes]

            if my_files:
                num_workers = min(len(my_files), max(1, (os.cpu_count() or 8) - 8))
                print(f"  Node {node_rank} (rank {rank}): tokenizing {len(my_files)} files with {num_workers} workers...")
                if num_workers > 1:
                    import multiprocessing as mp
                    with mp.Pool(processes=num_workers) as pool:
                        pool.starmap(
                            tokenize_shard_to_npy,
                            [(p, n, self.tokenizer_name, self.eos_token_id) for p, n in my_files],
                        )
                else:
                    for jsonl_path, npy_path in my_files:
                        tokenize_shard_to_npy(
                            jsonl_path, npy_path, self.tokenizer_name, self.eos_token_id,
                        )
                print(f"  Node {node_rank}: {len(my_files)} files done")

        # Wait for all nodes to finish tokenization.
        if dist.is_initialized():
            dist.barrier()

        # NOW initialize the parent class with real tokenized data
        NumpyFSLDataset.__init__(
            self,
            *self.paths,
            sequence_length=self._sequence_length,
            pad_token_id=self._pad_token_id,
            eos_token_id=self._eos_token_id,
            vocab_size=self._vocab_size,
            dtype=np.uint32,
            bos_token_id=self._bos_token_id,
        )
        self._initialized = True


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} [OPTIONS...]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", type=str, default="olmo3-7b-pretrain", help="Run name")
    parser.add_argument("--sequence-length", type=int, default=None, help="Sequence length")
    parser.add_argument("--data-root", type=str, required=True, help="Root of jsonl.zst files")
    parser.add_argument("--save-folder", type=str, required=True, help="Checkpoint folder")
    parser.add_argument("--work-dir", type=str, required=True, help="Cache directory")
    parser.add_argument("--max-tokens", type=int, default=int(5e12), help="Max tokens")
    parser.add_argument("--hard-stop-steps", type=int, default=597046, help="Hard stop steps")
    parser.add_argument("--tokenizer", type=str, default="allenai/dolma2-tokenizer", help="Tokenizer name or local path")
    parser.add_argument("--rank-microbatch-size", type=int, default=2 * 8192, help="Rank microbatch size in tokens")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument("--train-single", action="store_true", help="Single rank mode")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable W&B")
    parser.add_argument("--wandb-project", type=str, default="olmo3-7b-pretrain")
    return parser


def build_config(opts: argparse.Namespace, overrides: List[str]):
    """Build all training components."""
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = get_tokenizer_config(opts.tokenizer)
    work_dir = Path(opts.work_dir)

    # Find all jsonl.zst files
    jsonl_files = find_jsonl_zst_files(opts.data_root)
    print(f"Found {len(jsonl_files)} jsonl.zst files")

    # Model config
    model_config = TransformerConfig.olmo3_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )

    # Build dataset - inherits from NumpyFSLDataset for full compatibility
    cache_dir = work_dir / "tokenized_cache"
    dataset = LazyTokenizedDataset(
        jsonl_paths=jsonl_files,
        cache_dir=cache_dir,
        tokenizer_name=tokenizer_config.identifier,
        sequence_length=sequence_length,
        eos_token_id=tokenizer_config.eos_token_id,
        pad_token_id=tokenizer_config.pad_token_id,
        vocab_size=tokenizer_config.vocab_size,
        bos_token_id=tokenizer_config.bos_token_id,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_size,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR, weight_decay=0.1, betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=CosWithWarmup(warmup_steps=2000),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(opts.max_tokens),
            hard_stop=Duration.steps(opts.hard_stop_steps) if opts.hard_stop_steps > 0 else None,
        )
        .with_callback("monkey_patcher", MonkeyPatcherCallback())
        .with_callback("checkpointer", CheckpointerCallback(save_interval=1000, ephemeral_save_interval=None, save_async=False))
        .with_callback("wandb", WandBCallback(name=opts.name, project=opts.wandb_project, cancel_check_interval=10, enabled=opts.enable_wandb))
        .with_callback("config_saver", ConfigSaverCallback())
    )

    return model_config, dataset, trainer_config, train_module_config


def main(opts: argparse.Namespace) -> None:
    from olmo_core.data import NumpyDataLoaderConfig
    from olmo_core.io import is_url
    from olmo_core.script_utils import prepare_cli_environment, prepare_training_environment
    from olmo_core.utils import seed_all

    try:
        _main_inner(opts)
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Force flush so torchrun captures the traceback
        sys.stdout.flush()
        sys.stderr.flush()
        raise


def _main_inner(opts: argparse.Namespace) -> None:
    from olmo_core.data import NumpyDataLoaderConfig
    from olmo_core.io import is_url
    from olmo_core.script_utils import prepare_cli_environment, prepare_training_environment
    from olmo_core.utils import seed_all

    if opts.dry_run:
        prepare_cli_environment()

    model_config, dataset, trainer_config, train_module_config = build_config(opts, [])

    if opts.dry_run:
        print("=== Model Config ===")
        print(model_config)
        print(f"\n=== Dataset ===")
        print(f"Type: {type(dataset).__name__}")
        print(f"Sequence length: {dataset.sequence_length}")
        print(f"Num files: {len(dataset.paths)}")
        print(f"Cache dir: {dataset.cache_dir}")
        print("\n=== Trainer Config ===")
        print(trainer_config)
        return

    if opts.train_single:
        train_module_config.dp_config = None
        train_module_config.tp_config = None

    if torch.cuda.is_available():
        backend = "cpu:gloo,cuda:nccl"
    else:
        backend = None

    prepare_training_environment(shared_filesystem=not is_url(opts.save_folder), backend=backend)
    seed_all(12536)

    print("Building model...")
    model = model_config.build(init_device="meta")

    print("Building train module...")
    train_module = train_module_config.build(model)

    # Tokenize missing files (rank 0 only, others wait)
    print("Preparing dataset (tokenizing missing files)...")
    dataset.prepare()

    # Build data loader using the standard NumpyFSLDataLoader
    print("Building data loader...")
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=int(os.environ.get("GLOBAL_BATCH_SIZE", GLOBAL_BATCH_SIZE)),
        seed=34521,
        num_workers=int(os.environ.get("NUM_WORKERS", "8")),
        work_dir=Path(opts.work_dir),
    )
    data_loader = data_loader_config.build(
        dataset,
        dp_process_group=train_module.dp_process_group,
    )

    print("Building trainer...")
    trainer = trainer_config.build(train_module, data_loader)

    # Save config for checkpoints
    for callback in trainer.callbacks.values():
        if isinstance(callback, ConfigSaverCallback):
            callback.config = None  # We don't use ExperimentConfig here
            break

    print("Starting training...")
    trainer.fit()

    from olmo_core.script_utils import teardown_training_environment
    teardown_training_environment()


if __name__ == "__main__":
    parser = _get_parser()
    opts = parser.parse_args()
    main(opts)