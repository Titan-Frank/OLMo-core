#!/usr/bin/env python3
"""
OLMo3 7B Smoke Test Training Script.

This script trains OLMo3-7B using directly preprocessed ``.npy`` files
(instead of official ``DataMix`` paths). It is intended for verifying that
the full distributed training pipeline works end-to-end on your cluster
before preprocessing the full 6T token dataset.

Example (single-node, 8 GPUs):

    torchrun --nproc-per-node=8 scripts/train_olmo3_7b_smoke.py \
        --data-paths /path/to/tokenized/npy \
        --save-folder /path/to/checkpoints \
        --max-steps 50 \
        --global-batch-size 32768 \
        --rank-microbatch-size 8192

Example (multi-node via your cluster launcher):

    python scripts/train_olmo3_7b_smoke.py \
        --data-paths /path/to/tokenized/npy \
        --save-folder /path/to/checkpoints \
        --max-steps 50
"""

import argparse
import sys
from typing import List

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.script_utils import ExperimentConfig, main
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, ConfigSaverCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

DEFAULT_SEQUENCE_LENGTH = 8192
DEFAULT_LR = 3e-4


def _get_parser() -> argparse.ArgumentParser:
    """CLI parser that extends the official one with smoke-test specific args."""
    # We build a minimal parser compatible with olmo_core.script_utils.main().
    # For more advanced use-cases, inherit from script_utils.get_cli_parser().
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} [OPTIONS...] [CONFIG_OVERRIDES...]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--name",
        type=str,
        default="olmo3-7b-smoke",
        help="A name to assign the run for logging.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Sequence length for training.",
    )
    parser.add_argument(
        "--data-paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories/paths containing pre-tokenized ``.npy`` files. "
        "Globs like ``/data/*.npy`` are supported.",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        required=True,
        help="Directory to save checkpoints to.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Working directory for dataset preprocessing caches. "
        "Defaults to ``save-folder`` if not set.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum training steps for smoke test.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=32768,
        help="Global batch size in tokens. "
        "For a full run this is ~4M tokens (8192 * 512).",
    )
    parser.add_argument(
        "--rank-microbatch-size",
        type=int,
        default=8192,
        help="Microbatch size per rank in tokens.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the config and exit.",
    )
    parser.add_argument(
        "--train-single",
        action="store_true",
        help="Train on a single rank for debugging.",
    )
    return parser


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()

    # Model: OLMo3 7B
    model_config = TransformerConfig.olmo3_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    # Dataset: point directly at the preprocessed ``.npy`` files.
    # We accept a list of directories or glob patterns.
    glob_patterns = []
    for p in opts.data_paths:
        # If the path is a directory, append a wildcard for .npy files.
        if p.endswith("/") or not p.endswith(".npy"):
            glob_patterns.append(f"{p.rstrip('/')}/*.npy")
        else:
            glob_patterns.append(p)

    dataset_config = NumpyFSLDatasetConfig.glob(
        *glob_patterns,
        tokenizer=tokenizer_config,
        sequence_length=sequence_length,
        max_target_sequence_length=max(8192, sequence_length),
        work_dir=opts.work_dir,
        expand_glob=True,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size,
        seed=34521,
        num_workers=8,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_size,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=opts.lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=CosWithWarmup(warmup_steps=min(2000, opts.max_steps)),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            metrics_collect_interval=1,
            cancel_check_interval=10,
            max_duration=Duration.steps(opts.max_steps),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=max(1, opts.max_steps),
                ephemeral_save_interval=None,
                save_async=False,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


if __name__ == "__main__":
    parser = _get_parser()
    main(build_config, parser=parser)
