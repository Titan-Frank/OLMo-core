#!/usr/bin/env python3
"""
OLMo3-7B Stage 1 Pretraining Script (Local Data Version).

This script replicates the official OLMo-3-1025-7B-pretrain-1.py configuration
but uses pre-tokenized ``.npy`` files from a local filesystem instead of
fetching from remote URLs via ``DataMix``.

The data paths should point to directories containing ``.npy`` files produced
by ``scripts/pretokenize_dolma3_full.py``.

Usage (single-node debug):

    torchrun --nproc-per-node=8 \
        scripts/train_olmo3_7b_pretrain.py \
        --data-root /inspire/qb-ilm/project/ai4education/public/wwb/datasets/dolma3/pretrain/tokenized \
        --save-folder /inspire/qb-ilm/project/ai4education/public/wwb/olmo3-7b-checkpoints \
        --work-dir /tmp/olmo3-7b-work

Usage (cluster multi-node):

    python scripts/train_olmo3_7b_pretrain.py \
        --data-root /inspire/qb-ilm/project/ai4education/public/wwb/datasets/dolma3/pretrain/tokenized \
        --save-folder /inspire/qb-ilm/project/ai4education/public/wwb/olmo3-7b-checkpoints \
        --work-dir /tmp/olmo3-7b-work

Environment variables expected from your cluster launcher (set automatically):
    RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, LOCAL_RANK, etc.
"""

import argparse
import sys
from typing import List

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.script_utils import ExperimentConfig, main
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    LMEvaluatorCallbackConfig,
    MonkeyPatcherCallback,
    TensorBoardCallback,
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


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} [OPTIONS...] [CONFIG_OVERRIDES...]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--name",
        type=str,
        default="olmo3-7b-pretrain",
        help="A name to assign the run for logging.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="The sequence length to train and eval on. Defaults to 8192.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing preprocessed ``.npy`` files. "
        "Subdirectories are scanned recursively for ``*.npy`` files.",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        required=True,
        help="Directory to save checkpoints to. Should be on shared storage for multi-node.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Working directory for dataset preprocessing caches. "
        "If not set this will be inferred from the save folder.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(5e12),
        help="Maximum training tokens for this stage. Default 5T.",
    )
    parser.add_argument(
        "--hard-stop-steps",
        type=int,
        default=597046,
        help="Hard stop at this step count. Set 0 to disable hard stop.",
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
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="olmo3-7b-pretrain",
        help="WandB project name.",
    )
    parser.add_argument(
        "--enable-lm-eval",
        action="store_true",
        help="Enable LM perplexity evaluation. Requires a padded FSL validation dataset.",
    )
    parser.add_argument(
        "--eval-data-root",
        type=str,
        default=None,
        help="Root directory containing validation ``.npy`` files for LM eval. "
        "Defaults to ``<data-root>/v3-small-ppl-validation`` when LM eval is enabled.",
    )
    parser.add_argument(
        "--enable-downstream-eval",
        action="store_true",
        help="Enable downstream OLMES evaluations. Requires olmo-eval and task assets.",
    )
    return parser


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()

    # Model: OLMo3 7B
    model_config = TransformerConfig.olmo3_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )

    # Dataset: use all ``*.npy`` files under data-root recursively.
    # Each file is expected to be a 1D uint32 array produced by pretokenize_dolma3_full.py.
    dataset_config = NumpyFSLDatasetConfig.glob(
        f"{opts.data_root.rstrip('/')}/**/*.npy",
        tokenizer=tokenizer_config,
        sequence_length=sequence_length,
        max_target_sequence_length=max(8192, sequence_length),
        work_dir=opts.work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=8,
    )

    eval_data_root = opts.eval_data_root or f"{opts.data_root.rstrip('/')}/v3-small-ppl-validation"

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=2 * 8192,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
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
            hard_stop=(
                Duration.steps(opts.hard_stop_steps)
                if opts.hard_stop_steps > 0
                else None
            ),
        )
        .with_callback("monkey_patcher", MonkeyPatcherCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=None,
                save_async=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.name,
                project=opts.wandb_project,
                cancel_check_interval=10,
                enabled=opts.enable_wandb,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "tensorboard",
            TensorBoardCallback(
                enabled=True,
                log_dir=f"{opts.save_folder}/tensorboard",
            ),
        )
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.glob(
                    f"{eval_data_root.rstrip('/')}/**/*.npy",
                    tokenizer=tokenizer_config,
                    sequence_length=sequence_length,
                    work_dir=opts.work_dir,
                ),
                eval_interval=10_000,
                enabled=opts.enable_lm_eval,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=sorted(
                    [
                        "arc_challenge_test_bpb_5shot",
                        "arc_challenge_test_mc_5shot_fast",
                        "arc_easy_test_bpb_5shot",
                        "arc_easy_test_mc_5shot_fast",
                        "hellaswag_bpb_5shot",
                        "mmlu_humanities_test_bpb_5shot",
                        "mmlu_humanities_test_mc_5shot_fast",
                        "mmlu_other_test_bpb_5shot",
                        "mmlu_other_test_mc_5shot_fast",
                        "mmlu_social_sciences_test_bpb_5shot",
                        "mmlu_social_sciences_test_mc_5shot_fast",
                        "mmlu_stem_test_bpb_5shot",
                        "mmlu_stem_test_mc_5shot_fast",
                        "basic_skills_arithmetic_rc_5shot",
                        "basic_skills_coding_rc_5shot",
                        "basic_skills_common_knowledge_rc_5shot",
                        "basic_skills_logical_reasoning_rc_5shot",
                        "basic_skills_pattern_rc_5shot",
                        "basic_skills_string_operations_rc_5shot",
                        "codex_humaneval_gold_bpb_3shot",
                        "codex_mbpp_gold_bpb_3shot",
                        "minerva_math_500_gold_bpb_0shot",
                        "mt_mbpp_cpp_gold_bpb_3shot",
                        "mt_mbpp_java_gold_bpb_3shot",
                        "mt_mbpp_rust_gold_bpb_3shot",
                        "copycolors_10way_fast",
                    ]
                ),
                tokenizer=tokenizer_config,
                eval_interval=10_000,
                enabled=opts.enable_downstream_eval,
            ),
        )
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
