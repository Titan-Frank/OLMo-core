#!/usr/bin/env bash
# Distributed launcher for OLMo3-7B pretraining on raw jsonl.zst files with on-the-fly tokenization.
#
# Usage:
#   bash scripts/launch_olmo3_7b_pretrain_jsonl.sh
#
# Useful overrides:
#   SAVE_FOLDER=/path/to/checkpoints bash scripts/launch_olmo3_7b_pretrain_jsonl.sh
#   MAX_TOKENS=10000000000 HARD_STOP_STEPS=1000 bash scripts/launch_olmo3_7b_pretrain_jsonl.sh
#   DRY_RUN=1 bash scripts/launch_olmo3_7b_pretrain_jsonl.sh

set -euo pipefail

REPO_ROOT="/inspire/qb-ilm/project/ai4education/public/wwb/OLMo-core"
# Point to raw jsonl.zst files instead of tokenized .npy files
DATA_ROOT="${DATA_ROOT:-/inspire/qb-ilm/project/ai4education/public/wwb/datasets/dolma3/pretrain/dolma3_mix-6T-1025-7B/data}"
RUN_NAME="${RUN_NAME:-olmo3-7b-pretrain-dolma3-jsonl}"

JOB_ID="${TRAIN_JOB_ID:-${SLURM_JOB_ID:-manual}}"
SAVE_FOLDER="${SAVE_FOLDER:-/inspire/qb-ilm/project/ai4education/public/wwb/checkpoints/${RUN_NAME}/${JOB_ID}}"
WORK_DIR="${WORK_DIR:-/tmp/${RUN_NAME}-${JOB_ID}-${RANK:-0}}"

MAX_TOKENS="${MAX_TOKENS:-5000000000000}"
HARD_STOP_STEPS="${HARD_STOP_STEPS:-597046}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8192}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4194304}"
RANK_MICROBATCH_SIZE="${RANK_MICROBATCH_SIZE:-16384}"
NUM_WORKERS="${NUM_WORKERS:-8}"

export MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
export MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29500}}"

LOCAL_WORLD_SIZE_DEFAULT="${PET_NPROC_PER_NODE:-${LOCAL_WORLD_SIZE:-}}"
if [[ -z "${LOCAL_WORLD_SIZE_DEFAULT}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        LOCAL_WORLD_SIZE_DEFAULT="$(nvidia-smi -L | wc -l)"
    else
        LOCAL_WORLD_SIZE_DEFAULT="1"
    fi
fi
export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-${LOCAL_WORLD_SIZE_DEFAULT}}"

export NUM_NODES="${NUM_NODES:-${PET_NNODES:-${SLURM_NNODES:-1}}}"
NODE_RANK="${PET_NODE_RANK:-${SLURM_NODEID:-0}}"

if [[ -z "${WORLD_SIZE:-}" ]]; then
    export WORLD_SIZE="$((NUM_NODES * LOCAL_WORLD_SIZE))"
fi

if [[ -z "${RANK:-}" ]]; then
    export RANK="$((NODE_RANK * LOCAL_WORLD_SIZE))"
fi

if [[ -z "${LOCAL_RANK:-}" ]]; then
    export LOCAL_RANK="$((RANK % LOCAL_WORLD_SIZE))"
fi

export OLMO_SHARED_FS="${OLMO_SHARED_FS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
mkdir -p "${SAVE_FOLDER}" "${WORK_DIR}"

echo "============================================================"
echo "OLMo3-7B Dolma3 pretraining (on-the-fly tokenization)"
echo "host=$(hostname)"
echo "run_name=${RUN_NAME}"
echo "data_root=${DATA_ROOT}"
echo "save_folder=${SAVE_FOLDER}"
echo "work_dir=${WORK_DIR}"
echo "master=${MASTER_ADDR}:${MASTER_PORT}"
echo "rank=${RANK} world_size=${WORLD_SIZE} local_rank=${LOCAL_RANK} local_world_size=${LOCAL_WORLD_SIZE}"
echo "num_nodes=${NUM_NODES} node_rank=${NODE_RANK}"
echo "============================================================"

TOKENIZER_PATH="${TOKENIZER_PATH:-allenai/dolma2-tokenizer}"

COMMON_ARGS=(
    --name "${RUN_NAME}"
    --data-root "${DATA_ROOT}"
    --save-folder "${SAVE_FOLDER}"
    --work-dir "${WORK_DIR}"
    --sequence-length "${SEQUENCE_LENGTH}"
    --max-tokens "${MAX_TOKENS}"
    --hard-stop-steps "${HARD_STOP_STEPS}"
    train_module.rank_microbatch_size="${RANK_MICROBATCH_SIZE}"
    tokenizer.identifier="${TOKENIZER_PATH}"
)

if [[ "${ENABLE_WANDB:-0}" == "1" ]]; then
    COMMON_ARGS+=(--enable-wandb --wandb-project "${WANDB_PROJECT:-olmo3-7b-pretrain}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    COMMON_ARGS+=(--dry-run)
fi

EXTRA_ARGS=("$@")

if [[ "${LAUNCH_MODE:-auto}" == "direct" || ( "${LAUNCH_MODE:-auto}" == "auto" && -n "${TORCHELASTIC_RUN_ID:-}" ) ]]; then
    exec python scripts/train_olmo3_7b_pretrain_jsonl.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi

exec torchrun \
    --nnodes="${NUM_NODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc-per-node="${LOCAL_WORLD_SIZE}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    scripts/train_olmo3_7b_pretrain_jsonl.py \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"