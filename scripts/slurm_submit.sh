#!/bin/bash
# =============================================================================
# SLURM submission script for OLMo3-7B pretraining.
#
# Usage:
#   sbatch scripts/slurm_submit.sh
#
# Adjust SBATCH directives below to match your cluster quota.
# =============================================================================

#SBATCH -o /hpc_logs/slurm-%j.out
#SBATCH -e /hpc_logs/slurm-%j.err
#SBATCH --job-name=olmo3-7b-pretrain

# ---- Single-node example (8 GPUs) ----
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=120
#SBATCH --mem-per-cpu=3G
#SBATCH --time=7-00:00:00

# ---- Multi-node example (4 nodes × 8 GPUs = 32 GPUs) ----
# Uncomment below and comment out the single-node block above:
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --time=7-00:00:00

# =============================================================================
# Environment setup
# =============================================================================
set -euo pipefail

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "SLURM_NNODES: ${SLURM_NNODES:-N/A}"
echo "SLURM_NTASKS: ${SLURM_NTASKS:-N/A}"
echo "SLURM_NTASKS_PER_NODE: ${SLURM_NTASKS_PER_NODE:-N/A}"
echo "SLURM_PROCID: ${SLURM_PROCID:-N/A}"
echo "SLURM_LOCALID: ${SLURM_LOCALID:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"

# Optional: activate your conda/virtual environment
# source /path/to/your/conda/bin/activate olmo

cd /inspire/qb-ilm/project/ai4education/public/wwb/OLMo-core

# =============================================================================
# Distributed environment variable setup
# =============================================================================
# OLMo-core / PyTorch Distributed requires the following variables:
#   MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE
#
# Your platform already injects: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT,
# PET_NNODES, PET_NPROC_PER_NODE, etc.
#
# However, if LOCAL_RANK is missing, we derive it from SLURM_LOCALID or GPU ID.
# =============================================================================

# Ensure MASTER_ADDR and MASTER_PORT are set
if [ -z "${MASTER_ADDR:-}" ]; then
    # Derive from Slurm node list (first node is master)
    MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
    export MASTER_ADDR
    echo "Set MASTER_ADDR=${MASTER_ADDR}"
fi

if [ -z "${MASTER_PORT:-}" ]; then
    export MASTER_PORT=29500
    echo "Set MASTER_PORT=${MASTER_PORT}"
fi

# Detect launcher mode --------------------------------------------------------
# Mode A: Platform already launched via torchrun / elastic (variables present)
# Mode B: Standard Slurm (need srun or torchrun)
# -----------------------------------------------------------------------------

if [ -n "${LOCAL_RANK:-}" ]; then
    echo "Detected LOCAL_RANK=${LOCAL_RANK} (platform torchrun/elastic mode)"
    LAUNCH_MODE="direct"
elif [ -n "${SLURM_LOCALID:-}" ]; then
    echo "Detected SLURM_LOCALID=${SLURM_LOCALID} (standard Slurm mode)"
    export LOCAL_RANK=${SLURM_LOCALID}
    export LOCAL_WORLD_SIZE=${SLURM_NTASKS_PER_NODE:-1}
    export NUM_NODES=${SLURM_NNODES:-1}

    # torch.distributed needs RANK / WORLD_SIZE.
    # If the platform already set them, keep them. Otherwise derive from Slurm.
    [ -z "${RANK:-}" ] && export RANK=${SLURM_PROCID:-0}
    [ -z "${WORLD_SIZE:-}" ] && export WORLD_SIZE=${SLURM_NTASKS:-1}

    LAUNCH_MODE="direct"
else
    echo "No LOCAL_RANK or SLURM_LOCALID found. Falling back to torchrun."
    LAUNCH_MODE="torchrun"
fi

# =============================================================================
# Run training
# =============================================================================

# Paths (edit to match your local paths)
DATA_ROOT="/inspire/qb-ilm/project/ai4education/public/wwb/datasets/dolma3/pretrain/tokenized"
SAVE_FOLDER="/inspire/qb-ilm/project/ai4education/public/wwb/olmo3-7b-checkpoints"
WORK_DIR="/tmp/olmo3-7b-work-${SLURM_JOB_ID:-0}"

mkdir -p "${SAVE_FOLDER}"
mkdir -p "${WORK_DIR}"

# Training hyperparameters (matching official OLMo3 7B stage 1)
MAX_TOKENS=5000000000000
HARD_STOP_STEPS=597046

COMMON_ARGS=(
    --data-root "${DATA_ROOT}"
    --save-folder "${SAVE_FOLDER}"
    --work-dir "${WORK_DIR}"
    --max-tokens "${MAX_TOKENS}"
    --hard-stop-steps "${HARD_STOP_STEPS}"
)

# Optional: enable WandB
# COMMON_ARGS+=(--enable-wandb --wandb-project olmo3-7b-pretrain)

# Optional: config overrides (dot-list syntax)
# COMMON_ARGS+=(--train_module.rank_microbatch_size=4096)

if [ "${LAUNCH_MODE}" == "direct" ]; then
    echo "Launching training directly (distributed env vars already set)..."
    srun python scripts/train_olmo3_7b_pretrain.py "${COMMON_ARGS[@]}"
else
    # torchrun mode (single-node or multi-node via SSH if supported)
    NPROC_PER_NODE=${SLURM_NTASKS_PER_NODE:-8}
    NNODES=${SLURM_NNODES:-1}
    NODE_RANK=${SLURM_NODEID:-0}

    echo "Launching via torchrun (nnodes=${NNODES}, nproc_per_node=${NPROC_PER_NODE}, node_rank=${NODE_RANK})..."

    torchrun \
        --nnodes="${NNODES}" \
        --node_rank="${NODE_RANK}" \
        --nproc-per-node="${NPROC_PER_NODE}" \
        --master-addr="${MASTER_ADDR}" \
        --master-port="${MASTER_PORT}" \
        scripts/train_olmo3_7b_pretrain.py \
        "${COMMON_ARGS[@]}"
fi

echo "Job finished at $(date)"
