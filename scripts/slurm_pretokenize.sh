#!/bin/bash
# =============================================================================
# SLURM submission script for pretokenizing Dolma3 dataset.
#
# This is a CPU-only job. Run it in your HPC / high-performance computing slot.
# After completion, switch to the distributed training space to launch GPU training.
#
# Usage:
#   sbatch scripts/slurm_pretokenize.sh
#
# You can monitor progress with:
#   tail -f /inspire/qb-ilm/project/ai4education/public/wwb/pretokenize-slurm-*.log
# =============================================================================

#SBATCH -o /hpc_logs/slurm-pretokenize-%j.out
#SBATCH -e /hpc_logs/slurm-pretokenize-%j.err
#SBATCH --job-name=olmo3-pretokenize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=300G
#SBATCH --time=7-00:00:00

set -euo pipefail

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Allocated memory: ${SLURM_MEM_PER_NODE:-N/A}"

cd /inspire/qb-ilm/project/ai4education/public/wwb/OLMo-core

# Optional: activate conda environment
# source /path/to/conda/bin/activate olmo

# ---------------------------------------------------------------------------
# Data paths (edit if your paths differ)
# ---------------------------------------------------------------------------
DATA_ROOT="/inspire/qb-ilm/project/ai4education/public/wwb/datasets/dolma3/pretrain/dolma3_mix-6T-1025-7B/data"
OUTPUT_ROOT="/inspire/qb-ilm/project/ai4education/public/wwb/datasets/dolma3/pretrain/tokenized"

mkdir -p "${OUTPUT_ROOT}"

LOG_FILE="/inspire/qb-ilm/project/ai4education/public/wwb/pretokenize-slurm-${SLURM_JOB_ID}.log"

# ---------------------------------------------------------------------------
# Run pretokenization
# --workers matches SLURM_CPUS_PER_TASK for maximum parallelism
# ---------------------------------------------------------------------------
python scripts/pretokenize_dolma3_full.py \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    > "${LOG_FILE}" 2>&1

echo "Job finished at $(date)"
echo "Logs: ${LOG_FILE}"
