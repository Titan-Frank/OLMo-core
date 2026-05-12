#!/bin/bash
# =============================================================================
# 1 节点测试脚本：验证预处理逻辑和输出格式。
#
# 建议先用这个脚本跑通流程，确认 .npy 文件生成正确后，
# 再改用 slurm_pretokenize_distributed.sh 提交多节点全量任务。
#
# 使用方式：
#   sbatch scripts/slurm_pretokenize_test.sh
#
# 监控：
#   tail -f /hpc_logs/slurm-pretokenize-test-*.out
# =============================================================================

#SBATCH -o /hpc_logs/slurm-pretokenize-test-%j.out
#SBATCH -e /hpc_logs/slurm-pretokenize-test-%j.err
#SBATCH --job-name=olmo3-pretokenize-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=0
#SBATCH --time=12:00:00

set -euo pipefail

echo "=== Node $(hostname) starting at $(date) ==="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"

cd /inspire/qb-ilm/project/ai4education/public/wwb/OLMo-core

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
DATA_ROOT="/inspire/qb-ilm/project/ai4education/public/wwb/datasets/dolma3/pretrain/dolma3_mix-6T-1025-7B/data"
OUTPUT_ROOT="/inspire/qb-ilm/project/ai4education/public/wwb/datasets/dolma3/pretrain/tokenized"

mkdir -p "${OUTPUT_ROOT}"

# ---------------------------------------------------------------------------
# 先跑 Dry Run，确认分片和文件路径正确（不消耗实际算力）
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 1: Dry run (print only, no processing)"
python scripts/pretokenize_dolma3_full.py \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --dry-run

echo ""
echo ">>> Step 2: Actual tokenization (resume capable)"

# 单节点直接运行即可（无需 srun）
python scripts/pretokenize_dolma3_full.py \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --workers "${SLURM_CPUS_PER_TASK}"

echo ""
echo "=== Node $(hostname) finished at $(date) ==="

# ---------------------------------------------------------------------------
# 快速验证：统计生成的 .npy 数量和大小
# ---------------------------------------------------------------------------
echo ""
echo ">>> Verification: output statistics"
echo "  Total .npy files:"
find "${OUTPUT_ROOT}" -name "*.npy" | wc -l

echo ""
echo "  Disk usage by source:"
du -sh "${OUTPUT_ROOT}"/* | sort -hr | head -20
