#!/bin/bash
# =============================================================================
# 分布式 SLURM 预处理脚本：多节点同时 tokenize Dolma3 数据集。
#
# 每个节点启动一个 task，通过全局哈希分片各自处理一部分 shards。
# 所有节点写入同一共享目录。断点续传后可安全重新提交，已完成的文件自动跳过。
#
# 使用方式：
#   # 1 节点 × 120 CPU（适合小批量测试）
#   sbatch scripts/slurm_pretokenize_distributed.sh
#
#   # 4 节点 × 120 CPU（推荐，速度提升约 4 倍）
#   sbatch --nodes=4 --cpus-per-task=120 scripts/slurm_pretokenize_distributed.sh
#
#   # 8 节点 × 64 CPU（CPU 配额有限时）
#   sbatch --nodes=8 --cpus-per-task=64 scripts/slurm_pretokenize_distributed.sh
#
# 监控所有节点进度：
#   tail -f /hpc_logs/slurm-pretokenize-*.out
# =============================================================================

#SBATCH -o /hpc_logs/slurm-pretokenize-%j-%t.out
#SBATCH -e /hpc_logs/slurm-pretokenize-%j-%t.err
#SBATCH --job-name=olmo3-pretokenize-dist
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=0
#SBATCH --time=2-00:00:00

set -euo pipefail

echo "=== Node $(hostname) starting at $(date) ==="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_NNODES: ${SLURM_NNODES}"
echo "SLURM_NTASKS: ${SLURM_NTASKS}"
echo "SLURM_PROCID: ${SLURM_PROCID}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"

cd /inspire/qb-ilm/project/ai4education/public/wwb/OLMo-core

# ---------------------------------------------------------------------------
# 数据路径（按需修改）
# ---------------------------------------------------------------------------
DATA_ROOT="/inspire/qb-ilm/project/ai4education/public/wwb/datasets/dolma3/pretrain/dolma3_mix-6T-1025-7B/data"
OUTPUT_ROOT="/inspire/qb-ilm/project/ai4education/public/wwb/datasets/dolma3/pretrain/tokenized"

mkdir -p "${OUTPUT_ROOT}"

# ---------------------------------------------------------------------------
# 启动分布式预处理
# srun 会自动在所有节点上同步启动，每个节点一个 task。
# 脚本内部通过 SLURM_PROCID / SLURM_NTASKS 自动分片文件。
# ---------------------------------------------------------------------------

# --nodes / --ntasks-per-node 决定了总 task 数（world_size）。
# 每个 task 内部使用 --workers 个进程（等于分配的 CPU 数）。
srun python scripts/pretokenize_dolma3_full.py \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --workers "${SLURM_CPUS_PER_TASK}"

echo "=== Node $(hostname) finished at $(date) ==="
