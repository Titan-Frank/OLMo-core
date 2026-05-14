#!/usr/bin/env bash
# Standalone parallel tokenizer launcher for Dolma3.
#
# Usage (single node):
#   bash scripts/launch_tokenize_dolma3.sh
#
# Usage (multi-node): each node runs the same command with different NODE_RANK.
#   On启智平台, NODE_RANK / NUM_NODES are auto-detected from PET env vars.
#
# The script is fully idempotent: re-running skips already-tokenized files.
# Output .npy files are 100% compatible with train_olmo3_7b_pretrain_jsonl.py.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/inspire/qb-ilm/project/ai4education/public/wwb/datasets/dolma3/pretrain/dolma3_mix-6T-1025-7B/data}"
CACHE_DIR="${CACHE_DIR:-/inspire/qb-ilm/project/ai4education/public/wwb/olmo3-work/tokenized_cache}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/inspire/qb-ilm/project/ai4education/public/wwb/tokenizer/dolma2-tokenizer}"
WORKERS="${WORKERS:-}"

# Auto-detect node rank / num nodes from common schedulers
NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-${SLURM_NODEID:-0}}}"
NUM_NODES="${NUM_NODES:-${PET_NNODES:-${SLURM_NNODES:-1}}}"

REPO_ROOT="/inspire/qb-ilm/project/ai4education/public/wwb/OLMo-core"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "=== Dolma3 Parallel Tokenizer ==="
echo "data_root=${DATA_ROOT}"
echo "cache_dir=${CACHE_DIR}"
echo "tokenizer=${TOKENIZER_PATH}"
echo "node_rank=${NODE_RANK}  num_nodes=${NUM_NODES}"
echo "================================"

cd "${REPO_ROOT}"

ARGS=(
    --data-root "${DATA_ROOT}"
    --cache-dir "${CACHE_DIR}"
    --tokenizer "${TOKENIZER_PATH}"
    --node-rank "${NODE_RANK}"
    --num-nodes "${NUM_NODES}"
)

if [[ -n "${WORKERS}" ]]; then
    ARGS+=(--workers "${WORKERS}")
fi

exec python scripts/tokenize_dolma3_parallel.py "${ARGS[@]}" "$@"
