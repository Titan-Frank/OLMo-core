#!/usr/bin/env python3
"""
Standalone parallel tokenizer for Dolma3 jsonl.zst files.

Tokenizes all jsonl.zst shards into .npy cache files, fully decoupled from training.
Any number of nodes can run this script simultaneously — each node picks up its
own slice of untokenized files via a shared filesystem lock-free scheme:

  - Each node lists all jsonl.zst files, checks which .npy already exist,
    then takes files[node_rank::num_nodes] from the remaining list.
  - Atomic writes (write to .tmp, then os.replace) prevent partial-file races.
  - The output is 100% compatible with the LazyTokenizedDataset cache used
    by train_olmo3_7b_pretrain_jsonl.py.

Usage (single node):
  python scripts/tokenize_dolma3_parallel.py \
      --data-root /path/to/dolma3_mix-6T-1025-7B/data \
      --cache-dir /path/to/tokenized_cache \
      --tokenizer /path/to/dolma2-tokenizer

Usage (multi-node, e.g. via torchrun or启智平台):
  Each node runs the same command. Node rank and count are auto-detected from
  TORCHELASTIC / SLURM / PET env vars, or overridden with --node-rank / --num-nodes.

  # Example with 4 nodes on启智:
  for i in 0 1 2 3; do
    NODE_RANK=$i NUM_NODES=4 python scripts/tokenize_dolma3_parallel.py \
        --data-root ... --cache-dir ... --tokenizer ...
  done
"""

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _npy_cache_path(jsonl_path: Path, cache_dir: Path) -> Path:
    rel_parts = jsonl_path.name, jsonl_path.parent.name
    unique_key = "/".join(rel_parts)
    short_hash = hashlib.md5(unique_key.encode()).hexdigest()[:8]
    return cache_dir / f"{jsonl_path.parent.name}_{jsonl_path.stem.replace('.jsonl', '')}_{short_hash}.npy"


def _npy_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 132:
        return False
    try:
        np.load(path, mmap_mode="r")
        return True
    except Exception:
        return False


def tokenize_shard_to_npy(
    jsonl_path: Path,
    npy_path: Path,
    tokenizer_name: str,
    eos_token_id: int,
) -> int:
    import gzip
    import zstandard
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)

    all_token_ids: List[int] = []
    total_docs = 0

    with open(jsonl_path, "rb") as f:
        if jsonl_path.suffix == ".gz":
            text_stream = io.TextIOWrapper(gzip.open(f), encoding="utf-8")
        else:
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
    import threading
    tmp_path = npy_path.parent / f"{npy_path.stem}.{os.getpid()}.{threading.get_ident() % 100000}.tmp.npy"
    np.save(tmp_path, arr)
    os.replace(tmp_path, npy_path)
    return len(arr)


def get_node_info(args) -> Tuple[int, int]:
    """Return (node_rank, num_nodes) from args or env vars."""
    node_rank = args.node_rank
    num_nodes = args.num_nodes

    if node_rank is None:
        node_rank = int(os.environ.get("PET_NODE_RANK",
                         os.environ.get("SLURM_NODEID",
                         os.environ.get("NODE_RANK", 0))))
    if num_nodes is None:
        num_nodes = int(os.environ.get("PET_NNODES",
                       os.environ.get("SLURM_NNODES",
                       os.environ.get("NUM_NODES", 1))))

    return node_rank, num_nodes


def main():
    parser = argparse.ArgumentParser(description="Standalone parallel tokenizer for Dolma3")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory of jsonl.zst files (e.g. dolma3_mix-6T-1025-7B/data)")
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Output directory for .npy cache files")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Tokenizer path (local or HF identifier)")
    parser.add_argument("--eos-token-id", type=int, default=100257,
                        help="EOS token ID (default: 100257 for dolma2)")
    parser.add_argument("--node-rank", type=int, default=None,
                        help="Node rank (auto-detected from env if not set)")
    parser.add_argument("--num-nodes", type=int, default=None,
                        help="Total number of nodes (auto-detected from env if not set)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of multiprocessing workers per node (default: auto)")
    parser.add_argument("--check-only", action="store_true",
                        help="Only print progress statistics, don't tokenize")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Find all source files
    jsonl_files = sorted(list(data_root.rglob("*.jsonl.zst")) + list(data_root.rglob("*.jsonl.gz")))
    total_files = len(jsonl_files)
    print(f"Found {total_files} jsonl files under {data_root}")

    # Build mapping and check which are done
    file_pairs = []
    already_done = 0
    corrupted = 0
    for jsonl_path in jsonl_files:
        npy_path = _npy_cache_path(jsonl_path, cache_dir)
        if _npy_is_valid(npy_path):
            already_done += 1
        elif npy_path.exists():
            corrupted += 1
            file_pairs.append((jsonl_path, npy_path))
        else:
            file_pairs.append((jsonl_path, npy_path))

    remaining = len(file_pairs)
    progress_pct = already_done / total_files * 100 if total_files > 0 else 0
    print(f"Progress: {already_done}/{total_files} ({progress_pct:.1f}%) already tokenized")
    if corrupted:
        print(f"  {corrupted} corrupted/invalid .npy files will be re-tokenized")
    print(f"  {remaining} files remaining")

    if args.check_only:
        # Per-source breakdown
        source_counts: dict = {}
        for jsonl_path, npy_path in file_pairs:
            source = jsonl_path.parent.name
            source_counts[source] = source_counts.get(source, 0) + 1
        if source_counts:
            print(f"\nRemaining files by source (top 20):")
            for source, count in sorted(source_counts.items(), key=lambda x: -x[1])[:20]:
                print(f"  {source}: {count}")
        return

    if remaining == 0:
        print("All files already tokenized!")
        return

    # Distribute across nodes
    node_rank, num_nodes = get_node_info(args)
    my_files = file_pairs[node_rank::num_nodes]
    print(f"Node {node_rank}/{num_nodes}: assigned {len(my_files)} files")

    if not my_files:
        print("Nothing to do for this node.")
        return

    # Determine worker count
    if args.workers is not None:
        num_workers = args.workers
    else:
        num_workers = min(len(my_files), max(1, (os.cpu_count() or 8) - 4))

    # Load tokenizer once to verify it works
    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print("Tokenizer loaded successfully.")

    start_time = time.time()
    done_count = 0
    total_tokens = 0

    if num_workers > 1:
        import multiprocessing as mp
        print(f"Tokenizing with {num_workers} workers...")
        with mp.Pool(processes=num_workers) as pool:
            results = pool.starmap(
                tokenize_shard_to_npy,
                [(p, n, args.tokenizer, args.eos_token_id) for p, n in my_files],
            )
        done_count = len(results)
        total_tokens = sum(results)
    else:
        print("Tokenizing sequentially (1 worker)...")
        for i, (jsonl_path, npy_path) in enumerate(my_files):
            n_tokens = tokenize_shard_to_npy(jsonl_path, npy_path, args.tokenizer, args.eos_token_id)
            done_count += 1
            total_tokens += n_tokens
            if (i + 1) % 50 == 0 or (i + 1) == len(my_files):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{len(my_files)}] {rate:.1f} files/s, {total_tokens/1e9:.2f}B tokens so far")

    elapsed = time.time() - start_time
    print(f"\nNode {node_rank} done: {done_count} files, {total_tokens/1e9:.2f}B tokens in {elapsed:.0f}s")

    # Final progress check
    final_done = sum(1 for _, npy_path in file_pairs if _npy_is_valid(npy_path))
    # Also count the previously-done ones
    all_npy = list(cache_dir.glob("*.npy"))
    valid_all = sum(1 for p in all_npy if _npy_is_valid(p))
    print(f"Overall progress: {valid_all}/{total_files} ({valid_all/total_files*100:.1f}%)")


if __name__ == "__main__":
    main()
