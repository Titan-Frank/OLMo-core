#!/usr/bin/env python3
"""
预处理 Dolma3 jsonl.zst 数据为 OLMo-core 所需的 .npy 格式。
可以指定 source 子集来做 smoke test。

用法：
    python scripts/pretokenize_dolma3.py \
        --input /inspire/qb-ilm/project/ai4education/public/wwb/datasets/dolma3/pretrain/dolma3_mix-6T-1025-7B/data/dolma1_7-wiki-en \
        --output /inspire/qb-ilm/project/ai4education/public/wwb/datasets/dolma3/pretrain/tokenized/dolma1_7-wiki-en
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import zstandard
from tqdm import tqdm
from transformers import AutoTokenizer

# dolma2 tokenizer 特殊 token（与官方一致）
EOS_TOKEN_ID = 100257


def find_shards(input_dir: str):
    """递归查找所有 jsonl.zst 文件。"""
    return sorted(Path(input_dir).rglob("*.jsonl.zst"))


def process_shard(input_path: Path, output_path: Path, tokenizer_name: str):
    """处理单个 shard：解压 jsonl.zst -> tokenize -> 保存 .npy。"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    all_token_ids = []
    with open(input_path, "rb") as f:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            for line in reader:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = doc.get("text", "")
                if not text or text == "[REMOVED]":
                    continue
                # tokenize，不加特殊 token，稍后手动加 EOS
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                all_token_ids.extend(token_ids)
                all_token_ids.append(EOS_TOKEN_ID)

    if len(all_token_ids) == 0:
        return str(input_path), 0

    arr = np.array(all_token_ids, dtype=np.uint32)
    np.save(output_path, arr)
    return str(input_path), len(all_token_ids)


def main():
    parser = argparse.ArgumentParser(description="Tokenize Dolma3 jsonl.zst shards to .npy")
    parser.add_argument("--input", type=str, required=True, help="输入 source 目录，如 data/dolma1_7-wiki-en")
    parser.add_argument("--output", type=str, required=True, help="输出 .npy 目录")
    parser.add_argument("--tokenizer", type=str, default="allenai/dolma2-tokenizer")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--shard-pattern", type=str, default="*", help="只处理匹配此模式的 shard 文件名")
    opts = parser.parse_args()

    shards = [p for p in find_shards(opts.input) if opts.shard_pattern in p.name]
    if not shards:
        print(f"在 {opts.input} 下未找到 *.jsonl.zst 文件")
        return

    print(f"找到 {len(shards)} 个 shard，启动 {opts.workers} 进程并行处理...")

    os.makedirs(opts.output, exist_ok=True)
    tasks = []
    for shard in shards:
        out_name = shard.stem.replace(".jsonl", ".npy")  # shard_xxx.jsonl.zst -> shard_xxx.npy
        out_path = Path(opts.output) / out_name
        tasks.append((shard, out_path, opts.tokenizer))

    results = []
    if opts.workers > 1:
        with mp.Pool(processes=opts.workers) as pool:
            for res in tqdm(
                pool.starmap(process_shard, tasks),
                total=len(tasks),
                desc="Tokenizing",
            ):
                results.append(res)
    else:
        for shard, out, tok in tqdm(tasks, desc="Tokenizing"):
            results.append(process_shard(shard, out, tok))

    total_tokens = sum(r[1] for r in results)
    print(f"\n处理完成：{len(shards)} shards, 共 {total_tokens:,} tokens")
    print(f"输出目录: {opts.output}")


if __name__ == "__main__":
    main()
