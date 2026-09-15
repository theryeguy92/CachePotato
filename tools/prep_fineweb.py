"""prep_fineweb.py — FineWeb-Edu (sample-10BT) -> uint16 token bins (Phase 6).

Streams parquet shards in order, tokenizes with GPT-2 BPE, appends
<|endoftext|> (50256) after EVERY document (FineWeb has no inline
boundaries — without this the model never learns document structure).
Stops at TARGET_TRAIN train tokens + TARGET_VAL validation tokens.

Deterministic: shards in sorted order, documents in shard order; training
samples random windows with a fixed seed (the "deterministic shuffle" —
both lambda variants read the identical stream).

Output: data/fineweb_edu_train.bin, data/fineweb_edu_valid.bin
Run on the server:  python3 tools/prep_fineweb.py
"""

import os

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from transformers import GPT2TokenizerFast

REPO = "HuggingFaceFW/fineweb-edu"
PREFIX = "sample/10BT/"
TARGET_TRAIN = 3_000_000_000
TARGET_VAL = 15_000_000
EOS = 50256

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA, exist_ok=True)

tok = GPT2TokenizerFast.from_pretrained("gpt2")


def shard_list():
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in files if f.startswith(PREFIX) and f.endswith(".parquet"))


def tokenize_shard(path, out, budget):
    """Tokenize documents from one parquet shard into `out` (uint16).
    Returns tokens written (<= budget)."""
    written = 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=256, columns=["text"]):
        for text in batch.column("text").to_pylist():
            ids = tok.encode(text)
            ids.append(EOS)
            np.array(ids, dtype="<u2").tofile(out)
            written += len(ids)
            if written >= budget:
                return written
    return written


def main():
    shards = shard_list()
    print(f"{len(shards)} shards in {PREFIX}")
    train_path = os.path.join(DATA, "fineweb_edu_train.bin")
    val_path = os.path.join(DATA, "fineweb_edu_valid.bin")
    total = 0
    with open(train_path, "wb") as out:
        for i, shard in enumerate(shards):
            local = hf_hub_download(REPO, shard, repo_type="dataset")
            n = tokenize_shard(local, out, TARGET_TRAIN - total)
            total += n
            print(f"[{i + 1}/{len(shards)}] {shard}: +{n:,} -> {total:,} tokens", flush=True)
            if total >= TARGET_TRAIN:
                break
    print(f"train done: {total:,} tokens -> {train_path}")

    val_total = 0
    with open(val_path, "wb") as out:
        for shard in shards[i:]:
            local = hf_hub_download(REPO, shard, repo_type="dataset")
            n = tokenize_shard(local, out, TARGET_VAL - val_total)
            val_total += n
            if val_total >= TARGET_VAL:
                break
    print(f"val done: {val_total:,} tokens -> {val_path}")


if __name__ == "__main__":
    main()
