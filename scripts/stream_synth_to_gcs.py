#!/usr/bin/env python3
import argparse
import hashlib
import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path

from datasets import load_dataset


def run(cmd):
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_fields(raw):
    return [f.strip() for f in raw.split(",") if f.strip()]


def keep_by_sample_rate(example, sample_rate):
    if sample_rate >= 1.0:
        return True
    key = str(example.get("synth_id", ""))
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    return v < sample_rate


def select_fields(example, fields):
    return {k: example.get(k) for k in fields}


def flush_shard(shard_rows, shard_idx, tmp_dir, bucket_prefix):
    if not shard_rows:
        return 0
    local_name = f"synth_{shard_idx:06d}.jsonl.gz"
    local_path = tmp_dir / local_name
    with gzip.open(local_path, "wt", encoding="utf-8") as f:
        for row in shard_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    gs_uri = f"{bucket_prefix.rstrip('/')}/{local_name}"
    run(["gcloud", "storage", "cp", str(local_path), gs_uri])
    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"[upload] {local_name} -> {gs_uri} ({size_mb:.1f} MB)", flush=True)
    local_path.unlink(missing_ok=True)
    return len(shard_rows)


def main():
    p = argparse.ArgumentParser(description="Stream PleIAs/SYNTH to GCS in JSONL.GZ shards.")
    p.add_argument("--bucket-prefix", required=True, help="Destination, e.g. gs://needle-datasets-bucket/datasets/text_pretrain/synth/raw_jsonl_gz")
    p.add_argument("--split", default="train")
    p.add_argument("--rows-per-shard", type=int, default=50000)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--target-bytes", type=int, default=None, help="Stop after this many serialized UTF-8 bytes (approx token budget helper).")
    p.add_argument("--sample-rate", type=float, default=1.0, help="Deterministic keep fraction by synth_id hash, in (0,1].")
    p.add_argument("--fields", type=str, default="synth_id,language,model,seed_license,query,synthetic_answer,words")
    p.add_argument("--language", type=str, default=None, help="Optional exact language filter, e.g. en")
    p.add_argument("--tmp-dir", default="/tmp/synth_upload")
    args = p.parse_args()
    fields = parse_fields(args.fields)

    tmp_dir = Path(args.tmp_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("PleIAs/SYNTH", split=args.split, streaming=True)
    shard_rows = []
    shard_idx = 0
    total = 0
    total_bytes = 0

    try:
        for ex in ds:
            if args.language and ex.get("language") != args.language:
                continue
            if not keep_by_sample_rate(ex, args.sample_rate):
                continue

            row = select_fields(ex, fields)
            row_bytes = len(json.dumps(row, ensure_ascii=False).encode("utf-8")) + 1
            shard_rows.append(row)
            total += 1
            total_bytes += row_bytes

            if args.max_rows is not None and total >= args.max_rows:
                flush_shard(shard_rows, shard_idx, tmp_dir, args.bucket_prefix)
                shard_rows = []
                break

            if args.target_bytes is not None and total_bytes >= args.target_bytes:
                flush_shard(shard_rows, shard_idx, tmp_dir, args.bucket_prefix)
                shard_rows = []
                break

            if len(shard_rows) >= args.rows_per_shard:
                flush_shard(shard_rows, shard_idx, tmp_dir, args.bucket_prefix)
                shard_rows = []
                shard_idx += 1
                print(f"[progress] uploaded_rows={total} uploaded_bytes={total_bytes}", flush=True)

        if shard_rows:
            flush_shard(shard_rows, shard_idx, tmp_dir, args.bucket_prefix)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    approx_tokens = total_bytes / 4.0
    print(f"[done] uploaded_rows={total} uploaded_bytes={total_bytes} approx_tokens={int(approx_tokens)}", flush=True)


if __name__ == "__main__":
    main()
