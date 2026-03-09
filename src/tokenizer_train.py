"""Standalone tokenizer training pipeline: prepare corpus, train SP, validate.

Run on a GCP VM with direct GCS access:
    python -m src.tokenizer_train prepare --datasets synth tool_calls --output /tmp/corpus.txt
    python -m src.tokenizer_train train --corpus /tmp/corpus.txt
    python -m src.tokenizer_train validate --model tokenizer/needle.model --test-corpus /tmp/corpus.txt
"""

import argparse
import gzip
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor

import sentencepiece as spm

from src.tokenizer import pre_tokenize, train_tokenizer

DATASETS = {
    "synth": {
        "gcs_path": "needle-datasets-bucket/datasets/text_pretrain/synth/train/",
        "format": "jsonl_gz",
        "fields": ("query", "query_seed_text", "synthetic_reasoning", "synthetic_answer"),
    },
    "tool_calls": {
        "gcs_path": "gs://cactus-dataset/tool_calls",
        "format": "hf_dataset",
        "fields": ("query", "tools", "answers"),
    },
}

ISOLATED_CHARS = set('({[",]})')


def _process_shard(args):
    """Worker: stream one gzipped JSONL shard -> pre-tokenized temp file."""
    shard_path, fields, tmp_dir = args
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    tmp_fd, tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix=".txt")
    count = 0
    with os.fdopen(tmp_fd, "w") as out:
        with fs.open(shard_path, "rb") as f:
            with gzip.open(f, "rt") as gz:
                for line in gz:
                    example = json.loads(line)
                    for field in fields:
                        text = str(example.get(field, "")).strip()
                        if text:
                            out.write(pre_tokenize(text) + "\n")
                    count += 1
    return tmp_path, count


def write_corpus(output_path, datasets=None, num_workers=4):
    """Stream datasets from GCS -> pre-tokenize -> write combined corpus file."""
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    datasets = datasets or list(DATASETS.keys())
    tmp_dir = os.path.dirname(os.path.abspath(output_path))
    total = 0

    with open(output_path, "w") as out:
        for name in datasets:
            cfg = DATASETS[name]
            print(f"Streaming {name}...")

            if cfg["format"] == "jsonl_gz":
                shards = sorted(fs.ls(cfg["gcs_path"]))
                worker_args = [(s, cfg["fields"], tmp_dir) for s in shards]

                with ProcessPoolExecutor(max_workers=min(num_workers, len(shards))) as pool:
                    results = list(pool.map(_process_shard, worker_args))

                dataset_count = 0
                for tmp_path, count in results:
                    with open(tmp_path) as tmp_f:
                        for line in tmp_f:
                            out.write(line)
                    os.remove(tmp_path)
                    dataset_count += count

                print(f"  {name}: {dataset_count:,} examples ({len(shards)} shards, {num_workers} workers)")
                total += dataset_count

            else:  # hf_dataset
                from datasets import load_from_disk
                ds = load_from_disk(cfg["gcs_path"])
                text_cols = [c for c in cfg["fields"] if c in ds.column_names]
                ds = ds.select_columns(text_cols)
                count = 0
                for example in ds:
                    for field in cfg["fields"]:
                        text = str(example.get(field, "")).strip()
                        if text:
                            out.write(pre_tokenize(text) + "\n")
                    count += 1
                    if count % 100_000 == 0:
                        print(f"  {name}: {count:,} examples...")
                print(f"  {name}: {count:,} examples written")
                total += count

    print(f"Corpus written: {total:,} total examples from {len(datasets)} datasets")


def validate(model_path, test_corpus=None):
    """Validate trained tokenizer: isolated char constraint + compression ratio."""
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)

    # 1. Isolated character check
    violations = []
    for i in range(sp.GetPieceSize()):
        piece = sp.IdToPiece(i)
        for c in ISOLATED_CHARS:
            if c in piece and piece not in (c, f"▁{c}"):
                violations.append((i, piece, c))

    if violations:
        print("FAILED: Isolated character constraint violated!")
        for token_id, piece, char in violations:
            print(f"  Token {token_id} '{piece}' contains isolated char '{char}'")
        raise SystemExit(1)
    print(f"PASSED: All {sp.GetPieceSize()} tokens respect isolated char constraint")

    # 2. Compression ratio on corpus sample
    if test_corpus:
        total_chars = 0
        total_tokens = 0
        max_lines = 100_000
        with open(test_corpus) as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                total_chars += len(line)
                total_tokens += len(sp.Encode(line))

        if total_tokens > 0:
            ratio = total_chars / total_tokens
            print(f"Compression ratio: {ratio:.2f} chars/token ({total_chars:,} chars / {total_tokens:,} tokens, {min(max_lines, i+1):,} lines)")
        else:
            print("WARNING: No tokens produced from test corpus")

    print("Validation complete")


def main():
    parser = argparse.ArgumentParser(description="Needle tokenizer training pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Stream datasets from GCS and write corpus")
    p_prepare.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                           choices=list(DATASETS.keys()))
    p_prepare.add_argument("--output", default="/tmp/tokenizer_corpus.txt")
    p_prepare.add_argument("--num-workers", type=int, default=4)

    p_train = sub.add_parser("train", help="Train SentencePiece tokenizer")
    p_train.add_argument("--corpus", required=True)
    p_train.add_argument("--vocab-size", type=int, default=8192)

    p_validate = sub.add_parser("validate", help="Validate trained tokenizer")
    p_validate.add_argument("--model", default="tokenizer/needle.model")
    p_validate.add_argument("--test-corpus", default=None)

    args = parser.parse_args()

    if args.command == "prepare":
        write_corpus(args.output, datasets=args.datasets, num_workers=args.num_workers)
    elif args.command == "train":
        train_tokenizer(corpus_path=args.corpus, vocab_size=args.vocab_size)
    elif args.command == "validate":
        validate(args.model, test_corpus=args.test_corpus)


if __name__ == "__main__":
    main()
