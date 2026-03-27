#!/usr/bin/env python3
"""Merge Salesforce/xlam-function-calling-60k into the existing unified dataset
and re-upload to HuggingFace (Cactus-Compute/tool-calls).

Usage:
    python scripts/merge_xlam.py                    # merge + upload
    python scripts/merge_xlam.py --dry-run          # merge only, no upload
    python scripts/merge_xlam.py --max-samples 1000 # limit xlam samples (for testing)
"""

import argparse
import json
import os
import sys

from datasets import concatenate_datasets, load_dataset, load_from_disk

LOCAL_UNIFIED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tool_calls_unified")
HF_DATASET_REPO = "Cactus-Compute/tool-calls"


def load_existing():
    local = os.path.abspath(LOCAL_UNIFIED_DIR)
    if os.path.exists(local) and any(f.endswith(".arrow") for f in os.listdir(local)):
        ds = load_from_disk(local)
        print(f"Loaded existing dataset from disk: {len(ds)} rows")
    else:
        print(f"Downloading existing dataset from {HF_DATASET_REPO}...")
        ds = load_dataset(HF_DATASET_REPO, split="train", token=True)
        print(f"Downloaded: {len(ds)} rows")
    return ds


def load_xlam(max_samples=None):
    print("Loading Salesforce/xlam-function-calling-60k...")
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    print(f"Loaded xlam: {len(ds)} rows")

    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(max_samples))
        print(f"Sampled down to {len(ds)} rows")

    if "id" in ds.column_names:
        ds = ds.remove_columns(["id"])

    ds = ds.add_column("source", ["xlam-function-calling-60k"] * len(ds))
    ds = ds.add_column("model", ["human-curated"] * len(ds))

    return ds


def check_duplicates(existing, new):
    """Quick check for query overlap."""
    existing_queries = set(existing["query"][:10000]) 
    new_queries = set(new["query"])
    overlap = existing_queries & new_queries
    if overlap:
        print(f"Warning: {len(overlap)} overlapping queries found (sampling first 10k existing)")
    else:
        print("No duplicate queries detected (sampled check)")


def main():
    parser = argparse.ArgumentParser(description="Merge xlam into unified dataset")
    parser.add_argument("--dry-run", action="store_true", help="Skip upload")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit xlam samples")
    args = parser.parse_args()

    existing = load_existing()
    xlam = load_xlam(args.max_samples)

    xlam = xlam.select_columns(existing.column_names)

    check_duplicates(existing, xlam)

    merged = concatenate_datasets([existing, xlam])
    print(f"\nMerged dataset: {len(merged)} rows ({len(existing)} existing + {len(xlam)} new)")

    from collections import Counter
    counts = Counter(merged["source"])
    print("\nBreakdown by source:")
    for src, cnt in counts.most_common():
        print(f"  {src}: {cnt}")

    if args.dry_run:
        print("\n--dry-run: skipping save and upload")
        return

    import shutil
    local = os.path.abspath(LOCAL_UNIFIED_DIR)
    tmp_dir = local + "_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    print(f"\nSaving to {local}...")
    merged.save_to_disk(tmp_dir)
    if os.path.exists(local):
        shutil.rmtree(local)
    os.rename(tmp_dir, local)
    print("Saved.")

    # Upload to HuggingFace (train split only)
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(HF_DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)
    print(f"\nUploading to {HF_DATASET_REPO} (train split)...")
    merged.push_to_hub(HF_DATASET_REPO, split="train", token=True)
    print(f"Upload complete: {HF_DATASET_REPO}")
    print("NOTE: Run 'python scripts/split_dataset.py' to create the validation split.")


if __name__ == "__main__":
    main()