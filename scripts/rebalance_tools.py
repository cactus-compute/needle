#!/usr/bin/env python3
"""Rebalance tool count distribution by trimming over-represented bins.

- 2/3 of 3-tool examples → trimmed to 1 or 2 tools
- 2/3 of 10-tool examples → trimmed to 5-9 tools

Keeps called tools + random subset of uncalled tools to reach target count.
Uploads the rebalanced dataset back to HuggingFace.

Usage:
    python scripts/rebalance_tools.py
    python scripts/rebalance_tools.py --dry-run
"""

import argparse
import json
import os
import random
from collections import Counter

from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm

HF_REPO = "Cactus-Compute/tool-calls"


def _get_called_names(answers_str):
    """Extract set of called tool names from answers JSON."""
    try:
        calls = json.loads(answers_str)
    except (ValueError, TypeError):
        return set()
    if not isinstance(calls, list):
        return set()
    return {c["name"] for c in calls if isinstance(c, dict) and "name" in c}


def _trim_tools(tools_str, called_names, target_n, rng):
    """Trim tools list to target_n, keeping all called tools + random fillers."""
    try:
        tools = json.loads(tools_str)
    except (ValueError, TypeError):
        return tools_str
    if not isinstance(tools, list):
        return tools_str

    # Separate called vs uncalled
    called = [t for t in tools if isinstance(t, dict) and t.get("name") in called_names]
    uncalled = [t for t in tools if isinstance(t, dict) and t.get("name") not in called_names]

    # If we need fewer tools than called tools, can't trim — skip
    if target_n < len(called):
        return None

    # Pick random uncalled fillers to reach target
    n_fillers = target_n - len(called)
    if n_fillers > len(uncalled):
        return None  # not enough uncalled tools

    fillers = rng.sample(uncalled, n_fillers)
    new_tools = called + fillers
    rng.shuffle(new_tools)
    return json.dumps(new_tools, separators=(",", ":"))


def rebalance(dry_run=False):
    print(f"Loading dataset from {HF_REPO}...")
    ds = load_dataset(HF_REPO, split="train", token=True)
    print(f"Loaded {len(ds):,} examples")

    rng = random.Random(42)

    # Categorize by tool count
    indices_by_n = {}
    for i, ex in enumerate(tqdm(ds, desc="Categorizing")):
        try:
            tools = json.loads(ex["tools"])
            n = len(tools) if isinstance(tools, list) else 0
        except (ValueError, TypeError):
            n = 0
        indices_by_n.setdefault(n, []).append(i)

    print("\nBefore rebalancing:")
    for n in sorted(indices_by_n.keys()):
        if n <= 10:
            print(f"  {n:>2} tools: {len(indices_by_n[n]):>10,}")

    # Plan: which indices to modify
    modify_3 = []  # (index, target_n)
    modify_10 = []  # (index, target_n)

    # 2/3 of 3-tool → trim to 1 or 2
    if 3 in indices_by_n:
        pool = list(indices_by_n[3])
        rng.shuffle(pool)
        n_modify = len(pool) * 2 // 3
        for idx in pool[:n_modify]:
            target = rng.choice([1, 2])
            modify_3.append((idx, target))

    # 2/3 of 10-tool → trim to 5-9
    if 10 in indices_by_n:
        pool = list(indices_by_n[10])
        rng.shuffle(pool)
        n_modify = len(pool) * 2 // 3
        for idx in pool[:n_modify]:
            target = rng.randint(5, 9)
            modify_10.append((idx, target))

    all_modifications = modify_3 + modify_10
    print(f"\nModifications planned:")
    print(f"  3-tool → 1 or 2: {len(modify_3):,}")
    print(f"  10-tool → 5-9:   {len(modify_10):,}")
    print(f"  Total:            {len(all_modifications):,}")

    if dry_run:
        # Preview target distribution
        target_counts = Counter()
        for n in sorted(indices_by_n.keys()):
            target_counts[n] = len(indices_by_n[n])
        for idx, target in all_modifications:
            ex = ds[idx]
            try:
                tools = json.loads(ex["tools"])
                orig_n = len(tools)
            except:
                continue
            target_counts[orig_n] -= 1
            target_counts[target] += 1

        print("\nProjected distribution:")
        for n in sorted(target_counts.keys()):
            if n <= 10 and target_counts[n] > 0:
                total = sum(target_counts.values())
                print(f"  {n:>2} tools: {target_counts[n]:>10,} ({target_counts[n]/total*100:5.1f}%)")
        return

    # Apply modifications
    mod_map = {idx: target for idx, target in all_modifications}
    modified_indices = set(mod_map.keys())

    new_tools_col = []
    skipped = 0
    for i in tqdm(range(len(ds)), desc="Applying"):
        ex = ds[i]
        if i in modified_indices:
            called = _get_called_names(ex["answers"])
            new_tools = _trim_tools(ex["tools"], called, mod_map[i], rng)
            if new_tools is None:
                new_tools_col.append(ex["tools"])  # keep original
                skipped += 1
            else:
                new_tools_col.append(new_tools)
        else:
            new_tools_col.append(ex["tools"])

    if skipped:
        print(f"  Skipped {skipped} examples (couldn't trim)")

    # Build new dataset
    new_ds = ds.remove_columns(["tools"]).add_column("tools", new_tools_col)
    # Reorder columns to match original
    new_ds = new_ds.select_columns(ds.column_names)

    # Verify
    print("\nAfter rebalancing:")
    verify_counts = Counter()
    for ex in tqdm(new_ds, desc="Verifying"):
        try:
            tools = json.loads(ex["tools"])
            n = len(tools) if isinstance(tools, list) else 0
        except:
            n = 0
        verify_counts[n] += 1
    for n in sorted(verify_counts.keys()):
        if n <= 10:
            total = sum(verify_counts.values())
            print(f"  {n:>2} tools: {verify_counts[n]:>10,} ({verify_counts[n]/total*100:5.1f}%)")

    # Save locally then upload via HfApi to avoid push_to_hub dataset card bug
    import tempfile
    from huggingface_hub import HfApi, CommitOperationDelete

    local_dir = tempfile.mkdtemp(prefix="rebalanced_")
    print(f"\nSaving to {local_dir}...")
    new_ds.to_parquet(os.path.join(local_dir, "train.parquet"))

    api = HfApi()

    # Delete old train shards
    files = api.list_repo_files(HF_REPO, repo_type="dataset", token=True)
    old_train = [f for f in files if f.startswith("data/train-")]
    if old_train:
        print(f"Deleting {len(old_train)} old train shards...")
        ops = [CommitOperationDelete(path_in_repo=f) for f in old_train]
        api.create_commit(
            repo_id=HF_REPO, repo_type="dataset", operations=ops,
            commit_message="Remove old train shards", token=True,
        )

    # Upload new train data
    print(f"Uploading rebalanced train split...")
    api.upload_file(
        path_or_fileobj=os.path.join(local_dir, "train.parquet"),
        path_in_repo="data/train-00000-of-00001.parquet",
        repo_id=HF_REPO, repo_type="dataset", token=True,
        commit_message="Upload rebalanced train data",
    )

    # Clean up
    import shutil
    shutil.rmtree(local_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview without modifying")
    args = parser.parse_args()
    rebalance(dry_run=args.dry_run)
