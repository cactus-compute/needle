"""Split the existing Cactus-Compute/tool-calls dataset into train/validation splits.

Shuffles the full dataset with a fixed seed, then takes 10k rows as validation.
This ensures validation is representative of all data sources, not just the latest append.
Uploads via HfApi to avoid the push_to_hub dataset card bug.

Usage:
    python scripts/split_dataset.py
"""

import os
import shutil
import tempfile

from datasets import load_dataset


HF_DATASET_REPO = "Cactus-Compute/tool-calls"
VAL_SIZE = 10000
SEED = 42


def main():
    print(f"Downloading full dataset from {HF_DATASET_REPO}...")
    ds = load_dataset(HF_DATASET_REPO, split="train", token=True)
    n = len(ds)
    print(f"Total rows: {n:,}")

    # Shuffle with fixed seed so validation is representative of all sources
    ds = ds.shuffle(seed=SEED)

    val_size = min(VAL_SIZE, int(n * 0.1))
    train_ds = ds.select(range(n - val_size))
    val_ds = ds.select(range(n - val_size, n))

    print(f"Train: {len(train_ds):,} rows")
    print(f"Validation: {len(val_ds):,} rows")

    # Show source distribution in validation
    from collections import Counter
    val_sources = Counter(val_ds["source"])
    print(f"\nValidation source distribution:")
    for src, cnt in val_sources.most_common():
        print(f"  {src}: {cnt} ({cnt/len(val_ds)*100:.1f}%)")

    # Upload via HfApi (avoids push_to_hub dataset card bug)
    from huggingface_hub import HfApi, CommitOperationDelete

    api = HfApi()
    api.create_repo(HF_DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)

    parquet_dir = tempfile.mkdtemp(prefix="needle_split_")
    print(f"\nExporting to parquet...")
    train_ds.to_parquet(os.path.join(parquet_dir, "train.parquet"))
    val_ds.to_parquet(os.path.join(parquet_dir, "validation.parquet"))

    # Delete old data shards
    files = api.list_repo_files(HF_DATASET_REPO, repo_type="dataset", token=True)
    old_shards = [f for f in files if f.startswith("data/")]
    if old_shards:
        print(f"Deleting {len(old_shards)} old shards...")
        ops = [CommitOperationDelete(path_in_repo=f) for f in old_shards]
        api.create_commit(
            repo_id=HF_DATASET_REPO, repo_type="dataset", operations=ops,
            commit_message="Remove old shards before split upload", token=True,
        )

    print(f"Uploading train split ({len(train_ds):,} rows)...")
    api.upload_file(
        path_or_fileobj=os.path.join(parquet_dir, "train.parquet"),
        path_in_repo="data/train-00000-of-00001.parquet",
        repo_id=HF_DATASET_REPO, repo_type="dataset", token=True,
        commit_message=f"Upload train split ({len(train_ds):,} rows)",
    )

    print(f"Uploading validation split ({len(val_ds):,} rows)...")
    api.upload_file(
        path_or_fileobj=os.path.join(parquet_dir, "validation.parquet"),
        path_in_repo="data/validation-00000-of-00001.parquet",
        repo_id=HF_DATASET_REPO, repo_type="dataset", token=True,
        commit_message=f"Upload validation split ({len(val_ds):,} rows)",
    )

    shutil.rmtree(parquet_dir)
    print("Done.")


if __name__ == "__main__":
    main()
