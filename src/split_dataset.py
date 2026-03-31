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

HF_DATASET_REPO = "Cactus-Compute/tool-calls"
VAL_PER_SOURCE = {
    "gemini-3.1-flash-lite": 5000,
    "xlam": 2500,
    "xlam-translated": 2500,
}
SEED = 42


def main(args=None):
    val_per_source = dict(VAL_PER_SOURCE)
    if args is not None and getattr(args, "val_per_source", None) is not None:
        # CLI override applies uniformly to all sources
        val_per_source = {k: args.val_per_source for k in val_per_source}

    print(f"Downloading full dataset from {HF_DATASET_REPO} (train split only)...")
    from .data import download_hf_split
    ds = download_hf_split("train", HF_DATASET_REPO)
    n = len(ds)
    print(f"Total rows: {n:,}")

    # Stratified split: equal samples per source for balanced validation
    from collections import Counter
    source_counts = Counter(ds["source"])
    print(f"\nSource distribution:")
    for src, cnt in source_counts.most_common():
        print(f"  {src}: {cnt:,}")

    # Group indices by source
    source_indices = {}
    for i, src in enumerate(ds["source"]):
        source_indices.setdefault(src, []).append(i)

    import random
    rng = random.Random(SEED)

    val_indices = []
    for src, indices in sorted(source_indices.items()):
        rng.shuffle(indices)
        take = min(val_per_source.get(src, 0), len(indices))
        val_indices.extend(indices[:take])
        print(f"  {src}: taking {take} for validation")

    val_set = set(val_indices)
    train_indices = [i for i in range(n) if i not in val_set]

    train_ds = ds.select(train_indices)
    val_ds = ds.select(val_indices)

    print(f"\nTrain: {len(train_ds):,} rows")
    print(f"Validation: {len(val_ds):,} rows")

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
