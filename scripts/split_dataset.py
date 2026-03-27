"""Split the existing Cactus-Compute/tool-calls dataset into train/validation splits.

Takes the last 10k rows as validation, rest as train, and re-uploads as a DatasetDict.

Usage:
    python scripts/split_dataset.py
"""

from datasets import DatasetDict, load_dataset


HF_DATASET_REPO = "Cactus-Compute/tool-calls"
VAL_SIZE = 10000


def main():
    print(f"Downloading full dataset from {HF_DATASET_REPO}...")
    ds = load_dataset(HF_DATASET_REPO, split="train", token=True)
    n = len(ds)
    print(f"Total rows: {n:,}")

    val_size = min(VAL_SIZE, int(n * 0.1))
    train_ds = ds.select(range(n - val_size))
    val_ds = ds.select(range(n - val_size, n))

    print(f"Train: {len(train_ds):,} rows")
    print(f"Validation: {len(val_ds):,} rows")

    ds_dict = DatasetDict({"train": train_ds, "validation": val_ds})

    print(f"\nUploading to {HF_DATASET_REPO}...")
    ds_dict.push_to_hub(HF_DATASET_REPO, token=True)
    print("Done.")


if __name__ == "__main__":
    main()
