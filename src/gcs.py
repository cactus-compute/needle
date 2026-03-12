"""GCS sync utilities for persisting datasets to gs://cactus-dataset.

Uploads and downloads three artifact types:
  - raw_data/          unified HuggingFace dataset (.arrow files)
  - tokenized_data/    pre-tokenized .npy arrays + metadata JSONs
  - tokenizer/         SentencePiece model + vocab
"""

import os

import gcsfs

BUCKET = "cactus-dataset"

_RAW_DATA_PREFIX = "raw_data"
_TOKENIZED_DATA_PREFIX = "tokenized_data"
_TOKENIZER_PREFIX = "tokenizer"


def _get_fs():
    return gcsfs.GCSFileSystem()


def upload_directory(local_dir, gcs_prefix):
    """Upload all files in local_dir to gs://BUCKET/gcs_prefix/."""
    if not os.path.isdir(local_dir):
        print(f"[GCS] Skipping upload: {local_dir} does not exist")
        return
    fs = _get_fs()
    dest = f"{BUCKET}/{gcs_prefix}"
    count = 0
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, local_dir)
            remote_path = f"{dest}/{rel}"
            fs.put(local_path, remote_path)
            count += 1
    print(f"[GCS] Uploaded {count} files to gs://{dest}/")


def download_directory(gcs_prefix, local_dir):
    """Download gs://BUCKET/gcs_prefix/ to local_dir. Returns True if files were downloaded."""
    fs = _get_fs()
    src = f"{BUCKET}/{gcs_prefix}"
    try:
        remote_files = fs.ls(src, detail=False)
    except FileNotFoundError:
        return False
    if not remote_files:
        return False
    os.makedirs(local_dir, exist_ok=True)
    count = 0
    for remote_path in fs.find(src):
        rel = os.path.relpath(remote_path, src)
        local_path = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        fs.get(remote_path, local_path)
        count += 1
    print(f"[GCS] Downloaded {count} files from gs://{src}/ to {local_dir}/")
    return count > 0


# ── Convenience wrappers ──


def upload_raw_data(local_dir):
    """Upload unified dataset to GCS."""
    print("[GCS] Uploading raw dataset...")
    upload_directory(local_dir, _RAW_DATA_PREFIX)


def download_raw_data(local_dir):
    """Download unified dataset from GCS. Returns True if successful."""
    print("[GCS] Downloading raw dataset...")
    return download_directory(_RAW_DATA_PREFIX, local_dir)


def upload_tokenized_data(cache_dir):
    """Upload tokenized .npy files and metadata to GCS."""
    print("[GCS] Uploading tokenized data...")
    upload_directory(cache_dir, _TOKENIZED_DATA_PREFIX)


def download_tokenized_data(cache_dir):
    """Download tokenized data from GCS. Returns True if successful."""
    print("[GCS] Downloading tokenized data...")
    return download_directory(_TOKENIZED_DATA_PREFIX, cache_dir)


def upload_tokenizer(tokenizer_dir):
    """Upload tokenizer model and vocab to GCS."""
    print("[GCS] Uploading tokenizer...")
    upload_directory(tokenizer_dir, _TOKENIZER_PREFIX)


def download_tokenizer(tokenizer_dir):
    """Download tokenizer from GCS. Returns True if successful."""
    print("[GCS] Downloading tokenizer...")
    return download_directory(_TOKENIZER_PREFIX, tokenizer_dir)
