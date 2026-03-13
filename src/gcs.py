"""GCS sync utilities for persisting datasets to gs://cactus-dataset.

Uploads and downloads three artifact types:
  - synth_tool_calls/  synthesized tool-calling dataset (.arrow files)
  - tokenized_data/    pre-tokenized .npy arrays + metadata JSONs
  - tokenizer/         SentencePiece model + vocab

Uses `gcloud storage` CLI (authenticated via `gcloud auth login`).
"""

import os
import subprocess

BUCKET = "cactus-dataset"

_SYNTH_DATA_PREFIX = "synth_tool_calls"
_TOKENIZED_DATA_PREFIX = "tokenized_data"
_TOKENIZER_PREFIX = "tokenizer"


def upload_directory(local_dir, gcs_prefix):
    """Upload all files in local_dir to gs://BUCKET/gcs_prefix/."""
    if not os.path.isdir(local_dir):
        print(f"[GCS] Skipping upload: {local_dir} does not exist")
        return
    dest = f"gs://{BUCKET}/{gcs_prefix}/"
    print(f"[GCS] Uploading {local_dir} -> {dest}")
    subprocess.run(
        ["gcloud", "storage", "cp", "-r", f"{local_dir}/*", dest],
        check=True,
    )
    print(f"[GCS] Upload complete: {dest}")


def download_directory(gcs_prefix, local_dir):
    """Download gs://BUCKET/gcs_prefix/ to local_dir. Returns True if successful."""
    src = f"gs://{BUCKET}/{gcs_prefix}/"
    os.makedirs(local_dir, exist_ok=True)
    print(f"[GCS] Downloading {src} -> {local_dir}")
    result = subprocess.run(
        ["gcloud", "storage", "cp", "-r", f"{src}*", local_dir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        if "not found" in result.stderr.lower() or "CommandException" in result.stderr:
            print(f"[GCS] Nothing found at {src}")
            return False
        raise RuntimeError(f"gcloud storage download failed: {result.stderr}")
    print(f"[GCS] Download complete: {local_dir}")
    return True


# ── Convenience wrappers ──


def download_synth_data(local_dir):
    """Download synthesized tool-calling dataset from GCS. Returns True if successful."""
    return download_directory(_SYNTH_DATA_PREFIX, local_dir)


def upload_tokenized_data(cache_dir):
    """Upload tokenized .npy files and metadata to GCS."""
    upload_directory(cache_dir, _TOKENIZED_DATA_PREFIX)


def download_tokenized_data(cache_dir):
    """Download tokenized data from GCS. Returns True if successful."""
    return download_directory(_TOKENIZED_DATA_PREFIX, cache_dir)


def upload_tokenizer(tokenizer_dir):
    """Upload tokenizer model and vocab to GCS."""
    upload_directory(tokenizer_dir, _TOKENIZER_PREFIX)


def download_tokenizer(tokenizer_dir):
    """Download tokenizer from GCS. Returns True if successful."""
    return download_directory(_TOKENIZER_PREFIX, tokenizer_dir)
