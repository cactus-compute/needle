"""Standalone tokenization pipeline: train tokenizer and pre-tokenize all data.

Trains the SentencePiece tokenizer and tokenizes train + val splits for both
text and voice data, caching everything on GCS. Running fresh overwrites all
existing tokenizer + caches.

Usage:
    needle tokenize                         # full run
    needle tokenize --max-samples 1000      # dev/test
    needle tokenize --cleanup               # delete local cache after GCS upload
"""

import os
import shutil
import subprocess

from .data import (
    CACHE_DIR,
    GCS_CACHE_PATH,
    GCS_TOKENIZER_PATH,
    TOKENIZER_DIR,
    _cache_key,
    _save_cache_metadata,
    get_tokenizer,
    load_tool_calls,
    precompute_mels,
    prepare_tool_call_pairs,
    train_tokenizer,
    upload_tokenizer_to_gcs,
)


def _clear_gcs_caches():
    """Remove existing GCS cache and tokenizer files."""
    for path in [GCS_CACHE_PATH + "/*", GCS_TOKENIZER_PATH + "*"]:
        print(f"Clearing {path} ...")
        subprocess.run(
            ["gcloud", "storage", "rm", "-r", path],
            capture_output=True, text=True,
        )


def _clear_local_caches():
    """Remove local .data_cache/ and tokenizer/ directories."""
    for d in [CACHE_DIR, TOKENIZER_DIR]:
        if os.path.exists(d):
            print(f"Removing {d}/ ...")
            shutil.rmtree(d)


def tokenize(args):
    print("=== Clearing existing caches ===")
    _clear_gcs_caches()
    _clear_local_caches()

    print("\n=== Training tokenizer ===")
    train_tokenizer(max_samples=args.max_samples, force=True)
    upload_tokenizer_to_gcs()

    tokenizer = get_tokenizer()

    print("\n=== Tokenizing text data + precomputing mels ===")
    max_enc_len = getattr(args, "max_enc_len", 256)
    max_dec_len = getattr(args, "max_dec_len", 1024)
    n_mels = getattr(args, "n_mels", 80)
    max_mel_len = getattr(args, "max_mel_len", 1024)
    batch_size = getattr(args, "batch_size", 5000)

    for split in ("train", "val"):
        print(f"\n--- {split} split ---")
        ds, global_indices = load_tool_calls(
            split=split,
            max_samples=args.max_samples,
            return_global_indices=True,
            shuffle_before_split=getattr(args, "shuffle_before_split", False),
            shuffle_seed=getattr(args, "split_seed", 42),
        )
        _, _, _, _, kept_indices = prepare_tool_call_pairs(
            ds, tokenizer, max_enc_len=max_enc_len, max_dec_len=max_dec_len,
            batch_size=batch_size,
        )
        text_cache_id = _cache_key("toolcall", len(ds), max_enc_len, max_dec_len)

        mel_cache_id = precompute_mels(
            global_indices[kept_indices], n_mels=n_mels, max_mel_len=max_mel_len,
            cache_id_prefix=split, batch_size=batch_size,
        )

        _save_cache_metadata(split, text_cache_id, mel_cache_id, len(kept_indices),
                             max_enc_len, max_dec_len, n_mels, max_mel_len,
                             split_max_samples=args.max_samples,
                             shuffle_before_split=getattr(args, "shuffle_before_split", False),
                             split_seed=getattr(args, "split_seed", 42))

    if args.cleanup and os.path.exists(CACHE_DIR):
        print(f"\n=== Cleaning up {CACHE_DIR}/ ===")
        shutil.rmtree(CACHE_DIR)
        print("Done.")

    print("\n=== Tokenization pipeline complete ===")
