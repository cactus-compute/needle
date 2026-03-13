"""Standalone tokenization pipeline: train tokenizer and pre-tokenize all data.

Downloads the synthesized tool-calling dataset from GCS
(gs://cactus-dataset/synth_tool_calls/), trains the SentencePiece tokenizer,
and tokenizes train + val splits, caching everything locally.

Usage:
    needle tokenize                         # full run
    needle tokenize --max-samples 1000      # dev/test
"""

import os
import shutil

from .data import (
    CACHE_DIR,
    DEFAULT_MAX_DEC_LEN,
    DEFAULT_MAX_ENC_LEN,
    TOKENIZER_DIR,
    _cache_key,
    _save_cache_metadata,
    _DISK_UNIFIED_DIR,
    _SHM_UNIFIED_DIR,
    _shm_available,
    get_tokenizer,
    load_tool_calls,
    prepare_tool_call_pairs,
    train_tokenizer,
)


def _clear_local_caches():
    """Remove local .data_cache/, tokenizer/, and stale unified dataset directories."""
    for d in [CACHE_DIR, TOKENIZER_DIR, _DISK_UNIFIED_DIR, _SHM_UNIFIED_DIR]:
        if os.path.exists(d):
            print(f"Removing {d}/ ...")
            shutil.rmtree(d)


def _download_synth_dataset():
    """Download synthesized tool-calling dataset from GCS."""
    from .gcs import download_synth_data

    target = _SHM_UNIFIED_DIR if _shm_available() else _DISK_UNIFIED_DIR
    print(f"Downloading synth dataset to {target} ...")
    if not download_synth_data(target):
        raise FileNotFoundError(
            "Synth dataset not found at gs://cactus-dataset/synth_tool_calls/. "
            "Run 'python scripts/synthesize_tools_data.py' first."
        )


def tokenize(args):
    print("=== Clearing existing caches ===")
    _clear_local_caches()

    print("\n=== Downloading synth dataset from GCS ===")
    _download_synth_dataset()

    print("\n=== Training tokenizer ===")
    train_tokenizer(max_samples=args.max_samples, force=True)

    tokenizer = get_tokenizer()

    print("\n=== Tokenizing text data ===")
    max_enc_len = getattr(args, "max_enc_len", DEFAULT_MAX_ENC_LEN)
    max_dec_len = getattr(args, "max_dec_len", DEFAULT_MAX_DEC_LEN)

    for split in ("train", "val"):
        print(f"\n--- {split} split ---")
        ds, global_indices = load_tool_calls(
            split=split,
            max_samples=args.max_samples,
            return_global_indices=True,
        )
        w_name = getattr(args, "w_name", 3.0)
        w_value = getattr(args, "w_value", 2.0)
        w_key = getattr(args, "w_key", 1.5)
        shuffle_tools = getattr(args, "shuffle_tools", True)
        _, _, _, _, kept_indices, _ = prepare_tool_call_pairs(
            ds, tokenizer, max_enc_len=max_enc_len, max_dec_len=max_dec_len,
            w_name=w_name, w_value=w_value, w_key=w_key, shuffle_tools=shuffle_tools,
        )
        text_cache_id = _cache_key("toolcall", len(ds), max_enc_len, max_dec_len,
                                   w_name, w_value, w_key, shuffle_tools)

        _save_cache_metadata(split, text_cache_id, len(kept_indices),
                             max_enc_len, max_dec_len)

    from .gcs import upload_tokenized_data, upload_tokenizer
    upload_tokenizer(TOKENIZER_DIR)
    upload_tokenized_data(CACHE_DIR)

    print("\n=== Tokenization pipeline complete ===")
