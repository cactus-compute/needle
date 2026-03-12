"""Standalone tokenization pipeline: train tokenizer and pre-tokenize all data.

Trains the SentencePiece tokenizer and tokenizes train + val splits,
caching everything locally.

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
    get_tokenizer,
    load_tool_calls,
    prepare_tool_call_pairs,
    train_tokenizer,
)


def _clear_local_caches():
    """Remove local .data_cache/ and tokenizer/ directories."""
    for d in [CACHE_DIR, TOKENIZER_DIR]:
        if os.path.exists(d):
            print(f"Removing {d}/ ...")
            shutil.rmtree(d)


def tokenize(args):
    print("=== Clearing existing caches ===")
    _clear_local_caches()

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

    print("\n=== Tokenization pipeline complete ===")
