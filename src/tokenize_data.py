"""Standalone tokenization pipeline: train tokenizer and pre-tokenize all data.

Downloads the tool-calling dataset from HuggingFace (Cactus-Compute/tool-calls),
trains the SentencePiece tokenizer, and tokenizes train + val splits,
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
    LOCAL_UNIFIED_DIR,
    TOKENIZER_DIR,
    _cache_key,
    _save_cache_metadata,
    get_tokenizer,
    load_tool_calls,
    pack_sequences,
    prepare_tool_call_pairs,
    train_tokenizer,
)


_HF_TOKENIZER_REPO = "Cactus-Compute/needle-tokenizer"
_HF_TOKENIZED_REPO = "Cactus-Compute/tokenized-tool-calls"


def _push_to_hf(cache_dir, tokenizer_dir):
    """Upload tokenizer and tokenized data to their respective HuggingFace repos."""
    from huggingface_hub import HfApi

    api = HfApi()

    api.create_repo(_HF_TOKENIZER_REPO, repo_type="dataset", private=True, exist_ok=True)
    print(f"Uploading tokenizer to {_HF_TOKENIZER_REPO} ...")
    api.upload_folder(
        folder_path=tokenizer_dir,
        repo_id=_HF_TOKENIZER_REPO,
        repo_type="dataset",
        allow_patterns=["*.model", "*.vocab"],
    )

    api.create_repo(_HF_TOKENIZED_REPO, repo_type="dataset", private=True, exist_ok=True)
    print(f"Uploading tokenized data to {_HF_TOKENIZED_REPO} ...")
    api.upload_folder(
        folder_path=cache_dir,
        repo_id=_HF_TOKENIZED_REPO,
        repo_type="dataset",
        allow_patterns=["*.npy", "*.json"],
        delete_patterns=["*.npy", "*.json"],  
    )

    print("HuggingFace upload complete.")


def _clear_local_caches():
    """Remove local .data_cache/, tokenizer/, and unified dataset directories."""
    for d in [CACHE_DIR, TOKENIZER_DIR, LOCAL_UNIFIED_DIR]:
        if os.path.exists(d):
            print(f"Removing {d}/ ...")
            shutil.rmtree(d)


def _download_synth_dataset():
    """Download synthesized tool-calling dataset (train + validation) from HuggingFace."""
    from datasets import load_dataset

    print("Downloading dataset from HuggingFace (Cactus-Compute/tool-calls)...")
    for split in ("train", "validation"):
        try:
            ds = load_dataset("Cactus-Compute/tool-calls", split=split, token=True,
                              verification_mode="no_checks")
        except Exception as e:
            raise FileNotFoundError(
                f"Dataset split '{split}' not found on HuggingFace (Cactus-Compute/tool-calls): {e}\n"
                "Run 'python scripts/generate_data.py' first."
            )
        split_dir = os.path.join(LOCAL_UNIFIED_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
        ds.save_to_disk(split_dir)
        print(f"Saved {split}: {len(ds):,} rows to {split_dir}")


def tokenize(args):
    print("=== Clearing existing caches ===")
    _clear_local_caches()

    print("\n=== Downloading dataset from HuggingFace ===")
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
        shuffle_tools = getattr(args, "shuffle_tools", True)
        max_tool_len = getattr(args, "max_tool_len", 256)
        prepare_tool_call_pairs._max_tool_len = max_tool_len
        enc_vl, dec_in_vl, dec_tgt_vl, loss_vl, kept_indices, _ = prepare_tool_call_pairs(
            ds, tokenizer, max_enc_len=max_enc_len, max_dec_len=max_dec_len,
            shuffle_tools=shuffle_tools,
        )
        text_cache_id = _cache_key("toolcall", len(ds), max_enc_len, max_dec_len,
                                   1.0, 1.0, 1.0, shuffle_tools)
        cache_path = os.path.join(CACHE_DIR, text_cache_id)
        pack_sequences(cache_path, enc_vl, dec_in_vl, dec_tgt_vl, loss_vl)

        _save_cache_metadata(split, text_cache_id, len(kept_indices),
                             max_enc_len, max_dec_len, max_tool_len)

    _push_to_hf(CACHE_DIR, TOKENIZER_DIR)

    if os.path.exists(LOCAL_UNIFIED_DIR):
        print(f"\n=== Removing raw dataset ({LOCAL_UNIFIED_DIR}) ===")
        shutil.rmtree(LOCAL_UNIFIED_DIR)

    print("\n=== Tokenization pipeline complete ===")
