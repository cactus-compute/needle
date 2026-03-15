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


_HF_TOKENIZED_REPO = "Cactus-Compute/tool-calls-tokenized"


def _push_to_hf(cache_dir, tokenizer_dir):
    """Upload tokenized data and tokenizer to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(_HF_TOKENIZED_REPO, repo_type="dataset", private=True, exist_ok=True)

    print(f"Uploading tokenized data to {_HF_TOKENIZED_REPO} ...")
    api.upload_folder(
        folder_path=cache_dir,
        repo_id=_HF_TOKENIZED_REPO,
        repo_type="dataset",
        path_in_repo="tokenized_data",
        allow_patterns=["*.npy", "*.json"],
    )

    print(f"Uploading tokenizer to {_HF_TOKENIZED_REPO} ...")
    api.upload_folder(
        folder_path=tokenizer_dir,
        repo_id=_HF_TOKENIZED_REPO,
        repo_type="dataset",
        path_in_repo="tokenizer",
        allow_patterns=["*.model", "*.vocab"],
    )
    print("HuggingFace upload complete.")


def _clear_local_caches():
    """Remove local .data_cache/, tokenizer/, and stale unified dataset directories."""
    for d in [CACHE_DIR, TOKENIZER_DIR, _DISK_UNIFIED_DIR, _SHM_UNIFIED_DIR]:
        if os.path.exists(d):
            print(f"Removing {d}/ ...")
            shutil.rmtree(d)


def _download_synth_dataset():
    """Download synthesized tool-calling dataset from HuggingFace."""
    from datasets import load_dataset

    target = _SHM_UNIFIED_DIR if _shm_available() else _DISK_UNIFIED_DIR
    print(f"Downloading dataset from HuggingFace (Cactus-Compute/tool-calls)...")
    try:
        ds = load_dataset("Cactus-Compute/tool-calls", split="train", token=True)
    except Exception as e:
        raise FileNotFoundError(
            f"Dataset not found on HuggingFace (Cactus-Compute/tool-calls): {e}\n"
            "Run 'python scripts/push_to_hf.py' first."
        )
    import os
    os.makedirs(target, exist_ok=True)
    ds.save_to_disk(target)
    print(f"Saved {len(ds):,} rows to {target}")


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
        w_name = getattr(args, "w_name", 3.0)
        w_value = getattr(args, "w_value", 2.0)
        w_key = getattr(args, "w_key", 1.5)
        shuffle_tools = getattr(args, "shuffle_tools", True)
        max_tool_len = getattr(args, "max_tool_len", 256)
        prepare_tool_call_pairs._max_tool_len = max_tool_len
        _, _, _, _, kept_indices, _ = prepare_tool_call_pairs(
            ds, tokenizer, max_enc_len=max_enc_len, max_dec_len=max_dec_len,
            w_name=w_name, w_value=w_value, w_key=w_key, shuffle_tools=shuffle_tools,
        )
        text_cache_id = _cache_key("toolcall", len(ds), max_enc_len, max_dec_len,
                                   w_name, w_value, w_key, shuffle_tools)

        _save_cache_metadata(split, text_cache_id, len(kept_indices),
                             max_enc_len, max_dec_len)

    _push_to_hf(CACHE_DIR, TOKENIZER_DIR)

    print("\n=== Tokenization pipeline complete ===")
