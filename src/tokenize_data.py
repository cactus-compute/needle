"""Stage 2 tokenization pipeline for tool-call text data.

Retrains the local SentencePiece tokenizer, optionally updates the shared GCS
tokenizer, and prepares the train + val text tool-call caches used by Stage 2.
The old unified-dataset mel precompute path is intentionally not part of this
pipeline anymore.

Usage:
    needle tokenize                         # full run
    needle tokenize --max-samples 1000      # dev/test
    needle tokenize --overwrite-gcs-tokenizer
    needle tokenize --cleanup               # delete local cache after GCS upload
"""

import os
import shutil
import subprocess

from .data import (
    CACHE_DIR,
    GCS_TOKENIZER_PATH,
    EMILIA_SPEECH_GCS_PREFIX,
    TOKENIZER_DIR,
    _cache_key,
    _save_cache_metadata,
    get_tokenizer,
    load_emilia_speech_metadata,
    load_tool_calls,
    prefetch_emilia_mels,
    prepare_transcription_pairs,
    prepare_tool_call_pairs,
    save_prepared_transcription_data,
    train_tokenizer,
    upload_tokenizer_to_gcs,
)
from .toucan import cache_toucan_examples


def _clear_gcs_tokenizer():
    """Remove the shared tokenizer files."""
    path = GCS_TOKENIZER_PATH + "*"
    print(f"Clearing {path} ...")
    subprocess.run(
        ["gcloud", "storage", "rm", "-r", path],
        capture_output=True, text=True,
    )


def _clear_local_caches():
    """Remove local tokenizer and text caches, but preserve Emilia mel mirrors."""
    if os.path.exists(TOKENIZER_DIR):
        print(f"Removing {TOKENIZER_DIR}/ ...")
        shutil.rmtree(TOKENIZER_DIR)

    if os.path.exists(CACHE_DIR):
        for item in sorted(os.listdir(CACHE_DIR)):
            path = os.path.join(CACHE_DIR, item)
            if item.startswith("emilia_"):
                print(f"Preserving {path}/ (Emilia mel mirror)")
                continue
            if os.path.isdir(path):
                print(f"Removing {path}/ ...")
                shutil.rmtree(path)
            else:
                os.remove(path)


def tokenize(args):
    print("=== Clearing existing local caches ===")
    _clear_local_caches()
    if getattr(args, "overwrite_gcs_tokenizer", False):
        print("\n=== Clearing shared GCS tokenizer ===")
        _clear_gcs_tokenizer()

    print("\n=== Training local tokenizer ===")
    train_tokenizer(max_samples=args.max_samples, force=True)
    if getattr(args, "overwrite_gcs_tokenizer", False):
        upload_tokenizer_to_gcs()
    else:
        print("Skipping shared GCS tokenizer upload")

    tokenizer = get_tokenizer()
    print("\n=== Caching Toucan tool definitions for Stage 1 ===")
    toucan_path = cache_toucan_examples(
        config=args.toucan_config,
        split="train",
        max_samples=getattr(args, "toucan_max_samples", None),
        tokenizer=tokenizer,
        max_text_len=max(getattr(args, "max_enc_len", 256), getattr(args, "max_dec_len", 1024)),
    )
    print(f"Cached Toucan examples to {toucan_path}")

    print("\n=== Tokenizing Emilia transcription data for Stage 1 ===")
    speech_prefix = getattr(args, "speech_gcs_prefix", EMILIA_SPEECH_GCS_PREFIX)
    speech_val_ratio = getattr(args, "speech_val_ratio", 0.01)
    speech_max_samples = getattr(args, "max_speech_samples", None) or args.max_samples
    stage1_mel_uris = []
    stage1_counts = {}
    for split in ("train", "val"):
        rows = load_emilia_speech_metadata(
            split,
            gcs_prefix=speech_prefix,
            max_samples=speech_max_samples,
            val_ratio=speech_val_ratio,
            seed=getattr(args, "split_seed", 42),
        )
        stage1_counts[split] = len(rows)
        prepared = prepare_transcription_pairs(rows, tokenizer, args.max_enc_len, args.max_dec_len)
        cache_id = save_prepared_transcription_data(
            split,
            prepared,
            args.max_enc_len,
            args.max_dec_len,
            gcs_prefix=speech_prefix,
            val_ratio=speech_val_ratio,
            split_max_samples=speech_max_samples,
        )
        stage1_mel_uris.extend(prepared["mel_uris"].tolist())
        print(f"Cached {len(rows):,} Emilia {split} examples ({cache_id})")
    print(f"Stage 1 Emilia rows selected: {stage1_counts['train']:,} train / {stage1_counts['val']:,} val")

    print("\n=== Mirroring Emilia mel files for Stage 1 ===")
    use_rsync = speech_max_samples is None
    mirrored = prefetch_emilia_mels(stage1_mel_uris, gcs_prefix=speech_prefix, use_rsync=use_rsync)
    print(f"Mirrored {mirrored:,} Emilia mel files into local cache")

    print("\n=== Tokenizing Stage 2 text tool-call data ===")
    max_enc_len = getattr(args, "max_enc_len", 256)
    max_dec_len = getattr(args, "max_dec_len", 1024)
    batch_size = getattr(args, "batch_size", 5000)

    for split in ("train", "val"):
        print(f"\n--- {split} split ---")
        ds = load_tool_calls(
            split=split,
            max_samples=args.max_samples,
            shuffle_before_split=getattr(args, "shuffle_before_split", False),
            shuffle_seed=getattr(args, "split_seed", 42),
        )
        _, _, _, _, kept_indices = prepare_tool_call_pairs(
            ds, tokenizer, max_enc_len=max_enc_len, max_dec_len=max_dec_len,
            batch_size=batch_size,
        )
        text_cache_id = _cache_key("toolcall", len(ds), max_enc_len, max_dec_len)

        _save_cache_metadata(split, text_cache_id, None, len(kept_indices),
                             max_enc_len, max_dec_len, None, None,
                             split_max_samples=args.max_samples,
                             shuffle_before_split=getattr(args, "shuffle_before_split", False),
                             split_seed=getattr(args, "split_seed", 42),
                             toucan_cache_path=toucan_path)

    if args.cleanup and os.path.exists(CACHE_DIR):
        print(f"\n=== Cleaning up {CACHE_DIR}/ ===")
        shutil.rmtree(CACHE_DIR)
        print("Done.")

    print("\n=== Tokenization pipeline complete ===")
