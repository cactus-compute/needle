import hashlib
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".data_cache")


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_tinystories(split="train", max_samples=None):
    ds = load_dataset("roneneldan/TinyStories", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def _cache_key(n_samples, max_enc_len, max_dec_len):
    key = f"tinystories_{n_samples}_{max_enc_len}_{max_dec_len}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def prepare_encoder_decoder_pairs(ds, tokenizer, max_enc_len=128, max_dec_len=128, split_ratio=0.3):
    
    cache_id = _cache_key(len(ds), max_enc_len, max_dec_len)
    cache_path = os.path.join(CACHE_DIR, cache_id)
    if os.path.exists(cache_path + "_enc.npy"):
        print(f"Loading cached data ({cache_id})...")
        return (
            np.load(cache_path + "_enc.npy"),
            np.load(cache_path + "_dec_in.npy"),
            np.load(cache_path + "_dec_tgt.npy"),
        )

    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.eos_token_id
    max_total = max_enc_len + max_dec_len

    print("Tokenizing...")
    texts = list(ds["text"])
    all_tokens = tokenizer(
        texts, truncation=True, max_length=max_total,
        return_attention_mask=False, return_token_type_ids=False,
    )["input_ids"]

    n = len(all_tokens)
    enc_inputs = np.full((n, max_enc_len), pad_id, dtype=np.int32)
    dec_inputs = np.full((n, max_dec_len), pad_id, dtype=np.int32)
    dec_targets = np.full((n, max_dec_len), pad_id, dtype=np.int32)

    kept = 0
    for tokens in tqdm(all_tokens, desc="Building pairs"):
        if len(tokens) < 6:
            continue

        split_point = max(2, int(len(tokens) * split_ratio))
        enc_tokens = tokens[:split_point][:max_enc_len]
        dec_tokens = tokens[split_point:][:max_dec_len]

        if len(dec_tokens) < 2:
            continue

        enc_len = len(enc_tokens)
        dec_len = len(dec_tokens)
        enc_inputs[kept, :enc_len] = enc_tokens
        dec_inputs[kept, 0] = bos_id
        dec_inputs[kept, 1:dec_len] = dec_tokens[:dec_len - 1]
        dec_targets[kept, :dec_len] = dec_tokens
        kept += 1

    enc_inputs = enc_inputs[:kept]
    dec_inputs = dec_inputs[:kept]
    dec_targets = dec_targets[:kept]

    # Save cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_path + "_enc.npy", enc_inputs)
    np.save(cache_path + "_dec_in.npy", dec_inputs)
    np.save(cache_path + "_dec_tgt.npy", dec_targets)
    print(f"Cached data to {CACHE_DIR}/{cache_id}")

    return enc_inputs, dec_inputs, dec_targets


def get_batches(enc_inputs, dec_inputs, dec_targets, batch_size, shuffle=True):
    n = len(enc_inputs)
    if shuffle:
        perm = np.random.permutation(n)
        enc_inputs = enc_inputs[perm]
        dec_inputs = dec_inputs[perm]
        dec_targets = dec_targets[perm]

    for i in range(0, n - batch_size + 1, batch_size):
        yield enc_inputs[i : i + batch_size], dec_inputs[i : i + batch_size], dec_targets[i : i + batch_size]
