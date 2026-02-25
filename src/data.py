import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_tinystories(split="train", max_samples=None):
    ds = load_dataset("roneneldan/TinyStories", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def prepare_encoder_decoder_pairs(texts, tokenizer, max_enc_len=128, max_dec_len=128, split_ratio=0.3):
    enc_inputs = []
    dec_inputs = []
    dec_targets = []

    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.eos_token_id

    for text in texts:
        tokens = tokenizer.encode(text, truncation=True, max_length=max_enc_len + max_dec_len)
        if len(tokens) < 6:
            continue

        split_point = max(2, int(len(tokens) * split_ratio))

        enc_tokens = tokens[:split_point][:max_enc_len]
        dec_tokens = tokens[split_point:][:max_dec_len]

        if len(dec_tokens) < 2:
            continue

        dec_in = [bos_id] + dec_tokens[:-1]
        dec_tgt = dec_tokens

        enc_padded = enc_tokens + [pad_id] * (max_enc_len - len(enc_tokens))
        dec_in_padded = dec_in + [pad_id] * (max_dec_len - len(dec_in))
        dec_tgt_padded = dec_tgt + [pad_id] * (max_dec_len - len(dec_tgt))

        enc_inputs.append(enc_padded)
        dec_inputs.append(dec_in_padded)
        dec_targets.append(dec_tgt_padded)

    return np.array(enc_inputs), np.array(dec_inputs), np.array(dec_targets)


def get_batches(enc_inputs, dec_inputs, dec_targets, batch_size, shuffle=True):
    n = len(enc_inputs)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, n - batch_size + 1, batch_size):
        batch_idx = indices[i : i + batch_size]
        yield enc_inputs[batch_idx], dec_inputs[batch_idx], dec_targets[batch_idx]
