import hashlib
import multiprocessing as mp
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import sentencepiece as spm

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".data_cache")
TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tokenizer")
TOKENIZER_PREFIX = os.path.join(TOKENIZER_DIR, "needle")

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2


class NeedleTokenizer:
    """Wrapper around SentencePiece providing the interface the codebase expects."""

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    @property
    def pad_token_id(self):
        return PAD_ID

    @property
    def eos_token_id(self):
        return EOS_ID

    @property
    def bos_token_id(self):
        return BOS_ID

    @property
    def vocab_size(self):
        return self.sp.GetPieceSize()

    def encode(self, text):
        return self.sp.Encode(text, out_type=int)

    def decode(self, ids):
        if isinstance(ids, (list, tuple)) and len(ids) > 0 and isinstance(ids[0], (list, tuple)):
            return [self.sp.Decode(seq) for seq in ids]
        return self.sp.Decode(list(ids))

    def __call__(self, texts, truncation=True, max_length=None, **kwargs):
        all_ids = []
        for text in texts:
            ids = self.sp.Encode(text, out_type=int)
            if truncation and max_length:
                ids = ids[:max_length]
            all_ids.append(ids)
        return {"input_ids": all_ids}


_worker_sp = None
_worker_max_len = None


def _init_worker(model_path, max_length):
    """Initializer for multiprocessing pool — loads SP model once per worker."""
    global _worker_sp, _worker_max_len
    _worker_sp = spm.SentencePieceProcessor()
    _worker_sp.Load(model_path)
    _worker_max_len = max_length


def _tokenize_chunk(texts):
    """Encode a chunk of texts in a worker process."""
    return [_worker_sp.Encode(t, out_type=int)[:_worker_max_len] for t in texts]


def train_tokenizer(vocab_size=8192, max_samples=None):
    """Train a SentencePiece BPE tokenizer on TinyStories."""
    model_path = TOKENIZER_PREFIX + ".model"
    if os.path.exists(model_path):
        print(f"Tokenizer already exists at {model_path}")
        return model_path

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    ds = load_dataset("roneneldan/TinyStories", split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"Training SentencePiece BPE tokenizer (vocab_size={vocab_size}, samples={len(ds):,})...")

    corpus_path = os.path.join(TOKENIZER_DIR, "corpus.txt")
    with open(corpus_path, "w") as f:
        for example in tqdm(ds, desc="Writing corpus"):
            text = example["text"].strip()
            if text:
                f.write(text + "\n")

    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=PAD_ID,
        eos_id=EOS_ID,
        bos_id=BOS_ID,
        unk_id=3,
        byte_fallback=True,
        normalization_rule_name="identity",
        num_threads=os.cpu_count(),
        train_extremely_large_corpus=False,
        minloglevel=2,
    )

    os.remove(corpus_path)
    print(f"Tokenizer saved to {model_path}")
    return model_path


def get_tokenizer(max_samples=None):
    model_path = TOKENIZER_PREFIX + ".model"
    if not os.path.exists(model_path):
        train_tokenizer(max_samples=max_samples)
    return NeedleTokenizer(model_path)


def load_tinystories(split="train", max_samples=None):
    ds = load_dataset("roneneldan/TinyStories", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def _tokenizer_hash():
    """Hash the tokenizer model file to detect retraining."""
    model_path = TOKENIZER_PREFIX + ".model"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    return "none"


def _cache_key(n_samples, max_enc_len, max_dec_len):
    tok_hash = _tokenizer_hash()
    key = f"tinystories_{tok_hash}_{n_samples}_{max_enc_len}_{max_dec_len}"
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

    # --- Parallel tokenization ---
    texts = list(ds["text"])
    num_workers = min(os.cpu_count() or 1, 8)
    model_path = TOKENIZER_PREFIX + ".model"
    chunk_size = max(1, len(texts) // (num_workers * 4))
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    print(f"Tokenizing ({num_workers} workers)...")
    with mp.Pool(num_workers, initializer=_init_worker,
                 initargs=(model_path, max_total)) as pool:
        results = pool.map(_tokenize_chunk, chunks)
    all_tokens = [tok for chunk in results for tok in chunk]

    # --- Pre-filter by length to reduce loop iterations ---
    lengths = np.array([len(t) for t in all_tokens])
    valid_mask = lengths >= 6
    valid_indices = np.where(valid_mask)[0]

    split_points = np.maximum(2, (lengths[valid_indices] * split_ratio).astype(np.int32))
    enc_lens = np.minimum(split_points, max_enc_len)
    dec_lens = np.minimum(lengths[valid_indices] - split_points, max_dec_len)
    keep = dec_lens >= 2

    valid_indices = valid_indices[keep]
    split_points = split_points[keep]
    enc_lens = enc_lens[keep]
    dec_lens = dec_lens[keep]

    n_keep = len(valid_indices)
    enc_inputs = np.full((n_keep, max_enc_len), pad_id, dtype=np.int32)
    dec_inputs = np.full((n_keep, max_dec_len), pad_id, dtype=np.int32)
    dec_targets = np.full((n_keep, max_dec_len), pad_id, dtype=np.int32)

    for i, (idx, sp, el, dl) in enumerate(
        zip(valid_indices, split_points, enc_lens, dec_lens)
    ):
        tokens = all_tokens[idx]
        enc_inputs[i, :el] = tokens[:sp][:max_enc_len]
        dec_inputs[i, 0] = bos_id
        dec_inputs[i, 1:dl] = tokens[sp:][:dl - 1]
        dec_targets[i, :dl] = tokens[sp:][:max_dec_len]

    # Save cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_path + "_enc.npy", enc_inputs)
    np.save(cache_path + "_dec_in.npy", dec_inputs)
    np.save(cache_path + "_dec_tgt.npy", dec_targets)
    print(f"Cached {n_keep:,} pairs to {CACHE_DIR}/{cache_id}")

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
