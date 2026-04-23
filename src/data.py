import hashlib
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm
import sentencepiece as spm


DATASET_SPECS = {
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "name": None,
        "text_key": "text",
        "streaming": False,
    },
    "fineweb_edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "text_key": "text",
        "streaming": True,
    },
}


def _load_text_dataset(dataset_name, split="train", max_samples=None):
    """Return a non-streaming Dataset with a 'text' column, regardless of source."""
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Options: {list(DATASET_SPECS)}")
    spec = DATASET_SPECS[dataset_name]

    if spec["streaming"]:
        ds = load_dataset(spec["path"], name=spec["name"], split=split, streaming=True)
        if max_samples is None:
            raise ValueError(f"max_samples required for streaming dataset '{dataset_name}'")
        texts = []
        for row in tqdm(ds.take(max_samples), total=max_samples, desc=f"Streaming {dataset_name}"):
            t = row[spec["text_key"]]
            if t and t.strip():
                texts.append(t)
        return Dataset.from_dict({"text": texts})

    ds = load_dataset(spec["path"], name=spec["name"], split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    if spec["text_key"] != "text":
        ds = ds.rename_column(spec["text_key"], "text")
    return ds

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


def train_tokenizer(vocab_size=8192, max_samples=None, dataset="tinystories"):
    """Train a SentencePiece BPE tokenizer on the selected dataset."""
    model_path = TOKENIZER_PREFIX + ".model"
    stamp_path = TOKENIZER_PREFIX + ".dataset"
    if os.path.exists(model_path):
        existing = open(stamp_path).read().strip() if os.path.exists(stamp_path) else None
        if existing == dataset:
            print(f"Tokenizer already exists at {model_path} (dataset={dataset})")
            return model_path
        print(
            f"Tokenizer at {model_path} was trained on '{existing}', "
            f"but this run requests '{dataset}'. Retraining."
        )
        os.remove(model_path)
        vocab_file = TOKENIZER_PREFIX + ".vocab"
        if os.path.exists(vocab_file):
            os.remove(vocab_file)

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    tok_samples = max_samples or 200_000
    ds = _load_text_dataset(dataset, split="train", max_samples=tok_samples)

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
    with open(TOKENIZER_PREFIX + ".dataset", "w") as f:
        f.write(dataset)
    print(f"Tokenizer saved to {model_path}")
    return model_path


def get_tokenizer(max_samples=None, dataset="tinystories"):
    model_path = TOKENIZER_PREFIX + ".model"
    stamp_path = TOKENIZER_PREFIX + ".dataset"
    existing = (
        open(stamp_path).read().strip()
        if os.path.exists(model_path) and os.path.exists(stamp_path)
        else None
    )
    if not os.path.exists(model_path) or existing != dataset:
        train_tokenizer(max_samples=max_samples, dataset=dataset)
    return NeedleTokenizer(model_path)


def load_tinystories(split="train", max_samples=None):
    return _load_text_dataset("tinystories", split=split, max_samples=max_samples)


def load_text_dataset(dataset="tinystories", split="train", max_samples=None):
    return _load_text_dataset(dataset, split=split, max_samples=max_samples)


def _tokenizer_hash():
    """Hash the tokenizer model file to detect retraining."""
    model_path = TOKENIZER_PREFIX + ".model"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    return "none"


def _cache_key(dataset, n_samples, max_enc_len, max_dec_len):
    tok_hash = _tokenizer_hash()
    key = f"{dataset}_{tok_hash}_{n_samples}_{max_enc_len}_{max_dec_len}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def prepare_encoder_decoder_pairs(ds, tokenizer, max_enc_len=128, max_dec_len=128, split_ratio=0.3, dataset="tinystories"):

    cache_id = _cache_key(dataset, len(ds), max_enc_len, max_dec_len)
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
