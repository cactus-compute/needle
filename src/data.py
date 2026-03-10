import hashlib
import json as _json
import multiprocessing as mp
import os
import queue
import threading
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import numpy as np
from datasets import Audio, load_from_disk
from tqdm import tqdm
import sentencepiece as spm

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".data_cache")
TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tokenizer")
TOKENIZER_PREFIX = os.path.join(TOKENIZER_DIR, "needle")
LOCAL_UNIFIED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tool_calls_unified")
GCS_DATASET_PATH = "gs://cactus-dataset/tool_calls"

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2  # reserved for SentencePiece, unused at runtime (EOS_ID serves as SOS)
UNK_ID = 3
TOOL_CALL_ID = 4
TRANSCRIBE_ID = 5

_unified_dataset_cache = None


def _load_unified_dataset():
    """Load the unified dataset from GCS (via gcsfs) or local fallback.

    Caches the result in memory after first load.
    Uses soundfile as the audio decoding backend.
    """
    global _unified_dataset_cache
    if _unified_dataset_cache is not None:
        return _unified_dataset_cache

    if os.path.exists(LOCAL_UNIFIED_DIR) and any(
        f.endswith(".arrow") for f in os.listdir(LOCAL_UNIFIED_DIR)
    ):
        try:
            ds = load_from_disk(LOCAL_UNIFIED_DIR)
            print(f"Loaded unified dataset from {LOCAL_UNIFIED_DIR} ({len(ds)} rows)")
            _unified_dataset_cache = _set_audio_backend(ds)
            return _unified_dataset_cache
        except Exception:
            print(f"Local dataset at {LOCAL_UNIFIED_DIR} is incomplete, trying GCS...")

    try:
        ds = load_from_disk(GCS_DATASET_PATH)
        print(f"Loaded unified dataset from GCS ({len(ds)} rows)")
        _unified_dataset_cache = _set_audio_backend(ds)
        return _unified_dataset_cache
    except Exception as e:
        gcs_err = e

    raise FileNotFoundError(
        f"Unified dataset not found. Run scripts/build_dataset.py first, "
        f"or ensure GCS path {GCS_DATASET_PATH} is accessible (pip install gcsfs). "
        f"GCS error: {gcs_err}"
    )


def _set_audio_backend(ds):
    """Disable automatic audio decoding to avoid torchcodec dependency.

    Audio is decoded manually via soundfile in load_tool_call_audio().
    """
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))
    return ds


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
    def tool_call_token_id(self):
        return TOOL_CALL_ID

    @property
    def transcribe_token_id(self):
        return TRANSCRIBE_ID

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


def train_tokenizer(vocab_size=8192, max_samples=None, force=False):
    """Train a SentencePiece BPE tokenizer on tool-calling corpus."""
    model_path = TOKENIZER_PREFIX + ".model"
    if os.path.exists(model_path) and not force:
        print(f"Tokenizer already exists at {model_path}")
        return model_path

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    ds = _load_unified_dataset()
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"Training SentencePiece BPE tokenizer (vocab_size={vocab_size}, samples={len(ds):,})...")

    corpus_path = os.path.join(TOKENIZER_DIR, "corpus.txt")
    with open(corpus_path, "w") as f:
        for example in tqdm(ds, desc="Writing corpus"):
            for field in ("query", "tools", "answers"):
                text = example[field].strip()
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
        unk_id=UNK_ID,
        user_defined_symbols=["<tool_call>", "<transcribe>"],
        byte_fallback=True,
        normalization_rule_name="identity",
        num_threads=os.cpu_count(),
        train_extremely_large_corpus=False,
        minloglevel=2,
    )

    os.remove(corpus_path)
    print(f"Tokenizer saved to {model_path}")
    return model_path


GCS_TOKENIZER_PATH = "gs://cactus-dataset/tokenizer/"


def upload_tokenizer_to_gcs():
    """Upload local tokenizer files to GCS."""
    import subprocess
    import glob as globmod
    tok_files = globmod.glob(os.path.join(TOKENIZER_DIR, "*"))
    if tok_files:
        subprocess.run(
            ["gcloud", "storage", "cp"] + tok_files + [GCS_TOKENIZER_PATH],
            capture_output=True, text=True,
        )
        print(f"Uploaded tokenizer to {GCS_TOKENIZER_PATH}")


def get_tokenizer(max_samples=None):
    model_path = TOKENIZER_PREFIX + ".model"
    if not os.path.exists(model_path):
        # Try downloading from GCS first
        if _download_tokenizer_from_gcs():
            return NeedleTokenizer(model_path)
        train_tokenizer(max_samples=max_samples)
    return NeedleTokenizer(model_path)


def _download_tokenizer_from_gcs():
    """Download tokenizer files from GCS. Returns True on success."""
    import subprocess
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    result = subprocess.run(
        ["gcloud", "storage", "cp", "-r", GCS_TOKENIZER_PATH + "*", TOKENIZER_DIR + "/"],
        capture_output=True, text=True,
    )
    model_path = TOKENIZER_PREFIX + ".model"
    if result.returncode == 0 and os.path.exists(model_path):
        print(f"Downloaded tokenizer from {GCS_TOKENIZER_PATH}")
        return True
    return False


GCS_CACHE_PATH = "gs://cactus-dataset/cache"


def _gcs_cache_download(cache_id, suffixes):
    """Try downloading cached files from GCS. Returns True if all found."""
    import subprocess
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, cache_id)
    srcs = [f"{GCS_CACHE_PATH}/{cache_id}{s}" for s in suffixes]
    subprocess.run(
        ["gcloud", "storage", "cp"] + srcs + [CACHE_DIR + "/"],
        capture_output=True, text=True,
    )
    return all(os.path.exists(cache_path + s) for s in suffixes)


def _gcs_cache_upload(cache_id, suffixes):
    """Upload cached files to GCS."""
    import subprocess
    cache_path = os.path.join(CACHE_DIR, cache_id)
    files = [cache_path + s for s in suffixes if os.path.exists(cache_path + s)]
    if files:
        subprocess.run(
            ["gcloud", "storage", "cp"] + files + [GCS_CACHE_PATH + "/"],
            capture_output=True, text=True,
        )


def _gcs_download_shards(cache_id, n_shards, shard_suffixes):
    """Download sharded .npy files from GCS. Returns True if all found."""
    import subprocess
    os.makedirs(CACHE_DIR, exist_ok=True)
    srcs = []
    for suffix in shard_suffixes:
        for i in range(n_shards):
            srcs.append(f"{GCS_CACHE_PATH}/{cache_id}{suffix}_{i:05d}.npy")
    if not srcs:
        return True
    subprocess.run(
        ["gcloud", "storage", "cp"] + srcs + [CACHE_DIR + "/"],
        capture_output=True, text=True,
    )
    for suffix in shard_suffixes:
        for i in range(n_shards):
            if not os.path.exists(os.path.join(CACHE_DIR, f"{cache_id}{suffix}_{i:05d}.npy")):
                return False
    return True


def load_tool_calls(split="train", max_samples=None, return_global_indices=False):
    """Load tool-calling dataset, splitting 90/10 for train/val.

    If return_global_indices is True, also return a numpy array mapping each
    split-local row position back to its row id in the full unified dataset.
    """
    ds = _load_unified_dataset()
    n = len(ds)
    if split in ("validation", "val", "test"):
        start, end = int(n * 0.9), n
    elif split == "train":
        start, end = 0, int(n * 0.9)
    else:
        start, end = 0, n

    global_indices = np.arange(start, end, dtype=np.int64)
    ds = ds.select(range(start, end))

    if max_samples:
        limit = min(max_samples, len(ds))
        ds = ds.select(range(limit))
        global_indices = global_indices[:limit]

    if return_global_indices:
        return ds, global_indices
    return ds


def _tokenizer_hash():
    """Hash the tokenizer model file to detect retraining."""
    model_path = TOKENIZER_PREFIX + ".model"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    return "none"


def _cache_key(prefix, n_samples, max_enc_len, max_dec_len):
    tok_hash = _tokenizer_hash()
    key = f"{prefix}_{tok_hash}_{n_samples}_{max_enc_len}_{max_dec_len}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _mel_cache_key(prefix, n_samples, n_mels, max_mel_len):
    tok_hash = _tokenizer_hash()
    key = f"mel_{prefix}_{tok_hash}_{n_samples}_{n_mels}_{max_mel_len}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _save_cache_metadata(split, text_cache_id, mel_cache_id, n_samples,
                         max_enc_len, max_dec_len, n_mels, max_mel_len):
    """Save metadata JSON for a split, upload to GCS."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    meta = {
        "split": split,
        "text_cache_id": text_cache_id,
        "mel_cache_id": mel_cache_id,
        "n_samples": n_samples,
        "max_enc_len": max_enc_len,
        "max_dec_len": max_dec_len,
        "n_mels": n_mels,
        "max_mel_len": max_mel_len,
    }
    meta_path = os.path.join(CACHE_DIR, f"{split}_metadata.json")
    with open(meta_path, "w") as f:
        _json.dump(meta, f)
    _gcs_cache_upload(f"{split}_metadata", [".json"])


def _load_cache_metadata(split):
    """Load metadata JSON from local or GCS. Returns dict or None."""
    meta_path = os.path.join(CACHE_DIR, f"{split}_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return _json.load(f)
   
    import subprocess
    os.makedirs(CACHE_DIR, exist_ok=True)
    gcs_path = f"{GCS_CACHE_PATH}/{split}_metadata.json"
    subprocess.run(
        ["gcloud", "storage", "cp", gcs_path, meta_path],
        capture_output=True, text=True,
    )
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return _json.load(f)
    return None


def prepare_tool_call_pairs(ds, tokenizer, max_enc_len=256, max_dec_len=1024, batch_size=None):
    """Prepare tool-call encoder-decoder pairs with <tool_call> task token.

    Encoder input: tokenize(query), truncated to max_enc_len.
    Decoder input:  [BOS, <tool_call>, tools_tokens..., answer_tokens...]
    Decoder target: [<tool_call>, tools_tokens..., answer_tokens..., EOS]
    Loss mask: 1 only on answer + EOS positions (not tools prefix or padding).

    When batch_size is set, produces per-batch shard files uploaded to GCS
    incrementally. Returns (None, None, None, None, kept_indices).
    When batch_size is None, returns (enc_inputs, dec_inputs, dec_targets,
    loss_mask, kept_indices) as before.
    """

    cache_id = _cache_key("toolcall", len(ds), max_enc_len, max_dec_len)
    cache_path = os.path.join(CACHE_DIR, cache_id)

    if batch_size is None:
        tc_suffixes = ["_enc.npy", "_dec_in.npy", "_dec_tgt.npy", "_loss_mask.npy", "_kept_idx.npy"]

        def _load_tc_cache():
            return (
                np.load(cache_path + "_enc.npy"),
                np.load(cache_path + "_dec_in.npy"),
                np.load(cache_path + "_dec_tgt.npy"),
                np.load(cache_path + "_loss_mask.npy"),
                np.load(cache_path + "_kept_idx.npy"),
            )

        if os.path.exists(cache_path + "_enc.npy"):
            print(f"Loading cached tool-call data ({cache_id})...")
            return _load_tc_cache()

        if _gcs_cache_download(cache_id, tc_suffixes):
            print(f"Downloaded tool-call cache from GCS ({cache_id})...")
            return _load_tc_cache()

    if batch_size is not None:
        manifest_path = cache_path + "_manifest.json"
        if os.path.exists(manifest_path):
            kept_indices = np.load(cache_path + "_kept_idx.npy")
            print(f"Loading cached sharded tool-call data ({cache_id})...")
            return None, None, None, None, kept_indices

        if _gcs_cache_download(cache_id, ["_manifest.json", "_kept_idx.npy"]):
            if os.path.exists(manifest_path):
                kept_indices = np.load(cache_path + "_kept_idx.npy")
                print(f"Downloaded sharded tool-call cache from GCS ({cache_id})...")
                return None, None, None, None, kept_indices

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    tool_call_id = tokenizer.tool_call_token_id

    enc_texts = [ex["query"] for ex in ds]
    tools_texts = [ex["tools"] for ex in ds]
    ans_texts = [ex["answers"] for ex in ds]

    num_workers = min(os.cpu_count() or 1, 8)
    model_path = TOKENIZER_PREFIX + ".model"
    chunk_size = max(1, len(enc_texts) // (num_workers * 4))

    enc_chunks = [enc_texts[i:i + chunk_size] for i in range(0, len(enc_texts), chunk_size)]
    print(f"Tokenizing encoder inputs ({num_workers} workers)...")
    with mp.Pool(num_workers, initializer=_init_worker,
                 initargs=(model_path, max_enc_len)) as pool:
        enc_results = pool.map(_tokenize_chunk, enc_chunks)
    all_enc_tokens = [tok for chunk in enc_results for tok in chunk]

    tools_chunks = [tools_texts[i:i + chunk_size] for i in range(0, len(tools_texts), chunk_size)]
    print(f"Tokenizing tools ({num_workers} workers)...")
    with mp.Pool(num_workers, initializer=_init_worker,
                 initargs=(model_path, max_dec_len - 2)) as pool:
        tools_results = pool.map(_tokenize_chunk, tools_chunks)
    all_tools_tokens = [tok for chunk in tools_results for tok in chunk]

    ans_chunks = [ans_texts[i:i + chunk_size] for i in range(0, len(ans_texts), chunk_size)]
    print(f"Tokenizing answers ({num_workers} workers)...")
    with mp.Pool(num_workers, initializer=_init_worker,
                 initargs=(model_path, max_dec_len)) as pool:
        ans_results = pool.map(_tokenize_chunk, ans_chunks)
    all_ans_tokens = [tok for chunk in ans_results for tok in chunk]

    n = len(ds)

    def _fill_sample(j, e_tok, t_tok, a_tok, enc_arr, dec_in_arr, dec_tgt_arr, lm_arr):
        """Fill arrays at position j for one sample. Returns False if skipped."""
        prefix_len = 2 + len(t_tok)
        total_dec = prefix_len + len(a_tok) + 1
        if total_dec > max_dec_len:
            available_for_tools = max_dec_len - 2 - len(a_tok) - 1
            if available_for_tools < 1:
                return False
            t_tok = t_tok[:available_for_tools]
            prefix_len = 2 + len(t_tok)

        el = len(e_tok)
        enc_arr[j, :el] = e_tok

        dec_in_arr[j, 0] = eos_id
        dec_in_arr[j, 1] = tool_call_id
        tl = len(t_tok)
        if tl > 0:
            dec_in_arr[j, 2:2 + tl] = t_tok
        al = len(a_tok)
        if al > 0:
            dec_in_arr[j, prefix_len:prefix_len + al] = a_tok

        dec_tgt_arr[j, 0] = tool_call_id
        if tl > 0:
            dec_tgt_arr[j, 1:1 + tl] = t_tok
        if al > 0:
            dec_tgt_arr[j, prefix_len - 1:prefix_len - 1 + al] = a_tok
        dec_tgt_arr[j, prefix_len - 1 + al] = eos_id

        lm_arr[j, prefix_len - 1:prefix_len - 1 + al + 1] = 1.0
        return True

    os.makedirs(CACHE_DIR, exist_ok=True)

    if batch_size is not None:
        all_kept_indices = []
        shard_counts = []
        shard_idx = 0
        total_skipped = 0

        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_n = batch_end - batch_start

            enc_batch = np.full((batch_n, max_enc_len), pad_id, dtype=np.int32)
            dec_in_batch = np.full((batch_n, max_dec_len), pad_id, dtype=np.int32)
            dec_tgt_batch = np.full((batch_n, max_dec_len), pad_id, dtype=np.int32)
            loss_batch = np.zeros((batch_n, max_dec_len), dtype=np.float32)

            skipped = 0
            for j in range(batch_n):
                i = batch_start + j
                if not _fill_sample(j, all_enc_tokens[i], all_tools_tokens[i],
                                    all_ans_tokens[i], enc_batch, dec_in_batch,
                                    dec_tgt_batch, loss_batch):
                    skipped += 1

            if skipped > 0:
                keep = enc_batch[:, 0] != pad_id
                kept_in_batch = np.where(keep)[0]
                enc_batch = enc_batch[keep]
                dec_in_batch = dec_in_batch[keep]
                dec_tgt_batch = dec_tgt_batch[keep]
                loss_batch = loss_batch[keep]
                all_kept_indices.extend((batch_start + kept_in_batch).tolist())
                total_skipped += skipped
            else:
                all_kept_indices.extend(range(batch_start, batch_end))

            if len(enc_batch) == 0:
                continue

            shard_suffixes = []
            for suffix, arr in [("_enc", enc_batch), ("_dec_in", dec_in_batch),
                                ("_dec_tgt", dec_tgt_batch), ("_loss_mask", loss_batch)]:
                shard_suffix = f"{suffix}_{shard_idx:05d}.npy"
                np.save(os.path.join(CACHE_DIR, f"{cache_id}{shard_suffix}"), arr)
                shard_suffixes.append(shard_suffix)

            _gcs_cache_upload(cache_id, shard_suffixes)
            for s in shard_suffixes:
                fpath = os.path.join(CACHE_DIR, f"{cache_id}{s}")
                if os.path.exists(fpath):
                    os.remove(fpath)

            shard_counts.append(len(enc_batch))
            shard_idx += 1

        if total_skipped > 0:
            print(f"  Skipped {total_skipped} examples (too long for max_dec_len={max_dec_len})")

        kept_indices = np.array(all_kept_indices, dtype=np.int64)
        np.save(cache_path + "_kept_idx.npy", kept_indices)

        manifest = {
            "n_shards": shard_idx,
            "shard_counts": shard_counts,
            "total_kept": len(kept_indices),
        }
        manifest_path = cache_path + "_manifest.json"
        with open(manifest_path, "w") as f:
            _json.dump(manifest, f)

        _gcs_cache_upload(cache_id, ["_manifest.json", "_kept_idx.npy"])
        print(f"Cached {len(kept_indices):,} tool-call pairs in {shard_idx} shards ({cache_id})")

        return None, None, None, None, kept_indices

    enc_inputs = np.full((n, max_enc_len), pad_id, dtype=np.int32)
    dec_inputs = np.full((n, max_dec_len), pad_id, dtype=np.int32)
    dec_targets = np.full((n, max_dec_len), pad_id, dtype=np.int32)
    loss_mask = np.zeros((n, max_dec_len), dtype=np.float32)

    skipped = 0
    for i in range(n):
        if not _fill_sample(i, all_enc_tokens[i], all_tools_tokens[i],
                            all_ans_tokens[i], enc_inputs, dec_inputs,
                            dec_targets, loss_mask):
            skipped += 1

    if skipped > 0:
        keep = enc_inputs[:, 0] != pad_id
        kept_indices = np.where(keep)[0].astype(np.int64)
        enc_inputs = enc_inputs[keep]
        dec_inputs = dec_inputs[keep]
        dec_targets = dec_targets[keep]
        loss_mask = loss_mask[keep]
        print(f"  Skipped {skipped} examples (too long for max_dec_len={max_dec_len})")
    else:
        kept_indices = np.arange(n, dtype=np.int64)

    np.save(cache_path + "_enc.npy", enc_inputs)
    np.save(cache_path + "_dec_in.npy", dec_inputs)
    np.save(cache_path + "_dec_tgt.npy", dec_targets)
    np.save(cache_path + "_loss_mask.npy", loss_mask)
    np.save(cache_path + "_kept_idx.npy", kept_indices)
    tc_suffixes = ["_enc.npy", "_dec_in.npy", "_dec_tgt.npy", "_loss_mask.npy", "_kept_idx.npy"]
    _gcs_cache_upload(cache_id, tc_suffixes)
    print(f"Cached {len(enc_inputs):,} tool-call pairs to {CACHE_DIR}/{cache_id}")

    return enc_inputs, dec_inputs, dec_targets, loss_mask, kept_indices


def get_batches(enc_inputs, dec_inputs, dec_targets, batch_size, shuffle=True, loss_mask=None):
    n = len(enc_inputs)
    indices = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n - batch_size + 1, batch_size):
        idx = indices[i : i + batch_size]
        batch = (np.array(enc_inputs[idx]), np.array(dec_inputs[idx]), np.array(dec_targets[idx]))
        if loss_mask is not None:
            batch = batch + (np.array(loss_mask[idx]),)
        yield batch



def load_tool_call_audio(split="train", max_samples=None):
    """Return dataset-global indices for the given split.

    Applies the same 90/10 split as load_tool_calls. Audio is NOT loaded into memory.
    """
    ds = _load_unified_dataset()
    n = len(ds)

    if split in ("validation", "val", "test"):
        start, end = int(n * 0.9), n
    elif split == "train":
        start, end = 0, int(n * 0.9)
    else:
        start, end = 0, n

    indices = list(range(start, end))
    if max_samples:
        indices = indices[:max_samples]
    return indices


def load_audio_for_index(idx):
    """Load and decode audio for a single dataset index.

    Returns (audio_array, sampling_rate) or (None, None) if no audio.
    """
    import io
    import soundfile as sf

    ds = _load_unified_dataset()
    ex = ds[idx]
    audio_val = ex.get("audio")
    if audio_val is None:
        return None, None

    raw_bytes = None
    if isinstance(audio_val, dict):
        raw_bytes = audio_val.get("bytes")
    elif isinstance(audio_val, bytes):
        raw_bytes = audio_val
    if raw_bytes is None:
        return None, None

    audio_array, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    return audio_array.astype(np.float32), sr


def load_example_with_audio(idx):
    """Load a dataset example with decoded audio for eval use.

    Returns dict with {query, answers, tools, audio_array, sampling_rate}.
    """
    ds = _load_unified_dataset()
    ex = ds[idx]
    audio, sr = load_audio_for_index(idx)
    return {
        "query": ex["query"],
        "answers": ex["answers"],
        "tools": ex["tools"],
        "audio_array": audio,
        "sampling_rate": sr,
    }


def compute_mel_spectrogram(audio, sr=16000, n_mels=80, n_fft=400, hop_length=160):
    """Compute log-mel spectrogram using numpy/scipy. Returns (T_mel, n_mels) float32.

    25ms window (n_fft=400 at 16kHz), 10ms hop (hop_length=160) → ~100 frames/sec.
    """
    from scipy.signal import windows as scipy_windows
    from scipy.fft import rfft

    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    window = scipy_windows.hann(n_fft, sym=False).astype(np.float32)
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    if num_frames <= 0:
        return np.zeros((1, n_mels), dtype=np.float32)

    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(num_frames, n_fft),
        strides=(audio.strides[0] * hop_length, audio.strides[0]),
    ).copy()
    frames = frames * window
    spectrum = np.abs(rfft(frames, n=n_fft, axis=-1)) ** 2

    fmin, fmax = 0.0, sr / 2.0
    mel_low = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_high = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(np.int32)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        f_left, f_center, f_right = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        for k in range(f_left, f_center):
            if f_center > f_left:
                filterbank[m, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right > f_center:
                filterbank[m, k] = (f_right - k) / (f_right - f_center)

    mel_spec = spectrum @ filterbank.T
    log_mel = np.log(np.maximum(mel_spec, 1e-10))
    return log_mel.astype(np.float32)


def precompute_mels(kept_indices, n_mels=80, max_mel_len=1024, cache_id_prefix="", batch_size=None):
    """Precompute mel spectrograms.

    When batch_size is None: write a single (N, max_mel_len, n_mels) memmap file.
    When batch_size is set: produce per-batch shard .npy files, upload each to GCS.
    Returns the cache_id.
    """
    n_samples = len(kept_indices)
    cache_id = _mel_cache_key(cache_id_prefix, n_samples, n_mels, max_mel_len)
    cache_path = os.path.join(CACHE_DIR, cache_id)

    if batch_size is None:
        mel_file = cache_path + "_mels.npy"

        if os.path.exists(mel_file):
            print(f"  Mel cache already exists ({cache_id})")
            return cache_id

        if _gcs_cache_download(cache_id, ["_mels.npy"]):
            print(f"  Downloaded mel cache from GCS ({cache_id})")
            return cache_id

        os.makedirs(CACHE_DIR, exist_ok=True)
        shape = (n_samples, max_mel_len, n_mels)
        fp = np.lib.format.open_memmap(mel_file, mode='w+', dtype=np.float32, shape=shape)

        print(f"  Precomputing {n_samples} mel spectrograms (n_mels={n_mels}, max_mel_len={max_mel_len})...")
        for i, idx in enumerate(tqdm(kept_indices, desc="  Computing mels")):
            audio, sr = load_audio_for_index(int(idx))
            if audio is None:
                continue
            mel = compute_mel_spectrogram(audio, sr=sr, n_mels=n_mels)
            if mel.shape[0] > max_mel_len:
                mel = mel[:max_mel_len]
            t = mel.shape[0]
            fp[i, :t, :] = mel

        del fp
        _gcs_cache_upload(cache_id, ["_mels.npy"])
        print(f"  Cached mel spectrograms to {CACHE_DIR}/{cache_id}")
        return cache_id

    manifest_path = cache_path + "_mel_manifest.json"

    if os.path.exists(manifest_path):
        print(f"  Mel shard cache already exists ({cache_id})")
        return cache_id

    if _gcs_cache_download(cache_id, ["_mel_manifest.json"]):
        if os.path.exists(manifest_path):
            print(f"  Downloaded mel shard manifest from GCS ({cache_id})")
            return cache_id

    os.makedirs(CACHE_DIR, exist_ok=True)
    shard_counts = []
    shard_idx = 0

    print(f"  Precomputing {n_samples} mel spectrograms in shards of {batch_size}...")
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_indices = kept_indices[batch_start:batch_end]
        batch_n = len(batch_indices)

        mel_batch = np.zeros((batch_n, max_mel_len, n_mels), dtype=np.float32)

        for j, idx in enumerate(tqdm(batch_indices, desc=f"  Shard {shard_idx}", leave=False)):
            audio, sr = load_audio_for_index(int(idx))
            if audio is None:
                continue
            mel = compute_mel_spectrogram(audio, sr=sr, n_mels=n_mels)
            if mel.shape[0] > max_mel_len:
                mel = mel[:max_mel_len]
            t = mel.shape[0]
            mel_batch[j, :t, :] = mel

        shard_suffix = f"_mels_{shard_idx:05d}.npy"
        shard_path = os.path.join(CACHE_DIR, f"{cache_id}{shard_suffix}")
        np.save(shard_path, mel_batch)

        _gcs_cache_upload(cache_id, [shard_suffix])
        if os.path.exists(shard_path):
            os.remove(shard_path)

        shard_counts.append(batch_n)
        shard_idx += 1

    manifest = {
        "n_shards": shard_idx,
        "shard_counts": shard_counts,
        "total_samples": n_samples,
    }
    with open(manifest_path, "w") as f:
        _json.dump(manifest, f)

    _gcs_cache_upload(cache_id, ["_mel_manifest.json"])
    print(f"  Cached mel spectrograms in {shard_idx} shards ({cache_id})")
    return cache_id


class ShardedMmapArray:
    """Array-like wrapper over multiple .npy shard files with mmap support.

    Provides __len__ and __getitem__ (int, slice, numpy array) to transparently
    index across shards. Each shard is memory-mapped individually.
    """

    def __init__(self, paths, mmap_mode="r"):
        self._shards = [np.load(p, mmap_mode=mmap_mode) for p in paths]
        self._lengths = [len(s) for s in self._shards]
        self._cumulative = np.cumsum(self._lengths)
        self._total = int(self._cumulative[-1]) if len(self._cumulative) else 0

    def __len__(self):
        return self._total

    @property
    def shape(self):
        if not self._shards:
            return (0,)
        return (self._total,) + self._shards[0].shape[1:]

    @property
    def dtype(self):
        return self._shards[0].dtype if self._shards else np.float32

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            if idx < 0:
                idx += self._total
            si = int(np.searchsorted(self._cumulative, idx, side="right"))
            offset = int(self._cumulative[si - 1]) if si > 0 else 0
            return self._shards[si][idx - offset]

        if isinstance(idx, np.ndarray):
            result = np.empty((len(idx),) + self._shards[0].shape[1:],
                              dtype=self._shards[0].dtype)
            shard_ids = np.searchsorted(self._cumulative, idx, side="right")
            for si in range(len(self._shards)):
                mask = shard_ids == si
                if not mask.any():
                    continue
                offset = int(self._cumulative[si - 1]) if si > 0 else 0
                local = idx[mask] - offset
                result[mask] = self._shards[si][local]
            return result

        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._total)
            indices = np.arange(start, stop, step)
            return self[indices]

        raise TypeError(f"Unsupported index type: {type(idx)}")


def load_prepared_data(split, mmap=False):
    """Load pre-tokenized .npy files. If mmap=True, returns memory-mapped arrays.

    Supports both single-file and sharded (manifest-based) caches.
    Tries local cache first, then GCS download. Raises FileNotFoundError
    if not found (no fallback tokenization — run 'needle tokenize' first).

    Returns dict with keys: enc_inputs, dec_inputs, dec_targets, loss_mask,
    kept_indices, mel_cache_id.
    """
    meta = _load_cache_metadata(split)
    if meta is None:
        raise FileNotFoundError(
            f"No prepared data found for split '{split}'. Run 'needle tokenize' first."
        )

    text_cache_id = meta["text_cache_id"]
    cache_path = os.path.join(CACHE_DIR, text_cache_id)
    manifest_path = cache_path + "_manifest.json"

    if not os.path.exists(manifest_path):
        _gcs_cache_download(text_cache_id, ["_manifest.json"])

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = _json.load(f)
        n_shards = manifest["n_shards"]

        if not os.path.exists(cache_path + "_kept_idx.npy"):
            _gcs_cache_download(text_cache_id, ["_kept_idx.npy"])

        shard_suffixes = ["_enc", "_dec_in", "_dec_tgt", "_loss_mask"]
        _gcs_download_shards(text_cache_id, n_shards, shard_suffixes)

        def _shard_paths(suffix):
            return [os.path.join(CACHE_DIR, f"{text_cache_id}{suffix}_{i:05d}.npy")
                    for i in range(n_shards)]

        mmap_mode = "r" if mmap else None
        if mmap:
            result = {
                "enc_inputs": ShardedMmapArray(_shard_paths("_enc"), mmap_mode="r"),
                "dec_inputs": ShardedMmapArray(_shard_paths("_dec_in"), mmap_mode="r"),
                "dec_targets": ShardedMmapArray(_shard_paths("_dec_tgt"), mmap_mode="r"),
                "loss_mask": ShardedMmapArray(_shard_paths("_loss_mask"), mmap_mode="r"),
            }
        else:
            result = {
                "enc_inputs": np.concatenate([np.load(p) for p in _shard_paths("_enc")]),
                "dec_inputs": np.concatenate([np.load(p) for p in _shard_paths("_dec_in")]),
                "dec_targets": np.concatenate([np.load(p) for p in _shard_paths("_dec_tgt")]),
                "loss_mask": np.concatenate([np.load(p) for p in _shard_paths("_loss_mask")]),
            }

        result["kept_indices"] = np.load(cache_path + "_kept_idx.npy", mmap_mode=mmap_mode)
        result["mel_cache_id"] = meta.get("mel_cache_id")
        return result

    tc_suffixes = ["_enc.npy", "_dec_in.npy", "_dec_tgt.npy", "_loss_mask.npy", "_kept_idx.npy"]
    if not os.path.exists(cache_path + "_enc.npy"):
        if not _gcs_cache_download(text_cache_id, tc_suffixes):
            raise FileNotFoundError(
                f"Text cache '{text_cache_id}' not found. Run 'needle tokenize' first."
            )

    mmap_mode = "r" if mmap else None
    return {
        "enc_inputs": np.load(cache_path + "_enc.npy", mmap_mode=mmap_mode),
        "dec_inputs": np.load(cache_path + "_dec_in.npy", mmap_mode=mmap_mode),
        "dec_targets": np.load(cache_path + "_dec_tgt.npy", mmap_mode=mmap_mode),
        "loss_mask": np.load(cache_path + "_loss_mask.npy", mmap_mode=mmap_mode),
        "kept_indices": np.load(cache_path + "_kept_idx.npy", mmap_mode=mmap_mode),
        "mel_cache_id": meta.get("mel_cache_id"),
    }


def load_prepared_mels(mel_cache_id, mmap=False):
    """Load precomputed mel .npy file(s), optionally memory-mapped.

    Supports both single-file and sharded (manifest-based) caches.
    """
    cache_path = os.path.join(CACHE_DIR, mel_cache_id)
    manifest_path = cache_path + "_mel_manifest.json"

    if not os.path.exists(manifest_path):
        _gcs_cache_download(mel_cache_id, ["_mel_manifest.json"])

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = _json.load(f)
        n_shards = manifest["n_shards"]

        _gcs_download_shards(mel_cache_id, n_shards, ["_mels"])

        shard_paths = [os.path.join(CACHE_DIR, f"{mel_cache_id}_mels_{i:05d}.npy")
                       for i in range(n_shards)]

        if mmap:
            return ShardedMmapArray(shard_paths, mmap_mode="r")
        else:
            return np.concatenate([np.load(p) for p in shard_paths])

    mel_file = cache_path + "_mels.npy"
    if not os.path.exists(mel_file):
        if not _gcs_cache_download(mel_cache_id, ["_mels.npy"]):
            raise FileNotFoundError(
                f"Mel cache '{mel_cache_id}' not found. Run 'needle tokenize' first."
            )

    mmap_mode = "r" if mmap else None
    return np.load(mel_file, mmap_mode=mmap_mode)


def _fit_mel_frames_to_length(mel_frames, target_len, rng):
    """Crop or tile valid mel frames to target length."""
    mel_frames = np.asarray(mel_frames, dtype=np.float32)
    if target_len <= 0:
        return np.zeros((0, mel_frames.shape[-1]), dtype=np.float32)
    if mel_frames.shape[0] == 0:
        return np.zeros((target_len, mel_frames.shape[-1]), dtype=np.float32)
    if mel_frames.shape[0] == target_len:
        return mel_frames
    if mel_frames.shape[0] > target_len:
        start = int(rng.integers(0, mel_frames.shape[0] - target_len + 1))
        return mel_frames[start:start + target_len]
    reps = int(np.ceil(target_len / mel_frames.shape[0]))
    tiled = np.tile(mel_frames, (reps, 1))
    return tiled[:target_len].astype(np.float32)


def _split_valid_mel_frames(mel):
    """Return (valid_frames, frame_mask) where mask marks non-padding rows."""
    mel = np.asarray(mel, dtype=np.float32)
    frame_mask = np.any(mel != 0.0, axis=1)
    return mel[frame_mask], frame_mask


def _mix_log_mel_with_noise_power(clean_log_mel, noise_power, target_snr_db):
    """Mix additive noise power into log-mel features at a target SNR.

    Assumes the STFT cross-term is zero in expectation:
      |X + N|^2 ~= |X|^2 + |N|^2
    so we add noise in mel-power domain and map back to log-mel.
    """
    clean_log_mel = np.asarray(clean_log_mel, dtype=np.float32)
    noise_power = np.asarray(noise_power, dtype=np.float32)
    if clean_log_mel.size == 0 or noise_power.size == 0:
        return clean_log_mel

    clean_power = np.exp(np.clip(clean_log_mel, -30.0, 30.0))
    noise_power = np.maximum(noise_power, 0.0)

    signal_power = float(np.mean(clean_power))
    noise_mean = float(np.mean(noise_power))
    if signal_power <= 1e-12 or noise_mean <= 1e-12:
        return clean_log_mel

    desired_noise_power = signal_power / (10.0 ** (target_snr_db / 10.0))
    noise_scale = desired_noise_power / noise_mean
    mixed_power = clean_power + noise_power * noise_scale
    return np.log(np.maximum(mixed_power, 1e-10)).astype(np.float32)


class MelWhiteNoiseAugmenter:
    """White-noise augmenter operating directly on log-mel batches.

    Uses an independent random-phase approximation (zero expected cross-term)
    and mixes in mel-power domain at a sampled SNR.
    """

    def __init__(self, p=0.5, min_snr_db=8.0, max_snr_db=30.0, seed=None):
        self.p = float(np.clip(p, 0.0, 1.0))
        self.min_snr_db = float(min(min_snr_db, max_snr_db))
        self.max_snr_db = float(max(min_snr_db, max_snr_db))
        self._rng = np.random.default_rng(seed)
        self.name = "white"
        self.transforms = ("white_noise_mel",)

    def __call__(self, mel_batch):
        mel_batch = np.array(mel_batch, dtype=np.float32, copy=True)
        for i in range(mel_batch.shape[0]):
            if self._rng.random() > self.p:
                continue
            clean_valid, frame_mask = _split_valid_mel_frames(mel_batch[i])
            if clean_valid.shape[0] == 0:
                continue
            snr_db = float(self._rng.uniform(self.min_snr_db, self.max_snr_db))
            # Complex-white-noise power proxy: N_re^2 + N_im^2 (chi-square with 2 dof).
            n_re = np.asarray(self._rng.standard_normal(clean_valid.shape), dtype=np.float32)
            n_im = np.asarray(self._rng.standard_normal(clean_valid.shape), dtype=np.float32)
            white_power = n_re * n_re + n_im * n_im
            mel_batch[i, frame_mask, :] = _mix_log_mel_with_noise_power(clean_valid, white_power, snr_db)
        return mel_batch


class MelPersonNoiseAugmenter:
    """Background person-noise augmenter operating on precomputed log-mels.

    Mixes in mel-power domain with distance-weighted gains under the same
    zero-cross-term approximation used for white noise.
    """

    def __init__(self, mel_pool, n=10, r1=3.0, r2=10.0, r_ref=1.0,
                 min_snr_db=15.0, max_snr_db=40.0, seed=None):
        if r1 <= 0 or r2 <= r1 or r_ref <= 0:
            raise ValueError("Require r1 > 0, r2 > r1, and r_ref > 0 for person noise augmentation.")
        self.pool = mel_pool
        self.n = max(1, int(n))
        self.r1 = float(r1)
        self.r2 = float(r2)
        self.r_ref = float(r_ref)
        self.min_snr_db = float(min(min_snr_db, max_snr_db))
        self.max_snr_db = float(max(min_snr_db, max_snr_db))
        self._rng = np.random.default_rng(seed)
        self.name = "person"
        self.transforms = ("person_noise_mel",)

    def __call__(self, mel_batch):
        mel_batch = np.array(mel_batch, dtype=np.float32, copy=True)
        pool_len = len(self.pool)
        if pool_len == 0:
            return mel_batch

        for i in range(mel_batch.shape[0]):
            clean_valid, frame_mask = _split_valid_mel_frames(mel_batch[i])
            target_len = clean_valid.shape[0]
            if target_len == 0:
                continue

            replace = pool_len < self.n
            clip_ids = self._rng.choice(pool_len, size=self.n, replace=replace)
            z = self._rng.uniform(0.0, 1.0, size=self.n)
            r_z = np.sqrt(self.r1**2 + z * (self.r2**2 - self.r1**2))
            gain_power = (self.r_ref / np.maximum(r_z, 1e-6)) ** 2

            noise_power_sum = np.zeros_like(clean_valid, dtype=np.float32)
            for clip_id, g_pow in zip(clip_ids, gain_power):
                noise_mel = np.asarray(self.pool[int(clip_id)], dtype=np.float32)
                noise_valid, _ = _split_valid_mel_frames(noise_mel)
                if noise_valid.shape[0] == 0:
                    continue
                noise_seg = _fit_mel_frames_to_length(noise_valid, target_len, self._rng)
                noise_power_sum += float(g_pow) * np.exp(np.clip(noise_seg, -30.0, 30.0))

            if np.mean(noise_power_sum) <= 1e-12:
                continue

            snr_db = float(self._rng.uniform(self.min_snr_db, self.max_snr_db))
            mel_batch[i, frame_mask, :] = _mix_log_mel_with_noise_power(clean_valid, noise_power_sum, snr_db)

        return mel_batch


def build_audio_augmenter(sr=16000, mode="white", white_noise_p=0.5,
                          white_noise_min_snr_db=8.0, white_noise_max_snr_db=30.0,
                          person_noise_n=10, person_noise_r1=3.0, person_noise_r2=10.0,
                          person_noise_r_ref=1.0, person_noise_min_snr_db=15.0,
                          person_noise_max_snr_db=40.0, person_noise_pool=None):
    """Build a mel-domain augmentation pipeline for training."""
    del sr
    mode = (mode or "none").lower()
    if mode == "none":
        return None

    white_noise = MelWhiteNoiseAugmenter(
        p=white_noise_p,
        min_snr_db=white_noise_min_snr_db,
        max_snr_db=white_noise_max_snr_db,
    )
    if mode == "white":
        return white_noise

    if mode == "person":
        if person_noise_pool is None or len(person_noise_pool) == 0:
            print("  WARNING: person noise requested, but no mel pool was provided")
            return None
        return MelPersonNoiseAugmenter(
            mel_pool=person_noise_pool,
            n=person_noise_n,
            r1=person_noise_r1,
            r2=person_noise_r2,
            r_ref=person_noise_r_ref,
            min_snr_db=person_noise_min_snr_db,
            max_snr_db=person_noise_max_snr_db,
        )

    if mode == "full":
        print("  WARNING: full waveform augmentation is not used on precomputed mels — using white mel noise")
        return white_noise

    raise ValueError(f"Unknown audio augmentation mode: {mode}")


def get_speech_batches(mel_data, dec_inputs, dec_targets, batch_size,
                       shuffle=True, loss_mask=None, augmenter=None):
    """Yield speech batches from precomputed mel data.

    mel_data: array of shape (N, max_mel_len, n_mels), possibly memory-mapped.
    augmenter: callable applied to each mel batch on-the-fly.
    """
    n = len(mel_data)
    indices = np.random.permutation(n) if shuffle else np.arange(n)

    for i in range(0, n - batch_size + 1, batch_size):
        idx = indices[i : i + batch_size]
        batch_mel = np.array(mel_data[idx], dtype=np.float32)
        if augmenter is not None:
            batch_mel = augmenter(batch_mel)
        batch = (batch_mel, np.array(dec_inputs[idx]), np.array(dec_targets[idx]))
        if loss_mask is not None:
            batch = batch + (np.array(loss_mask[idx]),)
        yield batch


class PrefetchIterator:
    """Generic prefetch wrapper: runs any batch-generating callable in a background thread."""

    def __init__(self, generator_fn, prefetch=4):
        """generator_fn: callable that returns an iterable of batches."""
        self._queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._generator_fn = generator_fn
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _produce(self):
        try:
            for batch in self._generator_fn():
                if self._stop.is_set():
                    return
                self._queue.put(batch)
            self._queue.put(None)  # sentinel
        except Exception as e:
            self._queue.put(e)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self._stop.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=5)


def count_batches(n_samples, batch_size):
    """Return the number of full batches for a dataset of n_samples."""
    return n_samples // batch_size
