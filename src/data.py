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
UNK_ID = 3
STORY_ID = 4
TRANSCRIBE_ID = 5
MASK_ID = 6  # requires tokenizer retrain with <mask> in user_defined_symbols


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


def get_speech_batches(masked_mel, orig_mel, batch_size, shuffle=True):
    n = len(masked_mel)
    if shuffle:
        perm = np.random.permutation(n)
        masked_mel = masked_mel[perm]
        orig_mel = orig_mel[perm]
    for i in range(0, n - batch_size + 1, batch_size):
        yield masked_mel[i : i + batch_size], orig_mel[i : i + batch_size]


def load_librispeech(split="train.clean.100", max_samples=None):
    from datasets import load_dataset as _load_dataset
    ds = _load_dataset("librispeech_asr", "clean", split=split, trust_remote_code=True)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def load_librilight(subset="small", max_samples=None):
    """Load unlabeled audio from LibriSpeech (train.360 + train.100) as a LibriLight substitute.

    subset arg is ignored (kept for API compatibility); we always use LibriSpeech clean splits.
    Returns a dataset with 'audio' column containing {'array': np.ndarray, 'sampling_rate': int}.
    """
    import io
    import soundfile as sf
    from datasets import load_dataset as _load_dataset, Audio

    # train.360 ≈ 360h; train.100 ≈ 100h — enough for ablations
    splits = ["train.360", "train.100"]
    parts = []
    for spl in splits:
        try:
            ds = _load_dataset("openslr/librispeech_asr", "clean", split=spl)
            parts.append(ds)
        except Exception:
            pass
    if not parts:
        raise RuntimeError("Could not load any LibriSpeech split from openslr/librispeech_asr")

    from datasets import concatenate_datasets
    ds = concatenate_datasets(parts)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Decode FLAC bytes → float32 array using soundfile (no torch required)
    ds_raw = ds.cast_column("audio", Audio(decode=False))

    def _decode(batch):
        arrays, rates = [], []
        for item in batch["audio"]:
            data, sr = sf.read(io.BytesIO(item["bytes"]))
            if data.ndim > 1:
                data = data.mean(axis=1)  # stereo → mono
            arrays.append(data.astype(np.float32))
            rates.append(sr)
        return {"audio_array": arrays, "sampling_rate": rates}

    ds_decoded = ds_raw.map(
        _decode,
        batched=True,
        batch_size=64,
        remove_columns=ds_raw.column_names,
        desc="Decoding audio",
        num_proc=1,
    )

    # Wrap back into expected format: ds[i]['audio'] = {'array': ..., 'sampling_rate': ...}
    def _wrap(batch):
        return {"audio": [{"array": a, "sampling_rate": r}
                          for a, r in zip(batch["audio_array"], batch["sampling_rate"])]}

    return ds_decoded.map(_wrap, batched=True, batch_size=256, remove_columns=["audio_array", "sampling_rate"])


def _mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)  # (n_fft//2+1,)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    f = fft_freqs[np.newaxis, :]          # (1, n_fft//2+1)
    left   = hz_points[:-2, np.newaxis]   # (n_mels, 1)
    center = hz_points[1:-1, np.newaxis]  # (n_mels, 1)
    right  = hz_points[2:,  np.newaxis]   # (n_mels, 1)

    up   = np.maximum(0.0, (f - left)   / np.maximum(center - left,  1e-10))
    down = np.maximum(0.0, (right - f)  / np.maximum(right  - center, 1e-10))
    return np.minimum(up, down).astype(np.float32)  # (n_mels, n_fft//2+1)


def compute_mel_spectrogram(audio, sr, n_mels=80, n_fft=None, hop_length=None):
    """Pure numpy mel spectrogram: 25 ms window, 10 ms hop → ~100 frames/sec."""
    n_fft      = n_fft      or int(round(0.025 * sr))
    hop_length = hop_length or int(round(0.010 * sr))

    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))

    n_frames = 1 + (len(audio) - n_fft) // hop_length
    idx = np.arange(n_fft)[None, :] + hop_length * np.arange(n_frames)[:, None]
    frames = audio[idx] * np.hanning(n_fft).astype(np.float32)  # (n_frames, n_fft)

    fft_out = np.fft.rfft(frames, n=n_fft, axis=1)
    power = fft_out.real ** 2 + fft_out.imag ** 2  # (n_frames, n_fft//2+1)

    filters = _mel_filterbank(sr, n_fft, n_mels)
    mel_spec = power @ filters.T  # (n_frames, n_mels)
    return np.log1p(mel_spec).astype(np.float32)


def _masked_cache_key(n_samples, max_enc_len, max_dec_len, mask_ratio):
    tok_hash = _tokenizer_hash()
    key = f"masked_{tok_hash}_{n_samples}_{max_enc_len}_{max_dec_len}_{mask_ratio}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def prepare_text_masked(ds, tokenizer, max_enc_len=128, max_dec_len=128, mask_ratio=0.50):
    """BERT-style masked autoencoding pairs from TinyStories.

    Returns:
        masked_enc  (N, max_enc_len): encoder input with MASK_ID replacing mask_ratio of tokens
        dec_in      (N, max_dec_len): [BOS] + full unmasked sequence
        dec_tgt     (N, max_dec_len): full unmasked sequence (PAD for short stories)
    """
    cache_id = _masked_cache_key(len(ds), max_enc_len, max_dec_len, mask_ratio)
    cache_path = os.path.join(CACHE_DIR, cache_id)
    if os.path.exists(cache_path + "_menc.npy"):
        print(f"Loading cached masked text data ({cache_id})...")
        return (
            np.load(cache_path + "_menc.npy"),
            np.load(cache_path + "_dec_in.npy"),
            np.load(cache_path + "_dec_tgt.npy"),
        )

    # Parallel tokenization (reuse existing infrastructure)
    texts = list(ds["text"])
    num_workers = min(os.cpu_count() or 1, 8)
    model_path = TOKENIZER_PREFIX + ".model"
    chunk_size = max(1, len(texts) // (num_workers * 4))
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    max_total = max(max_enc_len, max_dec_len)

    print(f"Tokenizing for masked pretraining ({num_workers} workers)...")
    with mp.Pool(num_workers, initializer=_init_worker,
                 initargs=(model_path, max_total)) as pool:
        results = pool.map(_tokenize_chunk, chunks)
    all_tokens = [tok for chunk in results for tok in chunk]

    rng = np.random.RandomState(42)
    n = len(all_tokens)
    masked_enc = np.full((n, max_enc_len), PAD_ID, dtype=np.int32)
    dec_in     = np.full((n, max_dec_len), PAD_ID, dtype=np.int32)
    dec_tgt    = np.full((n, max_dec_len), PAD_ID, dtype=np.int32)

    for i, tokens in enumerate(tqdm(all_tokens, desc="Building masked pairs")):
        if not tokens:
            continue
        enc_tokens = np.array(tokens[:max_enc_len], dtype=np.int32)
        el = len(enc_tokens)
        mask_flags = rng.random(el) < mask_ratio
        masked = np.where(mask_flags, MASK_ID, enc_tokens)
        masked_enc[i, :el] = masked

        dec_tokens = tokens[:max_dec_len]
        dl = len(dec_tokens)
        dec_in[i, 0] = BOS_ID
        if dl > 1:
            dec_in[i, 1:dl] = dec_tokens[:dl - 1]
        dec_tgt[i, :dl] = dec_tokens

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_path + "_menc.npy", masked_enc)
    np.save(cache_path + "_dec_in.npy", dec_in)
    np.save(cache_path + "_dec_tgt.npy", dec_tgt)
    print(f"Cached {n:,} masked text pairs to {CACHE_DIR}/{cache_id}")
    return masked_enc, dec_in, dec_tgt


def _speech_cache_key(n_samples, n_mels, clip_frames, mask_ratio, patch_t, patch_f):
    key = f"speech_{n_samples}_{n_mels}_{clip_frames}_{mask_ratio}_{patch_t}_{patch_f}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def prepare_speech_masked(ds, tokenizer=None, n_mels=80, clip_frames=1000,
                          mask_ratio=0.50, patch_t=10, patch_f=10):
    """2D patch-masked mel spectrogram pairs from LibriSpeech.

    Returns:
        masked_mel  (N, clip_frames, n_mels): mel with time-strip patches zeroed
        orig_mel    (N, clip_frames, n_mels): original mel (reconstruction target)
        patch_mask  (N, clip_frames//patch_t, n_mels//patch_f): 1=masked
    """
    n_time_patches = clip_frames // patch_t
    n_freq_patches = n_mels // patch_f

    cache_id = _speech_cache_key(len(ds), n_mels, clip_frames, mask_ratio, patch_t, patch_f)
    cache_path = os.path.join(CACHE_DIR, cache_id)
    if os.path.exists(cache_path + "_mmel.npy"):
        print(f"Loading cached masked speech data ({cache_id})...")
        return (
            np.load(cache_path + "_mmel.npy"),
            np.load(cache_path + "_omel.npy"),
            np.load(cache_path + "_pmask.npy"),
        )

    rng = np.random.RandomState(42)
    n_mask = int(round(n_time_patches * mask_ratio))  # e.g. 50 out of 100

    orig_mels   = []
    masked_mels = []
    patch_masks = []

    print(f"Computing mel spectrograms for {len(ds):,} clips...")
    for example in tqdm(ds, desc="Processing audio"):
        audio_dict = example["audio"]
        audio = np.asarray(audio_dict["array"], dtype=np.float32)
        sr = int(audio_dict["sampling_rate"])

        mel = compute_mel_spectrogram(audio, sr, n_mels=n_mels)  # (T, n_mels)

        # Truncate or pad to exactly clip_frames
        if mel.shape[0] >= clip_frames:
            mel = mel[:clip_frames]
        else:
            mel = np.pad(mel, ((0, clip_frames - mel.shape[0]), (0, 0)))

        # Time-strip patch masking: select n_mask time patches to zero
        time_patch_idx = rng.permutation(n_time_patches)[:n_mask]
        pmask = np.zeros((n_time_patches, n_freq_patches), dtype=np.bool_)
        pmask[time_patch_idx] = True

        masked = mel.copy()
        for tp in time_patch_idx:
            t_start = tp * patch_t
            masked[t_start : t_start + patch_t, :] = 0.0

        orig_mels.append(mel)
        masked_mels.append(masked)
        patch_masks.append(pmask)

    orig_mel   = np.stack(orig_mels,   axis=0).astype(np.float32)
    masked_mel = np.stack(masked_mels, axis=0).astype(np.float32)
    patch_mask = np.stack(patch_masks, axis=0)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_path + "_mmel.npy", masked_mel)
    np.save(cache_path + "_omel.npy", orig_mel)
    np.save(cache_path + "_pmask.npy", patch_mask)
    print(f"Cached {len(orig_mel):,} masked mel pairs to {CACHE_DIR}/{cache_id}")
    return masked_mel, orig_mel, patch_mask
