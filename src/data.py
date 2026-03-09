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
from datasets import Audio, load_from_disk
from tqdm import tqdm

from .tokenizer import (  # noqa: F401 — re-exported for existing callers
    PAD_ID, EOS_ID, BOS_ID, UNK_ID, TOOL_CALL_ID, TRANSCRIBE_ID,
    TOKENIZER_DIR, TOKENIZER_PREFIX, GCS_TOKENIZER_PATH,
    pre_tokenize, NeedleTokenizer, _init_worker, _tokenize_chunk,
    _tokenizer_hash, train_tokenizer, get_tokenizer, _download_tokenizer_from_gcs,
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".data_cache")
LOCAL_UNIFIED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tool_calls_unified")
GCS_DATASET_PATH = "gs://cactus-dataset/tool_calls"

_unified_dataset_cache = None


def _load_unified_dataset():
    """Load the unified dataset from GCS (via gcsfs) or local fallback.

    Caches the result in memory after first load.
    Uses soundfile as the audio decoding backend.
    """
    global _unified_dataset_cache
    if _unified_dataset_cache is not None:
        return _unified_dataset_cache

    # Try GCS first
    try:
        ds = load_from_disk(GCS_DATASET_PATH)
        print(f"Loaded unified dataset from GCS ({len(ds)} rows)")
        # Cache locally for faster subsequent loads
        if not os.path.exists(LOCAL_UNIFIED_DIR):
            ds.save_to_disk(LOCAL_UNIFIED_DIR)
            print(f"  Cached locally to {LOCAL_UNIFIED_DIR}")
        _unified_dataset_cache = _set_audio_backend(ds)
        return _unified_dataset_cache
    except Exception:
        pass

    # Fallback to local
    if os.path.exists(LOCAL_UNIFIED_DIR):
        ds = load_from_disk(LOCAL_UNIFIED_DIR)
        print(f"Loaded unified dataset from {LOCAL_UNIFIED_DIR} ({len(ds)} rows)")
        _unified_dataset_cache = _set_audio_backend(ds)
        return _unified_dataset_cache

    raise FileNotFoundError(
        f"Unified dataset not found. Run scripts/build_dataset.py first, "
        f"or ensure GCS path {GCS_DATASET_PATH} is accessible (pip install gcsfs)."
    )


def _set_audio_backend(ds):
    """Disable automatic audio decoding to avoid torchcodec dependency.

    Audio is decoded manually via soundfile in load_tool_call_audio().
    """
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))
    return ds


def load_tool_calls(split="train", max_samples=None):
    """Load tool-calling dataset, splitting 90/10 for train/val."""
    ds = _load_unified_dataset()
    n = len(ds)
    if split in ("validation", "val", "test"):
        ds = ds.select(range(int(n * 0.9), n))
    elif split == "train":
        ds = ds.select(range(int(n * 0.9)))
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def _cache_key(prefix, n_samples, max_enc_len, max_dec_len):
    tok_hash = _tokenizer_hash()
    key = f"{prefix}_{tok_hash}_{n_samples}_{max_enc_len}_{max_dec_len}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def prepare_tool_call_pairs(ds, tokenizer, max_enc_len=256, max_dec_len=1024):
    """Prepare tool-call encoder-decoder pairs with <tool_call> task token.

    Encoder input: tokenize(query), truncated to max_enc_len.
    Decoder input:  [BOS, <tool_call>, tools_tokens..., answer_tokens...]
    Decoder target: [<tool_call>, tools_tokens..., answer_tokens..., EOS]
    Loss mask: 1 only on answer + EOS positions (not tools prefix or padding).

    Returns (enc_inputs, dec_inputs, dec_targets, loss_mask).
    """

    cache_id = _cache_key("toolcall", len(ds), max_enc_len, max_dec_len)
    cache_path = os.path.join(CACHE_DIR, cache_id)
    if os.path.exists(cache_path + "_enc.npy"):
        print(f"Loading cached tool-call data ({cache_id})...")
        return (
            np.load(cache_path + "_enc.npy"),
            np.load(cache_path + "_dec_in.npy"),
            np.load(cache_path + "_dec_tgt.npy"),
            np.load(cache_path + "_loss_mask.npy"),
        )

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    tool_call_id = tokenizer.tool_call_token_id

    enc_texts = [ex["query"] for ex in ds]
    tools_texts = [ex["tools"] for ex in ds]
    ans_texts = [ex["answers"] for ex in ds]

    num_workers = min(os.cpu_count() or 1, 8)
    model_path = TOKENIZER_PREFIX + ".model"
    chunk_size = max(1, len(enc_texts) // (num_workers * 4))

    # Tokenize queries (encoder)
    enc_chunks = [enc_texts[i:i + chunk_size] for i in range(0, len(enc_texts), chunk_size)]
    print(f"Tokenizing encoder inputs ({num_workers} workers)...")
    with mp.Pool(num_workers, initializer=_init_worker,
                 initargs=(model_path, max_enc_len)) as pool:
        enc_results = pool.map(_tokenize_chunk, enc_chunks)
    all_enc_tokens = [tok for chunk in enc_results for tok in chunk]

    # Tokenize tools (decoder prefix) — reserve 2 for BOS + <tool_call>, rest for tools + answer
    tools_chunks = [tools_texts[i:i + chunk_size] for i in range(0, len(tools_texts), chunk_size)]
    print(f"Tokenizing tools ({num_workers} workers)...")
    with mp.Pool(num_workers, initializer=_init_worker,
                 initargs=(model_path, max_dec_len - 2)) as pool:
        tools_results = pool.map(_tokenize_chunk, tools_chunks)
    all_tools_tokens = [tok for chunk in tools_results for tok in chunk]

    # Tokenize answers (decoder generation target)
    ans_chunks = [ans_texts[i:i + chunk_size] for i in range(0, len(ans_texts), chunk_size)]
    print(f"Tokenizing answers ({num_workers} workers)...")
    with mp.Pool(num_workers, initializer=_init_worker,
                 initargs=(model_path, max_dec_len)) as pool:
        ans_results = pool.map(_tokenize_chunk, ans_chunks)
    all_ans_tokens = [tok for chunk in ans_results for tok in chunk]

    n = len(ds)
    enc_inputs = np.full((n, max_enc_len), pad_id, dtype=np.int32)
    dec_inputs = np.full((n, max_dec_len), pad_id, dtype=np.int32)
    dec_targets = np.full((n, max_dec_len), pad_id, dtype=np.int32)
    loss_mask = np.zeros((n, max_dec_len), dtype=np.float32)

    skipped = 0
    for i in range(n):
        e_tok = all_enc_tokens[i]
        t_tok = all_tools_tokens[i]
        a_tok = all_ans_tokens[i]

        # Prefix length: BOS + <tool_call> + tools_tokens
        prefix_len = 2 + len(t_tok)
        # Total decoder length: prefix + answer + EOS
        total_dec = prefix_len + len(a_tok) + 1

        if total_dec > max_dec_len:
            # Truncate tools to fit answer + EOS
            available_for_tools = max_dec_len - 2 - len(a_tok) - 1
            if available_for_tools < 1:
                skipped += 1
                continue
            t_tok = t_tok[:available_for_tools]
            prefix_len = 2 + len(t_tok)
            total_dec = prefix_len + len(a_tok) + 1

        el = len(e_tok)
        enc_inputs[i, :el] = e_tok

        dec_inputs[i, 0] = eos_id
        dec_inputs[i, 1] = tool_call_id
        tl = len(t_tok)
        if tl > 0:
            dec_inputs[i, 2:2 + tl] = t_tok
        al = len(a_tok)
        if al > 0:
            dec_inputs[i, prefix_len:prefix_len + al] = a_tok

        # Decoder target: [<tool_call>, tools..., answer..., EOS]
        dec_targets[i, 0] = tool_call_id
        if tl > 0:
            dec_targets[i, 1:1 + tl] = t_tok
        if al > 0:
            dec_targets[i, prefix_len - 1:prefix_len - 1 + al] = a_tok
        dec_targets[i, prefix_len - 1 + al] = eos_id

        # Loss mask: 1 on answer positions + EOS only
        loss_mask[i, prefix_len - 1:prefix_len - 1 + al + 1] = 1.0

    if skipped > 0:
        # Remove skipped rows (all-pad)
        keep = enc_inputs[:, 0] != pad_id
        enc_inputs = enc_inputs[keep]
        dec_inputs = dec_inputs[keep]
        dec_targets = dec_targets[keep]
        loss_mask = loss_mask[keep]
        print(f"  Skipped {skipped} examples (too long for max_dec_len={max_dec_len})")

    # Save cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_path + "_enc.npy", enc_inputs)
    np.save(cache_path + "_dec_in.npy", dec_inputs)
    np.save(cache_path + "_dec_tgt.npy", dec_targets)
    np.save(cache_path + "_loss_mask.npy", loss_mask)
    print(f"Cached {len(enc_inputs):,} tool-call pairs to {CACHE_DIR}/{cache_id}")

    return enc_inputs, dec_inputs, dec_targets, loss_mask


def get_batches(enc_inputs, dec_inputs, dec_targets, batch_size, shuffle=True, loss_mask=None):
    n = len(enc_inputs)
    if shuffle:
        perm = np.random.permutation(n)
        enc_inputs = enc_inputs[perm]
        dec_inputs = dec_inputs[perm]
        dec_targets = dec_targets[perm]
        if loss_mask is not None:
            loss_mask = loss_mask[perm]

    for i in range(0, n - batch_size + 1, batch_size):
        batch = (enc_inputs[i : i + batch_size], dec_inputs[i : i + batch_size], dec_targets[i : i + batch_size])
        if loss_mask is not None:
            batch = batch + (loss_mask[i : i + batch_size],)
        yield batch



def load_tool_call_audio(split="train", max_samples=None):
    """Load tool-call dataset entries that have TTS audio.

    Applies the same 90/10 split as load_tool_calls, then filters to entries
    where the audio column is not None.

    Returns list of dicts: {query, answers, tools, audio_bytes, sampling_rate}.
    """
    ds = _load_unified_dataset()
    n = len(ds)

    if split in ("validation", "val", "test"):
        indices = list(range(int(n * 0.9), n))
    elif split == "train":
        indices = list(range(int(n * 0.9)))
    else:
        indices = list(range(n))

    import io
    import soundfile as sf

    pairs = []
    for idx in indices:
        ex = ds[idx]
        audio_val = ex.get("audio")
        if audio_val is None:
            continue
        # With decode=False, audio_val is {"bytes": b"...", "path": ...}
        raw_bytes = None
        if isinstance(audio_val, dict):
            raw_bytes = audio_val.get("bytes")
        elif isinstance(audio_val, bytes):
            raw_bytes = audio_val
        if raw_bytes is None:
            continue
        audio_array, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        pairs.append({
            "query": ex["query"],
            "answers": ex["answers"],
            "tools": ex["tools"],
            "audio_array": audio_array.astype(np.float32),
            "sampling_rate": sr,
        })
        if max_samples and len(pairs) >= max_samples:
            break

    return pairs


def compute_mel_spectrogram(audio, sr=16000, n_mels=80, n_fft=400, hop_length=160):
    """Compute log-mel spectrogram using numpy/scipy. Returns (T_mel, n_mels) float32.

    25ms window (n_fft=400 at 16kHz), 10ms hop (hop_length=160) → ~100 frames/sec.
    """
    from scipy.signal import windows as scipy_windows
    from scipy.fft import rfft

    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # STFT
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

    # Mel filterbank
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


def _speech_cache_key(n_samples, n_mels, max_mel_len, max_dec_len):
    tok_hash = _tokenizer_hash()
    key = f"speech_{tok_hash}_{n_samples}_{n_mels}_{max_mel_len}_{max_dec_len}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def build_audio_augmenter(sr=16000):
    """Build an audiomentations augmentation pipeline for training.

    Returns an augmenter callable or None if audiomentations is unavailable.
    """
    try:
        import audiomentations as A
    except ImportError:
        print("  WARNING: audiomentations not installed — no waveform augmentation")
        return None

    return A.Compose([
        A.AddGaussianSNR(min_snr_db=10.0, max_snr_db=35.0, p=0.5),
        A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
        A.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
        A.PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
        A.Gain(min_gain_db=-6, max_gain_db=6, p=0.4),
        A.LowPassFilter(min_cutoff_freq=3000, max_cutoff_freq=7500, p=0.2),
        A.HighPassFilter(min_cutoff_freq=50, max_cutoff_freq=400, p=0.2),
        A.ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=5, p=0.1),
    ])


def prepare_voice_tool_call_pairs(pairs, tokenizer, n_mels=80, max_mel_len=1024, max_dec_len=1024):
    """Prepare voice-tool-call decoder arrays with <tool_call> task token.

    Caches only decoder tokens + loss mask + audio arrays (NOT mels).
    Mel spectrograms are computed on-the-fly by get_speech_batches to allow
    online waveform augmentation during training.

    Returns (audio_arrays, dec_inputs, dec_targets, loss_mask).
    """

    cache_id = _speech_cache_key(len(pairs), n_mels, max_mel_len, max_dec_len)
    cache_path = os.path.join(CACHE_DIR, cache_id)
    arrays_file = cache_path + "_arrays.npy"
    if os.path.exists(arrays_file):
        print(f"Loading cached voice-tool-call data ({cache_id})...")
        return (
            np.load(arrays_file, allow_pickle=True),
            np.load(cache_path + "_dec_in.npy"),
            np.load(cache_path + "_dec_tgt.npy"),
            np.load(cache_path + "_loss_mask.npy"),
        )

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    tool_call_id = tokenizer.tool_call_token_id

    audio_arrays = []
    dec_inputs_list = []
    dec_targets_list = []
    loss_mask_list = []

    skipped = 0
    for pair in tqdm(pairs, desc="Processing voice-tool-call"):
        t_tok = tokenizer.encode(pair["tools"])[:max_dec_len - 2]
        a_tok = tokenizer.encode(pair["answers"])

        prefix_len = 2 + len(t_tok)
        total_dec = prefix_len + len(a_tok) + 1

        if total_dec > max_dec_len:
            available_for_tools = max_dec_len - 2 - len(a_tok) - 1
            if available_for_tools < 1:
                skipped += 1
                continue
            t_tok = t_tok[:available_for_tools]
            prefix_len = 2 + len(t_tok)

        dec_in = np.full(max_dec_len, pad_id, dtype=np.int32)
        dec_in[0] = eos_id
        dec_in[1] = tool_call_id
        tl = len(t_tok)
        if tl > 0:
            dec_in[2:2 + tl] = t_tok
        al = len(a_tok)
        if al > 0:
            dec_in[prefix_len:prefix_len + al] = a_tok

        # Decoder target: [<tool_call>, tools..., answer..., EOS]
        dec_tgt = np.full(max_dec_len, pad_id, dtype=np.int32)
        dec_tgt[0] = tool_call_id
        if tl > 0:
            dec_tgt[1:1 + tl] = t_tok
        if al > 0:
            dec_tgt[prefix_len - 1:prefix_len - 1 + al] = a_tok
        dec_tgt[prefix_len - 1 + al] = eos_id

        # Loss mask: 1 on answer positions + EOS only
        lm = np.zeros(max_dec_len, dtype=np.float32)
        lm[prefix_len - 1:prefix_len - 1 + al + 1] = 1.0

        audio_arrays.append(pair["audio_array"])
        dec_inputs_list.append(dec_in)
        dec_targets_list.append(dec_tgt)
        loss_mask_list.append(lm)

    if skipped > 0:
        print(f"  Skipped {skipped} examples (too long for max_dec_len={max_dec_len})")

    audio_arrays = np.array(audio_arrays, dtype=object)
    dec_inputs = np.stack(dec_inputs_list)
    dec_targets = np.stack(dec_targets_list)
    loss_mask = np.stack(loss_mask_list)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(arrays_file, audio_arrays)
    np.save(cache_path + "_dec_in.npy", dec_inputs)
    np.save(cache_path + "_dec_tgt.npy", dec_targets)
    np.save(cache_path + "_loss_mask.npy", loss_mask)
    print(f"Cached {len(audio_arrays):,} voice-tool-call pairs to {CACHE_DIR}/{cache_id}")

    return audio_arrays, dec_inputs, dec_targets, loss_mask


def _load_mel_batch(audio_arrays, n_mels, max_mel_len, augmenter=None, sr=16000):
    """Compute mel spectrograms for a batch of audio arrays.

    If augmenter is provided, applies waveform augmentation before mel computation.
    """
    mels = []
    for audio in audio_arrays:
        audio = np.array(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if augmenter is not None:
            audio = augmenter(samples=audio, sample_rate=sr)

        mel = compute_mel_spectrogram(audio, sr=sr, n_mels=n_mels)

        if mel.shape[0] > max_mel_len:
            mel = mel[:max_mel_len]
        elif mel.shape[0] < max_mel_len:
            pad_len = max_mel_len - mel.shape[0]
            mel = np.pad(mel, ((0, pad_len), (0, 0)))

        mels.append(mel)

    return np.stack(mels).astype(np.float32)


def get_speech_batches(audio_arrays, dec_inputs, dec_targets, batch_size,
                       shuffle=True, loss_mask=None, n_mels=80, max_mel_len=1024,
                       augmenter=None):
    """Yield speech batches with on-the-fly mel computation and optional augmentation.

    Args:
        audio_arrays: array of audio waveform arrays (numpy float32)
        dec_inputs, dec_targets, loss_mask: pre-tokenized decoder arrays
        augmenter: audiomentations Compose pipeline (None = no augmentation)
    """
    n = len(audio_arrays)
    indices = np.random.permutation(n) if shuffle else np.arange(n)

    for i in range(0, n - batch_size + 1, batch_size):
        batch_idx = indices[i : i + batch_size]
        batch_audio = audio_arrays[batch_idx]
        mel_batch = _load_mel_batch(batch_audio, n_mels, max_mel_len, augmenter=augmenter)

        batch = (mel_batch, dec_inputs[batch_idx], dec_targets[batch_idx])
        if loss_mask is not None:
            batch = batch + (loss_mask[batch_idx],)
        yield batch
