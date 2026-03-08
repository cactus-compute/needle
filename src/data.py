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
from datasets import Audio, load_dataset, load_from_disk
from tqdm import tqdm
import sentencepiece as spm

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".data_cache")
TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tokenizer")
TOKENIZER_PREFIX = os.path.join(TOKENIZER_DIR, "needle")
TOOL_CALLS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tool_calls")

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2
UNK_ID = 3
TOOL_CALL_ID = 4
TRANSCRIBE_ID = 5


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


def train_tokenizer(vocab_size=8192, max_samples=None):
    """Train a SentencePiece BPE tokenizer on tool-calling corpus."""
    model_path = TOKENIZER_PREFIX + ".model"
    if os.path.exists(model_path):
        print(f"Tokenizer already exists at {model_path}")
        return model_path

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    ds = load_from_disk(TOOL_CALLS_DIR)
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


def get_tokenizer(max_samples=None):
    model_path = TOKENIZER_PREFIX + ".model"
    if not os.path.exists(model_path):
        train_tokenizer(max_samples=max_samples)
    return NeedleTokenizer(model_path)


def load_tool_calls(split="train", max_samples=None):
    """Load tool-calling dataset, splitting 90/10 for train/val."""
    ds = load_from_disk(TOOL_CALLS_DIR)
    n = len(ds)
    if split in ("validation", "val", "test"):
        ds = ds.select(range(int(n * 0.9), n))
    elif split == "train":
        ds = ds.select(range(int(n * 0.9)))
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


def _cache_key(prefix, n_samples, max_enc_len, max_dec_len):
    tok_hash = _tokenizer_hash()
    key = f"{prefix}_{tok_hash}_{n_samples}_{max_enc_len}_{max_dec_len}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def prepare_tool_call_pairs(ds, tokenizer, max_enc_len=1024, max_dec_len=512):
    """Prepare tool-call encoder-decoder pairs with <tool_call> task token.

    Encoder input: tokenize(query + " " + tools), truncated to max_enc_len.
    Decoder input:  [BOS, <tool_call>, answer_tokens...]
    Decoder target: [<tool_call>, answer_tokens..., EOS]
    """

    cache_id = _cache_key("toolcall", len(ds), max_enc_len, max_dec_len)
    cache_path = os.path.join(CACHE_DIR, cache_id)
    if os.path.exists(cache_path + "_enc.npy"):
        print(f"Loading cached tool-call data ({cache_id})...")
        return (
            np.load(cache_path + "_enc.npy"),
            np.load(cache_path + "_dec_in.npy"),
            np.load(cache_path + "_dec_tgt.npy"),
        )

    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    tool_call_id = tokenizer.tool_call_token_id

    # Build encoder texts (query + tools) and decoder texts (answers)
    enc_texts = [ex["query"] + " " + ex["tools"] for ex in ds]
    dec_texts = [ex["answers"] for ex in ds]

    num_workers = min(os.cpu_count() or 1, 8)
    model_path = TOKENIZER_PREFIX + ".model"

    # Parallel tokenization for encoder inputs
    chunk_size = max(1, len(enc_texts) // (num_workers * 4))
    enc_chunks = [enc_texts[i:i + chunk_size] for i in range(0, len(enc_texts), chunk_size)]

    print(f"Tokenizing encoder inputs ({num_workers} workers)...")
    with mp.Pool(num_workers, initializer=_init_worker,
                 initargs=(model_path, max_enc_len)) as pool:
        enc_results = pool.map(_tokenize_chunk, enc_chunks)
    all_enc_tokens = [tok for chunk in enc_results for tok in chunk]

    # Parallel tokenization for decoder inputs (max_dec_len - 2 for BOS + task token)
    dec_chunks = [dec_texts[i:i + chunk_size] for i in range(0, len(dec_texts), chunk_size)]

    print(f"Tokenizing decoder inputs ({num_workers} workers)...")
    with mp.Pool(num_workers, initializer=_init_worker,
                 initargs=(model_path, max_dec_len - 2)) as pool:
        dec_results = pool.map(_tokenize_chunk, dec_chunks)
    all_dec_tokens = [tok for chunk in dec_results for tok in chunk]

    n = len(ds)
    enc_inputs = np.full((n, max_enc_len), pad_id, dtype=np.int32)
    dec_inputs = np.full((n, max_dec_len), pad_id, dtype=np.int32)
    dec_targets = np.full((n, max_dec_len), pad_id, dtype=np.int32)

    for i in range(n):
        # Encoder: truncated tokens
        e_tok = all_enc_tokens[i]
        el = len(e_tok)
        enc_inputs[i, :el] = e_tok

        # Decoder answer tokens
        d_tok = all_dec_tokens[i]
        dl = len(d_tok)

        # Decoder input: [BOS, <tool_call>, answer_tokens...]
        dec_inputs[i, 0] = bos_id
        dec_inputs[i, 1] = tool_call_id
        if dl > 0:
            dec_inputs[i, 2:2 + dl] = d_tok

        # Decoder target: [<tool_call>, answer_tokens..., EOS]
        dec_targets[i, 0] = tool_call_id
        if dl > 0:
            dec_targets[i, 1:1 + dl] = d_tok
        dec_targets[i, 1 + dl] = eos_id

    # Save cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_path + "_enc.npy", enc_inputs)
    np.save(cache_path + "_dec_in.npy", dec_inputs)
    np.save(cache_path + "_dec_tgt.npy", dec_targets)
    print(f"Cached {n:,} tool-call pairs to {CACHE_DIR}/{cache_id}")

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


get_text_batches = get_batches


# ─── Speech pipeline ───────────────────────────────────────────────────────────

def load_librispeech(split="train", max_samples=None):
    """Load LibriSpeech-100 from HuggingFace (SPRINGLab/LibriSpeech-100).

    The dataset has a single 'train' split, so we partition it:
      - train / train.clean.100: first 90%
      - validation.clean / test.clean / validation / test: last 10%

    Audio decoding is disabled (to avoid torchcodec/torch dependency);
    prepare_speech_pairs decodes audio bytes via soundfile instead.
    """
    ds = load_dataset("SPRINGLab/LibriSpeech-100", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    n = len(ds)
    if split in ("validation.clean", "validation", "test.clean", "test"):
        ds = ds.select(range(int(n * 0.9), n))
    elif split.startswith("train"):
        ds = ds.select(range(int(n * 0.9)))
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def _decode_audio(audio_data, target_sr=16000):
    """Decode raw audio bytes from a datasets Audio(decode=False) field using soundfile."""
    import io
    import soundfile as sf

    audio_bytes = audio_data["bytes"]
    if audio_bytes is None:
        path = audio_data["path"]
        audio, sr = sf.read(path, dtype="float32")
    else:
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    # Convert stereo to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        from scipy.signal import resample
        num_samples = int(len(audio) * target_sr / sr)
        audio = resample(audio, num_samples).astype(np.float32)
        sr = target_sr

    return audio.astype(np.float32), sr


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


def prepare_speech_pairs(ds, tokenizer, n_mels=80, max_mel_len=1024, max_dec_len=128):
    """Prepare speech encoder-decoder pairs with <transcribe> task token."""

    cache_id = _speech_cache_key(len(ds), n_mels, max_mel_len, max_dec_len)
    cache_path = os.path.join(CACHE_DIR, cache_id)
    if os.path.exists(cache_path + "_mel.npy"):
        print(f"Loading cached speech data ({cache_id})...")
        return (
            np.load(cache_path + "_mel.npy"),
            np.load(cache_path + "_dec_in.npy"),
            np.load(cache_path + "_dec_tgt.npy"),
        )

    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    transcribe_id = tokenizer.transcribe_token_id

    mel_features = []
    dec_inputs_list = []
    dec_targets_list = []

    for example in tqdm(ds, desc="Processing speech"):
        audio, sr = _decode_audio(example["audio"])

        mel = compute_mel_spectrogram(audio, sr=sr, n_mels=n_mels)

        # Pad or truncate mel to max_mel_len
        if mel.shape[0] > max_mel_len:
            mel = mel[:max_mel_len]
        elif mel.shape[0] < max_mel_len:
            pad_len = max_mel_len - mel.shape[0]
            mel = np.pad(mel, ((0, pad_len), (0, 0)))

        # Tokenize transcript
        text = example["text"].strip().lower()
        tokens = tokenizer.encode(text)
        # Leave room for <transcribe> prefix and EOS
        tokens = tokens[:max_dec_len - 2]

        # Decoder input: [BOS, <transcribe>, tok1, tok2, ...]
        dec_in = np.full(max_dec_len, pad_id, dtype=np.int32)
        dec_in[0] = bos_id
        dec_in[1] = transcribe_id
        n_tok = len(tokens)
        if n_tok > 0:
            dec_in[2:2 + n_tok] = tokens

        # Decoder target: [<transcribe>, tok1, tok2, ..., EOS]
        dec_tgt = np.full(max_dec_len, pad_id, dtype=np.int32)
        dec_tgt[0] = transcribe_id
        if n_tok > 0:
            dec_tgt[1:1 + n_tok] = tokens
        dec_tgt[1 + n_tok] = eos_id

        mel_features.append(mel)
        dec_inputs_list.append(dec_in)
        dec_targets_list.append(dec_tgt)

    mel_features = np.stack(mel_features).astype(np.float32)
    dec_inputs = np.stack(dec_inputs_list)
    dec_targets = np.stack(dec_targets_list)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_path + "_mel.npy", mel_features)
    np.save(cache_path + "_dec_in.npy", dec_inputs)
    np.save(cache_path + "_dec_tgt.npy", dec_targets)
    print(f"Cached {len(mel_features):,} speech pairs to {CACHE_DIR}/{cache_id}")

    return mel_features, dec_inputs, dec_targets


def get_speech_batches(mel_features, dec_inputs, dec_targets, batch_size, shuffle=True):
    n = len(mel_features)
    if shuffle:
        perm = np.random.permutation(n)
        mel_features = mel_features[perm]
        dec_inputs = dec_inputs[perm]
        dec_targets = dec_targets[perm]

    for i in range(0, n - batch_size + 1, batch_size):
        yield mel_features[i : i + batch_size], dec_inputs[i : i + batch_size], dec_targets[i : i + batch_size]
