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
from datasets import Audio, Dataset, load_from_disk
from tqdm import tqdm
import sentencepiece as spm

import re as _re

def to_snake_case(name):
    """Convert camelCase, PascalCase, or dot.notation name to snake_case."""
    # Replace any non-alphanumeric/underscore characters with underscores
    s = _re.sub(r'[^a-zA-Z0-9_]+', '_', name)
    # Insert underscore before uppercase letters that follow lowercase/digits
    s = _re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    # Insert underscore between consecutive uppercase and uppercase+lowercase
    s = _re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    # Collapse multiple underscores and strip edges
    s = _re.sub(r'_+', '_', s)
    return s.lower().strip('_')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_DISK_CACHE_DIR = os.path.join(_PROJECT_ROOT, ".data_cache")
TOKENIZER_DIR = os.path.join(_PROJECT_ROOT, "tokenizer")
TOKENIZER_PREFIX = os.path.join(TOKENIZER_DIR, "needle")
_DISK_UNIFIED_DIR = os.path.join(_PROJECT_ROOT, "data", "tool_calls_unified")
_SHM_UNIFIED_DIR = os.path.join("/dev/shm", "needle_data", "tool_calls_unified")


_MIN_SHM_BYTES = 20 * 1024**3


def _shm_available():
    """Check if /dev/shm has enough space for RAM-backed storage."""
    shm = "/dev/shm"
    if os.path.isdir(shm):
        try:
            st = os.statvfs(shm)
            return st.f_bavail * st.f_frsize >= _MIN_SHM_BYTES
        except OSError:
            pass
    return False


def _pick_unified_dir():
    """Use /dev/shm for unified dataset when RAM is plentiful, else disk."""
    if os.path.isdir(_SHM_UNIFIED_DIR) and any(
        f.endswith(".arrow") for f in os.listdir(_SHM_UNIFIED_DIR)
    ):
        return _SHM_UNIFIED_DIR
    if _shm_available():
        return _SHM_UNIFIED_DIR
    return _DISK_UNIFIED_DIR


LOCAL_UNIFIED_DIR = _pick_unified_dir()


def _pick_cache_dir():
    """Use /dev/shm (tmpfs/RAM) when available and large enough, else disk."""
    if _shm_available():
        d = os.path.join("/dev/shm", "needle_cache")
        os.makedirs(d, exist_ok=True)
        return d
    os.makedirs(_DISK_CACHE_DIR, exist_ok=True)
    return _DISK_CACHE_DIR


def _setup_hf_cache():
    """Point HuggingFace download cache to /dev/shm when RAM is plentiful."""
    if _shm_available():
        hf_cache = os.path.join("/dev/shm", "needle_hf_cache")
        os.makedirs(hf_cache, exist_ok=True)
        os.environ.setdefault("HF_HOME", hf_cache)
        os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_cache, "datasets"))


_setup_hf_cache()


CACHE_DIR = _pick_cache_dir()

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2
UNK_ID = 3
TOOL_CALL_ID = 4
TOOLS_ID = 5
DEFER_ID = 6

DEFAULT_MAX_ENC_LEN = 1024
DEFAULT_MAX_DEC_LEN = 512
DEFAULT_MAX_GEN_LEN = 512

_unified_dataset_cache = None


def _mark_json_value(s, char_w, key, value_str, weight):
    """Find '"key": "value_str"' or '"key": value_str' in s, mark value chars."""
    pattern_str = f'"{_re.escape(key)}"\\s*:\\s*"{_re.escape(value_str)}"'
    for m in _re.finditer(pattern_str, s):
        tail = s[m.start() + len(f'"{key}"'):m.end()]
        val_offset = tail.index(f'"{value_str}"') + 1
        val_start = m.start() + len(f'"{key}"') + val_offset
        val_end = val_start + len(value_str)
        char_w[val_start:val_end] = np.maximum(char_w[val_start:val_end], weight)
        return

    pattern_ns = f'"{_re.escape(key)}"\\s*:\\s*{_re.escape(value_str)}'
    for m in _re.finditer(pattern_ns, s):
        colon_offset = s[m.start():m.end()].index(':')
        val_start = m.start() + colon_offset + 1
        while val_start < m.end() and s[val_start] == ' ':
            val_start += 1
        val_end = m.end()
        char_w[val_start:val_end] = np.maximum(char_w[val_start:val_end], weight)
        return


def _mark_json_key_in_args(s, char_w, key, weight):
    """Mark the argument key string (inside quotes) in the JSON."""
    for m in _re.finditer(f'"{_re.escape(key)}"\\s*:', s):
        char_w[m.start() + 1:m.start() + 1 + len(key)] = np.maximum(
            char_w[m.start() + 1:m.start() + 1 + len(key)], weight)


def _count_tool_calls(answers_str):
    """Count the number of tool calls in an answers JSON string."""
    try:
        calls = _json.loads(answers_str)
    except (ValueError, TypeError):
        return 0
    return len(calls) if isinstance(calls, list) else 0


def _shuffle_tools_json(tools_str, seed):
    """Parse tools JSON array, shuffle order deterministically, re-serialize."""
    try:
        tools = _json.loads(tools_str)
    except (ValueError, TypeError):
        return tools_str
    if not isinstance(tools, list) or len(tools) <= 1:
        return tools_str
    rng = np.random.RandomState(seed)
    rng.shuffle(tools)
    return _json.dumps(tools, separators=(",", ":"))


def _token_weights_for_answer(answer_str, token_ids, sp_model,
                               w_name=3.0, w_value=2.0, w_key=1.5):
    """Compute per-token loss weights based on JSON structure.

    Parses the answer JSON to identify character spans of tool names,
    argument keys, and argument values, then maps to token-level weights
    using SentencePiece byte offsets.
    """
    n = len(token_ids)
    weights = np.ones(n, dtype=np.float32)

    try:
        calls = _json.loads(answer_str)
    except (ValueError, TypeError):
        return weights

    if not isinstance(calls, list):
        return weights

    char_w = np.ones(len(answer_str), dtype=np.float32)

    for call in calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name", "")
        if name:
            _mark_json_value(answer_str, char_w, "name", name, w_name)

        args = call.get("arguments", {})
        if isinstance(args, dict):
            for k, v in args.items():
                _mark_json_key_in_args(answer_str, char_w, k, w_key)
                v_str = _json.dumps(v) if not isinstance(v, str) else v
                _mark_json_value(answer_str, char_w, k, v_str, w_value)

    pieces = sp_model.Encode(answer_str, out_type=str)
    pos = 0
    for i, piece in enumerate(pieces):
        if i >= n:
            break
        raw = piece.replace('\u2581', ' ')
        plen = len(raw)
        if plen > 0 and pos + plen <= len(answer_str):
            token_w = char_w[pos:pos + plen].max()
            weights[i] = token_w
            pos += plen
        else:
            pos += max(plen, 1)

    return weights


def _load_unified_dataset():
    """Load the unified dataset from /dev/shm or disk.

    Checks /dev/shm first (RAM-backed), then falls back to disk.
    Caches the result in memory after first load.
    Uses soundfile as the audio decoding backend.
    """
    global _unified_dataset_cache
    if _unified_dataset_cache is not None:
        return _unified_dataset_cache

    for path in [_SHM_UNIFIED_DIR, _DISK_UNIFIED_DIR]:
        if os.path.exists(path) and any(
            f.endswith(".arrow") for f in os.listdir(path)
        ):
            try:
                ds = load_from_disk(path)
                print(f"Loaded unified dataset from {path} ({len(ds)} rows)")
                _unified_dataset_cache = _set_audio_backend(ds)
                return _unified_dataset_cache
            except Exception as e:
                print(f"Warning: failed to load from {path}: {e}")
                continue

    try:
        from .gcs import download_synth_data
        target = _SHM_UNIFIED_DIR if _shm_available() else _DISK_UNIFIED_DIR
        if download_synth_data(target):
            ds = load_from_disk(target)
            print(f"Loaded synth dataset from GCS -> {target} ({len(ds)} rows)")
            _unified_dataset_cache = _set_audio_backend(ds)
            return _unified_dataset_cache
    except Exception as e:
        print(f"Warning: GCS download failed: {e}")

    raise FileNotFoundError(
        f"Dataset not found at {_SHM_UNIFIED_DIR} or {_DISK_UNIFIED_DIR}. "
        f"Run 'needle tokenize' or 'python scripts/synthesize_tools_data.py' first."
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
    def tools_token_id(self):
        return TOOLS_ID

    @property
    def defer_token_id(self):
        return DEFER_ID

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
        user_defined_symbols=["<tool_call>", "<tools>", "<defer>"],
        byte_fallback=True,
        normalization_rule_name="identity",
        num_threads=min(20, max(1, (os.cpu_count() or 1) // 4)),
        train_extremely_large_corpus=False,
        minloglevel=2,
    )

    os.remove(corpus_path)
    print(f"Tokenizer saved to {model_path}")
    return model_path


def get_tokenizer(max_samples=None):
    model_path = TOKENIZER_PREFIX + ".model"
    if not os.path.exists(model_path):
        try:
            from .gcs import download_tokenizer
            download_tokenizer(TOKENIZER_DIR)
        except Exception:
            pass
    if not os.path.exists(model_path):
        train_tokenizer(max_samples=max_samples)
    return NeedleTokenizer(model_path)


def load_tool_calls(split="train", max_samples=None, return_global_indices=False):
    """Load tool-calling dataset with shuffled train/val split.

    Shuffles the full dataset with a fixed seed, then splits:
      - val: min(5000, 10% of total)
      - train: everything else

    If return_global_indices is True, also return a numpy array mapping each
    split-local row position back to its row id in the full unified dataset.
    """
    ds = _load_unified_dataset()
    n = len(ds)

    rng = np.random.RandomState(42)
    perm = rng.permutation(n)

    val_size = min(5000, int(n * 0.1))
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]

    if split in ("validation", "val", "test"):
        global_indices = val_indices.astype(np.int64)
    elif split == "train":
        global_indices = train_indices.astype(np.int64)
    else:
        global_indices = perm.astype(np.int64)

    ds = ds.select(global_indices.tolist())

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


def _cache_key(prefix, n_samples, max_enc_len, max_dec_len,
               w_name=3.0, w_value=2.0, w_key=1.5, shuffle_tools=True):
    tok_hash = _tokenizer_hash()
    key = f"{prefix}_{tok_hash}_{n_samples}_{max_enc_len}_{max_dec_len}_{w_name}_{w_value}_{w_key}_{shuffle_tools}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _save_cache_metadata(split, text_cache_id, n_samples, max_enc_len, max_dec_len):
    """Save metadata JSON for a split locally."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    meta = {
        "split": split,
        "text_cache_id": text_cache_id,
        "n_samples": n_samples,
        "max_enc_len": max_enc_len,
        "max_dec_len": max_dec_len,
    }
    meta_path = os.path.join(CACHE_DIR, f"{split}_metadata.json")
    with open(meta_path, "w") as f:
        _json.dump(meta, f)


def _load_cache_metadata(split):
    """Load metadata JSON from local cache. Returns dict or None."""
    meta_path = os.path.join(CACHE_DIR, f"{split}_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return _json.load(f)
    return None


def prepare_tool_call_pairs(ds, tokenizer, max_enc_len=DEFAULT_MAX_ENC_LEN, max_dec_len=DEFAULT_MAX_DEC_LEN,
                            w_name=3.0, w_value=2.0, w_key=1.5, shuffle_tools=True):
    """Prepare tool-call encoder-decoder pairs with <tool_call> task token.

    Encoder input: [query_tokens..., <tools>, tools_tokens...] truncated to max_enc_len.
    Decoder input:  [EOS, <tool_call>, answer_tokens..., PAD...]
    Decoder target: [<tool_call>, answer_tokens..., EOS, PAD...]
    Loss mask: token-level weights based on JSON structure (tool names, arg keys/values).

    Returns (enc_inputs, dec_inputs, dec_targets, loss_mask, kept_indices, tool_counts).
    """

    cache_id = _cache_key("toolcall", len(ds), max_enc_len, max_dec_len,
                          w_name, w_value, w_key, shuffle_tools)
    cache_path = os.path.join(CACHE_DIR, cache_id)

    tc_suffixes = ["_enc.npy", "_dec_in.npy", "_dec_tgt.npy", "_loss_mask.npy", "_kept_idx.npy", "_tool_count.npy"]

    def _load_tc_cache():
        tc_path = cache_path + "_tool_count.npy"
        tc = np.load(tc_path) if os.path.exists(tc_path) else None
        return (
            np.load(cache_path + "_enc.npy"),
            np.load(cache_path + "_dec_in.npy"),
            np.load(cache_path + "_dec_tgt.npy"),
            np.load(cache_path + "_loss_mask.npy"),
            np.load(cache_path + "_kept_idx.npy"),
            tc,
        )

    if os.path.exists(cache_path + "_enc.npy"):
        print(f"Loading cached tool-call data ({cache_id})...")
        return _load_tc_cache()

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    tool_call_id = tokenizer.tool_call_token_id
    tools_sep_id = tokenizer.tools_token_id

    defer_id = tokenizer.defer_token_id

    enc_texts = [ex["query"] for ex in ds]
    tools_texts = [ex["tools"] for ex in ds]
    ans_texts = [ex["answers"] for ex in ds]

    def _compact(s):
        try:
            return _json.dumps(_json.loads(s), separators=(",", ":"))
        except (ValueError, TypeError):
            return s

    ans_texts = [_compact(a) for a in ans_texts]
    tools_texts = [_compact(t) for t in tools_texts]

    empty_tools_mask = []
    for t in tools_texts:
        try:
            parsed = _json.loads(t)
            empty_tools_mask.append(isinstance(parsed, list) and len(parsed) == 0)
        except (ValueError, TypeError):
            empty_tools_mask.append(False)

    tool_counts = np.array([_count_tool_calls(a) for a in ans_texts], dtype=np.int32)

    if shuffle_tools:
        tools_texts = [_shuffle_tools_json(t, seed=i) for i, t in enumerate(tools_texts)]

    num_workers = min(20, max(1, (os.cpu_count() or 1) // 4))
    model_path = TOKENIZER_PREFIX + ".model"
    chunk_size = max(1, len(enc_texts) // (num_workers * 4))

    enc_chunks = [enc_texts[i:i + chunk_size] for i in range(0, len(enc_texts), chunk_size)]
    with mp.Pool(num_workers, initializer=_init_worker,
                 initargs=(model_path, max_enc_len)) as pool:
        enc_results = list(tqdm(pool.imap(_tokenize_chunk, enc_chunks),
                                total=len(enc_chunks), desc="Tokenizing queries"))
    all_enc_tokens = [tok for chunk in enc_results for tok in chunk]

    tools_chunks = [tools_texts[i:i + chunk_size] for i in range(0, len(tools_texts), chunk_size)]
    with mp.Pool(num_workers, initializer=_init_worker,
                 initargs=(model_path, max_enc_len)) as pool:
        tools_results = list(tqdm(pool.imap(_tokenize_chunk, tools_chunks),
                                  total=len(tools_chunks), desc="Tokenizing tools"))
    all_tools_tokens = [tok for chunk in tools_results for tok in chunk]

    ans_chunks = [ans_texts[i:i + chunk_size] for i in range(0, len(ans_texts), chunk_size)]
    with mp.Pool(num_workers, initializer=_init_worker,
                 initargs=(model_path, max_dec_len)) as pool:
        ans_results = list(tqdm(pool.imap(_tokenize_chunk, ans_chunks),
                                total=len(ans_chunks), desc="Tokenizing answers"))
    all_ans_tokens = [tok for chunk in ans_results for tok in chunk]

    n = len(ds)

    # Track which samples have empty answers (no tool calls to make)
    empty_ans_mask = []
    for a in ans_texts:
        stripped = a.strip()
        empty_ans_mask.append(stripped in ("", "[]"))

    def _fill_sample(j, e_tok, t_tok, a_tok, ans_text, is_empty_tools, is_empty_ans,
                     enc_arr, dec_in_arr, dec_tgt_arr, lm_arr):
        """Fill arrays at position j for one sample. Returns False if skipped."""
        # Encoder: [query..., <tools>, tools...] truncated to max_enc_len
        # For empty tools, use <defer> token instead of tokenized "[]"
        if is_empty_tools:
            t_tok = [defer_id]

        # Reserve 1 slot for <tools> separator + at least 1 token for tools
        max_query = max_enc_len - 1 - 1  # room for <tools> + at least 1 tools token
        if len(e_tok) > max_query:
            e_tok = e_tok[:max_query]
        remaining = max_enc_len - len(e_tok) - 1  # 1 for <tools>
        t_trunc = t_tok[:remaining]

        enc_seq = e_tok + [tools_sep_id] + t_trunc
        el = len(enc_seq)
        enc_arr[j, :el] = enc_seq

        if is_empty_ans:
            # Empty answer: decoder input [EOS, <defer>], target [<defer>, EOS]
            dec_in_arr[j, 0] = eos_id
            dec_in_arr[j, 1] = defer_id
            dec_tgt_arr[j, 0] = defer_id
            dec_tgt_arr[j, 1] = eos_id
            lm_arr[j, 0] = 3.0  # High weight on <defer> prediction
            lm_arr[j, 1] = 1.0  # EOS
        else:
            al = len(a_tok)
            # Decoder needs: [EOS, <tool_call>, answer..., EOS] = 2 + al + 1
            if 2 + al + 1 > max_dec_len:
                return False

            # Decoder input: [EOS, <tool_call>, answer...]
            dec_in_arr[j, 0] = eos_id
            dec_in_arr[j, 1] = tool_call_id
            if al > 0:
                dec_in_arr[j, 2:2 + al] = a_tok

            # Decoder target: [<tool_call>, answer..., EOS]
            dec_tgt_arr[j, 0] = tool_call_id
            if al > 0:
                dec_tgt_arr[j, 1:1 + al] = a_tok
            dec_tgt_arr[j, 1 + al] = eos_id

            # Token-level loss weighting
            token_weights = _token_weights_for_answer(ans_text, a_tok, tokenizer.sp,
                                                       w_name=w_name, w_value=w_value, w_key=w_key)
            lm_arr[j, 1:1 + len(token_weights)] = token_weights
            lm_arr[j, 1 + al] = 1.0  # EOS stays at baseline weight
        return True

    os.makedirs(CACHE_DIR, exist_ok=True)

    enc_inputs = np.full((n, max_enc_len), pad_id, dtype=np.int32)
    dec_inputs = np.full((n, max_dec_len), pad_id, dtype=np.int32)
    dec_targets = np.full((n, max_dec_len), pad_id, dtype=np.int32)
    loss_mask = np.zeros((n, max_dec_len), dtype=np.float32)

    skipped = 0
    for i in tqdm(range(n), desc="Building pairs"):
        if not _fill_sample(i, all_enc_tokens[i], all_tools_tokens[i],
                            all_ans_tokens[i], ans_texts[i], empty_tools_mask[i],
                            empty_ans_mask[i],
                            enc_inputs, dec_inputs, dec_targets, loss_mask):
            skipped += 1

    if skipped > 0:
        keep = enc_inputs[:, 0] != pad_id
        kept_indices = np.where(keep)[0].astype(np.int64)
        enc_inputs = enc_inputs[keep]
        dec_inputs = dec_inputs[keep]
        dec_targets = dec_targets[keep]
        loss_mask = loss_mask[keep]
        tool_counts = tool_counts[keep]
        print(f"  Skipped {skipped} examples (too long for max_dec_len={max_dec_len})")
    else:
        kept_indices = np.arange(n, dtype=np.int64)

    np.save(cache_path + "_enc.npy", enc_inputs)
    np.save(cache_path + "_dec_in.npy", dec_inputs)
    np.save(cache_path + "_dec_tgt.npy", dec_targets)
    np.save(cache_path + "_loss_mask.npy", loss_mask)
    np.save(cache_path + "_kept_idx.npy", kept_indices)
    np.save(cache_path + "_tool_count.npy", tool_counts)
    print(f"Cached {len(enc_inputs):,} tool-call pairs to {CACHE_DIR}/{cache_id}")

    max_tool_len = getattr(prepare_tool_call_pairs, '_max_tool_len', 256)
    _build_contrastive_arrays(ds, enc_texts, tools_texts, cache_path,
                              model_path, max_enc_len, max_tool_len,
                              num_workers, chunk_size, kept_indices)

    return enc_inputs, dec_inputs, dec_targets, loss_mask, kept_indices, tool_counts


def _build_contrastive_arrays(ds, enc_texts, tools_texts, cache_path,
                              model_path, max_enc_len, max_tool_len,
                              num_workers, chunk_size, kept_indices):
    """Build and save contrastive arrays: query-only tokens + individual tool tokens."""
    kept_set = set(kept_indices.tolist()) if kept_indices is not None else set(range(len(enc_texts)))
    kept_queries = [enc_texts[i] for i in range(len(enc_texts)) if i in kept_set]

    q_chunks = [kept_queries[i:i + chunk_size] for i in range(0, len(kept_queries), chunk_size)]
    with mp.Pool(num_workers, initializer=_init_worker,
                 initargs=(model_path, max_enc_len)) as pool:
        q_results = list(tqdm(pool.imap(_tokenize_chunk, q_chunks),
                              total=len(q_chunks), desc="Tokenizing queries (contrastive)"))
    query_tokens_list = [tok for chunk in q_results for tok in chunk]
    n_kept = len(query_tokens_list)
    query_arr = np.full((n_kept, max_enc_len), PAD_ID, dtype=np.int32)
    for i, toks in enumerate(query_tokens_list):
        l = min(len(toks), max_enc_len)
        query_arr[i, :l] = toks[:l]
    np.save(cache_path + "_query_only.npy", query_arr)

    # Individual tool tokens + mapping arrays
    tool_texts_individual = []
    tool_ex_idx = []
    tool_is_pos = []

    kept_list = sorted(kept_set)
    kept_to_local = {g: i for i, g in enumerate(kept_list)}

    for global_i in kept_list:
        local_i = kept_to_local[global_i]
        tools_str = tools_texts[global_i]
        ans_str = ds[global_i]["answers"] if global_i < len(ds) else "[]"

        try:
            tools = _json.loads(tools_str)
        except (ValueError, TypeError):
            tools = []
        if not isinstance(tools, list):
            tools = []

        try:
            calls = _json.loads(ans_str)
        except (ValueError, TypeError):
            calls = []
        pos_names = set()
        if isinstance(calls, list):
            for c in calls:
                if isinstance(c, dict) and "name" in c:
                    pos_names.add(c["name"])

        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_str = _json.dumps(tool, separators=(",", ":"))
            tool_texts_individual.append(tool_str)
            tool_ex_idx.append(local_i)
            tool_is_pos.append(tool.get("name", "") in pos_names)

    if tool_texts_individual:
        t_chunks = [tool_texts_individual[i:i + chunk_size]
                     for i in range(0, len(tool_texts_individual), chunk_size)]
        with mp.Pool(num_workers, initializer=_init_worker,
                     initargs=(model_path, max_tool_len)) as pool:
            t_results = list(tqdm(pool.imap(_tokenize_chunk, t_chunks),
                                  total=len(t_chunks), desc="Tokenizing tools (contrastive)"))
        tool_tokens_list = [tok for chunk in t_results for tok in chunk]

        n_tools = len(tool_tokens_list)
        tool_arr = np.full((n_tools, max_tool_len), PAD_ID, dtype=np.int32)
        for i, toks in enumerate(tool_tokens_list):
            l = min(len(toks), max_tool_len)
            tool_arr[i, :l] = toks[:l]
    else:
        tool_arr = np.zeros((0, max_tool_len), dtype=np.int32)

    np.save(cache_path + "_tool_individual.npy", tool_arr)
    np.save(cache_path + "_tool_ex_idx.npy", np.array(tool_ex_idx, dtype=np.int32))
    np.save(cache_path + "_tool_is_pos.npy", np.array(tool_is_pos, dtype=np.bool_))
    print(f"  Contrastive: {n_kept} queries, {len(tool_ex_idx)} individual tools")


def get_contrastive_batches(query_tokens, tool_tokens, tool_ex_idx, tool_is_pos, batch_size):
    """Yield (query_batch, tool_batch) for CLIP-style contrastive training.

    Each batch: B queries paired with 1 randomly-chosen positive tool each.
    In-batch negatives provide the contrastive signal.
    """
    n_queries = len(query_tokens)
    pos_map = {}
    for t_idx in range(len(tool_ex_idx)):
        if tool_is_pos[t_idx]:
            pos_map.setdefault(int(tool_ex_idx[t_idx]), []).append(t_idx)

    valid_queries = [i for i in range(n_queries) if i in pos_map]
    if not valid_queries:
        return

    indices = np.array(valid_queries)
    np.random.shuffle(indices)

    for start in range(0, len(indices) - batch_size + 1, batch_size):
        batch_q_idx = indices[start:start + batch_size]
        q_batch = np.array(query_tokens[batch_q_idx])
        t_indices = np.array([
            pos_map[int(qi)][np.random.randint(len(pos_map[int(qi)]))]
            for qi in batch_q_idx
        ])
        t_batch = np.array(tool_tokens[t_indices])
        yield q_batch, t_batch


def get_batches(enc_inputs, dec_inputs, dec_targets, batch_size, shuffle=True, loss_mask=None, tool_counts=None):
    n = len(enc_inputs)
    if tool_counts is not None:
        order = np.argsort(tool_counts, kind='stable')
        for c in np.unique(tool_counts):
            group = np.where(tool_counts[order] == c)[0]
            shuffled = order[group].copy()
            np.random.shuffle(shuffled)
            order[group] = shuffled
        indices = order
    elif shuffle:
        indices = np.random.permutation(n)
    else:
        indices = np.arange(n)
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


def load_example_with_audio(idx):
    """Load a dataset example by global index.

    Returns dict with {query, answers, tools, audio_array, sampling_rate}.
    Audio fields are None for text-only datasets.
    """
    ds = _load_unified_dataset()
    if idx < 0 or idx >= len(ds):
        raise IndexError(f"Index {idx} out of range (dataset has {len(ds)} rows)")
    ex = ds[idx]

    audio, sr = None, None
    audio_val = ex.get("audio")
    if audio_val is not None:
        import io
        import soundfile as sf

        raw_bytes = None
        if isinstance(audio_val, dict):
            raw_bytes = audio_val.get("bytes")
        elif isinstance(audio_val, bytes):
            raw_bytes = audio_val
        if raw_bytes is not None:
            audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

    return {
        "query": ex["query"],
        "answers": ex["answers"],
        "tools": ex["tools"],
        "audio_array": audio,
        "sampling_rate": sr,
    }


def compute_mel_spectrogram(audio, sr=16000, n_mels=80, n_fft=400, hop_length=160):
    """Compute log-mel spectrogram using numpy/scipy. Returns (T_mel, n_mels) float32.

    25ms window (n_fft=400 at 16kHz), 10ms hop (hop_length=160) -> ~100 frames/sec.
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

    Tries local cache only. Raises FileNotFoundError if not found
    (run 'needle tokenize' first).

    Returns dict with keys: enc_inputs, dec_inputs, dec_targets, loss_mask,
    kept_indices.
    """
    meta = _load_cache_metadata(split)
    if meta is None:
        try:
            from .gcs import download_tokenized_data
            download_tokenized_data(CACHE_DIR)
            meta = _load_cache_metadata(split)
        except Exception:
            pass
    if meta is None:
        raise FileNotFoundError(
            f"No prepared data found for split '{split}'. Run 'needle tokenize' first."
        )

    text_cache_id = meta["text_cache_id"]
    cache_path = os.path.join(CACHE_DIR, text_cache_id)

    if not os.path.exists(cache_path + "_enc.npy"):
        raise FileNotFoundError(
            f"Text cache '{text_cache_id}' not found. Run 'needle tokenize' first."
        )

    mmap_mode = "r" if mmap else None
    tc_path = cache_path + "_tool_count.npy"
    tc = np.load(tc_path, mmap_mode=mmap_mode) if os.path.exists(tc_path) else None
    def _load_optional(suffix):
        p = cache_path + suffix
        if os.path.exists(p):
            return np.load(p, mmap_mode=mmap_mode)
        return None

    return {
        "enc_inputs": np.load(cache_path + "_enc.npy", mmap_mode=mmap_mode),
        "dec_inputs": np.load(cache_path + "_dec_in.npy", mmap_mode=mmap_mode),
        "dec_targets": np.load(cache_path + "_dec_tgt.npy", mmap_mode=mmap_mode),
        "loss_mask": np.load(cache_path + "_loss_mask.npy", mmap_mode=mmap_mode),
        "kept_indices": np.load(cache_path + "_kept_idx.npy", mmap_mode=mmap_mode),
        "mel_cache_id": meta.get("mel_cache_id"),
        "tool_counts": tc,
        "query_only": _load_optional("_query_only.npy"),
        "tool_individual": _load_optional("_tool_individual.npy"),
        "tool_ex_idx": _load_optional("_tool_ex_idx.npy"),
        "tool_is_pos": _load_optional("_tool_is_pos.npy"),
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
