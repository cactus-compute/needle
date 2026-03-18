import hashlib
import json as _json
import multiprocessing as mp
import os
import queue
import threading
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

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
LOCAL_UNIFIED_DIR = os.path.join(_PROJECT_ROOT, "data", "tool_calls_unified")
CACHE_DIR = os.path.join(_PROJECT_ROOT, ".data_cache")

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2
UNK_ID = 3
TOOL_CALL_ID = 4
TOOLS_ID = 5

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
    """Load the unified dataset from disk.

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
        except Exception as e:
            print(f"Warning: failed to load from {LOCAL_UNIFIED_DIR}: {e}")

    try:
        from datasets import load_dataset as _hf_load
        print("Downloading dataset from HuggingFace (Cactus-Compute/tool-calls)...")
        ds = _hf_load("Cactus-Compute/tool-calls", split="train", token=True)
        os.makedirs(LOCAL_UNIFIED_DIR, exist_ok=True)
        ds.save_to_disk(LOCAL_UNIFIED_DIR)
        print(f"Loaded from HuggingFace -> {LOCAL_UNIFIED_DIR} ({len(ds)} rows)")
        _unified_dataset_cache = _set_audio_backend(ds)
        return _unified_dataset_cache
    except Exception as e:
        print(f"Warning: HuggingFace download failed: {e}")

    raise FileNotFoundError(
        f"Dataset not found at {LOCAL_UNIFIED_DIR}. "
        f"Run 'needle tokenize' first."
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
    try:
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
            user_defined_symbols=["<tool_call>", "<tools>"],
            byte_fallback=True,
            normalization_rule_name="identity",
            num_threads=min(20, max(1, (os.cpu_count() or 1) // 4)),
            train_extremely_large_corpus=False,
            minloglevel=2,
        )
    finally:
        if os.path.exists(corpus_path):
            os.remove(corpus_path)
    print(f"Tokenizer saved to {model_path}")
    return model_path


_HF_TOKENIZER_REPO = "Cactus-Compute/needle-tokenizer"
_HF_TOKENIZED_REPO = "Cactus-Compute/tokenized-tool-calls"


def _download_tokenized_from_hf():
    """Download tokenized .npy files from HuggingFace Hub into CACHE_DIR."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        _HF_TOKENIZED_REPO,
        repo_type="dataset",
        token=True,
    )
    os.makedirs(CACHE_DIR, exist_ok=True)
    for fname in os.listdir(local):
        if fname.startswith("."):
            continue
        dst = os.path.join(CACHE_DIR, fname)
        if not os.path.exists(dst):
            os.symlink(os.path.join(local, fname), dst)


def _download_tokenizer_from_hf():
    """Download tokenizer files from HuggingFace Hub into TOKENIZER_DIR."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        _HF_TOKENIZER_REPO,
        repo_type="dataset",
    )
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    for fname in os.listdir(local):
        if fname.startswith("."):
            continue
        dst = os.path.join(TOKENIZER_DIR, fname)
        if not os.path.exists(dst):
            os.symlink(os.path.join(local, fname), dst)


def get_tokenizer(max_samples=None):
    model_path = TOKENIZER_PREFIX + ".model"
    if not os.path.exists(model_path):
        print("Downloading pretrained tokenizer from HuggingFace...")
        _download_tokenizer_from_hf()
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


def _save_cache_metadata(split, text_cache_id, n_samples, max_enc_len, max_dec_len, max_tool_len=256):
    """Save metadata JSON for a split locally."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    meta = {
        "split": split,
        "text_cache_id": text_cache_id,
        "n_samples": n_samples,
        "max_enc_len": max_enc_len,
        "max_dec_len": max_dec_len,
        "max_tool_len": max_tool_len,
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

    Stores variable-length sequences as flat int16/float16 arrays with offsets
    for compact on-disk representation (~5-7x smaller than padded int32).
    Padding to fixed max lengths happens at batch time via VarLenArray.

    Returns (enc, dec_in, dec_tgt, loss, kept_indices, tool_counts) where
    the first four are VarLenArray objects when loaded from cache.
    """

    cache_id = _cache_key("toolcall", len(ds), max_enc_len, max_dec_len,
                          w_name, w_value, w_key, shuffle_tools)
    cache_path = os.path.join(CACHE_DIR, cache_id)

    if os.path.exists(cache_path + "_enc_data.npy"):
        print(f"Loading cached tool-call data ({cache_id})...")
        enc = VarLenArray.load(cache_path + "_enc", max_enc_len, pad_value=PAD_ID)
        dec_in = VarLenArray.load(cache_path + "_dec_in", max_dec_len, pad_value=PAD_ID)
        dec_tgt = VarLenArray.load(cache_path + "_dec_tgt", max_dec_len, pad_value=PAD_ID)
        loss = VarLenArray.load(cache_path + "_loss", max_dec_len, pad_value=0)
        kept = np.load(cache_path + "_kept_idx.npy")
        tc_path = cache_path + "_tool_count.npy"
        tc = np.load(tc_path) if os.path.exists(tc_path) else None
        return enc, dec_in, dec_tgt, loss, kept, tc

    eos_id = tokenizer.eos_token_id
    tool_call_id = tokenizer.tool_call_token_id
    tools_sep_id = tokenizer.tools_token_id

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
    enc_seqs = []
    dec_in_seqs = []
    dec_tgt_seqs = []
    loss_seqs = []
    query_lens_list = []
    kept_indices = []
    kept_tool_counts = []

    for i in tqdm(range(n), desc="Building pairs"):
        e_tok = all_enc_tokens[i]
        t_tok = all_tools_tokens[i]
        a_tok = all_ans_tokens[i]

        # Encoder: [query..., <tools>, tools...] truncated to max_enc_len
        max_query = max_enc_len - 1 - 1  # room for <tools> + at least 1 tools token
        if len(e_tok) > max_query:
            e_tok = e_tok[:max_query]
        remaining = max_enc_len - len(e_tok) - 1
        t_trunc = t_tok[:remaining]
        enc_seq = e_tok + [tools_sep_id] + t_trunc

        al = len(a_tok)
        if 2 + al + 1 > max_dec_len:
            continue  # skip samples too long for decoder

        # Decoder input: [EOS, <tool_call>, answer...]
        dec_in_seq = [eos_id, tool_call_id] + list(a_tok)
        # Decoder target: [<tool_call>, answer..., EOS]
        dec_tgt_seq = [tool_call_id] + list(a_tok) + [eos_id]

        # Token-level loss weighting
        weights = np.ones(len(dec_tgt_seq), dtype=np.float16)
        token_weights = _token_weights_for_answer(ans_texts[i], a_tok, tokenizer.sp,
                                                   w_name=w_name, w_value=w_value, w_key=w_key)
        weights[1:1 + len(token_weights)] = token_weights.astype(np.float16)
        weights[1 + al] = 1.0  # EOS at baseline

        enc_seqs.append(np.array(enc_seq, dtype=np.int16))
        dec_in_seqs.append(np.array(dec_in_seq, dtype=np.int16))
        dec_tgt_seqs.append(np.array(dec_tgt_seq, dtype=np.int16))
        loss_seqs.append(weights)
        query_lens_list.append(len(e_tok))
        kept_indices.append(i)
        kept_tool_counts.append(tool_counts[i])

    skipped = n - len(kept_indices)
    if skipped > 0:
        print(f"  Skipped {skipped} examples (too long for max_dec_len={max_dec_len})")

    kept_indices = np.array(kept_indices, dtype=np.int64)
    query_lens = np.array(query_lens_list, dtype=np.int32)
    kept_tool_counts = np.array(kept_tool_counts, dtype=np.int32)

    os.makedirs(CACHE_DIR, exist_ok=True)
    _save_varlen(cache_path + "_enc", enc_seqs)
    _save_varlen(cache_path + "_dec_in", dec_in_seqs)
    _save_varlen(cache_path + "_dec_tgt", dec_tgt_seqs)
    _save_varlen(cache_path + "_loss", loss_seqs)
    np.save(cache_path + "_query_len.npy", query_lens)
    np.save(cache_path + "_kept_idx.npy", kept_indices)
    np.save(cache_path + "_tool_count.npy", kept_tool_counts)
    print(f"Cached {len(enc_seqs):,} tool-call pairs to {CACHE_DIR}/{cache_id}")

    max_tool_len = getattr(prepare_tool_call_pairs, '_max_tool_len', 256)
    _build_contrastive_arrays(ds, enc_texts, tools_texts, cache_path,
                              model_path, max_enc_len, max_tool_len,
                              num_workers, chunk_size, kept_indices)

    return enc_seqs, dec_in_seqs, dec_tgt_seqs, loss_seqs, kept_indices, kept_tool_counts


def _build_contrastive_arrays(ds, enc_texts, tools_texts, cache_path,
                              model_path, max_enc_len, max_tool_len,
                              num_workers, chunk_size, kept_indices):
    """Build and save contrastive arrays: individual tool tokens (variable-length int16).

    Query-only tokens are NOT stored separately — they are derived at load time
    from the encoder data + query_len via QueryOnlyArray.
    """
    kept_set = set(kept_indices.tolist()) if kept_indices is not None else set(range(len(enc_texts)))
    n_kept = len(kept_set)

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
        tool_seqs = [np.array(tok[:max_tool_len], dtype=np.int16)
                     for chunk in t_results for tok in chunk]
    else:
        tool_seqs = []

    _save_varlen(cache_path + "_tool_ind", tool_seqs)
    np.save(cache_path + "_tool_ex_idx.npy", np.array(tool_ex_idx, dtype=np.int32))
    np.save(cache_path + "_tool_is_pos.npy", np.array(tool_is_pos, dtype=np.bool_))
    print(f"  Contrastive: {n_kept} queries (from enc), {len(tool_ex_idx)} individual tools")


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


def get_batches(enc_inputs, dec_inputs, dec_targets, batch_size, shuffle=True,
                loss_mask=None, tool_counts=None, enc_seg_ids=None, dec_seg_ids=None):
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
        if enc_seg_ids is not None:
            batch = batch + (np.array(enc_seg_ids[idx]), np.array(dec_seg_ids[idx]))
        yield batch


def _seq_lens(arr):
    """Get per-sequence lengths from a VarLenArray via its offsets."""
    return np.diff(arr._offsets).astype(np.int32)


def pack_sequences(cache_path, enc_vl, dec_in_vl, dec_tgt_vl, loss_vl):
    """Pre-pack variable-length sequences into dense bins with segment IDs.

    Uses first-fit-decreasing bin packing with vectorized bin search.
    Saves packed int16 arrays + segment ID arrays.

    Returns the number of packed bins.
    """
    n = len(enc_vl)
    max_enc = enc_vl._max_len
    max_dec = dec_in_vl._max_len

    enc_lens = _seq_lens(enc_vl)
    dec_lens = _seq_lens(dec_in_vl)

    # Sort by encoder length descending — first-fit-decreasing
    order = np.argsort(-enc_lens)

    # Vectorized bin packing: numpy arrays for bin capacities
    # Pre-allocate for worst case (every seq gets its own bin)
    bin_enc_rem = np.empty(n, dtype=np.int32)
    bin_dec_rem = np.empty(n, dtype=np.int32)
    bin_contents = [[] for _ in range(n)]
    n_bins = 0

    for idx in tqdm(order, desc="Bin packing"):
        el = int(enc_lens[idx])
        dl = int(dec_lens[idx])

        # Vectorized search: find all bins where both enc and dec fit
        if n_bins > 0:
            fits = (bin_enc_rem[:n_bins] >= el) & (bin_dec_rem[:n_bins] >= dl)
            candidates = np.flatnonzero(fits)
        else:
            candidates = np.array([], dtype=np.intp)

        if len(candidates) > 0:
            b = candidates[0]
            bin_enc_rem[b] -= el
            bin_dec_rem[b] -= dl
            bin_contents[b].append(int(idx))
        else:
            bin_enc_rem[n_bins] = max_enc - el
            bin_dec_rem[n_bins] = max_dec - dl
            bin_contents[n_bins].append(int(idx))
            n_bins += 1

    bin_enc_rem = bin_enc_rem[:n_bins]
    bin_dec_rem = bin_dec_rem[:n_bins]
    bin_contents = bin_contents[:n_bins]

    # Fill packed arrays — parallelized with multiprocessing
    packed_enc = np.zeros((n_bins, max_enc), dtype=np.int16)
    packed_dec_in = np.zeros((n_bins, max_dec), dtype=np.int16)
    packed_dec_tgt = np.zeros((n_bins, max_dec), dtype=np.int16)
    packed_loss = np.zeros((n_bins, max_dec), dtype=np.float16)
    packed_enc_seg = np.zeros((n_bins, max_enc), dtype=np.int16)
    packed_dec_seg = np.zeros((n_bins, max_dec), dtype=np.int16)

    for row in tqdm(range(n_bins), desc="Writing packed bins"):
        enc_pos = 0
        dec_pos = 0
        for seg_id, idx in enumerate(bin_contents[row], start=1):
            el = int(enc_lens[idx])
            dl = int(dec_lens[idx])

            e_start = int(enc_vl._offsets[idx])
            packed_enc[row, enc_pos:enc_pos + el] = enc_vl._data[e_start:e_start + el]
            packed_enc_seg[row, enc_pos:enc_pos + el] = seg_id

            di_start = int(dec_in_vl._offsets[idx])
            packed_dec_in[row, dec_pos:dec_pos + dl] = dec_in_vl._data[di_start:di_start + dl]

            dt_start = int(dec_tgt_vl._offsets[idx])
            packed_dec_tgt[row, dec_pos:dec_pos + dl] = dec_tgt_vl._data[dt_start:dt_start + dl]

            lm_start = int(loss_vl._offsets[idx])
            packed_loss[row, dec_pos:dec_pos + dl] = loss_vl._data[lm_start:lm_start + dl]

            packed_dec_seg[row, dec_pos:dec_pos + dl] = seg_id

            enc_pos += el
            dec_pos += dl

    np.save(cache_path + "_packed_enc.npy", packed_enc)
    np.save(cache_path + "_packed_dec_in.npy", packed_dec_in)
    np.save(cache_path + "_packed_dec_tgt.npy", packed_dec_tgt)
    np.save(cache_path + "_packed_loss.npy", packed_loss)
    np.save(cache_path + "_packed_enc_seg.npy", packed_enc_seg)
    np.save(cache_path + "_packed_dec_seg.npy", packed_dec_seg)

    total_cap = n_bins * (max_enc + max_dec)
    used = int(np.sum(max_enc - bin_enc_rem)) + int(np.sum(max_dec - bin_dec_rem))
    print(f"  Packed {n} sequences into {n_bins} bins ({used / total_cap:.0%} utilization)")
    return n_bins



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


class VarLenArray:
    """Variable-length storage with on-demand padding to fixed max_len.

    Stores sequences as a flat 1D array (int16 or float16) plus an offsets array.
    When indexed, returns padded arrays cast to int32/float32 for JAX compatibility.
    """

    def __init__(self, data, offsets, max_len, pad_value=0):
        self._data = data
        self._offsets = offsets
        self._max_len = max_len
        self._pad_value = pad_value
        self._n = len(offsets) - 1
        self._out_dtype = np.float32 if data.dtype == np.float16 else np.int32

    @classmethod
    def load(cls, prefix, max_len, pad_value=0, mmap_mode=None):
        data = np.load(prefix + "_data.npy", mmap_mode=mmap_mode)
        offsets = np.load(prefix + "_offsets.npy", mmap_mode=mmap_mode)
        return cls(data, offsets, max_len, pad_value)

    def __len__(self):
        return self._n

    @property
    def shape(self):
        return (self._n, self._max_len)

    @property
    def dtype(self):
        return self._out_dtype

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            if idx < 0:
                idx += self._n
            start, end = int(self._offsets[idx]), int(self._offsets[idx + 1])
            result = np.full(self._max_len, self._pad_value, dtype=self._out_dtype)
            seq = self._data[start:end]
            result[:len(seq)] = seq.astype(self._out_dtype)
            return result

        if isinstance(idx, np.ndarray):
            batch_size = len(idx)
            result = np.full((batch_size, self._max_len), self._pad_value, dtype=self._out_dtype)
            for j, i in enumerate(idx):
                start, end = int(self._offsets[i]), int(self._offsets[i + 1])
                seq = self._data[start:end]
                result[j, :len(seq)] = seq.astype(self._out_dtype)
            return result

        if isinstance(idx, slice):
            indices = np.arange(*idx.indices(self._n))
            return self[indices]

        raise TypeError(f"Unsupported index type: {type(idx)}")


class QueryOnlyArray:
    """Extracts query portions from encoder variable-length data without duplication.

    Queries are the prefix of each encoder sequence (before the <tools> separator).
    Instead of storing a separate copy, this derives them on-the-fly from encoder data.
    """

    def __init__(self, enc_data, enc_offsets, query_lens, max_len):
        self._enc_data = enc_data
        self._enc_offsets = enc_offsets
        self._query_lens = query_lens
        self._max_len = max_len
        self._n = len(query_lens)

    def __len__(self):
        return self._n

    @property
    def shape(self):
        return (self._n, self._max_len)

    @property
    def dtype(self):
        return np.int32

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            if idx < 0:
                idx += self._n
            start = int(self._enc_offsets[idx])
            qlen = int(self._query_lens[idx])
            result = np.full(self._max_len, PAD_ID, dtype=np.int32)
            result[:qlen] = self._enc_data[start:start + qlen].astype(np.int32)
            return result

        if isinstance(idx, np.ndarray):
            batch_size = len(idx)
            result = np.full((batch_size, self._max_len), PAD_ID, dtype=np.int32)
            for j, i in enumerate(idx):
                start = int(self._enc_offsets[i])
                qlen = int(self._query_lens[i])
                result[j, :qlen] = self._enc_data[start:start + qlen].astype(np.int32)
            return result

        if isinstance(idx, slice):
            indices = np.arange(*idx.indices(self._n))
            return self[indices]

        raise TypeError(f"Unsupported index type: {type(idx)}")


def _save_varlen(prefix, sequences):
    """Save variable-length sequences as flat data + offsets .npy files."""
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    for i, seq in enumerate(sequences):
        offsets[i + 1] = offsets[i] + len(seq)
    if sequences:
        data = np.concatenate(sequences)
    else:
        data = np.array([], dtype=np.int16)
    np.save(prefix + "_data.npy", data)
    np.save(prefix + "_offsets.npy", offsets)


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
    """Load pre-tokenized variable-length data. Returns VarLenArray objects
    that pad to fixed max lengths on-the-fly when indexed.

    If mmap=True, underlying flat arrays are memory-mapped.

    Returns dict with keys: enc_inputs, dec_inputs, dec_targets, loss_mask,
    kept_indices, tool_counts, query_only, tool_individual, tool_ex_idx, tool_is_pos.
    """
    meta = _load_cache_metadata(split)
    if meta is None:
        try:
            _download_tokenized_from_hf()
            meta = _load_cache_metadata(split)
        except Exception:
            pass
    if meta is None:
        raise FileNotFoundError(
            f"No prepared data found for split '{split}'. Run 'needle tokenize' first."
        )

    text_cache_id = meta["text_cache_id"]
    cache_path = os.path.join(CACHE_DIR, text_cache_id)
    max_enc_len = meta.get("max_enc_len", DEFAULT_MAX_ENC_LEN)
    max_dec_len = meta.get("max_dec_len", DEFAULT_MAX_DEC_LEN)
    max_tool_len = meta.get("max_tool_len", 256)

    if not os.path.exists(cache_path + "_enc_data.npy"):
        raise FileNotFoundError(
            f"Text cache '{text_cache_id}' not found. Run 'needle tokenize' first."
        )

    mmap_mode = "r" if mmap else None

    enc = VarLenArray.load(cache_path + "_enc", max_enc_len, pad_value=PAD_ID, mmap_mode=mmap_mode)
    dec_in = VarLenArray.load(cache_path + "_dec_in", max_dec_len, pad_value=PAD_ID, mmap_mode=mmap_mode)
    dec_tgt = VarLenArray.load(cache_path + "_dec_tgt", max_dec_len, pad_value=PAD_ID, mmap_mode=mmap_mode)
    loss = VarLenArray.load(cache_path + "_loss", max_dec_len, pad_value=0, mmap_mode=mmap_mode)
    kept = np.load(cache_path + "_kept_idx.npy", mmap_mode=mmap_mode)

    tc_path = cache_path + "_tool_count.npy"
    tc = np.load(tc_path, mmap_mode=mmap_mode) if os.path.exists(tc_path) else None

    # Query-only: derived from encoder data + query lengths (no duplication)
    query_only = None
    ql_path = cache_path + "_query_len.npy"
    if os.path.exists(ql_path):
        query_lens = np.load(ql_path, mmap_mode=mmap_mode)
        query_only = QueryOnlyArray(enc._data, enc._offsets, query_lens, max_enc_len)

    # Tool individual: variable-length
    tool_individual = None
    if os.path.exists(cache_path + "_tool_ind_data.npy"):
        tool_individual = VarLenArray.load(cache_path + "_tool_ind", max_tool_len,
                                           pad_value=PAD_ID, mmap_mode=mmap_mode)

    def _load_optional(suffix):
        p = cache_path + suffix
        if os.path.exists(p):
            return np.load(p, mmap_mode=mmap_mode)
        return None

    # Pre-packed arrays (if available from 'needle tokenize')
    packed = {}
    if os.path.exists(cache_path + "_packed_enc.npy"):
        packed = {
            "packed_enc": np.load(cache_path + "_packed_enc.npy", mmap_mode=mmap_mode),
            "packed_dec_in": np.load(cache_path + "_packed_dec_in.npy", mmap_mode=mmap_mode),
            "packed_dec_tgt": np.load(cache_path + "_packed_dec_tgt.npy", mmap_mode=mmap_mode),
            "packed_loss": np.load(cache_path + "_packed_loss.npy", mmap_mode=mmap_mode),
            "packed_enc_seg": np.load(cache_path + "_packed_enc_seg.npy", mmap_mode=mmap_mode),
            "packed_dec_seg": np.load(cache_path + "_packed_dec_seg.npy", mmap_mode=mmap_mode),
        }

    return {
        "enc_inputs": enc,
        "dec_inputs": dec_in,
        "dec_targets": dec_tgt,
        "loss_mask": loss,
        "kept_indices": kept,
        "tool_counts": tc,
        "query_only": query_only,
        "tool_individual": tool_individual,
        "tool_ex_idx": _load_optional("_tool_ex_idx.npy"),
        "tool_is_pos": _load_optional("_tool_is_pos.npy"),
        **packed,
    }


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


