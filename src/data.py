import csv
import hashlib
import json as _json
import multiprocessing as mp
import os
import queue
import subprocess
import threading
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import math
import numpy as np
from datasets import Audio, load_from_disk
from tqdm import tqdm
import sentencepiece as spm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_DISK_CACHE_DIR = os.path.join(_PROJECT_ROOT, ".data_cache")
_DEFAULT_TOKENIZER_DIR = os.path.join(_PROJECT_ROOT, "tokenizer")
_DEFAULT_LOCAL_UNIFIED_DIR = os.path.join(_PROJECT_ROOT, "data", "tool_calls_unified")
GCS_DATASET_PATH = "gs://cactus-dataset/tool_calls"
EMILIA_SPEECH_GCS_PREFIX = "gs://cactus-dataset/speech-datav1/emilia-large"

_MIN_SHM_BYTES = 200 * 1024**3 


def _pick_cache_dir():
    """Use /dev/shm (tmpfs/RAM) when available and large enough, else disk."""
    env_cache_dir = os.environ.get("NEEDLE_CACHE_DIR")
    if env_cache_dir:
        os.makedirs(env_cache_dir, exist_ok=True)
        return env_cache_dir

    shm = "/dev/shm"
    if os.path.isdir(shm):
        try:
            st = os.statvfs(shm)
            avail = st.f_bavail * st.f_frsize
            if avail >= _MIN_SHM_BYTES:
                d = os.path.join(shm, "needle_cache")
                os.makedirs(d, exist_ok=True)
                return d
        except OSError:
            pass
    os.makedirs(_DISK_CACHE_DIR, exist_ok=True)
    return _DISK_CACHE_DIR


CACHE_DIR = _pick_cache_dir()
TOKENIZER_DIR = os.environ.get("NEEDLE_TOKENIZER_DIR", _DEFAULT_TOKENIZER_DIR)
TOKENIZER_PREFIX = os.path.join(TOKENIZER_DIR, "needle")
LOCAL_UNIFIED_DIR = os.environ.get("NEEDLE_LOCAL_UNIFIED_DIR", _DEFAULT_LOCAL_UNIFIED_DIR)

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2  
UNK_ID = 3
TOOL_CALL_ID = 4
TRANSCRIBE_ID = 5

JSON_LBRACK = "<json_lbrack>"
JSON_RBRACK = "<json_rbrack>"
JSON_LBRACE = "<json_lbrace>"
JSON_RBRACE = "<json_rbrace>"
JSON_COLON = "<json_colon>"
JSON_COMMA = "<json_comma>"
JSON_QUOTE = "<json_quote>"
JSON_TRUE = "<json_true>"
JSON_FALSE = "<json_false>"
JSON_NULL = "<json_null>"
JSON_KEY_NAME = "<json_key_name>"
JSON_KEY_PARAMETERS = "<json_key_parameters>"
JSON_KEY_ARGUMENTS = "<json_key_arguments>"

JSON_SPECIAL_LITERALS = {
    JSON_LBRACK: "[",
    JSON_RBRACK: "]",
    JSON_LBRACE: "{",
    JSON_RBRACE: "}",
    JSON_COLON: ":",
    JSON_COMMA: ",",
    JSON_QUOTE: '"',
    JSON_TRUE: "true",
    JSON_FALSE: "false",
    JSON_NULL: "null",
    JSON_KEY_NAME: '"name"',
    JSON_KEY_PARAMETERS: '"parameters"',
    JSON_KEY_ARGUMENTS: '"arguments"',
}
JSON_SPECIAL_SYMBOLS = tuple(JSON_SPECIAL_LITERALS.keys())
JSON_KEY_SYMBOLS = {
    "name": JSON_KEY_NAME,
    "parameters": JSON_KEY_PARAMETERS,
    "arguments": JSON_KEY_ARGUMENTS,
}
TRAINABLE_SPECIAL_SYMBOLS = ["<tool_call>", "<transcribe>", *JSON_SPECIAL_SYMBOLS]

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


def _normalize_tool_schema_spec(tools_text):
    try:
        tools = _json.loads(tools_text)
    except (_json.JSONDecodeError, TypeError):
        return []

    if not isinstance(tools, list):
        return []

    normalized = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name", ""))
        params = tool.get("parameters")
        if not isinstance(params, dict):
            params = {}
        ordered_params = [(str(k), params[k]) for k in params.keys()]
        normalized.append({"name": name, "parameters": ordered_params})
    return normalized


def _normalize_tool_call_spec(answer_text, tools_text):
    schema = _normalize_tool_schema_spec(tools_text)
    param_order = {tool["name"]: [name for name, _ in tool["parameters"]] for tool in schema}

    try:
        calls = _json.loads(answer_text)
    except (_json.JSONDecodeError, TypeError):
        calls = []

    if not isinstance(calls, list):
        calls = []

    normalized = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name", ""))
        args = call.get("arguments")
        if not isinstance(args, dict):
            args = {}
        ordered = []
        seen = set()
        for arg_name in param_order.get(name, ()):
            if arg_name in args:
                ordered.append((arg_name, args[arg_name]))
                seen.add(arg_name)
        for arg_name, arg_value in args.items():
            if arg_name not in seen:
                ordered.append((str(arg_name), arg_value))
        normalized.append({"name": name, "arguments": ordered})
    return normalized


def _escape_json_string_content(text):
    content = _json.dumps(str(text), ensure_ascii=True)[1:-1]
    content = content.replace('\\"', '\\u0022')
    content = content.replace("<", "\\u003c").replace(">", "\\u003e")
    return content


def _number_to_text(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "0"
        text = format(value, ".15g")
        if text == "-0":
            return "0"
        return text
    return _json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _string_segments(text):
    return [JSON_QUOTE, _escape_json_string_content(text), JSON_QUOTE]


def _generic_json_segments(value):
    if isinstance(value, str):
        return _string_segments(value)
    if value is True:
        return [JSON_TRUE]
    if value is False:
        return [JSON_FALSE]
    if value is None:
        return [JSON_NULL]
    if isinstance(value, (int, float)):
        return [_number_to_text(value)]
    if isinstance(value, list):
        segments = [JSON_LBRACK]
        for i, item in enumerate(value):
            if i:
                segments.append(JSON_COMMA)
            segments.extend(_generic_json_segments(item))
        segments.append(JSON_RBRACK)
        return segments
    if isinstance(value, dict):
        segments = [JSON_LBRACE]
        for i, (key, item) in enumerate(value.items()):
            if i:
                segments.append(JSON_COMMA)
            segments.extend(_string_segments(str(key)))
            segments.append(JSON_COLON)
            segments.extend(_generic_json_segments(item))
        segments.append(JSON_RBRACE)
        return segments
    return _string_segments(str(value))


def _tool_schema_to_segments(tools_text):
    schema = _normalize_tool_schema_spec(tools_text)
    segments = [JSON_LBRACK]
    for i, tool in enumerate(schema):
        if i:
            segments.append(JSON_COMMA)
        segments.append(JSON_LBRACE)
        segments.extend([JSON_KEY_NAME, JSON_COLON, *_string_segments(tool["name"])])
        segments.append(JSON_COMMA)
        segments.extend([JSON_KEY_PARAMETERS, JSON_COLON, JSON_LBRACE])
        for j, (param_name, param_type) in enumerate(tool["parameters"]):
            if j:
                segments.append(JSON_COMMA)
            segments.extend(_string_segments(param_name))
            segments.append(JSON_COLON)
            segments.extend(_generic_json_segments(param_type))
        segments.extend([JSON_RBRACE, JSON_RBRACE])
    segments.append(JSON_RBRACK)
    return segments


def _tool_call_to_segments(answer_text, tools_text):
    calls = _normalize_tool_call_spec(answer_text, tools_text)
    segments = [JSON_LBRACK]
    for i, call in enumerate(calls):
        if i:
            segments.append(JSON_COMMA)
        segments.append(JSON_LBRACE)
        segments.extend([JSON_KEY_NAME, JSON_COLON, *_string_segments(call["name"])])
        segments.append(JSON_COMMA)
        segments.extend([JSON_KEY_ARGUMENTS, JSON_COLON, JSON_LBRACE])
        for j, (arg_name, arg_value) in enumerate(call["arguments"]):
            if j:
                segments.append(JSON_COMMA)
            segments.extend(_string_segments(arg_name))
            segments.append(JSON_COLON)
            segments.extend(_generic_json_segments(arg_value))
        segments.extend([JSON_RBRACE, JSON_RBRACE])
    segments.append(JSON_RBRACK)
    return segments


def _segments_to_training_text(segments):
    return " ".join(segments)


class NeedleTokenizer:
    """Wrapper around SentencePiece providing the interface the codebase expects."""

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.json_token_ids = {}
        for symbol in JSON_SPECIAL_SYMBOLS:
            piece_id = int(self.sp.PieceToId(symbol))
            if piece_id < 0 or self.sp.IdToPiece(piece_id) != symbol:
                raise ValueError(
                    "Tokenizer is missing structured JSON symbols. "
                    "Re-run `needle tokenize` to retrain the tokenizer."
                )
            self.json_token_ids[symbol] = piece_id
        self._special_id_to_literal = {
            self.json_token_ids[symbol]: literal
            for symbol, literal in JSON_SPECIAL_LITERALS.items()
        }
        excluded = {
            PAD_ID,
            EOS_ID,
            BOS_ID,
            TOOL_CALL_ID,
            TRANSCRIBE_ID,
            *self._special_id_to_literal.keys(),
        }
        self.regular_token_ids = np.array(
            [i for i in range(self.sp.GetPieceSize()) if i not in excluded],
            dtype=np.int32,
        )

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

    def _encode_segments(self, segments):
        ids = []
        for segment in segments:
            token_id = self.json_token_ids.get(segment)
            if token_id is not None:
                ids.append(token_id)
            elif segment:
                ids.extend(self.sp.Encode(segment, out_type=int))
        return ids

    def encode_tool_schema(self, tools_text):
        return self._encode_segments(_tool_schema_to_segments(tools_text))

    def encode_tool_call(self, answer_text, tools_text):
        return self._encode_segments(_tool_call_to_segments(answer_text, tools_text))

    def encode_json_string_content(self, text):
        return self.sp.Encode(_escape_json_string_content(text), out_type=int)

    def encode_json_number(self, value):
        return self.sp.Encode(_number_to_text(value), out_type=int)

    def decode_structured(self, ids):
        if isinstance(ids, (list, tuple)) and len(ids) > 0 and isinstance(ids[0], (list, tuple, np.ndarray)):
            return [self.decode_structured(seq) for seq in ids]

        out = []
        regular = []
        for token_id in list(ids):
            token_id = int(token_id)
            literal = self._special_id_to_literal.get(token_id)
            if literal is None:
                regular.append(token_id)
                continue
            if regular:
                out.append(self.sp.Decode(regular))
                regular = []
            out.append(literal)
        if regular:
            out.append(self.sp.Decode(regular))
        return "".join(out)

    def decode(self, ids):
        return self.decode_structured(ids)

    def token_surface(self, token_id):
        token_id = int(token_id)
        if token_id in self._special_id_to_literal:
            return self._special_id_to_literal[token_id]
        return self.sp.Decode([token_id])

    def quote_token_id(self):
        return self.json_token_ids[JSON_QUOTE]

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
            query = example["query"].strip()
            if query:
                f.write(query + "\n")

            tools_text = _segments_to_training_text(_tool_schema_to_segments(example["tools"]))
            if tools_text:
                f.write(tools_text + "\n")

            answers_text = _segments_to_training_text(_tool_call_to_segments(example["answers"], example["tools"]))
            if answers_text:
                f.write(answers_text + "\n")

    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=PAD_ID,
        eos_id=EOS_ID,
        bos_id=BOS_ID,
        unk_id=UNK_ID,
        user_defined_symbols=TRAINABLE_SPECIAL_SYMBOLS,
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


def load_tool_calls(split="train", max_samples=None, return_global_indices=False,
                    shuffle_before_split=False, shuffle_seed=42):
    """Load tool-calling dataset, splitting 90/10 for train/val.

    If return_global_indices is True, also return a numpy array mapping each
    split-local row position back to its row id in the full unified dataset.
    """
    ds = _load_unified_dataset()
    global_indices = _split_global_indices(
        len(ds),
        split=split,
        max_samples=max_samples,
        shuffle_before_split=shuffle_before_split,
        shuffle_seed=shuffle_seed,
    )
    ds = ds.select(global_indices.tolist())

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


def _split_global_indices(n, split="train", max_samples=None,
                          shuffle_before_split=False, shuffle_seed=42):
    """Return global row ids for the requested split of the unified dataset."""
    if shuffle_before_split:
        indices = np.random.default_rng(shuffle_seed).permutation(n).astype(np.int64)
    else:
        indices = np.arange(n, dtype=np.int64)

    cut = int(n * 0.9)
    if split in ("validation", "val", "test"):
        indices = indices[cut:]
    elif split == "train":
        indices = indices[:cut]

    if max_samples:
        indices = indices[:min(max_samples, len(indices))]

    return indices


def _save_cache_metadata(split, text_cache_id, mel_cache_id, n_samples,
                         max_enc_len, max_dec_len, n_mels, max_mel_len,
                         split_max_samples=None, shuffle_before_split=False,
                         split_seed=42, toucan_cache_path=None):
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
        "split_max_samples": split_max_samples,
        "shuffle_before_split": shuffle_before_split,
        "split_seed": split_seed,
        "toucan_cache_path": toucan_cache_path,
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

    print("Tokenizing tools (structured JSON)...")
    all_tools_tokens = [
        tokenizer.encode_tool_schema(text)[:max_dec_len - 2]
        for text in tqdm(tools_texts, desc="Encoding tools")
    ]

    print("Tokenizing answers (structured JSON)...")
    all_ans_tokens = [
        tokenizer.encode_tool_call(answer, tools)[:max_dec_len]
        for answer, tools in tqdm(zip(ans_texts, tools_texts), total=len(ans_texts), desc="Encoding answers")
    ]

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


def get_text_mel_batches(enc_inputs, mel_data, batch_size, shuffle=True):
    n = len(enc_inputs)
    indices = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n - batch_size + 1, batch_size):
        idx = indices[i : i + batch_size]
        yield np.array(enc_inputs[idx]), np.array(mel_data[idx])



def load_tool_call_audio(split="train", max_samples=None,
                         shuffle_before_split=False, shuffle_seed=42):
    """Return dataset-global indices for the given split.

    Applies the same 90/10 split as load_tool_calls. Audio is NOT loaded into memory.
    """
    ds = _load_unified_dataset()
    return _split_global_indices(
        len(ds),
        split=split,
        max_samples=max_samples,
        shuffle_before_split=shuffle_before_split,
        shuffle_seed=shuffle_seed,
    ).tolist()


_GCS_SHARD_ROWS = 5000
_GCS_N_SHARDS = 19
_shard_cache = {} 


def _load_shard(shard_idx):
    """Load a single arrow shard, from local or GCS. Caches in memory."""
    if shard_idx in _shard_cache:
        return _shard_cache[shard_idx]

    fname = f"data-{shard_idx:05d}-of-{_GCS_N_SHARDS:05d}.arrow"

    local_path = os.path.join(LOCAL_UNIFIED_DIR, fname)
    if os.path.exists(local_path):
        import pyarrow as pa
        source = pa.memory_map(local_path, "r")
        try:
            reader = pa.ipc.open_file(source)
        except pa.lib.ArrowInvalid:
            source.seek(0)
            reader = pa.ipc.open_stream(source)
        tbl = reader.read_all()
        from datasets import Dataset
        ds = Dataset(tbl)
        ds = _set_audio_backend(ds)
        _shard_cache[shard_idx] = ds
        return ds

    import subprocess
    shm_dir = os.path.join(CACHE_DIR, "arrow_shards")
    os.makedirs(shm_dir, exist_ok=True)
    dest = os.path.join(shm_dir, fname)
    if not os.path.exists(dest):
        gcs_src = f"{GCS_DATASET_PATH}/{fname}"
        subprocess.run(
            ["gcloud", "storage", "cp", gcs_src, dest],
            capture_output=True, text=True,
        )
    if not os.path.exists(dest):
        raise FileNotFoundError(f"Failed to download shard {fname} from GCS")

    import pyarrow as pa
    source = pa.memory_map(dest, "r")
    try:
        reader = pa.ipc.open_file(source)
    except pa.lib.ArrowInvalid:
        source.seek(0)
        reader = pa.ipc.open_stream(source)
    tbl = reader.read_all()
    from datasets import Dataset
    ds = Dataset(tbl)
    ds = _set_audio_backend(ds)
    _shard_cache[shard_idx] = ds
    return ds


def _load_example_by_index(idx):
    """Load a single example by global index, downloading only the needed shard."""
    shard_idx = idx // _GCS_SHARD_ROWS
    local_idx = idx % _GCS_SHARD_ROWS
    shard = _load_shard(shard_idx)
    if local_idx >= len(shard):
        raise IndexError(f"Index {idx} out of range (shard {shard_idx} has {len(shard)} rows)")
    return shard[local_idx]


def load_audio_for_index(idx):
    """Load and decode audio for a single dataset index.

    Returns (audio_array, sampling_rate) or (None, None) if no audio.
    """
    import io
    import soundfile as sf

    ex = _load_example_by_index(idx)
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

    Downloads only the needed arrow shard (~1-4GB) instead of the full dataset (~35GB).
    Returns dict with {query, answers, tools, audio_array, sampling_rate}.
    """
    ex = _load_example_by_index(idx)
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
        result["split_max_samples"] = meta.get("split_max_samples")
        result["shuffle_before_split"] = meta.get("shuffle_before_split", False)
        result["split_seed"] = meta.get("split_seed", 42)
        result["toucan_cache_path"] = meta.get("toucan_cache_path")
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
        "split_max_samples": meta.get("split_max_samples"),
        "shuffle_before_split": meta.get("shuffle_before_split", False),
        "split_seed": meta.get("split_seed", 42),
        "toucan_cache_path": meta.get("toucan_cache_path"),
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


def _load_audio_batch(ds_indices):
    """Load and decode audio for a batch of dataset indices."""
    import io
    import soundfile as sf

    ds = _load_unified_dataset()
    arrays = []
    for idx in ds_indices:
        ex = ds[int(idx)]
        audio_val = ex.get("audio")
        raw_bytes = None
        if isinstance(audio_val, dict):
            raw_bytes = audio_val.get("bytes")
        elif isinstance(audio_val, bytes):
            raw_bytes = audio_val
        if raw_bytes is None:
            arrays.append(np.zeros(16000, dtype=np.float32))
            continue
        audio_array, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        arrays.append(audio_array.astype(np.float32))
    return arrays


def get_speech_batches(mel_data, dec_inputs, dec_targets, batch_size,
                       shuffle=True, loss_mask=None):
    """Yield speech batches from precomputed mel data.

    mel_data: array of shape (N, max_mel_len, n_mels), possibly memory-mapped.
    Uses per-batch fancy indexing to avoid copying full arrays.
    """
    n = len(mel_data)
    indices = np.random.permutation(n) if shuffle else np.arange(n)

    for i in range(0, n - batch_size + 1, batch_size):
        idx = indices[i : i + batch_size]
        batch = (np.array(mel_data[idx]), np.array(dec_inputs[idx]), np.array(dec_targets[idx]))
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


def _gcs_slug(path):
    return hashlib.md5(path.encode()).hexdigest()[:12]


def _emilia_cache_dirs(gcs_prefix):
    root = os.path.join(CACHE_DIR, f"emilia_{_gcs_slug(gcs_prefix)}")
    meta_dir = os.path.join(root, "metadata")
    mel_dir = os.path.join(root, "mels")
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)
    return meta_dir, mel_dir


def _run_gcloud(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "gcloud command failed")
    return result


def _list_gcs_paths(prefix):
    result = _run_gcloud(["gcloud", "storage", "ls", prefix])
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _download_gcs_files(uris, dest_dir, chunk_size=64):
    os.makedirs(dest_dir, exist_ok=True)
    for start in range(0, len(uris), chunk_size):
        chunk = uris[start:start + chunk_size]
        if not chunk:
            continue
        _run_gcloud(["gcloud", "storage", "cp", *chunk, dest_dir])


def _rsync_gcs_dir(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    _run_gcloud(["gcloud", "storage", "rsync", src_dir, dest_dir, "--recursive"])


def _ensure_emilia_metadata_local(gcs_prefix):
    meta_dir, _ = _emilia_cache_dirs(gcs_prefix)
    remote_prefix = f"{gcs_prefix.rstrip('/')}/train/metadata/"
    remote_paths = _list_gcs_paths(remote_prefix)
    missing = []
    for uri in remote_paths:
        local_path = os.path.join(meta_dir, os.path.basename(uri))
        if not os.path.exists(local_path):
            missing.append(uri)
    if missing:
        _download_gcs_files(missing, meta_dir, chunk_size=32)
    return [os.path.join(meta_dir, os.path.basename(uri)) for uri in remote_paths]


def load_emilia_speech_metadata(split="train", gcs_prefix=EMILIA_SPEECH_GCS_PREFIX,
                                max_samples=None, val_ratio=0.01, seed=42):
    """Load Emilia speech metadata rows and create a deterministic train/val split."""
    if split not in ("train", "val", "validation", "test"):
        raise ValueError(f"Unsupported split: {split}")

    local_csvs = sorted(_ensure_emilia_metadata_local(gcs_prefix))
    rows = []
    for csv_path in local_csvs:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                transcript = (row.get("transcript") or "").strip()
                mel_uri = (row.get("mel_gcs_uri") or "").strip()
                if not transcript or not mel_uri.endswith(".npy"):
                    continue
                rows.append(row)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    cut = int(len(rows) * (1.0 - val_ratio))
    if split == "train":
        selected = order[:cut]
    else:
        selected = order[cut:]

    if max_samples is not None:
        selected = selected[:min(max_samples, len(selected))]

    return [rows[int(i)] for i in selected]


def prepare_transcription_pairs(rows, tokenizer, max_enc_len=256, max_dec_len=1024):
    """Prepare audio->transcript decoder targets plus transcript text encodings."""
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    transcribe_id = tokenizer.transcribe_token_id

    n = len(rows)
    text_inputs = np.full((n, max_enc_len), pad_id, dtype=np.int32)
    dec_inputs = np.full((n, max_dec_len), pad_id, dtype=np.int32)
    dec_targets = np.full((n, max_dec_len), pad_id, dtype=np.int32)
    loss_mask = np.zeros((n, max_dec_len), dtype=np.float32)
    mel_uris = []

    for i, row in enumerate(rows):
        transcript = row["transcript"].strip()
        text_tokens = tokenizer.encode(transcript)[:max_enc_len]
        if text_tokens:
            text_inputs[i, :len(text_tokens)] = text_tokens

        dec_tokens = tokenizer.encode(transcript)[:max(0, max_dec_len - 2)]
        dec_inputs[i, 0] = eos_id
        dec_inputs[i, 1] = transcribe_id
        if dec_tokens:
            dec_inputs[i, 2:2 + len(dec_tokens)] = dec_tokens

        dec_targets[i, 0] = transcribe_id
        if dec_tokens:
            dec_targets[i, 1:1 + len(dec_tokens)] = dec_tokens
        eos_pos = 1 + len(dec_tokens)
        if eos_pos < max_dec_len:
            dec_targets[i, eos_pos] = eos_id
            loss_mask[i, 1:eos_pos + 1] = 1.0

        mel_uris.append(row["mel_gcs_uri"])

    return {
        "text_inputs": text_inputs,
        "dec_inputs": dec_inputs,
        "dec_targets": dec_targets,
        "loss_mask": loss_mask,
        "mel_uris": np.array(mel_uris, dtype=object),
    }


def _transcription_cache_key(split, n_samples, max_enc_len, max_dec_len, gcs_prefix, val_ratio):
    tok_hash = _tokenizer_hash()
    key = (
        f"transcribe_{split}_{_gcs_slug(gcs_prefix)}_{tok_hash}_"
        f"{n_samples}_{max_enc_len}_{max_dec_len}_{val_ratio:.6f}"
    )
    return hashlib.md5(key.encode()).hexdigest()[:12]


def save_prepared_transcription_data(split, prepared, max_enc_len, max_dec_len,
                                     gcs_prefix=EMILIA_SPEECH_GCS_PREFIX, val_ratio=0.01,
                                     split_max_samples=None):
    """Persist tokenized Emilia transcript targets for Stage 1."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    n_samples = len(prepared["text_inputs"])
    cache_id = _transcription_cache_key(
        split, n_samples, max_enc_len, max_dec_len, gcs_prefix, val_ratio
    )
    cache_path = os.path.join(CACHE_DIR, cache_id)

    np.save(cache_path + "_text_inputs.npy", prepared["text_inputs"])
    np.save(cache_path + "_dec_inputs.npy", prepared["dec_inputs"])
    np.save(cache_path + "_dec_targets.npy", prepared["dec_targets"])
    np.save(cache_path + "_loss_mask.npy", prepared["loss_mask"])
    np.save(cache_path + "_mel_uris.npy", np.asarray(prepared["mel_uris"], dtype=np.str_))
    _gcs_cache_upload(
        cache_id,
        ["_text_inputs.npy", "_dec_inputs.npy", "_dec_targets.npy", "_loss_mask.npy", "_mel_uris.npy"],
    )

    meta = {
        "split": split,
        "cache_id": cache_id,
        "n_samples": n_samples,
        "max_enc_len": max_enc_len,
        "max_dec_len": max_dec_len,
        "speech_gcs_prefix": gcs_prefix,
        "speech_val_ratio": val_ratio,
        "split_max_samples": split_max_samples,
    }
    meta_path = os.path.join(CACHE_DIR, f"stage1_{split}_metadata.json")
    with open(meta_path, "w") as f:
        _json.dump(meta, f)
    _gcs_cache_upload(f"stage1_{split}_metadata", [".json"])
    return cache_id


def load_prepared_transcription_data(split, max_enc_len, max_dec_len,
                                     gcs_prefix=EMILIA_SPEECH_GCS_PREFIX, val_ratio=0.01,
                                     max_samples=None, mmap=False):
    """Load cached Stage 1 transcription targets if metadata matches."""
    meta_path = os.path.join(CACHE_DIR, f"stage1_{split}_metadata.json")
    if not os.path.exists(meta_path):
        import subprocess
        subprocess.run(
            ["gcloud", "storage", "cp", f"{GCS_CACHE_PATH}/stage1_{split}_metadata.json", meta_path],
            capture_output=True, text=True,
        )
    if not os.path.exists(meta_path):
        return None

    with open(meta_path) as f:
        meta = _json.load(f)

    if meta.get("speech_gcs_prefix") != gcs_prefix:
        return None
    if meta.get("max_enc_len") != max_enc_len or meta.get("max_dec_len") != max_dec_len:
        return None
    if abs(float(meta.get("speech_val_ratio", 0.01)) - float(val_ratio)) > 1e-12:
        return None

    cache_id = meta["cache_id"]
    cache_path = os.path.join(CACHE_DIR, cache_id)
    suffixes = [
        "_text_inputs.npy",
        "_dec_inputs.npy",
        "_dec_targets.npy",
        "_loss_mask.npy",
        "_mel_uris.npy",
    ]
    if not all(os.path.exists(cache_path + suffix) for suffix in suffixes):
        if not _gcs_cache_download(cache_id, suffixes):
            return None

    mmap_mode = "r" if mmap else None
    prepared = {
        "text_inputs": np.load(cache_path + "_text_inputs.npy", mmap_mode=mmap_mode),
        "dec_inputs": np.load(cache_path + "_dec_inputs.npy", mmap_mode=mmap_mode),
        "dec_targets": np.load(cache_path + "_dec_targets.npy", mmap_mode=mmap_mode),
        "loss_mask": np.load(cache_path + "_loss_mask.npy", mmap_mode=mmap_mode),
        "mel_uris": np.load(cache_path + "_mel_uris.npy", mmap_mode=mmap_mode),
    }

    if max_samples is not None:
        keep = min(max_samples, len(prepared["text_inputs"]))
        prepared = {key: np.array(value[:keep]) for key, value in prepared.items()}

    return prepared


def _ensure_emilia_mels_local(mel_uris, gcs_prefix=EMILIA_SPEECH_GCS_PREFIX, download_missing=True):
    _, mel_dir = _emilia_cache_dirs(gcs_prefix)
    missing = []
    local_paths = []
    for uri in mel_uris:
        local_path = os.path.join(mel_dir, os.path.basename(uri))
        local_paths.append(local_path)
        if not os.path.exists(local_path):
            missing.append(uri)
    if missing and download_missing:
        _download_gcs_files(missing, mel_dir, chunk_size=32)
    elif missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} local Emilia mel files under {mel_dir}. "
            "Run 'needle tokenize' to mirror Stage 1 mels locally before pretraining."
        )
    return local_paths


def prefetch_emilia_mels(mel_uris, gcs_prefix=EMILIA_SPEECH_GCS_PREFIX, use_rsync=False):
    """Mirror Emilia mel .npy files into the local cache."""
    _, mel_dir = _emilia_cache_dirs(gcs_prefix)
    if use_rsync:
        _rsync_gcs_dir(f"{gcs_prefix.rstrip('/')}/train/mels", mel_dir)
        return sum(1 for name in os.listdir(mel_dir) if name.endswith(".npy"))

    unique_uris = []
    seen = set()
    for uri in mel_uris:
        uri = str(uri)
        if uri not in seen:
            seen.add(uri)
            unique_uris.append(uri)
    _ensure_emilia_mels_local(unique_uris, gcs_prefix=gcs_prefix, download_missing=True)
    return len(unique_uris)


def _load_emilia_mel(local_path, n_mels=80, max_mel_len=1024):
    mel = np.load(local_path)
    if mel.ndim != 2:
        raise ValueError(f"Expected 2D mel array, got shape {mel.shape} for {local_path}")
    if mel.shape[0] == n_mels and mel.shape[1] != n_mels:
        mel = mel.T
    elif mel.shape[1] != n_mels:
        raise ValueError(f"Unexpected mel shape {mel.shape} for {local_path}")
    mel = mel.astype(np.float32)
    if mel.shape[0] > max_mel_len:
        mel = mel[:max_mel_len]
    elif mel.shape[0] < max_mel_len:
        mel = np.pad(mel, ((0, max_mel_len - mel.shape[0]), (0, 0)))
    return mel


def get_transcription_batches(prepared, batch_size, max_mel_len=1024, n_mels=80,
                              gcs_prefix=EMILIA_SPEECH_GCS_PREFIX, shuffle=True,
                              require_local_mels=False):
    """Yield batches of cached Emilia mels and transcript targets."""
    text_inputs = prepared["text_inputs"]
    dec_inputs = prepared["dec_inputs"]
    dec_targets = prepared["dec_targets"]
    loss_mask = prepared["loss_mask"]
    mel_uris = prepared["mel_uris"]

    n = len(text_inputs)
    indices = np.random.permutation(n) if shuffle else np.arange(n)

    for i in range(0, n - batch_size + 1, batch_size):
        idx = indices[i:i + batch_size]
        batch_uris = [str(mel_uris[j]) for j in idx]
        local_paths = _ensure_emilia_mels_local(
            batch_uris,
            gcs_prefix=gcs_prefix,
            download_missing=not require_local_mels,
        )
        mel_batch = np.stack([
            _load_emilia_mel(path, n_mels=n_mels, max_mel_len=max_mel_len)
            for path in local_paths
        ]).astype(np.float32)
        yield (
            mel_batch,
            np.array(text_inputs[idx]),
            np.array(dec_inputs[idx]),
            np.array(dec_targets[idx]),
            np.array(loss_mask[idx]),
        )
