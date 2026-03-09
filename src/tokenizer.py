import hashlib
import os
import re

import sentencepiece as spm

_PRE_TOK_RE = re.compile(r'([({}\[\]",)])')

TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tokenizer")
TOKENIZER_PREFIX = os.path.join(TOKENIZER_DIR, "needle")
GCS_TOKENIZER_PATH = "gs://needle-datasets-bucket/tokenizer/"

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2  # reserved for SentencePiece, unused at runtime (EOS_ID serves as SOS)
UNK_ID = 3
TOOL_CALL_ID = 4
TRANSCRIBE_ID = 5

_SP_TRAIN_KWARGS = dict(
    model_type="bpe",
    pad_id=PAD_ID,
    eos_id=EOS_ID,
    bos_id=BOS_ID,
    unk_id=UNK_ID,
    user_defined_symbols=[
        "<tool_call>", "<transcribe>",
        "(", ")", "{", "}", "[", "]", '"', ",",
    ],
    input_sentence_size=10_000_000,
    shuffle_input_sentence=True,
    train_extremely_large_corpus=True,
    max_sentence_length=32768,
    byte_fallback=True,
    character_coverage=0.9999,
    normalization_rule_name="identity",
    num_threads=os.cpu_count(),
    minloglevel=1,
)


def pre_tokenize(text: str) -> str:
    """Insert spaces around isolated chars so BPE never merges them."""
    return _PRE_TOK_RE.sub(r' \1 ', text)


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
        return self.sp.Encode(pre_tokenize(text), out_type=int)

    def decode(self, ids):
        if isinstance(ids, (list, tuple)) and len(ids) > 0 and isinstance(ids[0], (list, tuple)):
            return [self.sp.Decode(seq) for seq in ids]
        return self.sp.Decode(list(ids))

    def __call__(self, texts, truncation=True, max_length=None, **kwargs):
        all_ids = []
        for text in texts:
            ids = self.sp.Encode(pre_tokenize(text), out_type=int)
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
    return [_worker_sp.Encode(pre_tokenize(t), out_type=int)[:_worker_max_len] for t in texts]


def _tokenizer_hash():
    """Hash the tokenizer model file to detect retraining."""
    model_path = TOKENIZER_PREFIX + ".model"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    return "none"


def train_tokenizer(vocab_size=8192, max_samples=None, corpus_path=None):
    """Train a SentencePiece BPE tokenizer.

    If corpus_path is given, train directly on that file (skip corpus building).
    Otherwise, build a corpus from the unified tool-call dataset.
    """
    model_path = TOKENIZER_PREFIX + ".model"
    if os.path.exists(model_path):
        print(f"Tokenizer already exists at {model_path}")
        return model_path

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    cleanup_corpus = False
    if corpus_path is None:
        from datasets import load_from_disk
        from tqdm import tqdm
        from .data import _load_unified_dataset

        ds = _load_unified_dataset()
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        print(f"Training SentencePiece BPE tokenizer (vocab_size={vocab_size}, samples={len(ds):,})...")

        corpus_path = os.path.join(TOKENIZER_DIR, "corpus.txt")
        cleanup_corpus = True
        with open(corpus_path, "w") as f:
            for example in tqdm(ds, desc="Writing corpus"):
                for field in ("query", "tools", "answers"):
                    text = example[field].strip()
                    if text:
                        f.write(pre_tokenize(text) + "\n")
    else:
        print(f"Training SentencePiece BPE tokenizer (vocab_size={vocab_size}) on {corpus_path}...")

    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=vocab_size,
        **_SP_TRAIN_KWARGS,
    )

    if cleanup_corpus:
        os.remove(corpus_path)
    print(f"Tokenizer saved to {model_path}")
    return model_path


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


def get_tokenizer(max_samples=None):
    model_path = TOKENIZER_PREFIX + ".model"
    if not os.path.exists(model_path):
        if _download_tokenizer_from_gcs():
            return NeedleTokenizer(model_path)
        train_tokenizer(max_samples=max_samples)
    return NeedleTokenizer(model_path)
