import os
import re as _re

import sentencepiece as spm
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
TOKENIZER_DIR = os.path.join(_PROJECT_ROOT, "tokenizer")
TOKENIZER_PREFIX = os.path.join(TOKENIZER_DIR, "needle")

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2
UNK_ID = 3
TOOL_CALL_ID = 4
TOOLS_ID = 5

DEFAULT_MAX_ENC_LEN = 1024
DEFAULT_MAX_DEC_LEN = 512
DEFAULT_MAX_GEN_LEN = 512

_HF_MODEL_REPO = "Cactus-Compute/needle"


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


def _download_tokenizer_from_hf():
    """Download tokenizer files from HuggingFace Hub into TOKENIZER_DIR."""
    from huggingface_hub import hf_hub_download

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    for fname in ["needle.model", "needle.vocab"]:
        hf_hub_download(
            repo_id=_HF_MODEL_REPO,
            filename=f"tokenizer/{fname}",
            repo_type="model",
            local_dir=TOKENIZER_DIR,
            force_download=True,
        )
        # hf_hub_download puts it in tokenizer/tokenizer/needle.model, move up
        nested = os.path.join(TOKENIZER_DIR, "tokenizer", fname)
        dst = os.path.join(TOKENIZER_DIR, fname)
        if os.path.exists(nested) and not os.path.exists(dst):
            os.rename(nested, dst)


def get_tokenizer(max_samples=None):
    model_path = TOKENIZER_PREFIX + ".model"
    if not os.path.exists(model_path):
        print("Downloading pretrained tokenizer from HuggingFace...")
        _download_tokenizer_from_hf()
    return NeedleTokenizer(model_path)


def train_tokenizer(vocab_size=8192, max_samples=None, force=False):
    """Train a SentencePiece BPE tokenizer on tool-calling corpus."""
    model_path = TOKENIZER_PREFIX + ".model"
    if os.path.exists(model_path) and not force:
        print(f"Tokenizer already exists at {model_path}")
        return model_path

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    from datasets import concatenate_datasets
    from .dataset import _load_split_dataset

    train_ds = _load_split_dataset("train")
    val_ds = _load_split_dataset("validation")
    ds = concatenate_datasets([train_ds, val_ds])
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
            num_threads=min(128, max(1, (os.cpu_count() or 1) * 3 // 4)),
            train_extremely_large_corpus=False,
            minloglevel=2,
        )
    finally:
        if os.path.exists(corpus_path):
            os.remove(corpus_path)
    print(f"Tokenizer saved to {model_path}")
    return model_path
