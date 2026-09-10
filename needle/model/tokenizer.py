import os

import sentencepiece as spm

TOKENIZER_DIR = os.path.dirname(__file__)
TOKENIZER_PREFIX = os.path.join(TOKENIZER_DIR, "tokenizer")

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2
UNK_ID = 3
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_START = "<think>"
THINK_END = "</think>"
TOOLS_START = "<tools>"
TOOLS_END = "</tools>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_RESULT_START = "<tool_result>"
TOOL_RESULT_END = "</tool_result>"
CHAT_MARKERS = [
    IM_START, IM_END, THINK_START, THINK_END,
    TOOLS_START, TOOLS_END, TOOL_CALL_START, TOOL_CALL_END,
    TOOL_RESULT_START, TOOL_RESULT_END,
]
IM_START_ID, IM_END_ID, THINK_START_ID, THINK_END_ID = 4, 5, 6, 7
(TOOLS_START_ID, TOOLS_END_ID, TOOL_CALL_START_ID, TOOL_CALL_END_ID,
 TOOL_RESULT_START_ID, TOOL_RESULT_END_ID) = range(8, 14)

HF_REPO = "Cactus-Compute/needle2"
_HF_TOKENIZER_DIR = "tokenizer"


class SANTokenizer:

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.model_path = model_path

    @property
    def md5(self):
        import hashlib
        with open(self.model_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

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


def _download_tokenizer_from_hf(prefix):
    from huggingface_hub import hf_hub_download

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    base = os.path.basename(prefix)
    for ext in (".model", ".vocab"):
        fname = base + ext
        hf_hub_download(
            repo_id=HF_REPO,
            filename=f"{_HF_TOKENIZER_DIR}/{fname}",
            repo_type="model",
            local_dir=TOKENIZER_DIR,
        )
        nested = os.path.join(TOKENIZER_DIR, _HF_TOKENIZER_DIR, fname)
        dst = os.path.join(TOKENIZER_DIR, fname)
        if os.path.exists(nested) and not os.path.exists(dst):
            os.rename(nested, dst)


def get_tokenizer(vocab_size=None):
    prefix = TOKENIZER_PREFIX
    model_path = prefix + ".model"
    if not os.path.exists(model_path):
        try:
            print(f"  {'fetch':<9} {os.path.basename(prefix)} tokenizer  downloading from Hugging Face")
            _download_tokenizer_from_hf(prefix)
        except Exception as e:
            raise RuntimeError(
                f"No pretraining tokenizer at {model_path} and HF download failed ({e}). "
                f"It ships in the {HF_REPO} Hugging Face repo; retry once it is reachable."
            ) from e
    return SANTokenizer(model_path)
