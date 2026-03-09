# Tokenization Plan

## Overview

Train a SentencePiece BPE tokenizer (vocab_size=8192) on both available datasets, with the constraint that certain structural characters are always isolated tokens. Design for scale: extend to additional datasets (up to 1T tokens) later.

## Datasets

### 1. Synth (pretraining)

**Source**: `gs://needle-datasets-bucket/datasets/text_pretrain/synth/train/`
- 4 shards × 500K examples = 2M examples
- ~3.4 GiB compressed (jsonl.gz)
- Fields: `query`, `query_seed_text`, `synthetic_reasoning`, `synthetic_answer`
- ~1,200 tokens/example across those 4 fields → ~2.4B tokens total
- Multilingual (majority English, plus de/fr/es/it/pl/nl/la)
- Exercise types: memorization, rag, editing, cooking, mcq, math mcq

### 2. Tool calls (fine-tuning)

**Source**: `gs://cactus-dataset/tool_calls` (HuggingFace `load_from_disk` format)
- Fields: `query`, `tools`, `answers`
- Loaded via existing `_load_unified_dataset()` in `src/data.py`
- Contains structured JSON in `tools` and `answers` fields — critical for learning JSON structural tokens

### Why both matter

The tool-call dataset is heavy in JSON syntax (`{`, `}`, `[`, `]`, `"`, `,`) which is exactly the content the isolated-char constraint targets. Including it ensures BPE learns good sub-word units for the text *surrounding* those delimiters, not just for the synth pretraining text. Synth dominates in volume and provides broad language coverage.

## Hard Constraints

The following 8 characters must each be their own token and **never appear inside any other token**:

```
( ) { } [ ] " ,
```

### Why this matters

These characters serve as structural delimiters in tool-call JSON and query formatting. If they leak into multi-character BPE tokens (e.g. `("` or `],`), the model must learn fragile token-boundary rules for structured output. Isolating them makes structured generation trivially constrainable at decode time.

### Implementation: pre-tokenization

SentencePiece's `user_defined_symbols` guarantees a symbol is in the vocab but does **not** prevent BPE from learning merges that contain the character. The only reliable solution is **pre-tokenization**: ensure these characters are never adjacent to other characters in the training corpus.

```python
import re

ISOLATED_CHARS = set('({[",]})')
_PRE_TOK_RE = re.compile(r'([({}\[\]",)])')

def pre_tokenize(text: str) -> str:
    """Insert spaces around isolated chars so BPE never merges them."""
    return _PRE_TOK_RE.sub(r' \1 ', text)
```

This must be applied:
1. **During corpus preparation** (before SentencePiece training)
2. **During encoding** (in `NeedleTokenizer.encode`)
3. **NOT during decoding** (SentencePiece handles the spaces via its whitespace model)

After training, validate by scanning the vocab file to confirm no token contains these characters except the single-character tokens themselves.

## SentencePiece Config

```python
spm.SentencePieceTrainer.Train(
    input=corpus_path,
    model_prefix="tokenizer/needle",
    vocab_size=8192,
    model_type="bpe",

    # Special token IDs
    pad_id=0,
    eos_id=1,
    bos_id=2,
    unk_id=3,
    user_defined_symbols=[
        "<tool_call>", "<transcribe>",
        # Isolated structural chars (guaranteed in vocab)
        "(", ")", "{", "}", "[", "]", '"', ","
    ],

    # Corpus handling
    input_sentence_size=20_000_000,     # sample 20M sentences for BPE learning
    shuffle_input_sentence=True,        # random sample, not first-N
    train_extremely_large_corpus=True,  # memory-efficient mode

    # Character/byte handling
    byte_fallback=True,                 # UTF-8 byte tokens for OOV
    character_coverage=0.9999,          # cover all scripts (multilingual)
    normalization_rule_name="identity", # no unicode normalization

    # Performance
    num_threads=os.cpu_count(),

    # Logging
    minloglevel=1,
)
```

### Key parameter choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `vocab_size` | 8192 | Matches model embedding dim; small vocab = faster softmax, fits resource-constrained deployment |
| `input_sentence_size` | 20M | BPE merges plateau well before this; enough for stable vocab on 2.4B token corpus |
| `shuffle_input_sentence` | True | Fair sampling across all shards/languages/exercise types |
| `train_extremely_large_corpus` | True | Needed when corpus file exceeds available RAM |
| `character_coverage` | 0.9999 | High coverage for multilingual (default 0.9995 drops rare scripts) |
| `byte_fallback` | True | Graceful handling of unseen characters via UTF-8 bytes |
| `normalization_rule_name` | identity | Preserve exact text (critical for code, math, structured output) |

## Corpus Preparation

### Which fields to include

**Synth dataset:**
- `query` — encoder input
- `query_seed_text` — context/grounding text
- `synthetic_reasoning` — decoder reasoning trace
- `synthetic_answer` — decoder output
- Exclude: metadata fields (`synth_id`, `language`, `exercise`, `model`, URLs, `seed_license`, `words`)

**Tool-call dataset:**
- `query` — encoder input (natural language queries)
- `tools` — decoder prefix (JSON tool definitions)
- `answers` — decoder target (JSON tool call output)
- All three fields contain text the model sees at train/inference time

### Streaming pipeline

This runs on a GCP VM with direct access to GCS buckets (no download step needed). Stream shards via `gcsfs`, decompress in-process, pre-tokenize, and write the corpus file to the VM's local disk.

The synth dataset has 4 independent shards that can be processed in parallel. The tool-call dataset is a single HuggingFace dataset (no sharding to parallelize). Corpus prep uses `ProcessPoolExecutor` to decompress + pre-tokenize synth shards concurrently, then appends tool-call data sequentially.

```python
import gcsfs
import gzip
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datasets import load_from_disk

# Dataset registry: GCS path → (format, fields)
DATASETS = {
    "synth": {
        "gcs_path": "needle-datasets-bucket/datasets/text_pretrain/synth/train/",
        "format": "jsonl_gz",       # gzipped JSONL shards
        "fields": ("query", "query_seed_text", "synthetic_reasoning", "synthetic_answer"),
    },
    "tool_calls": {
        "gcs_path": "gs://cactus-dataset/tool_calls",
        "format": "hf_dataset",     # HuggingFace load_from_disk
        "fields": ("query", "tools", "answers"),
    },
}

def _process_shard(args):
    """Worker: stream one gzipped JSONL shard → pre-tokenized temp file.

    Runs in a child process. Returns (temp_path, example_count).
    """
    shard_path, fields, tmp_dir = args
    fs = gcsfs.GCSFileSystem()
    tmp_fd, tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix=".txt")
    count = 0
    with os.fdopen(tmp_fd, "w") as out:
        with fs.open(shard_path, "rb") as f:
            with gzip.open(f, "rt") as gz:
                for line in gz:
                    example = json.loads(line)
                    for field in fields:
                        text = str(example.get(field, "")).strip()
                        if text:
                            out.write(pre_tokenize(text) + "\n")
                    count += 1
    return tmp_path, count

def write_corpus(output_path: str, datasets: list[str] | None = None,
                 max_examples_per_dataset: int | None = None,
                 num_workers: int = 4):
    """Stream datasets from GCS → pre-tokenize → write combined corpus file.

    Synth shards are processed in parallel (one worker per shard).
    Tool-call dataset is appended sequentially.
    """
    fs = gcsfs.GCSFileSystem()
    datasets = datasets or list(DATASETS.keys())
    tmp_dir = os.path.dirname(output_path)
    total = 0

    with open(output_path, "w") as out:
        for name in datasets:
            cfg = DATASETS[name]
            print(f"Streaming {name}...")

            if cfg["format"] == "jsonl_gz":
                # --- Parallel: one worker per shard ---
                shards = sorted(fs.ls(cfg["gcs_path"]))
                worker_args = [(s, cfg["fields"], tmp_dir) for s in shards]

                with ProcessPoolExecutor(max_workers=min(num_workers, len(shards))) as pool:
                    results = list(pool.map(_process_shard, worker_args))

                # Concatenate temp files into final corpus (preserves shard order)
                dataset_count = 0
                for tmp_path, count in results:
                    with open(tmp_path) as tmp_f:
                        for line in tmp_f:
                            out.write(line)
                    os.remove(tmp_path)
                    dataset_count += count

                print(f"  {name}: {dataset_count:,} examples ({len(shards)} shards, {num_workers} workers)")
                total += dataset_count

            else:  # hf_dataset — sequential
                ds = load_from_disk(cfg["gcs_path"])
                count = 0
                for example in ds:
                    for field in cfg["fields"]:
                        text = str(example.get(field, "")).strip()
                        if text:
                            out.write(pre_tokenize(text) + "\n")
                    count += 1
                    if count % 100_000 == 0:
                        print(f"  {name}: {count:,} examples...")
                    if max_examples_per_dataset and count >= max_examples_per_dataset:
                        break
                print(f"  {name}: {count:,} examples written")
                total += count

    print(f"Corpus written: {total:,} total examples from {len(datasets)} datasets")
```

One text field = one "sentence" in SentencePiece terms. This means `input_sentence_size=20M` will sample ~5M examples (4 fields each), which is the full 2M-example corpus with headroom.

### Performance notes

- **Parallel shard processing**: 4 synth shards × 1 worker each = ~4x speedup on corpus prep (the I/O-bound step). Each worker streams from GCS, decompresses, pre-tokenizes, and writes to a temp file independently.
- **SentencePiece training itself is not parallelizable**: the core BPE merge loop is sequential. `num_threads` only accelerates initial corpus loading and frequency counting. No GPU support exists for BPE training in any library.
- **`input_sentence_size=20M`**: SentencePiece samples 20M sentences from the corpus file for BPE learning. Vocab quality plateaus well before this — the parameter avoids wasting time on redundant data while ensuring stable merges.
- **`train_extremely_large_corpus=True`**: Uses memory-mapped I/O internally so the full corpus file doesn't need to fit in RAM.

### System requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| vCPUs | 2 | 4 (one per synth shard for parallel prep) |
| RAM | 8 GiB | 16 GiB (headroom for SP sentence buffer) |
| Disk | 20 GiB free | 40 GiB free (corpus + temp files during parallel prep) |
| GPU | Not used | Not used |
| Instance type | n1-standard-4 | n1-standard-4 (4 vCPU, 15 GiB) is sufficient |

Estimated wall time on n1-standard-4:
- Corpus prep (parallel): ~10-15 min (GCS streaming + decompression + pre-tokenization)
- SentencePiece training: ~30-60 min (BPE merge loop on 20M sampled sentences)
- Validation: <1 min
- **Total: ~45-75 min**

### Sentence length consideration

SentencePiece has a default `max_sentence_length` of 4192 bytes. Some `query_seed_text` and `synthetic_reasoning` fields exceed this. Set `max_sentence_length=16384` or split long fields into paragraphs:

```python
max_sentence_length=16384,  # add to Train() call
```

## Scaling Strategy (future datasets)

When additional datasets are processed (up to 1T tokens total):

### Corpus sampling for tokenizer training

BPE vocab quality plateaus at ~5-10B tokens of well-sampled data. For 1T tokens across N datasets:

1. **Proportional sampling** (default):
   ```
   sample_i = min(dataset_size_i, 10B × (dataset_size_i / total_size))
   ```
   Each dataset contributes proportionally to its size.

2. **Temperature sampling** (if small datasets are important):
   ```
   weight_i = size_i^(1/T) / Σ(size_j^(1/T))    # T=0.7
   sample_i = 10B × weight_i
   ```
   Upsamples rare datasets. Use when some domains are small but critical (e.g., code, math).

3. **Cap per dataset**: No single dataset should exceed 70% of the tokenizer training corpus regardless of size, to prevent BPE merges being dominated by one domain.

### Process

1. Each dataset writes its sampled corpus to a shard file
2. Concatenate/interleave shards into final corpus file
3. SentencePiece trains on the combined file with `shuffle_input_sentence=True`
4. Validate vocab coverage per dataset (no dataset should have >5% UNK rate)

## Validation

After training, run these checks:

### 1. Isolated character check (hard requirement)
```python
sp = spm.SentencePieceProcessor()
sp.Load("tokenizer/needle.model")
isolated = set('({[",]})')
for i in range(sp.GetPieceSize()):
    piece = sp.IdToPiece(i)
    for c in isolated:
        if c in piece and piece not in (c, f"▁{c}"):
            raise ValueError(f"Token {i} '{piece}' contains isolated char '{c}'")
```

### 2. Roundtrip integrity
```python
# Sample 10K examples, encode → decode, check equality
for text in sample_texts:
    assert sp.Decode(sp.Encode(text)) == text
```

### 3. Vocab utilization
- No token should have 0 frequency on a held-out sample
- Check for excessive byte-fallback usage (should be <0.1% of tokens)
- Check UNK rate (should be 0% with byte_fallback=True)

### 4. Compression ratio
- Measure chars/token on held-out data per language
- English target: 3.5-4.5 chars/token at 8K vocab
- Flag any language with <2.0 chars/token (under-represented in training)

## Execution Plan

All steps run on a GCP VM with GCS bucket access. No local downloads needed.

### Phase 1: synth + tool_calls (now)

```bash
# On the GCP VM:

# 1. Stream both datasets from GCS, pre-tokenize, write combined corpus
python -m src.tokenizer_train prepare \
    --datasets synth tool_calls \
    --output /tmp/tokenizer_corpus.txt

# 2. Train tokenizer (reads corpus from local disk, model written to tokenizer/)
python -m src.tokenizer_train train \
    --corpus /tmp/tokenizer_corpus.txt \
    --vocab-size 8192

# 3. Validate isolated char constraint + compression ratio
python -m src.tokenizer_train validate \
    --model tokenizer/needle.model \
    --test-corpus /tmp/tokenizer_corpus.txt

# 4. Clean up corpus file (large)
rm /tmp/tokenizer_corpus.txt

# 5. Upload trained tokenizer to GCS
gsutil cp tokenizer/needle.{model,vocab} gs://needle-datasets-bucket/tokenizer/
```

### Phase 2: additional datasets (when ready)

1. Add new entries to the `DATASETS` registry (GCS path + format + fields)
2. Re-run `prepare` with all datasets listed
3. Re-train and re-validate
4. Re-tokenize all cached training data (MD5 invalidation handles this automatically)

## Changes to Existing Code

1. **New file `src/tokenizer_train.py`**: Standalone script for corpus streaming from GCS, pre-tokenization, training, and validation. Separate from the training loop so it can run independently on any GCP VM.
2. **`src/data.py`**: Update `NeedleTokenizer.encode()` and `__call__()` to apply `pre_tokenize()` before SentencePiece encoding. Update `train_tokenizer()` config to match the new parameters.
3. **`pyproject.toml`**: Add `gcsfs` dependency.
4. **GCS tokenizer path**: Update `GCS_TOKENIZER_PATH` to point to the new bucket location.

## Refactor: extract `src/tokenizer.py`

Tokenizer logic is currently split across `src/data.py` and `src/tokenizer_train.py`, with duplicate SP training configs. Consolidate all tokenizer code into a new `src/tokenizer.py` and keep `data.py` for data loading/preparation only.

### `src/tokenizer.py` (new file, extracted from `data.py`)

Move these from `data.py`:
- Token ID constants: `PAD_ID`, `EOS_ID`, `BOS_ID`, `UNK_ID`, `TOOL_CALL_ID`, `TRANSCRIBE_ID`
- `TOKENIZER_DIR`, `TOKENIZER_PREFIX`, `GCS_TOKENIZER_PATH`
- `pre_tokenize()` and `_PRE_TOK_RE`
- `NeedleTokenizer` class
- `_init_worker()`, `_tokenize_chunk()` (multiprocess tokenization helpers)
- `train_tokenizer()` (single source of truth for SP config)
- `get_tokenizer()`, `_download_tokenizer_from_gcs()`

### `src/tokenizer_train.py` (update)

- Import `pre_tokenize` from `src.tokenizer` instead of `src.data`
- Remove the duplicate `train()` function; replace with a call to `train_tokenizer()` from `src.tokenizer`
- Keep only GCS corpus streaming, parallel shard processing, and validation logic

### `src/data.py` (update)

- Remove all moved tokenizer code
- Add imports from `src.tokenizer`: `from src.tokenizer import (NeedleTokenizer, get_tokenizer, pre_tokenize, PAD_ID, EOS_ID, BOS_ID, UNK_ID, TOOL_CALL_ID, TRANSCRIBE_ID, _init_worker, _tokenize_chunk, _tokenizer_hash, TOKENIZER_PREFIX)`
- Everything else stays: `_load_unified_dataset()`, `prepare_tool_call_pairs()`, `prepare_voice_tool_call_pairs()`, cache key logic, etc.

### Import chain

```
src/tokenizer.py          ← owns all tokenizer logic
src/data.py               ← imports from src.tokenizer, owns data loading
src/tokenizer_train.py    ← imports from src.tokenizer, owns GCS corpus pipeline + CLI
src/train.py, src/run.py  ← import from src.data (no change needed if data.py re-exports)
```

To avoid breaking existing imports in `train.py`, `run.py`, etc., `data.py` should re-export the tokenizer symbols it previously owned (e.g. `get_tokenizer`, `PAD_ID`, etc.) so callers don't need updating.

## Dependencies

```
gcsfs          # GCS streaming (used on VM only, not needed for inference)
sentencepiece  # already in pyproject.toml
```
