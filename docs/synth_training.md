# Training On `PleIAs/SYNTH`

This repo can train directly from Hugging Face with:

- `--text-dataset PleIAs/SYNTH`
- `--text-column <column>`

For SYNTH, start with `synthetic_answer` as the text target.

## 1) Quick sanity run (small)

```bash
python -m src.cli train \
  --text-dataset PleIAs/SYNTH \
  --text-column synthetic_answer \
  --text-train-split train \
  --text-val-split train \
  --max-samples 20000 \
  --max-eval-samples 1000 \
  --no-speech \
  --epochs 1 \
  --batch-size 16
```

Notes:
- SYNTH may not have a separate `validation` split. Using `--text-val-split train` is fine for smoke tests.
- The loader will still use a held-out subset if a separate val split is unavailable and loading succeeds in map-style mode.

## 2) Medium run (text only)

```bash
python -m src.cli train \
  --text-dataset PleIAs/SYNTH \
  --text-column synthetic_answer \
  --text-train-split train \
  --text-val-split train \
  --max-samples 1000000 \
  --max-eval-samples 5000 \
  --no-speech \
  --epochs 1 \
  --batch-size 32
```

## 3) If you want instruction-style formatting

SYNTH has fields like:
- `query`
- `synthetic_answer`
- `synthetic_reasoning`

For instruction-style LM training, create a new `text` field such as:

```text
User: {query}
Assistant: {synthetic_answer}
```

Then train on that `text` field (either by publishing a transformed HF dataset, or by extending `src/data.py` to compose fields on the fly).

## 4) Low-disk tips (100GB VM)

- Always start with `--max-samples` and scale up.
- Keep `--no-speech` while validating SYNTH text training.
- Avoid downloading multiple huge datasets at once.
- Prefer uploading shards/checkpoints to GCS and deleting old local checkpoints.

