# Finetuning UI

A local web interface for finetuning needle on your own tools.

## Quick Start

```bash
needle ui --checkpoint <path/to/checkpoint.pkl>
```

Opens at `http://127.0.0.1:7860`. Use `--port` and `--host` to change.

## What It Does

1. **Load a base model** -- upload or specify a `.pkl` checkpoint
2. **Define your tools** -- paste your tool definitions as JSON
3. **Provide a Gemini API key** -- used to synthesize training data
4. **Click finetune** -- the UI handles everything:
   - Generates ~120 examples per tool via Gemini
   - Validates the data (checks format, deduplication, per-tool coverage)
   - Evaluates the base model on a held-out test set
   - Trains for 3 epochs
   - Evaluates the finetuned model on the same test set
   - Bundles the result into a downloadable `.zip`

## Tool Format

```json
[
  {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
      "type": "object",
      "properties": {
        "city": { "type": "string", "description": "City name" }
      },
      "required": ["city"]
    }
  }
]
```

## Download Bundle

After finetuning completes, download a `.zip` containing:

| File | Description |
|------|-------------|
| `checkpoint.pkl` | Finetuned model weights |
| `tools.json` | Tool definitions used |
| `train.jsonl` | Training split |
| `val.jsonl` | Validation split (used for checkpoint selection) |
| `test.jsonl` | Test split (held out, used for eval) |
| `eval_report.json` | Base vs. finetuned metrics |
| `README.md` | Summary with eval table |

## CLI Alternative

You can also finetune without the UI:

```bash
python -m src.finetune data.jsonl \
  --checkpoint <path/to/checkpoint.pkl> \
  --epochs 3 \
  --batch-size 32 \
  --max-enc-len 1024 \
  --max-dec-len 512
```

The JSONL file should have one example per line with `query`, `tools`, and `answers` fields.

## Notes

- All endpoints (finetune, upload, download) are localhost-only
- Training times out after 2 hours
- The base model stays loaded for interactive testing via the generate tab
