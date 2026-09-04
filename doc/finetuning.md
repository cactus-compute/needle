# Finetuning

Needle finetunes with LoRA adapters on the frozen base model: rank 16 on the five attention projections of every layer, trained on your JSONL, then merged into the weights at export. The engine, the tokenizer, and the confidence head are untouched. The output is a `.cact` archive you pass as `weights=`.

## Data format

Install the training dependencies before running the commands in this guide:

```sh
pip install "cactus-needle[train]"
```

One JSON object per line. `query` and `tools` describe the turn, `answers` lists the exact calls the model should emit, `reasoning` is one short line deriving each argument from its source span in the query. `reasoning` is optional but include it: the model produces the derivation before the call, and examples that show where each value comes from teach grounding, not just tool selection.

```json
{"query": "Bantilan, N. (2018). Themis. Journal of Technology in Human Services, 36(1).", "tools": [{"name": "extract_citation_data", "parameters": {"type": "object", "properties": {"authors": {"type": "string"}, "title": {"type": "string"}, "publisher": {"type": "string"}}, "required": ["authors", "title"]}}], "answers": [{"name": "extract_citation_data", "arguments": {"authors": "Bantilan, N.", "title": "Themis", "publisher": "Journal of Technology in Human Services, 36(1)."}}], "reasoning": "authors precede the year; title follows the year; publisher is the journal segment"}
```

Rules that matter:

1. Arguments contain only values present in the query. Omit optional fields with no evidence; never fill them with placeholders or empty strings.
2. Include off topic examples with `"answers": []`. The built in generator produces about 1 in 8. Without them the tuned model calls a tool on everything.
3. When the catalogue has similar tools, include ambiguous queries resolved to the correct one.
4. An optional `"system"` field per example becomes a system turn, matching `Needle(system=...)` at inference.
5. Each rendered example must fit within `--max-len` (default 1024) tokens; longer examples are silently truncated. Padding adjusts to the longest example automatically (rounded up to a power of two), so a short dataset trains fast regardless of the cap. This matters on CPU: a dataset of 245 token examples trains 6 times faster padded to 256 than to 1024.

## Commands

Train, export, load:

```sh
needle finetune data.jsonl --epochs 10 --out adapter.pkl
needle build checkpoints/needle2.pkl --lora adapter.pkl --out tuned.cact
```

```python
agent = needle.Needle(tools=[...], weights="tuned.cact")
```

Fine-tuning is quantization-aware by default. `--qat-bits auto` trains through
the checkpoint's declared CQ export scheme and records that scheme in the
adapter; `needle build` rejects a conflicting bit-width override. Use
`--qat-bits none` only when intentionally targeting float evaluation.

To share a tuned model, set `NEEDLE_HF_REPO=<you>/<model>` and pass `--upload` to `needle build`; on any other machine `needle download <you>/<model>/tuned.cact` pulls it back down (or pass just `<you>/<model>` when the repo holds a single archive).

Defaults: batch size 16, learning rate 0.0001 with warmup and cosine decay, gradient clipping at norm 1, rank 16, alpha 32, max length 1024, validation split 0.1, random seed 0. Use `--seed` to reproduce or intentionally vary LoRA initialization, validation selection, and epoch shuffling. The base checkpoint downloads from Hugging Face on first run.

To grow a small hand written set, seed the generator with it (needs `OPENROUTER_API_KEY`; set `OPENROUTER_URL` to use another OpenAI compatible gateway):

```sh
needle generate-data --augment data.jsonl --num-samples 1000
```

The playground button labelled Finetune on these tools runs the same pipeline from the browser.

Training is plain JAX, so it runs on any accelerator jax supports. On an NVIDIA machine install the CUDA build and the same command trains on the GPU, nothing else changes:

```sh
pip install "cactus-needle[train,gpu]"
```

Apple GPUs train through the jax metal plugin, which does not work past jax 0.4.38, so the `metal` extra pins an older stack:

```sh
pip install "cactus-needle[train,metal]"
```

Needle detects the Metal backend and adapts automatically: manual attention, no rematerialisation, unrolled layer stack, and the `ENABLE_PJRT_COMPATIBILITY` variable the plugin requires is set for you. Measured on an M5 Max: 0.71 seconds per step against 2.90 on CPU at the same shape, about 4 times faster, with a one time compile of about 23 seconds. Training runs in float32 on every backend.

## Reading the loss

The loss covers only the target: the reasoning line plus the JSON call. Much of the call is boilerplate the base model already predicts (the tool name, the braces, the field names), so training starts near 1.0 rather than near random. Judge a run by its trend, not its level.

Step count is what small datasets get wrong. 200 examples at batch 16 is 13 steps per epoch, and the default 3 epochs is 39 steps total, which barely moves a rank 16 adapter at the default learning rate. For a few hundred examples run 10 to 30 epochs and expect a clear downward trend. If the curve sits at its starting value after a few hundred steps, raise the epochs first, then the learning rate.

A validation loss prints at each epoch end (10 percent of examples are held out by default, `--val-split` to change). When it rises while the training loss keeps falling, the run is overfitting: stop there, or add data.

## Sizing the dataset

Tool selection moves first: a few hundred clean examples measurably improve which tool gets picked. Argument grounding moves later and needs more data, on the order of thousands of examples, with reasoning lines and varied phrasings and values. If evaluation shows correct tools with wrong argument values, the dataset is too small or too uniform, not mislabeled. For grounding heavy tasks `--lora-rank 32` doubles adapter capacity and the adapter stays tiny.

For a large catalogue, consider two passes at inference instead of more training: one turn against the full catalogue to pick the tool, then one turn declaring only that tool, which constrains the grammar to exactly that call.

## What finetuning does not change

The confidence head. Scores are calibrated for the base model on its training mix and finetuning does not update the head, so the package disables them for tuned weights: `Needle(weights=...)` warns once at construction and reports `confidence` as None. Non English deployments of the base model should also treat the score with caution (correct Spanish calls have been measured at confidence 0.0).

The tokenizer. Non English text fragments into roughly 1.7 times more tokens (measured on Spanish), which taxes both quality and the 256 token window.

## Troubleshooting

1. Loss goes NaN within the first steps on CPU: fixed, run `pip install --upgrade cactus-needle`.
2. The tuned model answers "Sorry, I can't help with that" on everything: an old engine gated low confidence responses; the gate is removed as of engine 2.0.1. Upgrade the package.
3. `failed to load weights`: the `.cact` format is tied to the engine version, so an archive exported by an older package will not load. Rebuild it with the current package version.
4. Loss hovers at its starting value: see Reading the loss. The run is undertrained, not broken.
