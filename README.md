![Needle](assets/banner.png)

# Needle 2

Needle 2 is an open 45M-parameter model for tool calling, device use and structured extraction. The whole model is a single 14MB binary that runs a full session in about 28MB of RAM. It is built on our Simple Attention Network findings, compressed to CQ2-bit with Cactus Quants, and baked into its own engine. On the benchmarks below, Needle 2 trades wins with other small models like FunctionGemma 270M, LFM2.5 230M and Apple FM, at 5x to 70x smaller, and 2 bits against their f16.

This repository is the Python package: inference, LoRA fine-tuning, and export. `pip install cactus-needle`, describe your tools, and call them from Python. The inference engine is fetched once from Hugging Face and cached; there is nothing else to build, and offline setup for air gapped devices is covered in [doc/apis.md](doc/apis.md).

- **Self-contained**: weights baked into a single 14MB engine; no separate model files to manage, and inference does no network.
- **Simple contract**: tool calls come back as structured data, text in, JSON out; a byte-level grammar compiled from your schemas constrains every token.
- **Confidence-gated**: every response carries a calibrated confidence score from a learned head; set a threshold, act above it, escalate below it.
- **Tool retrieval**: declare a large catalogue and a built-in retrieval head renders only the top five tools per turn, with the grammar constrained to that subset.
- **Bounded memory**: a 256-token sliding window with the tools pinned as KV sinks, so total memory stays near 28MB no matter how long the conversation runs.

Weights: [huggingface.co/Cactus-Compute/needle2](https://huggingface.co/Cactus-Compute/needle2) &middot; source: [github.com/cactus-compute/needle](https://github.com/cactus-compute/needle).

![Size-quality frontier: mobile-class and below](assets/frontier.png)

## Simple Attention Network

Needle 2 is a Simple Attention Network, our dense small-model recipe: a Hadamard MLP in place of the FFN, GQA attention, engram key-value memory, and multi-lane hyper-connections. See the paper for the design and ablations: [arXiv:2607.18363](https://arxiv.org/abs/2607.18363).

![Simple Attention Network architecture](assets/architecture.png)

Each block carries its update rule. Here x̂ is the RMS-normalised flattening of the four residual streams, H the orthonormal Walsh-Hadamard transform (a fixed matrix, applied in n log n time with no weights to read), (kₜ, vₜ) rows gathered from hashed n-gram tables, and P the doubly-stochastic normalisation of the routing logits A, computed by Sinkhorn iteration; a, b, g and all σ-gates are learned and input-dependent. Both attention and MLP residuals are sandwich-normed and gated, the engram sites fire at two layers, and decoding is constrained by a byte-level grammar compiled from the declared schemas.

## Quickstart

```sh
pip install cactus-needle
```

The runtime package does not install the training stack. Add the `train` extra
when using fine-tuning or checkpoint export:

```sh
pip install "cactus-needle[train]"
```

Needle reads your tool descriptions to decide what to call and how to fill arguments, so describing them well is the whole game.

**Simple**: decorate a function. The signature gives the argument types, the docstring is the tool description, and `run()` completes the loop: model picks the call, Needle executes your function, feeds the result back, and returns the final response with the executed tool results attached as `results`.

```python
import needle

@needle.tool
def get_weather(city: str):
    "Get the current weather for a city."
    return {"city": city, "temp_c": 27, "sky": "clear"}

agent = needle.Needle(tools=[get_weather])
print(agent.run("what's it like in Lagos right now?")["results"])
# [{'city': 'Lagos', 'temp_c': 27, 'sky': 'clear'}]
```

**Extraction**: to pull structured data out of text, declare the shape and call `extract()`. Pass a Pydantic model and you get a typed object back.

```python
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    total: float
    due_date: str

invoice = needle.extract("Invoice from Acme Corp, $1,200.00, due 2026-09-01", Invoice)
print(invoice.vendor, invoice.total)   # -> Acme Corp 1200.0
```

Per argument descriptions and choices, value constraints compiled into the decode grammar, raw JSON schemas, driving the loop with `complete()`, the response contract, system facts, tool retrieval, and confidence gating are all covered in [doc/apis.md](doc/apis.md).

## Playground

Try any model in the browser: pick a preset, edit the tools or prompt, and Run. Follow-up queries continue the same conversation.

```sh
needle playground                      # base model, http://127.0.0.1:7860
needle playground --weights my.cact    # a tuned model
```

The server downloads and initializes the model before serving, so the first query is instant. The **Finetune on these tools** button runs the fine-tuning pipeline below from the UI and hands back a downloadable `.cact`.

## Environments

Ready-made tool surfaces in `needle.environments`: `smart_home`, `media_player`, `productivity`, `wearable`, `kitchen_appliance`, and `data_capture`. Each is a hand-curated set of tools whose enums, bounds, and descriptions map cleanly onto Needle's constrained decoding, with a ready agent and a frozen acceptance suite.

```python
from needle.environments import smart_home

smart_home.agent.complete("dim the study lights to 30 percent")
smart_home.run_tests()
```

`python -m needle.environments.smart_home` runs a suite from the shell. To adapt an environment to your product, swap the `Literal` values (rooms, contacts, categories) for your own and keep the shapes: closed sets as enums, bounded numbers, verbatim copy for free text, five tools or fewer. The full tool surfaces and the suite contract are in [doc/environments.md](doc/environments.md).

## Fine-tuning

Needle fine-tunes with LoRA on the frozen base and merges the adapter at export, so a run is cheap and the tuned model is still a single `.cact` that runs on the same engine. The workflow is: (optionally) synthesize data, LoRA fine-tune, then build a tuned `.cact`. See [doc/finetuning.md](doc/finetuning.md) for dataset sizing, reading the loss curve, and troubleshooting.

**Data format.** A JSONL file, one example per line. `reasoning` is optional; an off-topic example has `answers: []`.

```json
{"query": "dim the kitchen to 10", "tools": [{"name": "set_lights", "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "brightness": {"type": "integer"}}, "required": ["room"]}}], "answers": [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 10}}], "reasoning": "'kitchen' -> room; 'dim to 10' -> brightness 10"}
```

**1. Synthesize data (optional).** Needs an API key for the gateway you use. [OrcaRouter](https://www.orcarouter.ai) and OpenRouter both work; OpenRouter is the default. Seed from a tool schema file, or expand an existing set:

```sh
export OPENROUTER_API_KEY=sk-or-...            # or: export ORCAROUTER_API_KEY=sk-orca-...
needle generate-data --tools my_tools.json --num-samples 500 --output data.jsonl
needle generate-data --augment data.jsonl --num-samples 500      # expand an existing JSONL
```

Pass `--provider orcarouter` to synthesize through `https://api.orcarouter.ai/v1`, an OpenAI-compatible
gateway for both models and agents. Set `OPENROUTER_URL` to use a different OpenAI-compatible gateway
instead of the default OpenRouter endpoint.

**OrcaRouter.** Rather than pasting a key, you can authorize with your own account; the consent screen
issues an `sk-orca-…` key that is billed to you and revocable from your console at any time:

```sh
needle connect                                  # OAuth 2.0 + PKCE, loopback redirect
needle connect --oob                            # no callback reachable: paste the code instead
needle models --provider orcarouter             # the real catalog, filtered by capability
needle generate-data --tools my_tools.json --provider orcarouter --model orcarouter/auto
```

`needle models` reads `GET /v1/models` with your key, so the list is what your workspace can actually
call, and it keeps the vendor/model namespace. The model selection in the playground is that same live
list, filtered by capability — not a hand-written sample and not a free-text field. `ORCA_AUTH_BASE_URL`
and `ORCA_API_BASE_URL` override the two origins separately, and `ORCA_BASE_URL` supplies both for a
self-hosted deployment.

**Image inputs.** Prompts can carry one image alongside the text, which makes the run multimodal. The
image requirement narrows the model list to the models whose catalog entry declares an `image` input —
declared ones only, never inferred from a model's name — and a model that does not declare it is refused
before any request is sent:

```sh
needle models --provider orcarouter --input-modality image            # what may receive an image
needle generate-data --tools my_tools.json --provider orcarouter \
  --model deepseek/deepseek-v4.1-flash --input-modality image --image-url receipt.png
```

`--image-url` takes an `https://` URL or a `data:` URI. The playground exposes the same thing as a
checkbox in the finetune dialog: turning it on re-filters the model dropdown and clears a selection
that no longer qualifies.

**2. LoRA fine-tune.** The base checkpoint auto-downloads from Hugging Face if you do not pass `--checkpoint`. `--generate N` first synthesizes N more examples from the tools in your data (also needs an API key).

```sh
needle finetune data.jsonl --epochs 10
needle finetune data.jsonl --epochs 10 --generate 300 --lora-rank 16 --lora-alpha 32
```

Key options: `--epochs` (default 3), `--layers <n>` (fine-tune the n-layer rung of the base, see below), `--lora-rank` (16), `--lora-alpha` (32), `--lr` (1e-4), `--batch-size` (16), `--max-len` (1024), `--val-split` (0.1), `--checkpoint <base.safetensors or .pkl>`, `--checkpoint-dir <dir>` (default `checkpoints`), `--out <adapter.safetensors or .pkl>`, `--generate <n>`, `--provider <id>`, `--model <id>` (default: the provider's own), and `--workers <n>` (default 8). `--generate` uses the configured gateway to synthesize extra examples before training. The adapter is written to `checkpoints/needle_lora.pkl` by default. A validation loss prints each epoch from the held out split.

Training is plain JAX and runs on any accelerator jax supports. On an NVIDIA machine install the CUDA build and the same command trains on the GPU:

```sh
pip install "cactus-needle[train,gpu]"
```

On Apple Silicon the `metal` extra trains on the GPU:

```sh
pip install "cactus-needle[train,metal]"
```

The base is Needle 3, a depth ladder: every rung from 2 layers up to the full stack is a trained model whose blocks are a nested subset of the full one. `--layers n` slices the base to its n-layer rung before training, trains that rung at full depth, and records the depth in the adapter, so `needle build` exports the same rung. A smaller rung trains and runs faster and fits a smaller device at some accuracy cost; without `--layers` the full base is fine-tuned.

```sh
needle finetune data.jsonl --epochs 10 --layers 8
```

**3. Build a tuned `.cact`.** Merge the adapter into the base and quantize. The checkpoint is optional: `needle build` uses the base the adapter was trained on, and auto-downloads the Needle 3 base if absent.

```sh
needle build --lora checkpoints/needle_lora.safetensors --out my_needle.cact
needle build checkpoints/needle3_enterprise.safetensors --lora adapter.safetensors --out my_needle.cact   # 20L base
needle build --layers 4 --out needle3_4l.cact                                                            # untuned 4-layer rung
```

The engine is weights-agnostic and never rebuilt: one engine library per platform (under 1MB) runs any archive, and the archive shrinks with the rung (about 9MB at 2 layers, 29MB at 16). `needle download needle3` fetches the base 16-layer archive by itself, `needle download needle3.safetensors` (or `needle3_enterprise.safetensors`) the checkpoint to fine-tune.

Add `--bits 2` for a smaller model (by default the export follows the checkpoint's declared per-layer bit map, falling back to 4 when the checkpoint declares none), or set `NEEDLE_HF_REPO=<you>/<model>` and pass `--upload` to publish the `.cact`. The counterpart `needle download <you>/<model>/my_needle.cact` pulls a published archive on any machine, and `needle download <platform>` (e.g. `macos-arm64`) fetches that platform's engine runner.

**4. Run it.** The engine is weights-agnostic, so a tuned `.cact` runs on it directly - no recompilation:

```python
import needle
agent = needle.Needle(weights="my_needle.cact", tools=[...])
agent = needle.Needle(tools=[...], generation=3)   # the base Needle 3 archive, fetched once and cached
agent.run("...")
```

## Telemetry

Cactus Compute collects strictly anonymous usage telemetry (function name, package version, OS — never your prompts, outputs, or data); opt out with `NEEDLE_TELEMETRY=0` or `DO_NOT_TRACK=1`.

## Citation

Needle 2 is built by the Cactus Compute team. If you use it in your work, please cite:

```bibtex
@misc{needle2_2026,
  title        = {Needle 2: A 45M-Parameter Foundation Tool-Calling Model for Tiny Devices},
  author       = {Ndubuaku, Henry and Mosoyan, Karen and Mroz, Jakub and Cylich, Noah and
                  Kumar, Satyajit and Sandhu, Parkirat and Shemet, Roman and Lee, Justin H.},
  year         = {2026},
  organization = {Cactus Compute, Inc.},
  howpublished = {\url{https://github.com/cactus-compute/needle}}
}
```

Reach out on founders@cactuscompute.com for partnerships, collaborations, synergies and deploying Needle2 in your product.
