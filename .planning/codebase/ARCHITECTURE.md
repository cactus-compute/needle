<!-- refreshed: 2026-08-31 -->
# Architecture

**Analysis Date:** 2026-08-31

## System Overview

```text
                         Python public API / CLI
       `needle/__init__.py`       `needle/cli.py`
                |                         |
                v                         v
       Native C engine (ctypes)    JAX/Flax reference + training
       `needle/agent/fetch.py`     `needle/model/*.py`
                |                         |
                v                         v
       `.cact` weights + grammar   `.pkl` checkpoints / LoRA
                |                         |
                +------------+------------+
                             v
                    Device inference/runtime
                    `needle.Needle`, playground,
                    ready-made environments
```

The package has two intentionally separate execution paths. Production inference uses a platform-specific shared library (`libneedle.so`, `.dylib`, or `.dll`) loaded by `ctypes` and a self-contained `.cact` archive. Training, checkpoint inspection, reference generation, and export use Python, JAX, Flax, and NumPy under `needle/model`.

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Public agent API | Resolve Python callables/Pydantic models to JSON schemas, bind the engine, complete requests, execute tool loops | `needle/__init__.py` |
| Tool schema builder | Convert annotations, `Literal`, enums, `Field`, docstrings, and Pydantic models into schemas | `needle/agent/tools.py` |
| Engine fetcher | Select platform tag, download/cache the native engine from Hugging Face | `needle/agent/fetch.py` |
| CLI router | Parse `run`, `finetune`, `generate-data`, `build`, `download`, `fetch`, and `playground` commands | `needle/cli.py` |
| Reference model | Define TransformerConfig, Simple Attention Network modules, masks, KV-window sizing | `needle/model/architecture.py` |
| Reference decode | Load `.pkl` checkpoints and run greedy/temperature generation in JAX | `needle/model/run.py`, `needle/model/decode.py` |
| Fine-tuning pipeline | Render JSONL examples, synthesize data, train LoRA, merge adapter, call export | `needle/model/finetune.py` |
| Quantization | Fake-QAT and Cactus Quants codebooks, mixed bit maps, deployment quantization | `needle/model/quantize.py` |
| Export format | Pack tensors, metadata, codebooks, and SentencePiece tokenizer into `.cact`; read it back | `needle/model/export.py` |
| Playground | Threaded HTTP server around `Needle`, model loading, completion, and background fine-tune | `needle/playground/server.py` |
| Environments | Curated tool schemas and frozen acceptance cases | `needle/environments/*.py`, `needle/environments/_harness.py` |

## Pattern Overview

**Overall:** Thin public facade over a native runtime, with a Flax reference/training implementation and an offline binary export boundary.

**Key Characteristics:**
- The native engine is process-global: `needle/__init__.py` keeps one loaded library, active agent, and active weight blob. A tuned archive cannot be unloaded in-process.
- Tool declarations are data at the engine boundary. Python callables are retained locally for `Needle.run()` execution, while schemas are serialized to JSON for constrained decoding.
- The model output head is tied to `embedding.embedding`; `.cact` export transposes and quantizes runtime matrices in a fixed positional tensor order.
- The reference stack is scanned over `num_layers` with per-layer parameter axes, optional rematerialization, and configurable flash attention.

## Layers

**Public API and orchestration:**
- Purpose: User-facing completion, extraction, and agentic tool execution.
- Location: `needle/__init__.py`, `needle/agent/tools.py`.
- Contains: `Needle`, `extract`, `tool`, `Field`, schema conversion and result normalization.
- Depends on: `ctypes`, native engine symbols, Hugging Face fetch fallback.
- Used by: README examples, environments, playground, downstream applications.

**Native engine adapter:**
- Purpose: Load the platform binary and expose `needle_init`, `needle_complete`, `needle_reset`, and `needle_load`.
- Location: `needle/__init__.py`, `needle/agent/fetch.py`.
- Contains: cache lookup, download, symbol signatures, output buffer handling, process-global weight state.
- Depends on: platform detection and Hugging Face model artifacts.
- Used by: every production `Needle` call and the playground.

**Model definition:**
- Purpose: Compute logits and auxiliary heads in JAX/Flax.
- Location: `needle/model/architecture.py`.
- Contains: `ZCRMSNorm`, `MultiHeadAttention`, `HadamardMLP`, `Block`, `Stack`, `Engram`, mHC routing, contrastive/confidence heads, masks.
- Depends on: JAX, Flax, quantization helpers.
- Used by: fine-tuning, checkpoint generation, reference decode, export metadata.

**Training/export:**
- Purpose: Convert JSONL supervision into a LoRA adapter and deployable archive.
- Location: `needle/model/finetune.py`, `needle/model/quantize.py`, `needle/model/export.py`.
- Contains: prompt rendering, masked causal loss, LoRA over attention projections, CQ packing, tokenizer embedding.
- Depends on: base `.pkl` checkpoint, SentencePiece tokenizer, optional OpenRouter API for data generation.
- Used by: CLI `finetune`, `generate-data`, and `build`; playground background fine-tune.

## Data Flow

### Production Completion Path

1. `Needle.__init__` resolves tools to schemas and records callable implementations (`needle/__init__.py:56-114`).
2. `_bind` locates/downloads the native library, optionally loads `.cact` bytes, and calls `needle_init` with system text, tools JSON, and optional tool index (`needle/__init__.py:75-99`).
3. `Needle.complete` sends UTF-8 text and a bounded output buffer to `needle_complete` (`needle/__init__.py:119-137`).
4. The native engine returns a JSON envelope containing `type`, text/function calls, and confidence; tuned weights force confidence to `None` because the confidence head is not adapted.
5. `Needle.run` invokes local Python functions for returned calls, serializes results, feeds them back through `_complete`, and attaches all executed results (`needle/__init__.py:139-160`).

### Reference Model Path

1. `load_checkpoint` reads a format-v2 pickle and reconstructs `TransformerConfig` (`needle/model/run.py:37-72`).
2. `SimpleAttentionNetwork.__call__` embeds token IDs, computes RoPE and engram key/value vectors, runs `Stack`, applies final normalization, and projects against the tied embedding table (`needle/model/architecture.py:478-540`).
3. Each scanned `Block` performs ZCRMS pre-normalization, GQA attention with RoPE and sigmoid output gating, a residual attention gate, then a sandwich-normalized Hadamard MLP (`needle/model/architecture.py:305-375`).
4. `generate`/`batch_generate` repeatedly call the JIT model, append tokens until EOS/max length, and decode with SentencePiece (`needle/model/run.py:89-183`).

### Fine-Tune and Deployment Path

1. JSONL examples are rendered with chat/tool markers by `render_example`; only target tokens contribute to masked cross-entropy (`needle/model/finetune.py:194-243`).
2. `finetune_local` loads the frozen base checkpoint, initializes LoRA A/B matrices only for `q_proj`, `k_proj`, `v_proj`, `gate_proj`, and `out_proj`, then optimizes them with clipped AdamW and a warmup/cosine schedule (`needle/model/finetune.py:254-390`).
3. `build_main` merges the adapter into the base parameters (`merge_lora`) and calls `write_export` (`needle/model/finetune.py:404-440`).
4. `write_export` emits a fixed-header `.cact` containing quantized layer-major tensors, codebooks, optional probe heads, and a raw tokenizer blob (`needle/model/export.py:340-393`).
5. `Needle(weights=...)` loads that archive into the same native engine without recompilation.

**State Management:** JAX model parameters are immutable pytrees; optimizer state is local to `finetune_local`. Native runtime state is mutable and process-global (`_active`, `_active_weights`, `_active_blob`), and `Needle.reset()` resets the native conversation/KV state. Playground serializes engine operations with a lock.

## Key Abstractions

**`TransformerConfig`:** Dataclass carrying model geometry, dtype, RoPE, engram, mHC, KV-window, and quantization settings (`needle/model/architecture.py:58-94`). Keep checkpoint config and export header geometry aligned.

**`SimpleAttentionNetwork`:** Flax top-level model. It owns embedding, `Stack`, auxiliary heads, engram sites, and optional MTP block (`needle/model/architecture.py:478-579`).

**`Needle`:** Runtime facade that owns schemas and Python callables while delegating decoding to native C (`needle/__init__.py:56-169`). Use `complete` for one response, `run` for tool execution, and `extract` for one-shot schema output.

**`.cact` positional archive:** Binary contract between Python export and the native engine. Tensor order and header geometry are defined in `needle/model/export.py:1-80`; do not reorder tensors without a matching engine change.

## Entry Points

**Python package:** `needle/__init__.py` exports `Needle`, `tool`, `Field`, `extract`.

**CLI:** `needle/cli.py:main` is registered as the `needle` console script in `pyproject.toml`; command handlers delegate to model/runtime modules.

**Reference generation:** `needle/model/run.py:main` backs `needle run` and requires a `.pkl` checkpoint.

**Playground:** `needle/playground/server.py:main` serves static UI and HTTP endpoints on `127.0.0.1:7860` by default.

**Environment suites:** `python -m needle.environments.smart_home` (and sibling modules) exercises the native agent through `_harness.run_tests`.

## Architectural Constraints

- **Process-global native state:** One native engine/library and one active weight archive are shared per process; construct base agents before tuned agents or isolate processes.
- **Format compatibility:** `.cact` archives are tied to engine version (`ENGINE_VERSION` in `needle/agent/fetch.py`); rebuild archives after package/engine upgrades.
- **Shape constraints:** Export currently requires equal query/key and value head dimensions and rejects unsupported lexicon or local/global sliding-window configurations (`needle/model/export.py:89-111`).
- **Memory constraint:** KV cache sizing is computed from an ~11.5 MiB budget and aligned to `KV_GROUP`; `effective_kv_window` caps it to the configured maximum (`needle/model/architecture.py:603-620`).
- **Tokenizer contract:** Export embeds the SentencePiece vocabulary and special-token IDs; tokenizer vocabulary must equal `config.vocab_size` (`needle/model/export.py:340-346`).
- **Single-threaded model update:** JAX training mutates no shared model state, while playground native calls are explicitly locked (`needle/playground/server.py:17-54`).

## Anti-Patterns

### Loading Base Weights After Tuned Weights

**What happens:** Constructing a base `Needle` after a tuned agent raises instead of unloading the tuned archive (`needle/__init__.py:79-84`).
**Why it's wrong:** The native engine cannot unload weights and would silently answer with the wrong model.
**Do this instead:** Instantiate base agents first, use one tuned archive per process, or run separate processes.

### Bypassing Schema Generation

**What happens:** Passing arbitrary callable objects without usable annotations/docstrings produces weak schemas (`needle/agent/tools.py:100-145`).
**Why it's wrong:** Constrained decoding depends on accurate JSON types, enums, bounds, and descriptions.
**Do this instead:** Annotate every argument, use `Literal`/`Field` for constraints, and document callable behavior.

### Editing `.cact` Tensor Order Independently

**What happens:** A custom export that changes positional tensor order can still produce a file but native loading interprets weights incorrectly.
**Why it's wrong:** The runtime directory is intentionally nameless and positional (`needle/model/export.py:367-379`).
**Do this instead:** Extend both export and engine format together, with round-trip tests in `tests/test_build.py`.

## Error Handling

**Strategy:** Raise explicit Python exceptions at boundaries; return structured tool errors inside `Needle.run`; return JSON error bodies from playground handlers.

**Patterns:**
- Missing/invalid checkpoints raise `ValueError` with format-version details (`needle/model/run.py:54-72`).
- Native negative return codes become `RuntimeError` (`needle/__init__.py:123-134`).
- Tool lookup/execution errors are appended as `{"error": ...}` results rather than aborting the loop (`needle/__init__.py:147-158`).
- Playground catches request exceptions and responds with an error envelope (`needle/playground/server.py:133-170`).

## Cross-Cutting Concerns

**Logging:** CLI/model paths print aligned progress lines; native XLA noise is filtered at startup in `needle/cli.py:20-111`; playground suppresses HTTP access logs.

**Validation:** Tool schema constraints are compiled into the engine's constrained decoder; environment suites compare exact function-call JSON and optionally apply confidence gates (`needle/environments/_harness.py:11-46`).

**Authentication:** The runtime itself has no user auth. Optional OpenRouter data synthesis uses `OPENROUTER_API_KEY` in `needle/model/finetune.py`; Hugging Face access uses the ambient `huggingface_hub` configuration.

---

*Architecture analysis: 2026-08-31*
