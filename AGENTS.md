<!-- GSD:project-start source:PROJECT.md -->

## Project

**Needle**

Needle is a lightweight Python package for running a compact language model and
agent workflows on constrained devices. It provides a native inference facade,
JAX/Flax reference and training code, LoRA fine-tuning, quantization/export to
`.cact`, a CLI, and a local playground.

This milestone focuses on making the existing capabilities understandable and
usable by Python and machine-learning beginners through Chinese-first
documentation and reproducible end-to-end examples.

**Core Value:** A beginner can install Needle and reliably go from a first inference request to
a fine-tuned, exported model without guessing which assets, commands, or
runtime constraints apply.

### Constraints

- **Audience**: Write for Python/ML beginners, while linking to source paths for
  readers who need implementation detail.
- **Language**: Chinese is the primary user-facing documentation language; keep
  API names, commands, paths, and code identifiers exact.
- **Platforms**: Cover both CPU-first setup and supported NVIDIA CUDA/Apple
  Metal acceleration without claiming unsupported combinations.
- **Compatibility**: Preserve public APIs, CLI behavior, checkpoint formats, and
  `.cact` tensor ordering while improving documentation.
- **Verification**: Every tutorial must state prerequisites, expected output,
  and a practical way to verify success.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Python (>=3.9) - Public package API, CLI, inference orchestration, training and export in `needle/`.
- JAX/Python numerical code - Neural-network definition, decoding, quantization and LoRA updates in `needle/model/architecture.py`, `needle/model/decode.py`, `needle/model/quantize.py`, and `needle/model/finetune.py`.
- C/C++ shared library (prebuilt, outside this repository) - Native inference engine loaded through `ctypes` in `needle/__init__.py`; the Python package does not compile it locally.
- HTML/CSS/JavaScript - Browser playground assets in `needle/playground/index.html`, `needle/playground/app.js`, and `needle/playground/style.css`.

## Runtime

- CPython 3.9 or newer (declared by `requires-python` in `pyproject.toml`).
- Native engine selected by platform/architecture and loaded with `ctypes.CDLL` from `needle/__init__.py`.
- `pip`/PEP 517 setuptools build (`setuptools>=68.0`, `pyproject.toml`).
- Lockfile: missing; dependency versions are specified as unconstrained or minimum versions in `pyproject.toml` and `requirements*.txt`.

## Frameworks

- Flax Linen (`flax>=0.10.2`, training extra) - Transformer/Simple Attention Network modules in `needle/model/architecture.py`.
- JAX (`jax`, `jaxlib`, training extra) - Array operations, autodiff, JIT and accelerator execution in `needle/model/`.
- SentencePiece (`sentencepiece`, training extra) - Tokenizer model loading/encoding in `needle/model/tokenizer.py`.
- Python standard library `argparse` and `http.server` - CLI in `needle/cli.py` and local playground server in `needle/playground/server.py`.
- Pytest (`pytest`, test extra) - Tests under `tests/`, configured by `tool.pytest.ini_options` in `pyproject.toml`.
- Pydantic (`pydantic`, test extra/runtime-optional) - Typed extraction schemas and test fixtures; integrated dynamically in `needle/agent/tools.py` and `needle/__init__.py`.
- Setuptools package discovery and package data, configured in `pyproject.toml`.
- Cactus Quants export format (`.cact`) implemented in `needle/model/export.py` and quantization utilities in `needle/model/quantize.py`.

## Key Dependencies

- `huggingface_hub` - Downloads the base checkpoint, tokenizer, native engine wheels and published `.cact` archives (`needle/model/run.py`, `needle/model/tokenizer.py`, `needle/agent/fetch.py`, `needle/cli.py`).
- `jax`/`jaxlib` - Required for checkpoint inference utilities and all fine-tuning/export paths (`needle/model/run.py`, `needle/model/finetune.py`).
- `flax` - Parameterized neural-network modules and tree traversal for LoRA (`needle/model/architecture.py`, `needle/model/finetune.py`).
- `optax` - AdamW, warmup/cosine schedule, gradient clipping and loss helpers in `needle/model/finetune.py`.
- `sentencepiece` - Required to train or load the model tokenizer (`needle/model/tokenizer.py`).
- `numpy` - Checkpoint conversion, array serialization and export packing across `needle/model/`.
- `pydantic` (optional) - Converts `BaseModel` schemas into tool contracts and typed extraction results (`needle/agent/tools.py`, `needle/__init__.py`).

## Configuration

- `NEEDLE_LIB_PATH` overrides native engine lookup; otherwise the package directory and `~/.cache/cactus-needle/<engine-version>/` are searched (`needle/__init__.py`).
- `HF_HUB_OFFLINE=1` prevents Hugging Face network access for air-gapped operation (documented in `doc/apis.md`).
- `OPENROUTER_API_KEY` authorizes optional synthetic data generation; `OPENROUTER_URL` overrides the OpenAI-compatible endpoint (`needle/model/finetune.py`).
- `NEEDLE_HF_REPO` selects the Hugging Face destination for `needle build --upload` (`needle/model/finetune.py`).
- `NEEDLE_TELEMETRY=0` or `DO_NOT_TRACK=1` disables anonymous telemetry; `CI` also disables it (`needle/_telemetry.py`).
- `ENABLE_PJRT_COMPATIBILITY` is set automatically on macOS before JAX initialization for the Metal plugin (`needle/model/finetune.py`).
- `pyproject.toml` defines package metadata, optional extras (`train`, `gpu`, `metal`, `test`), console script `needle = needle.cli:main`, package data and pytest paths.
- `requirements.txt` contains runtime installation; `requirements-train.txt` extends it with JAX/Flax/Optax/SentencePiece training dependencies.
- Model/tokenizer assets (`*.model`, `*.vocab`) are included as package data for `needle.model`; playground static assets are included for `needle.playground`.

## Platform Requirements

- Python 3.9+ and a platform-supported JAX backend. Install `cactus-needle[train,gpu]` for NVIDIA CUDA 12 or `cactus-needle[train,metal]` for Apple Silicon Metal (pins JAX 0.4.38).
- Network access is needed once to fetch the native engine/checkpoint/tokenizer from Hugging Face unless assets are pre-populated in cache.
- A supported native engine binary for the target platform (downloaded by `needle fetch` or `needle download <platform>`).
- Runtime RAM target is approximately 28 MB for the bundled 14 MB Needle 2 engine/weights, as described in `README.md`; tuned `.cact` archives use the same engine.

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming Patterns

- Python modules use lowercase `snake_case.py`, grouped by package responsibility, for example `needle/model/finetune.py` and `needle/agent/fetch.py`.
- Tests use `test_<subject>.py` and functions use `test_<behavior>()`, for example `tests/test_render.py::test_encode_loss_mask_targets_only`.
- Public and private functions use `snake_case`; private implementation helpers begin with `_`, such as `_parse_array` in `needle/model/finetune.py` and `_library_path` in `needle/__init__.py`.
- Boolean/configuration helpers use descriptive predicates or accessors (`_engine_available`, `has_default`, `effective_kv_window`).
- Local variables and parameters use `snake_case`; short mathematical names are used in tensor code where shape context is clear (`B`, `T`, `D`, `q`, `k`, `v` in `needle/model/architecture.py`).
- Module constants use uppercase with underscores (`PAD_ID`, `EOS_ID`, `DEFAULT_BASE`, `LORA_TARGETS`).
- Classes use `PascalCase` (`TransformerConfig`, `SimpleAttentionNetwork`, `SANTokenizer`, `Needle`).
- Type annotations use built-in generics where practical (`list`, `dict`) and `typing`/`Annotated` for Python 3.9-compatible unions and schema metadata, as shown in `needle/agent/tools.py` and `needle/environments/smart_home.py`.
- Dataclass configuration belongs in `@dataclass` classes; model hyperparameters are centralized in `needle/model/architecture.py::TransformerConfig`.

## Code Style

- No Black, Ruff, Flake8, isort, or formatter configuration is present in `pyproject.toml` or the repository root. Preserve the existing four-space indentation, blank-line grouping, and manually wrapped calls.
- Keep imports at module scope and group standard-library imports before third-party imports and local relative imports, following `needle/model/architecture.py` and `needle/__init__.py`.
- Use trailing commas in multiline calls/collections where the surrounding file does; keep lines readable rather than introducing a new formatter dependency.
- No lint command or enforced lint rules are configured. New code should still avoid unused imports, wildcard imports, mutable default arguments, and broad exception handling except at explicit process/network boundaries.
- The release workflow validates behavior with `pytest -q -m "not slow"` in `.github/workflows/release.yaml`; it does not run a linter.

## Import Organization

- No import aliases or package path aliases are configured. Use package imports such as `from needle.model...` in tests and relative imports within package modules.

## Error Handling

- Raise `ValueError` for invalid user/configuration data and incompatible checkpoint/export formats, for example `needle/model/run.py::load_checkpoint` and `needle/model/export.py::_geometry`.
- Raise `RuntimeError` when an external engine, tokenizer download, native call, or response envelope fails; preserve the original exception with `raise ... from e` where useful (`needle/model/tokenizer.py::get_tokenizer`, `needle/__init__.py::Needle._complete`).
- Catch narrowly when the failure is expected (`JSONDecodeError`, `OSError`, `EntryNotFoundError`). Broad `except Exception` is reserved for isolation boundaries such as tool execution in `Needle.run`, telemetry, and optional platform probing.
- Tool execution errors are converted into structured `{"error": ...}` results so one failing tool does not abort the agent loop (`needle/__init__.py::Needle.run`).
- Validate external schemas and arguments before acting; environment definitions encode bounds/enums using `needle.Field` and `typing.Literal` (`needle/environments/smart_home.py`).

## Logging

- CLI and download/training progress uses aligned `print` messages with labels such as `fetch`, `file`, `weights`, and `next` (`needle/cli.py`, `needle/model/run.py`, `needle/model/tokenizer.py`).
- Streaming generation writes incremental text directly to stdout and flushes (`needle/model/run.py::generate`).
- User-facing warnings use `warnings.warn` for tuned-weight confidence limitations (`needle/__init__.py::Needle.__init__`).
- Anonymous telemetry is isolated in `needle/_telemetry.py`; failures are swallowed there so instrumentation cannot break inference.

## Comments

- Comment non-obvious runtime constraints, binary formats, backend workarounds, or algorithmic invariants. Examples include the Metal PJRT compatibility note in `needle/model/finetune.py` and quantization format documentation in `needle/model/export.py`.
- Keep comments close to the implementation and avoid narrating straightforward assignments.
- Python docstrings document public behavior and tool schemas. Function docstrings in environment modules include an overview and an `Args:` section consumed by `needle/agent/tools.py::build_schema`.
- Public helpers such as `needle.extract` and `needle.environments._harness.run_tests` have concise behavioral docstrings; private tensor helpers generally rely on names and nearby comments.

## Function Design

- Keep orchestration in small helpers and isolate serialization, tokenization, model math, and CLI dispatch in their existing modules. Large model routines may be compact tensor pipelines, but avoid mixing CLI parsing with numerical implementation.
- Prefer explicit keyword arguments for configuration-heavy APIs and defaults that preserve current behavior (`Needle(..., max_new_tokens=256)`, `TransformerConfig`).
- Use annotations on public/tool-facing parameters; use `Annotated[..., needle.Field(...)]` for validation constraints and `Literal` for closed sets.
- Return plain dictionaries/lists for JSON/native boundaries (`Needle.complete`, tool results, generated examples).
- Return typed Pydantic instances only when the caller supplies a Pydantic schema (`needle.extract`).
- Preserve array dtypes/shapes at numerical boundaries and use NumPy/JAX conversion explicitly rather than implicit Python coercion.

## Module Design

- `needle/__init__.py` defines the public surface through `__all__` (`Needle`, `tool`, `Field`, `extract`, `__version__`).
- Model internals are imported from their focused modules; avoid adding engine, training, or quantization implementation to the package root.
- There are no broad barrel modules. `needle/model/__init__.py`, `needle/agent/__init__.py`, and `needle/environments/__init__.py` provide lightweight package entry points/registries only.

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```text

```

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

- The native engine is process-global: `needle/__init__.py` keeps one loaded library, active agent, and active weight blob. A tuned archive cannot be unloaded in-process.
- Tool declarations are data at the engine boundary. Python callables are retained locally for `Needle.run()` execution, while schemas are serialized to JSON for constrained decoding.
- The model output head is tied to `embedding.embedding`; `.cact` export transposes and quantizes runtime matrices in a fixed positional tensor order.
- The reference stack is scanned over `num_layers` with per-layer parameter axes, optional rematerialization, and configurable flash attention.

## Layers

- Purpose: User-facing completion, extraction, and agentic tool execution.
- Location: `needle/__init__.py`, `needle/agent/tools.py`.
- Contains: `Needle`, `extract`, `tool`, `Field`, schema conversion and result normalization.
- Depends on: `ctypes`, native engine symbols, Hugging Face fetch fallback.
- Used by: README examples, environments, playground, downstream applications.
- Purpose: Load the platform binary and expose `needle_init`, `needle_complete`, `needle_reset`, and `needle_load`.
- Location: `needle/__init__.py`, `needle/agent/fetch.py`.
- Contains: cache lookup, download, symbol signatures, output buffer handling, process-global weight state.
- Depends on: platform detection and Hugging Face model artifacts.
- Used by: every production `Needle` call and the playground.
- Purpose: Compute logits and auxiliary heads in JAX/Flax.
- Location: `needle/model/architecture.py`.
- Contains: `ZCRMSNorm`, `MultiHeadAttention`, `HadamardMLP`, `Block`, `Stack`, `Engram`, mHC routing, contrastive/confidence heads, masks.
- Depends on: JAX, Flax, quantization helpers.
- Used by: fine-tuning, checkpoint generation, reference decode, export metadata.
- Purpose: Convert JSONL supervision into a LoRA adapter and deployable archive.
- Location: `needle/model/finetune.py`, `needle/model/quantize.py`, `needle/model/export.py`.
- Contains: prompt rendering, masked causal loss, LoRA over attention projections, CQ packing, tokenizer embedding.
- Depends on: base `.pkl` checkpoint, SentencePiece tokenizer, optional OpenRouter API for data generation.
- Used by: CLI `finetune`, `generate-data`, and `build`; playground background fine-tune.

## Data Flow

### Production Completion Path

### Reference Model Path

### Fine-Tune and Deployment Path

## Key Abstractions

## Entry Points

## Architectural Constraints

- **Process-global native state:** One native engine/library and one active weight archive are shared per process; construct base agents before tuned agents or isolate processes.
- **Format compatibility:** `.cact` archives are tied to engine version (`ENGINE_VERSION` in `needle/agent/fetch.py`); rebuild archives after package/engine upgrades.
- **Shape constraints:** Export currently requires equal query/key and value head dimensions and rejects unsupported lexicon or local/global sliding-window configurations (`needle/model/export.py:89-111`).
- **Memory constraint:** KV cache sizing is computed from an ~11.5 MiB budget and aligned to `KV_GROUP`; `effective_kv_window` caps it to the configured maximum (`needle/model/architecture.py:603-620`).
- **Tokenizer contract:** Export embeds the SentencePiece vocabulary and special-token IDs; tokenizer vocabulary must equal `config.vocab_size` (`needle/model/export.py:340-346`).
- **Single-threaded model update:** JAX training mutates no shared model state, while playground native calls are explicitly locked (`needle/playground/server.py:17-54`).

## Anti-Patterns

### Loading Base Weights After Tuned Weights

### Bypassing Schema Generation

### Editing `.cact` Tensor Order Independently

## Error Handling

- Missing/invalid checkpoints raise `ValueError` with format-version details (`needle/model/run.py:54-72`).
- Native negative return codes become `RuntimeError` (`needle/__init__.py:123-134`).
- Tool lookup/execution errors are appended as `{"error": ...}` results rather than aborting the loop (`needle/__init__.py:147-158`).
- Playground catches request exceptions and responds with an error envelope (`needle/playground/server.py:133-170`).

## Cross-Cutting Concerns

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `$gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `$gsd-debug` for investigation and bug fixing
- `$gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `$gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
