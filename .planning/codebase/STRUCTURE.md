# Codebase Structure

**Analysis Date:** 2026-08-31

## Directory Layout

```text
needle/
├── __init__.py             # Public API and ctypes native-engine facade
├── cli.py                  # Console command parser/router
├── _telemetry.py           # Anonymous usage tracking/opt-out
├── agent/
│   ├── tools.py            # Callable/Pydantic -> JSON schema conversion
│   └── fetch.py            # Platform engine download/cache
├── model/
│   ├── architecture.py     # Flax Simple Attention Network and config
│   ├── decode.py           # Cached JAX decode implementation
│   ├── run.py              # Checkpoint loading and reference generation
│   ├── finetune.py         # Data synthesis, JSONL loading, LoRA, build CLI
│   ├── quantize.py         # Fake-QAT and CQ quantization utilities
│   ├── export.py           # .cact binary writer/reader and tokenizer blob
│   └── tokenizer.py        # SentencePiece tokenizer and special IDs
├── environments/           # Tool surfaces and acceptance suites
└── playground/             # Browser assets and threaded HTTP server
tests/                      # Pytest unit, integration, and slow training tests
doc/                        # API, environment, and fine-tuning guides
assets/                     # README architecture/frontier images
```

## Directory Purposes

**`needle/`:** Installable package. Keep public API changes in `needle/__init__.py` and CLI changes in `needle/cli.py`; avoid importing JAX from the runtime facade so production installs remain lightweight.

**`needle/agent/`:** Runtime integration boundary. `tools.py` owns schema semantics and `fetch.py` owns platform/Hugging Face artifact retrieval. Native shared libraries are downloaded into the user cache, not committed here.

**`needle/model/`:** Training/reference implementation. `architecture.py` is the source of truth for parameter names and geometry; `finetune.py` assumes those names when selecting LoRA targets; `export.py` assumes the same tree when serializing.

**`needle/environments/`:** Product-like examples. Each module defines `SYSTEM`, `TOOLS`, and `TEST_CASES`; `_harness.py` supplies lazy agent construction and exact acceptance scoring.

**`needle/playground/`:** Self-contained static frontend plus `server.py`. It stores uploaded/tuned temporary files in the system temp directory and runs fine-tuning in a daemon thread.

**`tests/`:** Pytest suite configured by `pyproject.toml`. `tests/conftest.py` supplies shared fixtures such as tiny checkpoints; engine-dependent tests use the `requires_engine` marker.

**`doc/`:** User-facing operational guidance. Read `doc/apis.md` for runtime contracts, `doc/finetuning.md` for dataset/training details, and `doc/environments.md` for tool-surface design.

## Key File Locations

**Entry Points:**
- `needle/__init__.py`: `Needle`, `extract`, `tool`, and `Field` public API.
- `needle/cli.py`: `main()` and all console subcommands.
- `needle/playground/server.py`: local HTTP server entry point.
- `needle/model/run.py`: reference checkpoint generation entry point.

**Configuration:**
- `pyproject.toml`: package metadata, Python >=3.9, dependencies/extras, console script, package data, pytest config.
- `needle/model/architecture.py`: `TransformerConfig` and `PRESETS` model geometry.
- `needle/agent/fetch.py`: `HF_REPO`, `ENGINE_VERSION`, platform tags.
- `needle/model/tokenizer.py`: tokenizer path and special token IDs.

**Core Logic:**
- `needle/model/architecture.py`: model block, engram, mHC, masks, KV budgeting.
- `needle/model/decode.py`: KV-cached forward pass and generation internals.
- `needle/__init__.py`: native completion and agent loop.
- `needle/agent/tools.py`: schema reflection and validation metadata.

**Training/Deployment:**
- `needle/model/finetune.py`: JSONL encoding, LoRA optimizer, adapter serialization, `.cact` build orchestration.
- `needle/model/export.py`: binary archive contract and round-trip reader.
- `needle/model/quantize.py`: CQ/fake quantization algorithms.

**Testing:**
- `tests/test_inference.py`: native completion, extraction, loops, and multiple-agent behavior.
- `tests/test_finetune.py`: adapter contents and merge/build integration.
- `tests/test_build.py`: archive creation, bit widths, and projection round trips.
- `tests/test_run.py`: prompt rendering and reference CLI behavior.
- `tests/test_tools.py`: schema reflection and constraints.
- `tests/test_environments.py`: environment acceptance contracts.

## Naming Conventions

**Files:**
- Lowercase snake_case for Python modules (`finetune.py`, `data_capture.py`).
- Leading underscore for private helpers/modules (`_harness.py`, `_telemetry.py`).
- Tests use `test_<area>.py`; fixtures and test data are kept under `tests/` or temporary paths.

**Directories:**
- Lowercase package names (`agent`, `model`, `environments`, `playground`).
- No source-generated build directory is required; checkpoints and `.cact` outputs are user-selected paths (README defaults to `checkpoints/`).

**Python symbols:**
- Classes use PascalCase (`SimpleAttentionNetwork`, `TransformerConfig`, `Needle`).
- Functions and variables use snake_case (`load_checkpoint`, `merge_lora`).
- Constants use uppercase (`BOS_ID`, `ENGINE_VERSION`, `LORA_TARGETS`).

## Where to Add New Code

**New Public Runtime Feature:**
- Primary code: `needle/__init__.py` for API behavior; add schema support in `needle/agent/tools.py` when required.
- Tests: `tests/test_inference.py` or `tests/test_tools.py`.
- Documentation: `doc/apis.md` and the relevant README section.

**New Model Layer/Head:**
- Implementation: `needle/model/architecture.py`, with config fields in `TransformerConfig`.
- Reference execution: update `needle/model/decode.py` if cached/native-equivalent inference needs the layer.
- Export: update tensor order/header handling in `needle/model/export.py` and add round-trip coverage to `tests/test_build.py`.

**New Fine-Tune Behavior:**
- Data/rendering/optimizer: `needle/model/finetune.py`.
- Quantization changes: `needle/model/quantize.py`.
- Tests: `tests/test_finetune.py`; use the `slow` marker for accelerator-dependent work.

**New CLI Command:**
- Parser and dispatch: `needle/cli.py`.
- Implementation: keep command-specific logic in `needle/model/` or `needle/agent/`, not in the parser function.
- Test: add argument/behavior coverage under `tests/test_*.py`.

**New Environment:**
- Add `needle/environments/<name>.py` defining `SYSTEM`, `TOOLS`, and `TEST_CASES`.
- Reuse `needle.environments._harness.agent_for` and `run_tests`; add a focused module test in `tests/test_environments.py`.

**Utilities:**
- Shared schema helpers belong in `needle/agent/tools.py`; model math belongs in `needle/model/architecture.py` or `needle/model/quantize.py` according to ownership.

## Special Directories

**`checkpoints/` (runtime-created):**
- Purpose: default destination for downloaded/base checkpoints and LoRA adapters.
- Generated: Yes.
- Committed: No by convention; use explicit paths for reproducible artifacts.

**User cache `~/.cache/cactus-needle/<ENGINE_VERSION>/`:**
- Purpose: downloaded native engine library.
- Generated: Yes.
- Committed: No.

**`needle/model` package data:**
- `*.model` and `*.vocab` tokenizer files are declared in `pyproject.toml` and may be downloaded from Hugging Face when absent.

**System temp directory:**
- Playground uploads, generated JSONL, adapters, and tuned `.cact` files are placed in `tempfile.gettempdir()` by `needle/playground/server.py`; they are ephemeral and not source-controlled.

**`assets/`:**
- Purpose: README visual assets (`banner.png`, `architecture.png`, frontier images).
- Generated: No for normal development.
- Committed: Yes.

---

*Structure analysis: 2026-08-31*
