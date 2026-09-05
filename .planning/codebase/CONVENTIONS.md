# Coding Conventions

**Analysis Date:** 2026-08-31

## Naming Patterns

**Files:**
- Python modules use lowercase `snake_case.py`, grouped by package responsibility, for example `needle/model/finetune.py` and `needle/agent/fetch.py`.
- Tests use `test_<subject>.py` and functions use `test_<behavior>()`, for example `tests/test_render.py::test_encode_loss_mask_targets_only`.

**Functions:**
- Public and private functions use `snake_case`; private implementation helpers begin with `_`, such as `_parse_array` in `needle/model/finetune.py` and `_library_path` in `needle/__init__.py`.
- Boolean/configuration helpers use descriptive predicates or accessors (`_engine_available`, `has_default`, `effective_kv_window`).

**Variables:**
- Local variables and parameters use `snake_case`; short mathematical names are used in tensor code where shape context is clear (`B`, `T`, `D`, `q`, `k`, `v` in `needle/model/architecture.py`).
- Module constants use uppercase with underscores (`PAD_ID`, `EOS_ID`, `DEFAULT_BASE`, `LORA_TARGETS`).

**Types:**
- Classes use `PascalCase` (`TransformerConfig`, `SimpleAttentionNetwork`, `SANTokenizer`, `Needle`).
- Type annotations use built-in generics where practical (`list`, `dict`) and `typing`/`Annotated` for Python 3.9-compatible unions and schema metadata, as shown in `needle/agent/tools.py` and `needle/environments/smart_home.py`.
- Dataclass configuration belongs in `@dataclass` classes; model hyperparameters are centralized in `needle/model/architecture.py::TransformerConfig`.

## Code Style

**Formatting:**
- No Black, Ruff, Flake8, isort, or formatter configuration is present in `pyproject.toml` or the repository root. Preserve the existing four-space indentation, blank-line grouping, and manually wrapped calls.
- Keep imports at module scope and group standard-library imports before third-party imports and local relative imports, following `needle/model/architecture.py` and `needle/__init__.py`.
- Use trailing commas in multiline calls/collections where the surrounding file does; keep lines readable rather than introducing a new formatter dependency.

**Linting:**
- No lint command or enforced lint rules are configured. New code should still avoid unused imports, wildcard imports, mutable default arguments, and broad exception handling except at explicit process/network boundaries.
- The release workflow validates behavior with `pytest -q -m "not slow"` in `.github/workflows/release.yaml`; it does not run a linter.

## Import Organization

**Order:**
1. Standard library (`os`, `json`, `typing`, `dataclasses`, etc.).
2. Third-party dependencies (`numpy`, `jax`, `flax`, `pytest`, `pydantic`).
3. Local package imports, using relative imports inside `needle` (`from .tokenizer import ...`, `from . import quantize`).

**Path Aliases:**
- No import aliases or package path aliases are configured. Use package imports such as `from needle.model...` in tests and relative imports within package modules.

## Error Handling

**Patterns:**
- Raise `ValueError` for invalid user/configuration data and incompatible checkpoint/export formats, for example `needle/model/run.py::load_checkpoint` and `needle/model/export.py::_geometry`.
- Raise `RuntimeError` when an external engine, tokenizer download, native call, or response envelope fails; preserve the original exception with `raise ... from e` where useful (`needle/model/tokenizer.py::get_tokenizer`, `needle/__init__.py::Needle._complete`).
- Catch narrowly when the failure is expected (`JSONDecodeError`, `OSError`, `EntryNotFoundError`). Broad `except Exception` is reserved for isolation boundaries such as tool execution in `Needle.run`, telemetry, and optional platform probing.
- Tool execution errors are converted into structured `{"error": ...}` results so one failing tool does not abort the agent loop (`needle/__init__.py::Needle.run`).
- Validate external schemas and arguments before acting; environment definitions encode bounds/enums using `needle.Field` and `typing.Literal` (`needle/environments/smart_home.py`).

## Logging

**Framework:** `print()` for CLI/progress output; no logging framework is configured.

**Patterns:**
- CLI and download/training progress uses aligned `print` messages with labels such as `fetch`, `file`, `weights`, and `next` (`needle/cli.py`, `needle/model/run.py`, `needle/model/tokenizer.py`).
- Streaming generation writes incremental text directly to stdout and flushes (`needle/model/run.py::generate`).
- User-facing warnings use `warnings.warn` for tuned-weight confidence limitations (`needle/__init__.py::Needle.__init__`).
- Anonymous telemetry is isolated in `needle/_telemetry.py`; failures are swallowed there so instrumentation cannot break inference.

## Comments

**When to Comment:**
- Comment non-obvious runtime constraints, binary formats, backend workarounds, or algorithmic invariants. Examples include the Metal PJRT compatibility note in `needle/model/finetune.py` and quantization format documentation in `needle/model/export.py`.
- Keep comments close to the implementation and avoid narrating straightforward assignments.

**JSDoc/TSDoc:**
- Python docstrings document public behavior and tool schemas. Function docstrings in environment modules include an overview and an `Args:` section consumed by `needle/agent/tools.py::build_schema`.
- Public helpers such as `needle.extract` and `needle.environments._harness.run_tests` have concise behavioral docstrings; private tensor helpers generally rely on names and nearby comments.

## Function Design

**Size:**
- Keep orchestration in small helpers and isolate serialization, tokenization, model math, and CLI dispatch in their existing modules. Large model routines may be compact tensor pipelines, but avoid mixing CLI parsing with numerical implementation.

**Parameters:**
- Prefer explicit keyword arguments for configuration-heavy APIs and defaults that preserve current behavior (`Needle(..., max_new_tokens=256)`, `TransformerConfig`).
- Use annotations on public/tool-facing parameters; use `Annotated[..., needle.Field(...)]` for validation constraints and `Literal` for closed sets.

**Return Values:**
- Return plain dictionaries/lists for JSON/native boundaries (`Needle.complete`, tool results, generated examples).
- Return typed Pydantic instances only when the caller supplies a Pydantic schema (`needle.extract`).
- Preserve array dtypes/shapes at numerical boundaries and use NumPy/JAX conversion explicitly rather than implicit Python coercion.

## Module Design

**Exports:**
- `needle/__init__.py` defines the public surface through `__all__` (`Needle`, `tool`, `Field`, `extract`, `__version__`).
- Model internals are imported from their focused modules; avoid adding engine, training, or quantization implementation to the package root.

**Barrel Files:**
- There are no broad barrel modules. `needle/model/__init__.py`, `needle/agent/__init__.py`, and `needle/environments/__init__.py` provide lightweight package entry points/registries only.

---

*Convention analysis: 2026-08-31*
