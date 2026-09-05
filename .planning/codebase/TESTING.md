# Testing Patterns

**Analysis Date:** 2026-08-31

## Test Framework

**Runner:**
- `pytest` is configured in `pyproject.toml` with `testpaths = ["tests"]`.
- The only registered marker is `slow`, documented as end-to-end JAX build/finetune tests that initialize a tiny model.
- CI runs `pytest -q -m "not slow"` in `.github/workflows/release.yaml` after installing `.[test,train]`.

**Assertion Library:**
- Use plain Python `assert` for scalar/structure checks and `numpy.testing.assert_allclose` for numerical tensors (`tests/test_lora.py`, `tests/test_build.py`).
- `pytest.raises(..., match=...)` verifies exception type and message; `pytest.mark.skipif` gates optional runtime features.

**Run Commands:**
```bash
pytest -q                         # Run the complete suite
pytest -q -m "not slow"           # Fast/CI suite without JAX fine-tune and build tests
pytest -q -m slow                  # Explicitly run slow model-training/export tests
pytest -q tests/test_tools.py     # Run one focused module
pytest -q tests/test_lora.py::test_merge_lora_adds_scaled_delta  # Run one test
```

## Test File Organization

**Location:**
- Tests are in the top-level `tests/` directory, separate from implementation modules. There are no co-located `needle/**/test_*.py` files.
- `tests/conftest.py` contains shared engine detection and the session-scoped `tiny_checkpoint` fixture.

**Naming:**
- Files are `test_<feature>.py`; tests are descriptive `test_<behavior>()` functions. Test classes are not used.

**Structure:**
```
tests/
├── conftest.py                 # shared fixtures and optional-engine marker
├── test_tools.py               # schema/decorator unit tests
├── test_lora.py                # LoRA tensor math unit tests
├── test_render.py              # prompt/token/loss-mask unit tests
├── test_generate.py            # data synthesis and JSON parsing tests
├── test_run.py, test_weights.py # checkpoint/CLI/runtime behavior
├── test_build.py, test_finetune.py # slow JAX/export integration tests
└── test_inference.py, test_environments.py # native-engine and environment suites
```

## Test Structure

**Suite Organization:**
```python
def test_merge_lora_adds_scaled_delta():
    import jax.numpy as jnp
    from needle.model.finetune import merge_lora

    params = {"w": {"kernel": jnp.zeros((3, 4))}}
    lora = {("w", "kernel"): {"A": jnp.ones((3, 2)), "B": jnp.ones((2, 4))}}
    merged = merge_lora(params, lora, scale=0.5)
    np.testing.assert_allclose(np.asarray(merged["w"]["kernel"]), np.ones((3, 4)))
```

**Patterns:**
- Keep each test focused on one behavior and construct minimal local inputs.
- Import implementation modules inside tests when optional dependencies or monkeypatch setup must occur first (`tests/test_finetune.py`, `tests/test_tools.py`).
- Use fixtures for reusable state (`tok`, `tiny_checkpoint`, `engine`, `tuned`), with `scope="session"` only for expensive immutable setup.
- Assert externally visible contracts: exact schema fields, output types, file existence, checkpoint metadata, array shape/dtype, or allowed response types.

## Mocking

**Framework:**
- Use pytest's `monkeypatch` fixture; no standalone mock library is configured.

**Patterns:**
```python
monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
monkeypatch.setattr(finetune, "generate_examples", fake_generator)
monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
```

- Native engine behavior is replaced with a small `_Stub` object in `tests/test_weights.py`; its methods record calls and populate a JSON envelope in the provided ctypes buffer.
- Use `warnings.catch_warnings()` to suppress expected tuned-weight warnings while testing behavior.

**What to Mock:**
- Mock network/API calls (`needle.model.finetune._openrouter`), environment variables, platform/native library discovery, and ctypes engine bindings when testing control flow.
- Mock expensive generated-data calls with deterministic rows and test deduplication/termination separately (`tests/test_generate.py`).

**What NOT to Mock:**
- Keep pure schema, rendering, LoRA math, quantization, and serialization logic real. Slow tests intentionally initialize a tiny JAX model rather than mocking model parameters (`tests/conftest.py::tiny_checkpoint`).

## Fixtures and Factories

**Test Data:**
- Use `tmp_path`/`tmp_path_factory` for JSONL, checkpoint, adapter, and `.cact` files; tests write only ephemeral files.
- Build minimal dictionaries inline for tool schemas and calls. `tests/test_finetune.py::TOOLS` and `_write_data` are representative fixtures/factories.
- `tiny_checkpoint` creates a 2-layer, 64-dimension `SimpleAttentionNetwork`, converts leaves to NumPy, and serializes a format-v2 pickle for training/export tests.

**Location:**
- Shared fixtures belong in `tests/conftest.py`; feature-specific helpers stay at the top of each test module (`_finetune_args`, `_build_args`, `_parse_array` inputs).

## Coverage

**Requirements:**
- No coverage target or coverage configuration is enforced. CI only runs the non-slow pytest selection.

**View Coverage:**
```bash
pytest --cov=needle --cov-report=term-missing
```
This command is available only when a coverage plugin is installed; it is not declared in `pyproject.toml`.

## Test Types

**Unit Tests:**
- Most tests cover deterministic helpers without the native engine: tool schema generation (`tests/test_tools.py`), prompt rendering/token masks (`tests/test_render.py`), LoRA operations (`tests/test_lora.py`), quantized export round trips (`tests/test_build.py`), and CLI/checkpoint parsing (`tests/test_run.py`, `tests/test_fetch.py`).

**Integration Tests:**
- `tests/test_finetune.py` runs one-epoch JAX LoRA training and then exports/reads a `.cact`; marked `slow`.
- `tests/test_environments.py` validates every declared environment's tool surface, frozen cases, and direct tool execution. The smoke inference test is engine-gated.

**E2E Tests:**
- `tests/test_inference.py` is marked with `requires_engine`, which skips the module when the platform-specific native engine is absent. It exercises `Needle.complete`, `run`, extraction, and multiple agents against the installed engine.

## Common Patterns

**Async Testing:**
- No async test functions or async fixtures are present. Concurrent generation is tested synchronously by monkeypatching worker-facing functions (`tests/test_generate.py`).

**Error Testing:**
```python
with pytest.raises(RuntimeError, match="cannot unload"):
    needle.Needle(tools="[]")
```
- Assert invalid/missing input behavior with `pytest.raises`, and use exact/regex message fragments for important operational diagnostics (`tests/test_weights.py`, `tests/test_generate.py`).
- Engine-optional tests should use the shared `requires_engine` marker rather than failing on machines without the downloaded native library.

---

*Testing analysis: 2026-08-31*
