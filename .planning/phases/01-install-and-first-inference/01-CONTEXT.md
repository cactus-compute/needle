# Phase 1: Install and First Inference - Context

**Gathered:** 2026-08-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a Chinese-first, CPU-only onboarding path that creates a Python
environment, installs the current repository, explicitly fetches native assets,
and verifies a first CLI inference before showing the equivalent Python API.
CUDA and Metal execution are not part of this phase's acceptance criteria.

</domain>

<decisions>
## Implementation Decisions

### Environment Installation
- **D-01:** Use `uv` to create and manage the local virtual environment.
- **D-02:** Install the current repository in editable mode rather than using the published PyPI package.
- **D-03:** Phase 1 installs the complete `.[train,test]` extras so the environment is ready for later training documentation.
- **D-04:** Keep the package's declared Python 3.9+ support; record the exact Python version used during verification instead of narrowing support to one interpreter version.
- **D-05:** Environment setup is accepted only after `needle fetch` and a real CPU inference succeed; an import-only smoke test is insufficient.

### Asset Acquisition
- **D-06:** Explain the native engine, checkpoint, tokenizer, and `.cact` artifact roles before asking the reader to download anything.
- **D-07:** Use `needle fetch` as the beginner path so platform selection is automatic; reserve `needle download <platform>` for advanced/cross-platform cases.
- **D-08:** Use the default cache location and document how to inspect and clear it; do not require a custom cache directory.
- **D-09:** Fetch online once, then verify cached/offline operation with `HF_HUB_OFFLINE=1`.
- **D-10:** Validate fetched assets by checking file presence, size/version information, and offline loading. Do not invent a checksum command or checksum table that the project does not currently provide.

### CPU Inference Verification
- **D-11:** Phase 1 verifies CPU only. CUDA and Metal setup/verification are deferred as explicit TODOs.
- **D-12:** Use a fixed short prompt and fixed `max_new_tokens`; acceptance checks successful exit and non-empty output, not exact generated text.
- **D-13:** Present the CLI first as the lowest-friction entry point, then show the equivalent `Needle` Python API.
- **D-14:** If assets are missing, the documented flow must fail clearly and direct the user to run `needle fetch`; do not rely on implicit downloading in the quickstart.

### the agent's Discretion
- Exact short prompt and `max_new_tokens` value, provided they are stable and fast on CPU.
- Exact document split between `README.md` and `doc/`, because the user did not select the document-entry gray area.
- Exact formatting of cache inspection output and expected-output callouts.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope and Requirements
- `.planning/ROADMAP.md` — Phase 1 goal, requirements, success criteria, and two planned work streams.
- `.planning/REQUIREMENTS.md` — `INST-01..03`, `INFR-01..03`, and `DOCS-01..02` acceptance scope.
- `.planning/PROJECT.md` — Chinese-first beginner audience, platform constraints, and verification expectations.

### Existing User Documentation and Packaging
- `README.md` — current installation, product positioning, and first-use examples to preserve or reorganize.
- `pyproject.toml` — Python baseline, optional extras, package metadata, and `needle` console entry point.
- `doc/apis.md` — current API and offline-mode documentation.
- `doc/environments.md` — current environment-related guidance and examples.

### Runtime and Verification Behavior
- `needle/cli.py` — authoritative CLI commands and argument behavior.
- `needle/agent/fetch.py` — native engine platform selection, cache path, and fetch behavior.
- `needle/__init__.py` — native library lookup and `Needle` inference behavior.
- `tests/test_fetch.py` — existing fetch expectations and reusable test patterns.
- `tests/test_inference.py` — existing inference behavior and verification patterns.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `pyproject.toml` extras: reuse the existing `train` and `test` dependency groups in the uv editable-install command.
- `needle.agent.fetch`: reuse automatic platform detection and the default `~/.cache/cactus-needle/<engine-version>/` cache behavior.
- CLI entry point `needle = needle.cli:main`: use the real installed console command for fetch and inference validation.
- `tests/test_fetch.py` and `tests/test_inference.py`: adapt their observable checks rather than defining incompatible examples.

### Established Patterns
- The project supports CPython 3.9+ and does not maintain a lockfile.
- Production inference loads a platform-specific native library via ctypes; JAX/Flax dependencies serve reference/training paths.
- CLI progress uses short aligned labels, so tutorials should quote actual output rather than invent a new format.
- Artifact downloads use Hugging Face configuration and can be forced offline with `HF_HUB_OFFLINE=1`.

### Integration Points
- Installation guidance connects `uv` commands to `pyproject.toml` extras.
- Asset guidance connects `needle fetch` to `needle/agent/fetch.py` and the default cache.
- Quickstart guidance connects the CLI handler in `needle/cli.py` to `Needle` in `needle/__init__.py`.
- Verification should run on CPU and remain compatible with the existing pytest suite.

</code_context>

<specifics>
## Specific Ideas

- The user explicitly wants environment initialization to use uv, not pip-only instructions.
- The download step must be explained before execution so beginners know what the native engine, checkpoint, tokenizer, and `.cact` files are.
- A successful first inference is the setup completion signal.

</specifics>

<deferred>
## Deferred Ideas

- CUDA environment setup and runtime verification — record as a TODO for a later phase/milestone.
- Apple Metal environment setup and runtime verification — record as a TODO for a later phase/milestone.

</deferred>

---

*Phase: 1-install-and-first-inference*
*Context gathered: 2026-08-31*
