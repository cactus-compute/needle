---
phase: 01-install-and-first-inference
status: passed
verified: 2026-08-31
verifier: inline goal-backward verification
---

# Phase 1 Verification

## Goal

A beginner can install Needle on CPU, obtain the required runtime assets, and run
a first inference request with a documented CLI or `Needle` API example.

## Must-Haves

| Must-have | Evidence | Status |
| --- | --- | --- |
| Python 3.9+ uv environment and editable train/test install | `doc/installation.md` documents `uv venv`, activation, `uv pip install -e ".[train,test]"`; CPython 3.11.16 environment installed 50 packages | PASS |
| Asset roles, cache, explicit fetch, and offline mode are clear | Installation guide explains native engine, checkpoint, tokenizer, `.cact`, default cache, `needle fetch`, `HF_HUB_OFFLINE=1`, and `NEEDLE_LIB_PATH` | PASS |
| CPU first inference works with bounded output | `needle fetch` succeeded; native `Needle.complete(..., max_new_tokens=16)` returned a response envelope; CLI `needle run ... --max-len 4` exited 0 with generated output | PASS |
| CLI and equivalent Python API are discoverable | `doc/inference.md` contains CLI `needle run`, native `Needle` API, and a minimal `@needle.tool` example | PASS |
| Missing assets and exact-text drift are addressed | Guide directs readers to run `needle fetch` or restore checkpoint assets and explicitly avoids exact generated-text assertions | PASS |
| Documentation links and commands remain testable | `tests/test_docs_examples.py` checks install/fetch/offline commands, CLI/API sections, typed tool example, README links, and no-exact-text rule; 4 tests passed | PASS |

## Requirement Traceability

- INST-01, INST-02, INST-03: covered by `doc/installation.md` and verified with `uv pip check`.
- INFR-01, INFR-02, INFR-03: covered by `doc/inference.md`, verified with CLI/API runs and smoke tests.
- DOCS-01, DOCS-02: Chinese-first guides preserve exact commands, prerequisites, expected output, cache behavior, and verification steps; CPU acceptance is explicit and CUDA/Metal are marked TODO.

## Automated Checks

```text
uv pip check                         -> All installed packages are compatible
.venv/bin/pytest -q tests/test_docs_examples.py -> 4 passed
needle fetch                         -> libneedle.so cached under ~/.cache/cactus-needle/2.0.3/
HF_HUB_OFFLINE=1 Needle.complete     -> response envelope, exit 0
needle run --max-len 4               -> bounded output, exit 0
```

## Human Verification

None required. The documented CPU commands were executed successfully; generated text is intentionally treated as variable output.

## Residual Risk

CUDA and Apple Metal installation/verification remain deferred to a later phase, as
decided in the phase context. No claim is made that those branches were tested here.
