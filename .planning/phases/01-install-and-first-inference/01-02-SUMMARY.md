---
phase: 01-install-and-first-inference
plan: 02
subsystem: documentation
tags: [inference, cli, python-api, cpu, pytest]

requires: [01-01]
provides:
  - CLI-first CPU reference inference quickstart with bounded generation
  - Native Needle Python API example with offline verification
  - Documentation drift smoke tests for onboarding commands and links
affects: [02-model-and-runtime-concepts]

tech-stack:
  added: []
  patterns: [bounded prompts, response-envelope assertions, docs smoke tests]

key-files:
  created: [doc/inference.md, tests/test_docs_examples.py]
  modified: [README.md]

key-decisions:
  - "Document needle run as the JAX/Flax checkpoint CLI path and Needle.complete as the native engine path; do not imply needle fetch downloads checkpoints."
  - "Verify non-empty response envelopes and successful exit codes rather than exact generated text."

requirements-completed: []

coverage:
  - id: I1
    description: "CLI reference inference with fixed prompt and bounded max-len"
    verification:
      - kind: e2e
        ref: ".venv/bin/needle run --checkpoint checkpoints/needle2.pkl --query 'hello' --max-len 4 --temperature 0"
        status: pass
  - id: I2
    description: "Native API CPU response after explicit fetch and offline mode"
    verification:
      - kind: e2e
        ref: "HF_HUB_OFFLINE=1 Needle().complete(..., max_new_tokens=16)"
        status: pass
  - id: I3
    description: "Documentation drift smoke checks"
    verification:
      - kind: unit
        ref: ".venv/bin/pytest -q tests/test_docs_examples.py"
        status: pass
---

# Phase 1 Plan 2: CLI and Python Inference Quickstart Summary

## Accomplishments

- Added Chinese-first `doc/inference.md` with explicit fetch prerequisite, CLI reference-model invocation, native API invocation, missing-asset guidance, and offline verification.
- Linked the inference guide from the README installation entry point.
- Added four lightweight tests that check required commands, links, CLI/API sections, and the no-exact-text acceptance rule without downloading assets.
- Initialized `.venv` with `uv`, installed `.[train,test]`, and verified `uv pip check`.

## Verification

- `uv pip check` passed with CPython 3.11.16 and package `2.0.8`.
- `needle fetch` downloaded `/home/dr/.cache/cactus-needle/2.0.3/libneedle.so`.
- `HF_HUB_OFFLINE=1` native API check returned a valid non-empty response envelope (`type: call`, empty tool calls) with exit code 0.
- CLI reference run downloaded `checkpoints/needle2.pkl`, loaded the tokenizer, generated bounded output, and exited 0.
- `.venv/bin/pytest -q tests/test_docs_examples.py`: 4 passed.

## Deviations from Plan

None - plan executed exactly as written.

## Known Limitations

- CLI `needle run` output depends on the checkpoint/tokenizer and may contain non-ASCII or special token fragments; the guide intentionally checks only process success and non-empty output.
- CUDA and Metal remain deferred to later phases as agreed.

## Self-Check: PASSED

- Both created files exist and README links resolve from the repository root.
- Plan verification commands passed.
