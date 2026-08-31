# Needle

## What This Is

Needle is a lightweight Python package for running a compact language model and
agent workflows on constrained devices. It provides a native inference facade,
JAX/Flax reference and training code, LoRA fine-tuning, quantization/export to
`.cact`, a CLI, and a local playground.

This milestone focuses on making the existing capabilities understandable and
usable by Python and machine-learning beginners through Chinese-first
documentation and reproducible end-to-end examples.

## Core Value

A beginner can install Needle and reliably go from a first inference request to
a fine-tuned, exported model without guessing which assets, commands, or
runtime constraints apply.

## Requirements

### Validated

- [x] Native `Needle` API supports completion, tool execution, schema extraction,
  reset, and weight loading — existing in `needle/__init__.py`.
- [x] CLI exposes fetch/download, run, fine-tune, data generation, build,
  playground, and related workflows — existing in `needle/cli.py`.
- [x] JAX/Flax reference model defines the Simple Attention Network, decoding,
  LoRA training, quantization, and `.cact` export paths — existing in
  `needle/model/`.
- [x] Test suite covers model, inference, fine-tuning, export, packaging,
  environments, and CLI behavior — existing in `tests/`.

### Active

- [ ] DOC-01: Provide a Chinese-first installation and environment guide that
  explains CPU and GPU options, optional extras, model assets, caches, and
  offline operation.
- [ ] DOC-02: Provide a copy-paste quickstart that runs a pre-trained inference
  example and explains the public `Needle` API and CLI equivalents.
- [ ] DOC-03: Provide an end-to-end fine-tuning tutorial covering JSONL data
  format, prompt rendering, LoRA training, checkpoint outputs, and common
  resource/configuration choices.
- [ ] DOC-04: Provide an export/deployment tutorial covering merge, quantization,
  `.cact` generation, engine compatibility, and running the exported artifact.
- [ ] DOC-05: Document the model structure and data flow, including the native
  runtime path versus the JAX/Flax training path, attention/MLP components,
  tokenizer contract, and process-global state constraints.
- [ ] DOC-06: Add troubleshooting and safety notes for download verification,
  pickle checkpoints, playground exposure, concurrency, dependency drift, and
  known training/export limitations; track code hardening as later work.
- [ ] DOC-07: Keep examples and commands testable on CPU and GPU where supported,
  with a clear verification checklist for each tutorial.

### Out of Scope

- Rewriting the native inference engine or changing the `.cact` binary contract
  — documentation should reflect the current implementation first.
- Adding authentication, hosted multi-user serving, or a production web service
  — the current playground remains a local development tool.
- Solving every security or performance concern in this documentation milestone
  — risks are recorded and prioritized for later implementation phases.

## Context

- The package targets CPython 3.9+ and uses setuptools with optional `train`,
  `gpu`, `metal`, and `test` extras.
- Production inference loads a platform-specific native library through ctypes;
  JAX/Flax/Optax/SentencePiece are used for reference inference, fine-tuning,
  and export.
- Model and tokenizer artifacts are commonly fetched from Hugging Face and may
  be cached locally; `HF_HUB_OFFLINE=1` supports air-gapped use.
- The native engine keeps process-global active state, so base and tuned agents
  have ordering and process-isolation constraints.
- The codebase map in `.planning/codebase/` is the evidence source for current
  structure, integrations, conventions, testing, and concerns.

## Constraints

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

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Chinese-first documentation | The intended onboarding audience asked for Chinese guidance | - Pending |
| Cover inference, LoRA, and deployment in one path | Users need a complete journey from install to usable tuned artifact | - Pending |
| Support CPU and GPU guidance | Hardware availability varies and beginner setup should not assume CUDA | - Pending |
| Record hardening risks before fixing them | Documentation can prevent misuse without expanding the first milestone into a security rewrite | - Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check - still the right priority?
3. Audit Out of Scope - reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-08-31 after initialization*
