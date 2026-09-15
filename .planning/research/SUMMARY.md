# Project Research Summary

**Project:** Needle
**Domain:** Beginner onboarding for an edge language-model inference and fine-tuning toolkit
**Researched:** 2026-08-31
**Confidence:** MEDIUM

## Executive Summary

Needle already has a coherent two-path architecture: a ctypes-backed native
runtime for compact deployment and a JAX/Flax reference/training path for
decoding, LoRA, quantization, and export. The documentation should mirror that
boundary and lead beginners through named artifacts rather than hiding them.

The recommended milestone is a Chinese-first, CPU-friendly journey with
accelerator branches: install, fetch assets, run inference, prepare JSONL and
train LoRA, then merge/export/load a `.cact` artifact. Every stage needs an
expected output and a verification check. Security and reliability warnings are
part of the guide, while deeper hardening remains later scope.

## Key Findings

### Recommended Stack

Keep Python 3.9+, JAX, Flax, Optax, SentencePiece, NumPy, and pytest aligned
with the repository declarations. Use `.[train]`, `.[gpu]`, and `.[metal]` as
explicit variants. Do not promise backend compatibility beyond the tested
matrix; `.cact` archives must remain coupled to engine version and geometry.

### Expected Features

**Must have:** installation/backend guide, asset/cache explanation, inference
quickstart, JSONL and LoRA tutorial, export/deployment tutorial, troubleshooting.

**Should have:** architecture diagrams, CPU/GPU parity guidance, and safety
notes next to risky commands.

**Defer:** hosted multi-user service, GUI training management, and automatic
artifact registry/signing.

### Architecture Approach

Organize docs around user goals while preserving source boundaries: public API
and CLI, native engine adapter, JAX/Flax model, fine-tuning, and export. The
build order is install/assets -> inference -> training -> export/deployment.

### Critical Pitfalls

1. Do not present the local playground as production hosting.
2. Warn that untrusted pickle/checkpoint or native downloads are unsafe.
3. Verify engine version, tokenizer vocabulary, tensor order, and geometry for
   every `.cact` export.
4. State that interrupted training cannot currently be treated as resumable.

## Implications for Roadmap

### Phase 1: Installation and First Inference

**Rationale:** Establishes the minimum success path and resolves asset/backend
confusion before training instructions.
**Delivers:** Chinese install guide, CPU/GPU matrix, asset/cache notes, API/CLI
quickstart, and verification checks.
**Addresses:** DOC-01, DOC-02.

### Phase 2: Model and Runtime Concepts

**Rationale:** Beginners need a mental model before changing weights.
**Delivers:** architecture/data-flow guide, artifact glossary, and troubleshooting
entry points.
**Addresses:** DOC-05, part of DOC-06.

### Phase 3: LoRA Fine-Tuning

**Rationale:** Training depends on the installation and artifact knowledge from
Phases 1-2.
**Delivers:** JSONL schema, rendering/masking explanation, reproducible LoRA
command, outputs, resource knobs, and safety notes.
**Addresses:** DOC-03, DOC-06.

### Phase 4: Export and Deployment Verification

**Rationale:** Export is a separate compatibility boundary and should be taught
after a working adapter exists.
**Delivers:** merge/quantize/build/load tutorial, `.cact` compatibility checks,
CPU/GPU verification, and release checklist.
**Addresses:** DOC-04, DOC-07.

### Phase Ordering Rationale

- Asset and backend choices precede every runtime path.
- Architecture concepts explain why checkpoint, tokenizer, adapter, and `.cact`
  artifacts differ.
- Export verification is last because it consumes the output of fine-tuning.

### Research Flags

- **Phase 1:** validate current JAX backend installation commands on supported CI
  and hardware.
- **Phase 3:** validate current CLI flags and checkpoint output names with a
  small CPU run.
- **Phase 4:** validate `.cact` round-trip behavior against the active engine.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Strong repository evidence; external version checks pending |
| Features | MEDIUM | Derived from user scope and existing docs/code |
| Architecture | HIGH | Mapped directly from source and codebase map |
| Pitfalls | HIGH | Confirmed in codebase concerns and boundaries |

**Overall confidence:** MEDIUM

### Gaps to Address

- Exact tested CUDA/Metal version matrix needs a live environment check.
- Tutorial commands need execution on clean CPU and accelerator environments.
- Current artifact provenance/checksum policy is not implemented and must remain
  an explicit limitation.

## Sources

### Primary (HIGH confidence)

- `.planning/codebase/*.md` - generated source-grounded map
- `README.md`, `doc/`, `pyproject.toml`, `needle/`, `tests/` - repository evidence

### Secondary (MEDIUM confidence)

- JAX, Flax, Hugging Face Hub, and Python pickle official documentation links in
  `STACK.md` and `PITFALLS.md`.

---
*Research completed: 2026-08-31*
*Ready for roadmap: yes*
