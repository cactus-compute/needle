# Requirements: Needle Documentation and Onboarding

**Defined:** 2026-08-31
**Core Value:** A beginner can install Needle and reliably go from a first inference request to a fine-tuned, exported model without guessing which assets, commands, or runtime constraints apply.

## v1 Requirements

### Installation and Environment

- [x] **INST-01**: A beginner can create an isolated Python 3.9+ environment and install the correct base, training, test, GPU, or Metal extras for their platform.
- [x] **INST-02**: The guide explains native engine, checkpoint, tokenizer, Hugging Face cache locations, offline mode, and required environment variables.
- [x] **INST-03**: The guide states a supported CPU/GPU/Metal matrix and gives a CPU fallback when acceleration is unavailable.

### Inference Quickstart

- [x] **INFR-01**: A beginner can run a copy-paste pre-trained inference example and identify the expected successful output.
- [x] **INFR-02**: The guide demonstrates equivalent CLI and `Needle` Python API flows, including a minimal typed tool call or extraction example.
- [x] **INFR-03**: Each quickstart includes a verification command and links common errors to troubleshooting guidance.

### Model and Runtime Concepts

- [ ] **MODL-01**: The documentation explains the native ctypes runtime path versus the JAX/Flax reference and training path.
- [ ] **MODL-02**: The documentation describes the Simple Attention Network components, tokenizer contract, checkpoint formats, and `.cact` artifact roles with source links.
- [ ] **MODL-03**: The documentation calls out process-global native state, base/tuned agent ordering, and when to use a fresh process.

### LoRA Fine-Tuning

- [ ] **LORA-01**: A beginner can create a valid JSONL fine-tuning dataset with documented fields, chat/tool markers, and target masking behavior.
- [ ] **LORA-02**: A beginner can run a small local LoRA fine-tuning job, understand its prerequisites and resource knobs, and locate the adapter/checkpoint outputs.
- [ ] **LORA-03**: The guide explains how to inspect or validate a trained adapter and documents the current limitation around interrupted-run resume.

### Export and Deployment

- [ ] **DEPL-01**: A beginner can merge a LoRA adapter, quantize it, and produce a `.cact` archive using the supported CLI workflow.
- [ ] **DEPL-02**: The guide explains engine version, tensor order, model geometry, and tokenizer vocabulary compatibility requirements.
- [ ] **DEPL-03**: A beginner can load the exported archive in a fresh process and compare its output with the reference or base path using a verification checklist.

### Troubleshooting and Safety

- [ ] **SAFE-01**: Troubleshooting maps installation, asset, backend, checkpoint, tokenizer, and export failures to observable symptoms and fixes.
- [ ] **SAFE-02**: Documentation warns about untrusted pickle checkpoints, native downloads, API keys, and exposing the unauthenticated local playground.
- [ ] **SAFE-03**: Documentation records known concurrency, dependency drift, performance, and test-coverage limitations without implying they are solved.

### Documentation Quality

- [x] **DOCS-01**: The primary onboarding path is Chinese-first, preserves exact commands/API identifiers, and links every conceptual claim to a source path or official reference.
- [x] **DOCS-02**: Every tutorial states prerequisites, expected output, cleanup/cache behavior, and a practical verification step for CPU and supported accelerator branches.
- [ ] **DOCS-03**: Examples are organized by user goal and can be checked in automated tests or a documented manual verification pass.

## v2 Requirements

### Productization

- **PROD-01**: Provide authenticated hosted or multi-user serving guidance.
- **PROD-02**: Provide a GUI for training job management and artifact browsing.
- **PROD-03**: Add automatic artifact signing, checksums, or registry provenance enforcement.

### Advanced Learning Material

- **LEARN-01**: Provide notebook-based walkthroughs and interactive visualizations.
- **LEARN-02**: Publish benchmark tables across model sizes, hardware backends, and quantization levels.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Native engine rewrite | This milestone documents the current runtime contract rather than changing it. |
| New `.cact` format or tensor ordering | Format changes require coordinated engine/export work and are not documentation-only. |
| Production security hardening | Risks are documented and tracked, but code remediation is a later phase. |
| Universal OS/GPU support claims | Backend availability changes; docs will state tested/supportable combinations only. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INST-01 | Phase 1 | Complete |
| INST-02 | Phase 1 | Complete |
| INST-03 | Phase 1 | Complete |
| INFR-01 | Phase 1 | Complete |
| INFR-02 | Phase 1 | Complete |
| INFR-03 | Phase 1 | Complete |
| MODL-01 | Phase 2 | Pending |
| MODL-02 | Phase 2 | Pending |
| MODL-03 | Phase 2 | Pending |
| LORA-01 | Phase 3 | Pending |
| LORA-02 | Phase 3 | Pending |
| LORA-03 | Phase 3 | Pending |
| DEPL-01 | Phase 4 | Pending |
| DEPL-02 | Phase 4 | Pending |
| DEPL-03 | Phase 4 | Pending |
| SAFE-01 | Phase 2 | Pending |
| SAFE-02 | Phase 2 | Pending |
| SAFE-03 | Phase 2 | Pending |
| DOCS-01 | Phase 1 | Complete |
| DOCS-02 | Phase 1 | Complete |
| DOCS-03 | Phase 4 | Pending |

**Coverage:**

- v1 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0

---
*Requirements defined: 2026-08-31*
*Last updated: 2026-08-31 after initial definition*
