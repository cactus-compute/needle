# Roadmap: Needle Documentation and Onboarding

## Overview

Deliver a Chinese-first beginner journey through the existing Needle lifecycle:
install and fetch assets, run inference, understand the model/runtime boundary,
fine-tune with LoRA, and export/deploy a verified `.cact` artifact. Each phase
ships a usable documentation slice with executable examples and explicit limits.

## Phases

- [x] **Phase 1: Install and First Inference** - Make a clean environment and first response repeatable. (completed 2026-08-31)
- [ ] **Phase 2: Model and Runtime Concepts** - Explain architecture, artifacts, troubleshooting, and safety boundaries.
- [ ] **Phase 3: LoRA Fine-Tuning** - Document data preparation and a reproducible local adaptation run.
- [ ] **Phase 4: Export and Deployment Verification** - Teach merge, quantize, `.cact` loading, and end-to-end checks.

## Phase Details

### Phase 1: Install and First Inference

**Goal**: A beginner can install Needle on CPU or a supported accelerator, obtain required assets, and run a first inference request.
**Mode**: mvp
**Depends on**: Nothing (first phase)
**Requirements**: [INST-01, INST-02, INST-03, INFR-01, INFR-02, INFR-03, DOCS-01, DOCS-02]
**Success Criteria** (what must be TRUE):

  1. A clean Python 3.9+ environment can be installed using documented commands for CPU and supported accelerator branches.
  2. A reader can fetch or locate engine/checkpoint/tokenizer assets and knows the cache and offline controls.
  3. A reader can copy a CLI or `Needle` API example, observe the expected response, and run a verification check.
  4. Common install and first-run failures link to actionable troubleshooting entries.

**Plans**: 1/2 plans executed

Plans:

- [x] 01-01-PLAN.md
- [x] 01-02-PLAN.md
- [x] 01-01: Write installation, asset/cache, and backend matrix guide.
- [x] 01-02: Write and verify CLI/API inference quickstart with expected output.

### Phase 2: Model and Runtime Concepts

**Goal**: A beginner can explain which runtime path and artifact applies to inference, training, and deployment, and can avoid known unsafe usage.
**Mode**: mvp
**Depends on**: Phase 1
**Requirements**: [MODL-01, MODL-02, MODL-03, SAFE-01, SAFE-02, SAFE-03]
**Success Criteria** (what must be TRUE):

  1. Architecture documentation traces a request through `Needle`/ctypes and separately through JAX/Flax reference code.
  2. The roles and compatibility constraints of `.pkl`, tokenizer, LoRA adapter, and `.cact` artifacts are clear with source links.
  3. A reader can identify process-global state, pickle/download risks, playground exposure risks, and documented performance/dependency limitations.
  4. Troubleshooting is organized by symptom and includes a safe next action.

**Plans**: 2 plans

Plans:

- [ ] 02-01: Write model architecture, data-flow, and artifact glossary.
- [ ] 02-02: Write troubleshooting and safety reference from codebase concerns.

### Phase 3: LoRA Fine-Tuning

**Goal**: A beginner can prepare valid supervision data and complete a small, reproducible LoRA fine-tuning run.
**Mode**: mvp
**Depends on**: Phase 2
**Requirements**: [LORA-01, LORA-02, LORA-03]
**Success Criteria** (what must be TRUE):

  1. A documented JSONL example renders to the expected chat/tool markers and explains which tokens contribute to loss.
  2. A small local fine-tuning command runs with stated CPU/GPU prerequisites and resource knobs.
  3. Adapter/checkpoint outputs have documented paths and a validation or inspection step.
  4. The guide clearly states current interrupted-run resume limitations and safe checkpoint handling.

**Plans**: 2 plans

Plans:

- [ ] 03-01: Document JSONL schema, rendering/masking, and dataset validation.
- [ ] 03-02: Document and run the LoRA CLI workflow with output verification.

### Phase 4: Export and Deployment Verification

**Goal**: A beginner can turn a trained adapter into a compatible `.cact` artifact and verify it in a fresh runtime process.
**Mode**: mvp
**Depends on**: Phase 3
**Requirements**: [DEPL-01, DEPL-02, DEPL-03, DOCS-03]
**Success Criteria** (what must be TRUE):

  1. Merge, quantize, and build commands produce a named `.cact` archive from the documented adapter output.
  2. Engine version, tensor order, geometry, and tokenizer vocabulary checks are explicit and testable.
  3. A fresh process loads the archive and compares output against a reference/base path using a checklist.
  4. Examples have a documented manual or automated verification route suitable for CI follow-up.

**Plans**: 2 plans

Plans:

- [ ] 04-01: Write merge/quantize/export/deployment tutorial and compatibility checklist.
- [ ] 04-02: Add or document example verification commands and review all links/commands.

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Install and First Inference | 2/2 | Complete    | 2026-08-31 |
| 2. Model and Runtime Concepts | 0/2 | Not started | - |
| 3. LoRA Fine-Tuning | 0/2 | Not started | - |
| 4. Export and Deployment Verification | 0/2 | Not started | - |
