---
phase: 01-install-and-first-inference
status: secured
threats_open: 0
asvs_level: 1
block_on: high
audited: 2026-08-31
---

# Phase 1 Security Verification

This phase changes onboarding documentation and adds documentation-only smoke
tests. It does not change the native engine, checkpoint loader, CLI behavior, or
network protocol. The review below is a retroactive STRIDE-style check because
the phase plans did not contain a formal threat model.

## Threat Register

| ID | Category | Component | Severity | Status | Evidence / disposition |
| --- | --- | --- | --- | --- | --- |
| T-01 | Tampering / supply chain | `needle fetch` and Hugging Face assets | high | CLOSED | Installation guide makes fetch explicit, documents the cache path and offline check, and does not claim checksum verification. Users must obtain assets from the configured repository; checksum/signing is accepted as a future productization risk. |
| T-02 | Code execution | JAX `.pkl` checkpoint loading | high | CLOSED (accepted risk) | Checkpoint files are loaded with Python pickle by the existing reference path. Phase 1 does not alter that behavior; the risk is accepted for this documentation milestone and is deferred to the Phase 2 safety reference, where trusted-source handling will be documented. |
| T-03 | Availability | Missing engine/checkpoint/tokenizer assets | medium | CLOSED | Guides distinguish native engine from training assets, require explicit `needle fetch`, and provide an actionable offline failure path instead of silently assuming network access. |
| T-04 | Information disclosure | Telemetry and prompt data | medium | CLOSED | Installation guide documents `NEEDLE_TELEMETRY=0` and `DO_NOT_TRACK=1`; the package telemetry contract records function metadata, not prompts or outputs. |
| T-05 | Spoofing / local exposure | CLI and local runtime | medium | CLOSED (accepted risk) | Phase 1 documents local CPU execution and does not expose a new server endpoint. Playground authentication and production serving hardening remain outside this phase and are tracked for later safety documentation. |

## Accepted Risks

- Native downloads are trusted based on the configured Hugging Face repository; artifact signing/checksums are not implemented in this milestone.
- `.pkl` files are executable serialization and must come from a trusted source. Do not load arbitrary user-supplied checkpoints.
- CUDA/Metal setup and broader production serving controls are intentionally deferred.

## Audit Trail

| Date | Reviewer | Result |
| --- | --- | --- |
| 2026-08-31 | inline retroactive review | `threats_open: 0`; documentation and smoke tests reviewed against Phase 1 scope |
