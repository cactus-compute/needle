# Phase 1: Install and First Inference - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md; this log preserves alternatives considered.

**Date:** 2026-08-31
**Phase:** 1-install-and-first-inference
**Areas discussed:** Installation strategy, Asset acquisition, Backend verification

---

## Installation Strategy

| Decision | Options considered | Selected |
|----------|--------------------|----------|
| uv install form | editable repository; published package; both | editable repository |
| dependency extras | test only; train+test; minimal plus optional full | `.[train,test]` |
| Python baseline | fixed 3.10/3.11; declared 3.9+; current machine only | declared 3.9+ |
| success criterion | import smoke; real fetch+inference; layered checks | real fetch+CPU inference |

**User's choice:** Initialize with uv, editable install, full train/test extras, retain Python 3.9+, and require real inference success.
**Notes:** The user wants the environment ready for later training work rather than installing training dependencies in a later phase.

---

## Asset Acquisition

| Decision | Options considered | Selected |
|----------|--------------------|----------|
| beginner command | `needle fetch`; `needle download <platform>`; both equally | `needle fetch` |
| cache behavior | default cache; mandatory custom cache; default plus example | default cache |
| offline path | online then offline check; online only; both mandatory | online then `HF_HUB_OFFLINE=1` check |
| integrity check | file/size/version/offline load; SHA256 table; inference only | file/size/version/offline load |

**User's choice:** Explain assets first, then use automatic-platform fetch and the default cache, followed by an offline verification.
**Notes:** The user paused the option selection to ask what the assets and commands mean; the final documentation must preserve that explanatory order.

---

## Backend Verification

| Decision | Options considered | Selected |
|----------|--------------------|----------|
| supported backends in Phase 1 | CPU only; all backends; CPU plus conditional smoke | CPU only |
| generated-output check | exit/non-empty; exact text; machine+human review | exit/non-empty |
| entry point order | CLI first; Python first; parallel | CLI first |
| missing assets | explicit failure/fetch instruction; implicit download; both | explicit failure/fetch instruction |

**User's choice:** Verify only CPU with a short bounded prompt, lead with CLI, and require explicit asset setup.
**Notes:** CUDA and Metal should be TODOs for later work rather than Phase 1 acceptance criteria.

---

## the agent's Discretion

- Exact quickstart prompt, token limit, and README/doc split.
- Exact presentation format for expected output and cache inspection.

## Deferred Ideas

- CUDA setup and verification.
- Apple Metal setup and verification.
