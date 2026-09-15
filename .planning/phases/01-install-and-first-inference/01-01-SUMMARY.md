---
phase: 01-install-and-first-inference
plan: 01
subsystem: documentation
tags: [uv, cpu, installation, huggingface, offline]

requires: []
provides:
  - Chinese-first uv installation guide for Python 3.9+ CPU environments
  - Explicit native engine asset fetch, cache inspection, and offline verification flow
  - README entry point linking source users to the canonical installation guide
affects: [01-02-inference, fine-tuning, onboarding]

tech-stack:
  added: []
  patterns: [uv editable installs, explicit fetch-before-inference, HF_HUB_OFFLINE verification]

key-files:
  created: [doc/installation.md]
  modified: [README.md]

key-decisions:
  - "Use uv venv and editable uv pip install -e \".[train,test]\" as the source checkout baseline."
  - "Fetch only the platform native engine in Phase 1; explain checkpoint, tokenizer, and .cact roles for later workflows."
  - "Verify online fetch followed by HF_HUB_OFFLINE=1 loading on CPU, without asserting exact generated text."

requirements-completed: []

coverage:
  - id: D1
    description: "Chinese-first CPU installation guide with uv environment creation and train/test extras"
    verification:
      - kind: integration
        ref: "uv venv + uv pip install -e .\"[train,test]\" in /tmp/needle-uv-ddSeBK"
        status: pass
      - kind: unit
        ref: "uv pip check --python /tmp/needle-uv-ddSeBK/.venv/bin/python"
        status: pass
    human_judgment: false
  - id: D2
    description: "Explicit engine fetch, cache inspection, and HF_HUB_OFFLINE inference check"
    verification:
      - kind: e2e
        ref: "HOME=/tmp/needle-home-03UaCo /tmp/needle-uv-ddSeBK/.venv/bin/needle fetch"
        status: pass
      - kind: e2e
        ref: "HF_HUB_OFFLINE=1 Needle.complete('hello', max_new_tokens=16)"
        status: pass
    human_judgment: false
  - id: D3
    description: "README Chinese installation entry links to doc/installation.md and preserves API quickstart"
    verification:
      - kind: other
        ref: "git diff --check and README relative link inspection"
        status: pass
    human_judgment: false

duration: 14min
completed: 2026-08-31
status: complete
---

# Phase 1 Plan 1: Install, Assets, and CPU Baseline Summary

**uv-based CPU onboarding with explicit Needle engine fetching, cache/offline checks, and a Chinese README entry path**

## Performance

- **Duration:** 14 min
- **Started:** 2026-08-31T13:48:00Z
- **Completed:** 2026-08-31T14:04:00Z
- **Tasks:** 3
- **Files modified:** 2 (plus this summary and the Windows ledger)

## Accomplishments

- Added `doc/installation.md` covering Python 3.9+, `uv venv`, editable `.[train,test]` installation, CPU scope, telemetry opt-out, and expected checks.
- Documented the distinction between native engine, JAX checkpoint, SentencePiece tokenizer, and `.cact` archive, including default cache inspection and cleanup.
- Added an online-first `needle fetch` then `HF_HUB_OFFLINE=1` verification flow and a Chinese-first README entry that points to the guide.

## Task Commits

1. **Write the uv CPU installation guide** - `fd0202a` (`docs`)
2. **Document explicit asset fetch, cache, and offline verification** - `b174059` (`docs`)
3. **Add the installation entry link to README** - `786d75d` (`docs`)
4. **Auto-fix temporary README断链** - `4819483` (`fix`)

## Files Created/Modified

- `doc/installation.md` - Chinese-first setup, artifact lifecycle, cache, fetch, and offline CPU verification instructions.
- `README.md` - Source-checkout installation entry with the shortest uv/fetch path and published-package distinction.
- `.planning/WINDOWS.md` - Broken-windows ledger entry for intentionally deferred CUDA/Metal verification.

## Decisions Made

- Keep the repository's declared Python 3.9+ range and install the complete training/test extras for future phases.
- Treat `needle fetch` as an explicit prerequisite; do not rely on implicit downloads during the quickstart.
- Keep CUDA and Metal out of Phase 1 acceptance while naming them as later work.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Avoided a README link to a not-yet-created inference guide**

- **Found during:** Task 3 (Add the installation entry link to README)
- **Issue:** The initial entry linked `doc/inference.md`, which belongs to dependent plan 01-02 and did not yet exist.
- **Fix:** Removed that link while retaining the instruction to continue to the later inference guide; plan 01-02 can add its link when the file is created.
- **Files modified:** `README.md`
- **Verification:** `git diff --check`; installation link resolves from repository root.
- **Committed in:** `4819483`

**Total deviations:** 1 auto-fixed (Rule 1)
**Impact on plan:** No scope creep; the plan's required installation link remains valid and no broken link is left behind.

## Issues Encountered

- The first isolated fetch command selected an empty virtual-environment path due to an overly shallow `find`; rerunning with the known environment path succeeded. No repository change was needed.
- Hugging Face emitted an unauthenticated-rate-limit warning during fetch; public assets downloaded successfully, so no authentication gate was required.

## Known Stubs

- `doc/installation.md:5` - CUDA and Metal installation/runtime verification are intentionally deferred to a later phase, matching the Phase 1 CPU-only boundary.

## User Setup Required

None - no external service configuration is required for this CPU baseline.

## Next Phase Readiness

The source checkout can be installed reproducibly with uv, the native CPU engine was fetched into the documented default cache, and offline loading was verified. Plan 01-02 can add the first fixed CLI inference example and its documentation smoke tests; it should restore a README link to `doc/inference.md` after creating that file.

---
*Phase: 01-install-and-first-inference*
*Completed: 2026-08-31*

## Self-Check: PASSED

- `01-01-SUMMARY.md` exists at the expected phase path.
- Task commits `fd0202a`, `b174059`, `786d75d`, and `4819483` are present in git history.
