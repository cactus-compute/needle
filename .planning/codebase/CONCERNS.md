# Codebase Concerns

**Analysis Date:** 2026-08-31

## Tech Debt

**Fine-tune checkpointing and resume:**
- Issue: `finetune_local` accepts `checkpoint_dir` and creates the directory only when training finishes; it never writes step checkpoints or supports resuming an interrupted run.
- Files: `needle/model/finetune.py:294-400`, `needle/cli.py:111-132`
- Impact: Long JAX/accelerator runs lose all progress on interruption, and the advertised checkpoint directory can appear healthy while remaining empty until completion.
- Fix approach: Persist adapter parameters, optimizer state, config, and step at a configurable interval using an atomic rename; add an explicit `--resume` path and a recovery test.

**Silent configuration drops:**
- Issue: `TransformerConfig.__init__` ignores unknown checkpoint keys instead of rejecting or recording them.
- Files: `needle/model/architecture.py:58-90`
- Impact: A misspelled or newer config field silently falls back to a default, potentially constructing a model with incompatible dimensions or attention/engram behavior before export.
- Fix approach: Validate keys and dimension invariants (head divisibility, engram layers, dtype) and fail with the offending field; keep a migration path for intentionally deprecated fields.

**Unpinned training stack and release metadata drift:**
- Issue: JAX, jaxlib, optax, numpy, and sentencepiece are broadly specified without a lockfile; package metadata reports `2.0.8` while the checked-out repository is tagged `v2.0.11`, and the native engine is independently pinned to `2.0.3`.
- Files: `pyproject.toml:3-31`, `requirements-train.txt:1-8`, `needle/__init__.py:12`, `needle/agent/fetch.py:8`
- Impact: Reproducing a fine-tune/export environment is difficult, and package/engine/CACT compatibility can be misdiagnosed from inconsistent version identifiers.
- Fix approach: Generate a supported lock/test matrix, make the package version single-sourced, and publish an explicit package-to-engine/format compatibility table.

## Known Bugs

**Empty batch generation is not handled:**
- Symptoms: Calling `batch_generate` with an empty `prompts` sequence raises `ValueError` from `max(plens)` instead of returning an empty result.
- Files: `needle/model/run.py:119-126`
- Trigger: Any caller builds a batch dynamically and passes no requests.
- Workaround: Guard at the API boundary and return `[]` (or reject with a clear `ValueError`) before computing `max()`.

**Engine calls race across agents/threads:**
- Symptoms: Concurrent calls can overwrite process-global active agent/weights and the per-agent response buffer, producing responses for the wrong tool set or corrupted JSON.
- Files: `needle/__init__.py:34-99`, `needle/__init__.py:119-168`
- Trigger: Two threads call `Needle.complete`, `run`, or `reset`, or bind agents with different tool/weight sets in one process; `_bind` has no lock.
- Workaround: Serialize all calls in the application (the playground does this only around its own `Engine.complete`); isolate incompatible workloads in separate processes.

## Security Considerations

**Unsafe pickle deserialization:**
- Risk: Checkpoints and LoRA adapters are loaded with `pickle.load`, which can execute arbitrary code when an attacker controls a file or a downloaded artifact.
- Files: `needle/model/run.py:31-35`, `needle/model/run.py:84-85`, `needle/model/finetune.py:412-417`
- Current mitigation: A format-version field is checked only after deserialization; no signature or trusted-source check is performed.
- Recommendations: Replace the interchange format with a safe tensor container (for example, NPZ/safetensors plus JSON metadata), or require signatures and an explicit trusted-file policy before loading pickle.

**Unverified native artifact and tokenizer downloads:**
- Risk: The package downloads a wheel/engine and tokenizer from Hugging Face, extracts a shared library, and writes it to the cache/package without a project-level digest or signature verification.
- Files: `needle/agent/fetch.py:57-100`, `needle/model/tokenizer.py:82-112`, `needle/__init__.py:16-31`
- Current mitigation: Hugging Face client cache/ETag handling is delegated to the SDK; the archive member path is fixed.
- Recommendations: Pin and verify SHA-256/signatures for every native and tokenizer artifact, reject unexpected archive metadata, and expose an offline verification failure.

**Unauthenticated playground control plane:**
- Risk: When bound beyond localhost, any caller can upload arbitrary model bytes, invoke OpenRouter-backed fine-tuning with their own or a supplied API key, and consume unbounded memory/disk/network resources.
- Files: `needle/playground/server.py:99-170`, `needle/playground/server.py:176-186`
- Current mitigation: CLI defaults to `127.0.0.1`; uploaded filenames are reduced with `basename`.
- Recommendations: Keep localhost as the only default, require an auth token for non-loopback hosts, enforce request/body/sample limits, rate-limit `/complete` and `/finetune`, and never accept API keys from unauthenticated remote clients.

## Performance Bottlenecks

**Per-event telemetry thread creation:**
- Problem: Every tracked operation creates a new daemon thread and performs a network request, including frequent `complete` calls.
- Files: `needle/_telemetry.py:59-84`, `needle/__init__.py:119-121`, `needle/__init__.py:139-141`
- Cause: There is no bounded queue or shared sender; high-QPS applications can accumulate threads and connection overhead.
- Improvement path: Use a bounded queue and one sender thread (or sampled/batched events), with explicit shutdown behavior and back-pressure/drop metrics.

**Dense Walsh-Hadamard materialization during model application:**
- Problem: `HadamardMLP` constructs a dense `n x n` NumPy matrix (next power of two of `d_model`) in the module call path.
- Files: `needle/model/architecture.py:287-302`
- Cause: The matrix is represented explicitly even though the model description calls for an O(n log n) transform; repeated tracing/compilation multiplies memory and compile cost.
- Improvement path: Cache the matrix per dimension/dtype or implement a structured Walsh-Hadamard transform; benchmark compile and peak memory on the 45M-parameter preset.

## Fragile Areas

**CACT exporter supports only a subset of model configurations:**
- Files: `needle/model/export.py:102-118`, `needle/model/export.py:420-433`
- Why fragile: Export rejects split qk/v head dimensions, lexicon layers, and sliding-window layer patterns, while `TransformerConfig` can represent related fields. A checkpoint can train/load but fail only at deployment build time.
- Safe modification: Run `_geometry` validation before expensive quantization, and add fixture checkpoints for every supported/unsupported configuration with actionable migration messages.
- Test coverage: Existing build tests cover basic tiny exports only (`tests/test_build.py:13-37`); no negative geometry cases are exercised.

**Shared temporary filenames in playground fine-tuning:**
- Files: `needle/playground/server.py:14-15`, `needle/playground/server.py:66-93`
- Why fragile: All sessions use fixed files in the process temp directory (`needle_playground_data.jsonl`, adapter, and output), so retries or another process can overwrite active artifacts.
- Safe modification: Allocate a per-job temporary directory, retain an explicit job id in status, and clean up only after download/expiry.
- Test coverage: No playground HTTP or concurrent fine-tune tests exist.

## Scaling Limits

**Single-process engine/session state:**
- Current capacity: One active native engine state and one `_FT` fine-tune job per Python process.
- Limit: Distinct weight sets cannot be unloaded/rebound safely, and simultaneous playground users share one conversation/engine lock and global status.
- Scaling path: Move model sessions/jobs behind worker processes with explicit job/session identifiers, or document a strict single-session deployment contract.

## Dependencies at Risk

**JAX/Flax/Metal compatibility:**
- Risk: The training extras leave most versions unconstrained while the Metal path relies on an exact JAX/JAXLIB pair and environment mutation before import.
- Impact: Backend upgrades can break compilation or silently change numerics; CPU/GPU/Metal behavior is not reproducible from the repository alone.
- Migration plan: Maintain tested constraints per backend, publish lockfiles, and run a small model smoke test for each supported Python/platform combination.

## Missing Critical Features

**Production request hardening:**
- Problem: The playground has no authentication, quotas, request size limits, cancellation, or durable job storage.
- Blocks: Safely exposing the demo server to a team/network and operating fine-tuning as a shared service.

## Test Coverage Gaps

**Native fetch, security, and concurrency paths:**
- What's not tested: Artifact digest/signature validation, malicious pickle rejection, tokenizer/engine cache failures, concurrent `Needle` calls, and playground endpoint authorization/resource limits.
- Files: `needle/agent/fetch.py`, `needle/model/tokenizer.py`, `needle/model/run.py`, `needle/playground/server.py`, `needle/__init__.py`
- Risk: Supply-chain, data-corruption, and denial-of-service regressions can ship unnoticed; most engine tests are skipped when the native library is absent (`tests/conftest.py:7-19`).
- Priority: High

**Configuration/export edge cases:**
- What's not tested: Unknown config keys, invalid dimension combinations, empty `batch_generate`, and unsupported CACT geometry.
- Files: `needle/model/architecture.py`, `needle/model/run.py`, `needle/model/export.py`, `tests/test_build.py`, `tests/test_run.py`
- Risk: Failures occur late during training/export or only in production deployment.
- Priority: Medium

---

*Concerns audit: 2026-08-31*
