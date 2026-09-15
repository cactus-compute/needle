# External Integrations

**Analysis Date:** 2026-08-31

## APIs & External Services

**Model and artifact distribution:**
- Hugging Face Hub (`Cactus-Compute/needle2`) - Publishes native engine wheels, base checkpoints, tokenizer files and optional tuned `.cact` archives. Used by `needle/agent/fetch.py`, `needle/model/run.py`, `needle/model/tokenizer.py`, and `needle/cli.py`.
  - SDK/Client: `huggingface_hub` (`hf_hub_download`, `list_repo_files`, `HfApi`)
  - Auth: public downloads require no configured secret; upload uses the Hugging Face client/token environment handled by `HfApi`.

**Synthetic fine-tuning data:**
- OpenRouter (default `https://openrouter.ai/api/v1/chat/completions`) - Optional generation/augmentation of tool-calling JSONL examples in `needle/model/finetune.py` and the playground worker in `needle/playground/server.py`.
  - SDK/Client: Python `urllib.request` with an OpenAI-compatible JSON request.
  - Auth: `OPENROUTER_API_KEY` (Bearer token); endpoint can be changed with `OPENROUTER_URL`.
  - Default model: `deepseek/deepseek-v4-flash` (`DEFAULT_MODEL` in `needle/model/finetune.py`).

**Native inference runtime:**
- Cactus Needle engine - A platform-specific shared library fetched from Hugging Face and loaded with `ctypes` in `needle/__init__.py`.
  - SDK/Client: C ABI functions `needle_init`, `needle_complete`, `needle_reset`, and `needle_load`.
  - Auth: none; local binary and `.cact` bytes are supplied by the caller.

## Data Storage

**Databases:**
- None detected. Runtime state is in-memory Python objects and native engine state (`needle/__init__.py`, `needle/playground/server.py`).

**File Storage:**
- Local filesystem only for checkpoints, LoRA adapters, `.cact` exports, JSONL datasets and playground downloads.
- Hugging Face cache is used for remote artifacts; native engine defaults to `~/.cache/cactus-needle/<ENGINE_VERSION>/` (`needle/agent/fetch.py`, `needle/__init__.py`).
- Telemetry anonymous ID is stored at `~/.cactus_needle/telemetry_id` (`needle/_telemetry.py`).

**Caching:**
- Hugging Face's local cache backs downloads; `tool_index_path` optionally persists tool embeddings keyed by schema/model fingerprint (documented in `doc/apis.md`).
- No Redis, Memcached or remote cache integration is present.

## Authentication & Identity

**Auth Provider:**
- No user authentication or identity provider is implemented.
- External credentials are limited to `OPENROUTER_API_KEY` for data synthesis and Hugging Face credentials used implicitly by `HfApi` when `--upload` is requested.
- Telemetry uses a locally generated random anonymous ID, not an account identity (`needle/_telemetry.py`).

## Monitoring & Observability

**Error Tracking:**
- None detected (no Sentry or hosted error tracker).

**Logs:**
- CLI and training progress use `print()`/stderr; `needle/cli.py` installs an XLA stderr noise filter.
- Anonymous usage counts are sent asynchronously to a Supabase Edge Function endpoint (`needle/_telemetry.py`), containing function name, package/engine versions, OS/arch/Python and optional non-content properties; prompts and outputs are not sent.

## CI/CD & Deployment

**Hosting:**
- The package is installable from Python packaging infrastructure; model/engine artifacts are hosted on Hugging Face. No application hosting configuration is included.

**CI Pipeline:**
- No CI workflow files were detected at repository root. Tests are run locally with Pytest (`pyproject.toml`, `tests/`).

## Environment Configuration

**Required env vars:**
- Normal inference: none if engine/checkpoint assets are cached or downloadable.
- Synthetic generation: `OPENROUTER_API_KEY`; optional `OPENROUTER_URL`.
- Telemetry opt-out: `NEEDLE_TELEMETRY=0` or `DO_NOT_TRACK=1`.
- Air-gapped deployment: optionally `HF_HUB_OFFLINE=1` and/or `NEEDLE_LIB_PATH`.

**Secrets location:**
- Supplied through process environment or Hugging Face's standard local credential configuration; no secret files are read by the package. Never commit API keys to datasets or source.

## Webhooks & Callbacks

**Incoming:**
- None. The local playground exposes HTTP endpoints (`/complete`, `/reset`, `/finetune`, `/load-model`) but they are local server routes, not third-party webhooks (`needle/playground/server.py`).

**Outgoing:**
- OpenRouter HTTPS POST requests for synthetic examples (`needle/model/finetune.py`).
- Supabase telemetry HTTPS POST requests to the configured/default endpoint (`needle/_telemetry.py`).
- Hugging Face Hub download/upload API requests for model artifacts (`needle/agent/fetch.py`, `needle/model/run.py`, `needle/model/tokenizer.py`, `needle/cli.py`).

---

*Integration audit: 2026-08-31*
