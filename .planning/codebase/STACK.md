# Technology Stack

**Analysis Date:** 2026-08-31

## Languages

**Primary:**
- Python (>=3.9) - Public package API, CLI, inference orchestration, training and export in `needle/`.
- JAX/Python numerical code - Neural-network definition, decoding, quantization and LoRA updates in `needle/model/architecture.py`, `needle/model/decode.py`, `needle/model/quantize.py`, and `needle/model/finetune.py`.

**Secondary:**
- C/C++ shared library (prebuilt, outside this repository) - Native inference engine loaded through `ctypes` in `needle/__init__.py`; the Python package does not compile it locally.
- HTML/CSS/JavaScript - Browser playground assets in `needle/playground/index.html`, `needle/playground/app.js`, and `needle/playground/style.css`.

## Runtime

**Environment:**
- CPython 3.9 or newer (declared by `requires-python` in `pyproject.toml`).
- Native engine selected by platform/architecture and loaded with `ctypes.CDLL` from `needle/__init__.py`.

**Package Manager:**
- `pip`/PEP 517 setuptools build (`setuptools>=68.0`, `pyproject.toml`).
- Lockfile: missing; dependency versions are specified as unconstrained or minimum versions in `pyproject.toml` and `requirements*.txt`.

## Frameworks

**Core:**
- Flax Linen (`flax>=0.10.2`, training extra) - Transformer/Simple Attention Network modules in `needle/model/architecture.py`.
- JAX (`jax`, `jaxlib`, training extra) - Array operations, autodiff, JIT and accelerator execution in `needle/model/`.
- SentencePiece (`sentencepiece`, training extra) - Tokenizer model loading/encoding in `needle/model/tokenizer.py`.
- Python standard library `argparse` and `http.server` - CLI in `needle/cli.py` and local playground server in `needle/playground/server.py`.

**Testing:**
- Pytest (`pytest`, test extra) - Tests under `tests/`, configured by `tool.pytest.ini_options` in `pyproject.toml`.
- Pydantic (`pydantic`, test extra/runtime-optional) - Typed extraction schemas and test fixtures; integrated dynamically in `needle/agent/tools.py` and `needle/__init__.py`.

**Build/Dev:**
- Setuptools package discovery and package data, configured in `pyproject.toml`.
- Cactus Quants export format (`.cact`) implemented in `needle/model/export.py` and quantization utilities in `needle/model/quantize.py`.

## Key Dependencies

**Critical:**
- `huggingface_hub` - Downloads the base checkpoint, tokenizer, native engine wheels and published `.cact` archives (`needle/model/run.py`, `needle/model/tokenizer.py`, `needle/agent/fetch.py`, `needle/cli.py`).
- `jax`/`jaxlib` - Required for checkpoint inference utilities and all fine-tuning/export paths (`needle/model/run.py`, `needle/model/finetune.py`).
- `flax` - Parameterized neural-network modules and tree traversal for LoRA (`needle/model/architecture.py`, `needle/model/finetune.py`).
- `optax` - AdamW, warmup/cosine schedule, gradient clipping and loss helpers in `needle/model/finetune.py`.
- `sentencepiece` - Required to train or load the model tokenizer (`needle/model/tokenizer.py`).

**Infrastructure:**
- `numpy` - Checkpoint conversion, array serialization and export packing across `needle/model/`.
- `pydantic` (optional) - Converts `BaseModel` schemas into tool contracts and typed extraction results (`needle/agent/tools.py`, `needle/__init__.py`).

## Configuration

**Environment:**
- `NEEDLE_LIB_PATH` overrides native engine lookup; otherwise the package directory and `~/.cache/cactus-needle/<engine-version>/` are searched (`needle/__init__.py`).
- `HF_HUB_OFFLINE=1` prevents Hugging Face network access for air-gapped operation (documented in `doc/apis.md`).
- `OPENROUTER_API_KEY` authorizes optional synthetic data generation; `OPENROUTER_URL` overrides the OpenAI-compatible endpoint (`needle/model/finetune.py`).
- `NEEDLE_HF_REPO` selects the Hugging Face destination for `needle build --upload` (`needle/model/finetune.py`).
- `NEEDLE_TELEMETRY=0` or `DO_NOT_TRACK=1` disables anonymous telemetry; `CI` also disables it (`needle/_telemetry.py`).
- `ENABLE_PJRT_COMPATIBILITY` is set automatically on macOS before JAX initialization for the Metal plugin (`needle/model/finetune.py`).

**Build:**
- `pyproject.toml` defines package metadata, optional extras (`train`, `gpu`, `metal`, `test`), console script `needle = needle.cli:main`, package data and pytest paths.
- `requirements.txt` contains runtime installation; `requirements-train.txt` extends it with JAX/Flax/Optax/SentencePiece training dependencies.
- Model/tokenizer assets (`*.model`, `*.vocab`) are included as package data for `needle.model`; playground static assets are included for `needle.playground`.

## Platform Requirements

**Development:**
- Python 3.9+ and a platform-supported JAX backend. Install `cactus-needle[train,gpu]` for NVIDIA CUDA 12 or `cactus-needle[train,metal]` for Apple Silicon Metal (pins JAX 0.4.38).
- Network access is needed once to fetch the native engine/checkpoint/tokenizer from Hugging Face unless assets are pre-populated in cache.

**Production:**
- A supported native engine binary for the target platform (downloaded by `needle fetch` or `needle download <platform>`).
- Runtime RAM target is approximately 28 MB for the bundled 14 MB Needle 2 engine/weights, as described in `README.md`; tuned `.cact` archives use the same engine.

---

*Stack analysis: 2026-08-31*
