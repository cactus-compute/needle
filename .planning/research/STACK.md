# Stack Research

**Domain:** Local/edge language-model inference and fine-tuning toolkit
**Researched:** 2026-08-31
**Confidence:** MEDIUM (repository evidence; external search unavailable)

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | >=3.9 | Public API, CLI, orchestration | Matches package metadata and beginner accessibility |
| JAX + jaxlib | project minimums | Arrays, autodiff, JIT, CPU/GPU/Metal execution | Existing training and reference inference path |
| Flax Linen | >=0.10.2 | Transformer module definitions and parameter trees | Existing `needle/model/architecture.py` contract |
| Optax | project minimums | AdamW, schedules, clipping | Existing LoRA optimizer path |

### Supporting Libraries

| Library | Purpose | When to Use |
|---------|---------|-------------|
| SentencePiece | Tokenizer loading and encoding | Any checkpoint inference or export |
| NumPy | Serialization, conversion, export packing | Checkpoint and `.cact` tooling |
| huggingface_hub | Model/engine artifact downloads | Online setup or explicit asset fetch |
| pytest | Unit and integration verification | Running the documented examples and regressions |

## Installation Guidance

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
# Add `.[train]` for JAX/Flax/Optax/SentencePiece workflows.
# Choose `.[gpu]` or `.[metal]` only for a supported accelerator.
```

Document CPU first, then accelerator extras. Explain that native engine assets
and tokenizer/checkpoint files may be fetched and cached separately.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| JAX/Flax | PyTorch | Use only if a separate training ecosystem is required; it does not match current checkpoints |
| Native `.cact` runtime | Python-only inference | Use Python reference path for debugging or training, not constrained deployment |
| SentencePiece | BPE tokenizer library | Only when checkpoint vocabulary and export contract are changed together |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Unpinned ad-hoc dependency upgrades | JAX/Flax and export geometry can drift | Keep project constraints and test after upgrades |
| Editing `.cact` tensor order manually | Native loader is positional | Extend exporter and engine together |
| Downloading artifacts without provenance checks | Corrupt or malicious assets can be loaded | Verify source, checksum/provenance, and cache path |

## Version Compatibility

- CPython 3.9+ is the declared baseline.
- GPU and Metal installs must follow the JAX backend compatibility matrix; do
  not promise one command for every platform.
- `.cact` files are coupled to `ENGINE_VERSION` and model geometry.

## Sources

- `pyproject.toml`, `requirements.txt`, `requirements-train.txt` - repository dependency declarations (HIGH)
- `needle/model/architecture.py`, `finetune.py`, `export.py` - active architecture and export contracts (HIGH)
- JAX installation guide: https://docs.jax.dev/en/latest/installation.html (MEDIUM; validate during implementation)
- Flax Linen guide: https://flax.readthedocs.io/en/latest/ (MEDIUM; validate during implementation)
- Hugging Face Hub docs: https://huggingface.co/docs/huggingface_hub/ (MEDIUM; validate download details)

---
*Stack research for: Needle*
*Researched: 2026-08-31*
