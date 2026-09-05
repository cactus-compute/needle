# Onboarding Summary

## Project State
- PROJECT.md: present
- REQUIREMENTS.md: present
- ROADMAP.md: present
- STATE.md: present

## Codebase Context
- Brownfield repo: yes
- Map readiness: complete
- Codebase map: `.planning/codebase/` (complete codebase map)
- Fast map available: yes

## Docs Context
- Existing ADR/PRD/SPEC/RFC candidates: 0

## What This Project Contains

Needle is a Python package with two main execution paths:

- Production inference: `needle.Needle` loads a platform native engine through
  ctypes and runs against a `.cact` archive.
- Reference/training: JAX/Flax code defines the Simple Attention Network,
  decoding, LoRA fine-tuning, quantization, and export.

The CLI in `needle/cli.py` covers fetching assets, inference, data generation,
fine-tuning, building exports, downloading engines, and the playground.

## How To Start

1. Read `.planning/codebase/STACK.md` for Python extras, backends, assets, and
   environment variables.
2. Read `.planning/codebase/ARCHITECTURE.md` for model structure and native vs
   JAX data flow.
3. Use the first roadmap phase to turn the current README/docs into a tested
   Chinese-first install and inference quickstart.
4. Continue through LoRA and `.cact` deployment only after the first inference
   path is verified.

## Fine-Tuning Entry Points

- `needle/model/finetune.py`: JSONL rendering, masked causal loss, LoRA update,
  merge, and build orchestration.
- `doc/finetuning.md`: existing fine-tuning notes to reconcile with the new
  beginner tutorial.
- `tests/test_finetune.py` and `tests/test_build.py`: executable behavior and
  export checks to reuse for documentation verification.

## Important Constraints

- Native state is process-global; isolate base and tuned agents when needed.
- `.cact` tensor order, geometry, tokenizer vocabulary, and engine version must
  remain aligned.
- Treat downloaded native artifacts and pickle checkpoints as trusted inputs
  only; the local playground is not an authenticated production server.
- Git commits require a writable `.git` index; this environment now has a
  successful initialization commit.

## Recommended Next Step
- `$gsd-manager` (then `$gsd-discuss-phase 1` or `$gsd-plan-phase 1`)
