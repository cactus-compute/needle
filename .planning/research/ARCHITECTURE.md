# Architecture Research

**Domain:** Edge language-model toolkit documentation and workflows
**Researched:** 2026-08-31
**Confidence:** HIGH for current repository, MEDIUM for external conventions

## Standard Architecture

```text
User docs / CLI examples
          |
          +--> Native runtime facade (`Needle`, ctypes, `.cact`)
          |
          +--> Reference/training path (JAX + Flax + checkpoints)
                         |
                         +--> LoRA adapter -> merge -> quantize/export
```

### Component Responsibilities

| Component | Responsibility | Repository implementation |
|-----------|----------------|---------------------------|
| Public facade | Completion, tools, extraction, reset | `needle/__init__.py` |
| CLI | Route user workflows and environment setup | `needle/cli.py` |
| Native adapter | Load engine and active weights | `needle/__init__.py`, `needle/agent/fetch.py` |
| Reference model | SAN layers, masks, decode | `needle/model/architecture.py`, `run.py` |
| Fine-tuning | Render JSONL, train LoRA, merge | `needle/model/finetune.py` |
| Export | Quantize and pack `.cact` | `needle/model/quantize.py`, `export.py` |

## Recommended Documentation Structure

```text
README.md                 # 10-minute success path
doc/
├── installation.md       # CPU/GPU/Metal, assets, cache, offline
├── inference.md          # API, CLI, tools, expected output
├── finetuning.md         # JSONL, LoRA, checkpoints, resource knobs
├── deployment.md         # merge, quantize, `.cact`, compatibility
├── architecture.md       # model and runtime data flow
└── troubleshooting.md    # symptoms, causes, fixes, safety warnings
```

Keep source links next to claims. Each tutorial should state prerequisites,
commands, expected output, and a verification command.

## Data Flow and Build Order

1. Select Python/backend and install extras.
2. Fetch or locate engine, checkpoint, and tokenizer assets.
3. Run native or reference inference.
4. Render JSONL examples and train LoRA against a frozen base checkpoint.
5. Merge adapter, quantize, write `.cact`, and load it in a fresh process.
6. Verify output and compatibility against the target engine version.

This ordering follows existing boundaries and prevents teaching deployment
before the user understands the asset and tokenizer contracts.

## Architectural Patterns

### Pattern 1: Two explicit execution paths

Document native production inference and JAX/Flax training separately. They share
model concepts but not the same runtime dependencies or artifact formats.

### Pattern 2: Artifact-driven handoff

Treat checkpoint, tokenizer, LoRA adapter, and `.cact` archive as named handoff
artifacts. Record where each is written and which command consumes it.

### Pattern 3: Verification at every boundary

After install, inference, training, and export, show one observable check. This
is more reliable for beginners than a final end-to-end claim only.

## Scaling Considerations

- Small local use: keep the single-process facade and local cache.
- Repeated inference: isolate tuned/base agents by process because native state
  is global and cannot unload a tuned archive.
- Larger training: document accelerator memory and checkpoint retention before
  suggesting distributed changes; current training has no robust resume flow.

## Sources

- `.planning/codebase/ARCHITECTURE.md` and `STRUCTURE.md` (HIGH)
- `needle/model/finetune.py`, `export.py`, `needle/__init__.py` (HIGH)
- JAX/Flax official guides listed in `research/STACK.md` (MEDIUM)

---
*Architecture research for: Needle*
*Researched: 2026-08-31*
