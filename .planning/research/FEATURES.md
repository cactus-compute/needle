# Feature Research

**Domain:** Beginner onboarding for an edge language-model toolkit
**Researched:** 2026-08-31
**Confidence:** MEDIUM

## Feature Landscape

### Table Stakes (Users Expect These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Install and environment guide | New users must know extras and backend choices | LOW | CPU-first plus GPU/Metal branches |
| Copy-paste inference quickstart | Immediate proof that installation works | LOW | Cover CLI and `Needle` API |
| Asset/cache explanation | Model files are not all bundled | MEDIUM | Explain Hugging Face fetch and offline mode |
| Training data schema | Fine-tuning fails without exact JSONL shape | MEDIUM | Show rendered prompt and masked targets |
| LoRA tutorial | Users need a practical low-memory adaptation path | MEDIUM | Include output files and reproducibility knobs |
| Export/deploy verification | A trained adapter is not yet a runtime artifact | HIGH | Merge, quantize, build `.cact`, load it |
| Troubleshooting | Native/JAX failures are environment-specific | MEDIUM | Error symptoms mapped to fixes |

### Differentiators (Competitive Advantage)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| One journey from inference to deployment | Teaches the actual lifecycle, not isolated snippets | MEDIUM | Organize docs by user goal |
| Model architecture visual explanation | Makes unusual SAN components teachable | MEDIUM | Link concepts to source paths |
| CPU and accelerator parity checks | Reduces hardware assumptions | MEDIUM | Expected outputs should be comparable |
| Safety notes beside commands | Prevents unsafe pickle/download/playground use | LOW | Risks are documented before code hardening |

### Anti-Features

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Promise every OS/GPU combination | Sounds beginner-friendly | Backend support changes and failures become opaque | Declare supported matrices and fallback to CPU |
| Hide model artifacts and formats | Keeps quickstart short | Users cannot debug cache or compatibility failures | Explain `.pkl`, tokenizer, engine, and `.cact` roles |
| Production hosting tutorial | Broadens audience | Playground has no auth/rate limits and is local-only | Keep a local playground guide and label it clearly |

## Feature Dependencies

```text
Install/backend guide
    -> asset fetch/cache guide
        -> inference quickstart
            -> JSONL data guide -> LoRA tutorial -> export/deployment guide
Architecture explanation -> troubleshooting and safe usage notes
```

## MVP Definition

### Launch With (v1)

- [ ] Chinese-first install and CPU/GPU environment guide
- [ ] Working inference quickstart
- [ ] Complete LoRA and export/deployment tutorial
- [ ] Architecture, troubleshooting, and safety references

### Add After Validation (v1.x)

- [ ] Notebook-based walkthroughs
- [ ] Automated docs examples in CI
- [ ] More benchmark and memory tables

### Future Consideration (v2+)

- [ ] Hosted multi-user service and authentication
- [ ] GUI training management
- [ ] Automatic artifact signing and registry integration

## Sources

- Existing README and `doc/finetuning.md`, `doc/apis.md` (HIGH)
- Existing CLI and test suite (`needle/cli.py`, `tests/`) (HIGH)
- General ML toolkit onboarding conventions (MEDIUM; validate with user feedback)

---
*Feature research for: Needle*
*Researched: 2026-08-31*
