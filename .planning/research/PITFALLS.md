# Pitfalls Research

**Domain:** Edge language-model inference, LoRA fine-tuning, and deployment
**Researched:** 2026-08-31
**Confidence:** HIGH for repository-specific risks

## Critical Pitfalls

### Pitfall 1: Treating the playground as production service

**What goes wrong:** Local HTTP endpoints are exposed without authentication or
resource limits.
**Why it happens:** A demo feels like a finished server.
**How to avoid:** Label it local-only, bind to loopback, and document network
exposure risks; hardening belongs in a later phase.
**Warning signs:** Binding to a public interface or accepting unbounded prompts.
**Phase to address:** Documentation quickstart and later security hardening.

### Pitfall 2: Loading untrusted pickle checkpoints

**What goes wrong:** Python deserialization can execute attacker-controlled code.
**Why it happens:** `.pkl` is convenient for JAX parameter trees.
**How to avoid:** Use trusted artifacts, verify provenance/checksums, and explain
the risk before the fine-tuning guide.
**Warning signs:** Checkpoints downloaded from unknown URLs or shared blindly.
**Phase to address:** Installation and fine-tuning safety notes.

### Pitfall 3: Export/runtime incompatibility

**What goes wrong:** A `.cact` archive loads with wrong tensors or is rejected.
**Why it happens:** Tensor order, geometry, tokenizer size, and engine version
are positional contracts.
**How to avoid:** Rebuild after engine upgrades and run round-trip tests.
**Warning signs:** Shape errors, nonsense output, or mismatched vocab size.
**Phase to address:** Deployment tutorial and verification checklist.

### Pitfall 4: Assuming training can resume safely

**What goes wrong:** Interrupted LoRA training loses optimizer/step state.
**Why it happens:** The current local training flow does not expose robust resume.
**How to avoid:** Document checkpoint outputs and current limitation explicitly.
**Warning signs:** Long runs with no retained adapter or optimizer snapshots.
**Phase to address:** Fine-tuning guide; implementation follow-up later.

## Technical Debt and Performance Traps

| Shortcut | Long-term cost | Prevention |
|----------|----------------|------------|
| Unconstrained dependency upgrades | Backend/export drift | Record tested versions and rerun tests |
| Dense Hadamard operations at large width | Memory/latency spikes | Explain geometry and benchmark before scaling |
| Per-event telemetry threads | Overhead and flaky shutdown | Document opt-out variables and CI behavior |
| Global native state in one process | Base/tuned ordering races | Use fresh processes for isolated artifacts |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Trusting downloaded native binaries | Code execution/supply-chain risk | Verify source and checksums |
| Publishing playground port | Unauthenticated inference abuse | Keep loopback binding and warn clearly |
| Sharing API keys in examples | Credential leakage | Use placeholders and environment variables |

## Looks Done But Isn't

- [ ] Install works without explaining native engine and tokenizer assets.
- [ ] LoRA command runs but does not explain masked target format or output path.
- [ ] `.cact` is produced but engine version and tokenizer geometry are unchecked.
- [ ] GPU instructions omit CPU fallback and backend compatibility caveats.

## Sources

- `.planning/codebase/CONCERNS.md` (HIGH)
- `needle/model/run.py`, `finetune.py`, `export.py`, `playground/server.py` (HIGH)
- Python pickle security guidance: https://docs.python.org/3/library/pickle.html (MEDIUM)

---
*Pitfalls research for: Needle*
*Researched: 2026-08-31*
