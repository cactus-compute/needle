# Learned Matryoshka Dimension Selection Plan

## Overview

Replace the current static MRL weight-slicing approach with **learned per-dimension masks** using Gumbel-Softmax selection, adapted from `../matformer-olmo`. The key insight: instead of `x[..., :d] @ emb[:, :d].T` (prefix slicing), we learn a binary mask over all `d_model` dimensions that selects which dimensions participate in each sub-model. This allows the model to keep non-contiguous, high-saliency dimensions rather than being forced to pack everything into a prefix.

**Critical design decision (from matformer-olmo Phase 2.3d):** Masks must multiply the **full-width** hidden states — NOT slice weights. If you slice weights, the learned logits have no gradient path and the whole system fails. Apply mask to `x` only (not both `x` and `emb`) to avoid squaring the mask during soft-mask training.

**Adaptation from matformer-olmo:** We use **uniform allocation** across all layers for each target width (same `k` for every decoder block), rather than the heterogeneous per-layer allocation in matformer-olmo. This simplifies the system and matches our existing MRL export semantics where each sub-model has a single `d_prime`.

---

## Implementation Status

### Completed
- TopK mask logic with differentiable sigmoid threshold
- tau/hard passed as JAX array args through pmap (not globals — captured at compile time)
- Three-phase training: warmup (no MRL) → learning (soft masks, tau annealing) → freeze (hard STE masks)
- Mask logit optimizer (Adam, host-side, non-replicated)
- Dedicated warmup train step functions (no MRL, not global-dependent)
- Single masking: `(x * mask) @ emb.T` not `(x * mask) @ (emb * mask).T`
- TopK export: full model + mask indices (not smaller architecture, preserves attention head structure)
- Spread penalty (lambda=0.01, matching matformer-olmo R6)

### Key Bugs Fixed
1. τ/hard were Python globals captured at pmap compile time → now JAX array args
2. pmap traces lazily at first call, not creation → dedicated warmup functions
3. Double masking squashed gradients for soft masks → mask x only
4. TopK export gathered non-contiguous dims across head boundaries → save full model + mask
5. mask_opt_state unnecessarily replicated → plain host-side value
6. MRL val loss fn closed over mask (retrace each epoch) → pass as arg

---

## Experiment Log

### Baseline Config
- d_model=512, heads=8, kv=2, enc=6, dec=6, 75.6M params
- batch=48×8=384, ~7358 total steps, 1 epoch
- MRL dims: 256, 128, 64
- Speech enabled (every 3 text steps)
- Seed 42

### Experiment 1: Slice vs TopK (Random Init)

**Config:** Default hyperparams (tau 0.5→0.1, warmup 15%, freeze 20%, spread λ=0.01, init normal σ=0.5)

| Dim | Slice PPL | TopK PPL | Δ |
|-----|-----------|----------|---|
| 512 | 4.49 | **4.44** | -1.1% |
| 256 | **4.50** | 4.58 | +1.8% |
| 128 | **4.57** | 4.96 | +8.5% |
| 64 | **4.82** | 5.52 | +14.5% |

**Analysis:** Full model better with topk (warmup has no MRL penalty). Sub-models significantly worse, gap grows with compression ratio. Hypothesis: with only ~4800 learning steps (65% of 7358), the soft masks create a moving target that the model can't fully adapt to. The slice approach has a fixed selection from step 0.

---

## Hypotheses & Experiment Queue

### H1: Reduce warmup, extend learning phase
**Rationale:** 15% warmup (1104 steps) wastes training budget on no-MRL. The model doesn't need 1k+ steps to stabilize before masks start. Reducing to 5% gives masks ~700 more learning steps.
**Config:** `--mrl-warmup-frac 0.05`

### H2: Slower tau annealing (wider soft regime)
**Rationale:** Aggressive tau annealing (0.5→0.1) forces hard decisions too early. Higher tau_end keeps masks softer longer, allowing more exploration.
**Config:** `--mrl-tau-start 1.0 --mrl-tau-end 0.3`

### H3: Prefix initialization
**Rationale:** Initialize mask logits so initial top-k matches prefix dims. Smooth transition from warmup (no MRL) → learning (masks start at prefix, can deviate). Avoids random-init exploration cost.
**Config:** `--mrl-init-mode prefix`

### H4: Higher mask learning rate
**Rationale:** Mask logits use Adam at 1e-3. The model params use much higher LR (2.4e-3 Adam, 0.057 Muon). Masks may not adapt fast enough to keep up with model changes.
**Config:** Increase mask_tx LR to 3e-3 or 1e-2.

### H5: Saliency-initialized masks (from plan)
**Rationale:** During warmup, accumulate per-dimension gradient importance on the embedding. At warmup→learning transition, initialize mask logits from saliency ranking. Gives masks informed starting point.
**Config:** `--mrl-init-mode saliency` (requires implementation)

### H6: No warmup — topk from step 0
**Rationale:** Slice trains MRL from step 0. TopK's warmup gives fewer MRL training steps. What if we skip warmup entirely and use soft topk masks from the start?
**Config:** `--mrl-warmup-frac 0.0`

### H7: Combined: prefix init + no warmup + slow tau
**Rationale:** Start with prefix-matching masks (smooth), no wasted warmup steps, slow annealing for gradual deviation. Maximizes both MRL training steps and exploration time.
**Config:** `--mrl-init-mode prefix --mrl-warmup-frac 0.0 --mrl-tau-start 1.0 --mrl-tau-end 0.2`

---

## Experiment Results (Full Scale: 75.6M params, 7358 steps)

### Summary Table (seed 42)

| Dim | Slice | TopK Exp1 | H7 | H8 | H9 | **H10** | H11 | H12 |
|-----|-------|-----------|------|------|------|---------|------|------|
| 512 | 4.49 | 4.44 | 4.43 | 4.41 | 4.42 | **4.42** | 4.52 | 4.58 |
| 256 | 4.50 | 4.58 | 4.55 | 4.60 | 4.58 | **4.45** | 4.54 | 4.61 |
| 128 | 4.57 | 4.96 | 4.77 | 4.87 | 4.81 | **4.55** | 4.64 | 4.72 |
| 64 | 4.82 | 5.52 | 5.28 | 5.35 | 5.36 | **4.83** | 4.90 | 4.99 |

### Experiment Details

| ID | Warmup | Init | tau | freeze | mask_lr | spread | Notes |
|----|--------|------|-----|--------|---------|--------|-------|
| Exp1 | 15% | normal | 0.5→0.1 | 20% | 1e-3 | 0.01 | First attempt, all defaults |
| H7 | 0% | prefix | 1.0→0.2 | 20% | 3e-3 | 0.01 | No warmup + prefix init |
| H8 | 0% | prefix | 2.0→0.1 | 10% | 1e-2 | 0.01 | Too aggressive LR, negative loss |
| H9 | 0% | prefix | 1.0→0.2 | 15% | 5e-3 | 0.005 | Slightly worse than H7 |
| **H10** | **0%** | **prefix** | **1.0→0.2** | **60%** | **3e-3** | **0.01** | **Best: matches slice** |
| H11 | 0% | prefix | 1.0→0.2 | 70% | 3e-3 | 0.01 | Too little learning time |
| H12 | 0% | normal | 1.0→0.2 | 60% | 3e-3 | 0.01 | Random init worse than prefix |

### Cross-Seed Validation (H10 config)

| Dim | Slice s42 | TopK s42 | Slice s123 | TopK s123 |
|-----|-----------|----------|------------|-----------|
| 512 | 4.49 | **4.42** | 4.48 | **4.45** |
| 256 | 4.50 | **4.45** | **4.49** | **4.49** |
| 128 | 4.57 | **4.55** | **4.56** | 4.58 |
| 64 | **4.82** | 4.83 | **4.81** | 4.86 |

**Conclusion:** TopK H10 achieves **parity with slice** on average. The key hyperparameter is `--mrl-freeze-frac 0.6` — a long freeze phase gives the model time to fully adapt to the locked hard masks, mimicking slice's stability advantage. Prefix init provides a smooth starting point. Slow tau annealing (1.0→0.2) allows gradual exploration before freezing.

### 2-Epoch Comparison (14,716 total steps)

| Dim | Slice 2ep | TopK H10 2ep |
|-----|-----------|-------------|
| 512 | 4.11 | **4.09** |
| 256 | 4.11 | 4.11 |
| 128 | **4.17** | 4.18 |
| 64 | **4.38** | 4.39 |

With 2 epochs, topk fully converges to match slice (within 0.01 PPL at every dim). The full model is consistently better with topk due to no MRL penalty during mask learning. The theoretical advantage of non-contiguous dimension selection may require even longer training or larger models to manifest.

### Best TopK Config (H10)
```bash
needle train --mrl-method topk \
    --mrl-init-mode prefix \
    --mrl-warmup-frac 0.0 \
    --mrl-tau-start 1.0 --mrl-tau-end 0.2 \
    --mrl-freeze-frac 0.6 \
    --mrl-mask-lr 0.003 \
    --mrl-spread-lambda 0.01
```
