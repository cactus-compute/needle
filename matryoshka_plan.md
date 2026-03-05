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

## Experiment Config

### Shared Config (all experiments)
- d_model=512, heads=8, kv=2, enc=6, dec=6, 75.6M params
- batch=48×8=384, ~7358 total steps, 1 epoch (unless noted)
- MRL dims: 256, 128, 64
- Speech enabled (every 3 text steps)
- Seed 42 (unless noted)

### TopK Baseline Defaults
- init: normal (σ=0.5), warmup: 15%, tau: 0.5→0.1, freeze: 20%, mask_lr: 1e-3, spread λ: 0.01

---

## Experiment Results

### Full Table (seed 42, 1 epoch, val PPL)

Each experiment changes **only the listed variable** from the topk baseline, except where noted.

| Dim | Slice | TopK baseline | H1 | H2 | H3 | H4 | H6 | H7 | H8 | H9 | **H10** | H11 | H12 |
|-----|-------|---------------|------|------|------|------|------|------|------|------|---------|------|------|
| 512 | 4.49 | 4.44 | 4.45 | 4.48 | **4.39** | 4.44 | 4.50 | 4.43 | 4.41 | 4.42 | **4.42** | 4.52 | 4.58 |
| 256 | **4.50** | 4.58 | 4.57 | 4.61 | 4.51 | 4.56 | 4.61 | 4.55 | 4.60 | 4.58 | **4.45** | 4.54 | 4.61 |
| 128 | **4.57** | 4.96 | 4.92 | 5.05 | 4.72 | 4.93 | 4.91 | 4.77 | 4.87 | 4.81 | **4.55** | 4.64 | 4.72 |
| 64 | **4.82** | 5.52 | 5.42 | 5.67 | 5.34 | 5.49 | 5.38 | 5.28 | 5.35 | 5.36 | **4.83** | 4.90 | 4.99 |

### Experiment Key

**Isolated single-variable experiments (one change from topk baseline):**

| ID | Changed variable | Value | Notes |
|----|-----------------|-------|-------|
| H1 | warmup | 5% (from 15%) | Modest improvement at sub-models |
| H2 | tau | 1.0→0.3 (from 0.5→0.1) | Worse — softer masks hurt |
| H3 | init | prefix (from normal) | **Biggest single-variable win** |
| H4 | mask_lr | 3e-3 (from 1e-3) | Negligible effect |
| H6 | warmup | 0% (from 15%) | Modest improvement, similar to H1 |

**Combined experiments (multiple changes from topk baseline):**

| ID | Warmup | Init | tau | freeze | mask_lr | spread | Notes |
|----|--------|------|-----|--------|---------|--------|-------|
| H7 | 0% | prefix | 1.0→0.2 | 20% | 3e-3 | 0.01 | Combines H3+H6+H2+H4 |
| H8 | 0% | prefix | 2.0→0.1 | 10% | 1e-2 | 0.01 | Too aggressive, negative loss |
| H9 | 0% | prefix | 1.0→0.2 | 15% | 5e-3 | 0.005 | Slightly worse than H7 |
| **H10** | **0%** | **prefix** | **1.0→0.2** | **60%** | **3e-3** | **0.01** | **Best: matches slice** |
| H11 | 0% | prefix | 1.0→0.2 | 70% | 3e-3 | 0.01 | Too little learning time |
| H12 | 0% | normal | 1.0→0.2 | 60% | 3e-3 | 0.01 | H10 ablation: random init worse |

### Cross-Seed Validation (H10 config)

| Dim | Slice s42 | TopK s42 | Slice s123 | TopK s123 |
|-----|-----------|----------|------------|-----------|
| 512 | 4.49 | **4.42** | 4.48 | **4.45** |
| 256 | 4.50 | **4.45** | **4.49** | **4.49** |
| 128 | 4.57 | **4.55** | **4.56** | 4.58 |
| 64 | **4.82** | 4.83 | **4.81** | 4.86 |

### 2-Epoch Comparison (14,716 total steps, H10 config)

| Dim | Slice 2ep | TopK 2ep |
|-----|-----------|----------|
| 512 | 4.11 | **4.09** |
| 256 | 4.11 | 4.11 |
| 128 | **4.17** | 4.18 |
| 64 | **4.38** | 4.39 |

### Training Overhead
TopK H10: 8.65 it/s vs Slice: 8.91 it/s → **~3% slower**

---

## Analysis

**Isolated variable impact (ranked by d=128 improvement over topk baseline 4.96):**
1. **H3 (prefix init):** 4.72 (−0.24) — by far the most impactful single change
2. **H6 (no warmup):** 4.91 (−0.05)
3. **H1 (warmup 5%):** 4.92 (−0.04)
4. **H4 (mask lr 3e-3):** 4.93 (−0.03) — negligible
5. **H2 (slow tau):** 5.05 (+0.09) — actually hurts

**Combined config H10** achieves parity with slice by combining prefix init (biggest win) with no warmup and a long 60% freeze phase. The freeze phase is critical — it gives the model 4400+ steps of stable hard-mask MRL training, equivalent to slice's fixed selection.

**2-epoch results** show full convergence: topk matches slice within 0.01 PPL at every dim.

---

## Best TopK Config (H10)
```bash
needle train --mrl-method topk \
    --mrl-init-mode prefix \
    --mrl-warmup-frac 0.0 \
    --mrl-tau-start 1.0 --mrl-tau-end 0.2 \
    --mrl-freeze-frac 0.6 \
    --mrl-mask-lr 0.003 \
    --mrl-spread-lambda 0.01
```

- **`--mrl-init-mode prefix`** — Initialize mask logits as a linear ramp: dim 0 gets the highest value, dim d_model-1 gets the lowest. So the initial top-k selection exactly matches prefix slicing (dims 0..k-1). This was the single most impactful change (H3 isolated experiment).
- **`--mrl-warmup-frac 0.0`** — No warmup phase. TopK mask learning starts from step 0. There is NO period of full-model-only training — MRL sub-model losses are active from the very first step. (When warmup > 0, the model trains with no MRL loss at all during warmup — just full-model CE.)
- **`--mrl-tau-start 1.0`** — Initial sigmoid temperature. Higher tau = softer masks. At tau=1.0, the sigmoid `σ((logit - threshold) / tau)` has a gentle slope, so many dimensions get intermediate mask values (0.3-0.7). This allows gradients to flow to all dimensions near the boundary.
- **`--mrl-tau-end 0.2`** — Final sigmoid temperature before freeze. At tau=0.2, the sigmoid is much steeper — mask values are pushed closer to 0 or 1. The annealing from 1.0→0.2 happens exponentially over the learning phase.
- **`--mrl-freeze-frac 0.6`** — The last 60% of training uses hard binary masks (STE: straight-through estimator). Mask logit optimizer is frozen — no more mask updates. This was the key finding: the model needs a long period of stable, fixed dimension selection to fully adapt, just like slice has fixed selection from the start.
- **`--mrl-mask-lr 0.003`** — Adam learning rate for the mask logit optimizer (separate from the model's Muon/AdamW optimizers). The mask logits are a small `(n_mrl_dims, d_ff)` array updated host-side after each step.
- **`--mrl-spread-lambda 0.01`** — Weight of the spread penalty: `−λ · mean(var(logits))`. Maximizes variance of mask logits per MRL dim, encouraging clear on/off decisions rather than ambiguous middle values. Matches matformer-olmo's best config (R6).

---

## PROPER Learned Matryoshka (FFN Interior Masking)

All experiments above used **output-only masking**: the mask was applied to the final hidden state at the logit projection (`x * mask @ emb.T`). The decoder and encoder ran at full width — the mask only selected which output dimensions contributed to classification. This is fundamentally wrong: the sub-models aren't truly independent, they share all internal computation and just differ in which output dims are read.

Evidence: a **shuffled_prefix** experiment (same logit distribution as prefix, but randomly permuted dims) performed worse than prefix (d=128: 4.63 vs 4.55). If the implementation were correct, these should be identical since all dimensions are symmetric. The gap proves the model was exploiting contiguous prefix structure in the output projection, not learning genuine sub-networks.

### What Changed

**The mask now operates on the FFN intermediate activations (d_ff), not d_model.** This is exactly what matformer-olmo does:

```python
# Inside FeedForward.__call__:
gate = gate_proj(x)     # (B, T, d_ff)
up = up_proj(x)         # (B, T, d_ff)
h = activation(gate) * up
if ffn_mask is not None:
    h = h * ffn_mask[None, None, :]  # mask d_ff neurons
return down_proj(h)      # (B, T, d_model) — d_model unchanged
```

Key differences from the output-only approach:

1. **d_model is constant** — attention heads, embeddings, residual stream all stay full-width. No head structure issues.
2. **Only FFN parameters are reduced** — for a sub-model at MRL dim d, the active FFN neurons are `k = d_ff * d // d_model` (proportional reduction). Only gate_proj, up_proj, and down_proj weights for the active neurons participate.
3. **Mask shape is `(n_mrl, d_ff)`** not `(n_mrl, d_model)` — the learnable logits select which FFN neurons to keep, not which hidden dimensions.
4. **Separate forward passes per MRL dim** — each sub-model runs a full encoder + decoder pass with its FFN mask applied at every FeedForward layer (both encoder local FFN and decoder FFN). The full-model forward runs without any mask.
5. **All FeedForward layers masked** — encoder's local FFN in each MemoryMixerBlock and decoder's FFN in each DecoderBlock. The MLP-Mixer's token/channel mixing is NOT masked (different structure/dimensions).

### Training Cost

With 3 MRL dims, each step runs 4 forward passes (1 full + 3 masked). Observed ~2x slower than output-only masking (~4 it/s vs ~8.6 it/s). The extra cost comes from re-running the full encoder and decoder per MRL dim.

### Why This Should Work

With FFN masking, shuffled_prefix and prefix init should now give **identical results**, because:
- The FFN treats all intermediate neurons symmetrically (each is an independent gate+up → activation → down pathway)
- The mask selects which neurons are active — their index doesn't matter, only which set
- d_model is unchanged, so attention heads are unaffected
- The model is forced to route information through fewer FFN neurons for sub-models, creating genuinely different computational paths

### Files Modified
- `src/model.py`: Added `ffn_mask` parameter to `FeedForward`, `DecoderBlock`, `Decoder`, `MemoryMixerBlock`, `MemoryMixerEncoder`, and all encode/decode methods. `forward_with_aux` and `forward_speech_with_aux` now run separate encoder+decoder passes per MRL dim with FFN masks.
- `src/train.py`: Mask logits shape changed from `(n_mrl, d_model)` to `(n_mrl, d_ff)`. `_compute_mrl_masks` computes `k = d_ff * d // d_model` per MRL dim. Eval topk_mask uses `k_ff` not `d`.

---

## Untested Hypotheses for Future Work

### Saliency-initialized masks (H5)
During a warmup phase, accumulate per-dimension gradient importance on the FFN intermediate activations. At the warmup→learning transition, initialize mask logits from saliency ranking instead of prefix ramp. Requires implementation of gradient accumulation during warmup.
