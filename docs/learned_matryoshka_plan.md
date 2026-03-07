# Learned Matryoshka (TopK FFN Masking)

> **Note:** This document was ported from the `learned-mat-old` branch. The CLI args
> use the old `--mrl-*` naming — the current codebase uses `--mat-*` instead (e.g.
> `--mat-method topk`, `--mat-tau-start`, `--mat-mask-lr`). The architecture has also
> changed: masks are now **per-layer** (one mask per encoder/decoder block) and
> training uses the single-forward `forward_masked()` approach with heterogeneous
> `(n_blocks, batch, d_ff)` masks, rather than the N+1 `forward_with_aux()` approach
> described below. **The experimental results and lessons learned remain valid.**

## Overview

Extension of the FFN interior matryoshka (see `matryoshka_results.md`). Instead of fixed prefix masks (first k FFN neurons), we learn WHICH neurons to keep via differentiable top-k selection. The model discovers the optimal neuron selection rather than being constrained to a prefix.

## How It Works

The base FFN matryoshka uses `_make_ffn_mask(batch_size, d_ff, mrl_ff_slices)` to create a `(batch, d_ff)` mask where different batch items get different widths, each using a **prefix mask**: `arange < k`.

Learned matryoshka replaces this with `_make_ffn_mask_topk(mask_logits, ...)` which uses **differentiable top-k** to select neurons:

```python
def topk_mask(logits, k, tau, hard):
    topk_vals = jax.lax.top_k(logits, k)[0]
    threshold = topk_vals[-1]
    y_soft = sigmoid((logits - threshold) / tau)
    y_hard = (y_soft >= 0.5).astype(y_soft.dtype)
    ste = y_hard - stop_gradient(y_soft) + y_soft
    return where(hard, ste, y_soft)
```

- `mask_logits`: `(n_mrl, d_ff)` — learned per-width logit vectors
- `tau`: sigmoid temperature — controls softness of selection
- `hard`: boolean — use STE for exact binary masks
- Gradients flow through the soft sigmoid back to the logits

The same heterogeneous-batch structure is used: batch is split into N sections (full + one per MRL width), each section gets its mask. The only difference is WHERE the mask comes from.

## Training Schedule

Two-phase (no warmup — MRL active from step 0):

1. **Learning phase** (first 40% of steps): soft masks with tau annealing (1.0 → 0.2). Mask logits updated by their own Adam optimizer after each step.
2. **Freeze phase** (last 60% of steps): hard binary masks via STE. Mask logit optimizer frozen — the model adapts to the locked neuron selection.

> **Current implementation note:** The codebase now supports a three-phase schedule
> with an explicit warmup phase (`--mat-warmup-frac`) before mask learning begins.

## CLI Args

> **Outdated naming.** See table below for current equivalents.
>
> | Old (`learned-mat-old`) | Current (`fixed-mrl`) |
> |---|---|
> | `--mrl-method topk` | `--mat-method topk` |
> | `--mrl-tau-start` | `--mat-tau-start` |
> | `--mrl-tau-end` | `--mat-tau-end` |
> | `--mrl-freeze-frac` | `--mat-freeze-frac` |
> | `--mrl-mask-lr` | `--mat-mask-lr` |
> | `--mrl-spread-lambda` | `--mat-spread-lambda` |
> | `--mrl-init-mode` | `--mat-init-mode` |
> | `--mrl-warmup-frac` | `--mat-warmup-frac` |
> | `--mrl-saliency-scale` | `--mat-saliency-scale` |
> | *(N/A)* | `--mat-gumbel` (new: per-item Gumbel noise) |

```
--mrl-method topk                   # Enable learned masks (default: prefix)
--mrl-tau-start 1.0                 # Initial sigmoid temperature
--mrl-tau-end 0.2                   # Final temperature before freeze
--mrl-freeze-frac 0.6              # Fraction of training with hard frozen masks
--mrl-mask-lr 0.003                # Mask logit optimizer learning rate
--mrl-spread-lambda 0.01           # Spread penalty weight
--mrl-init-mode shuffled_prefix    # Mask logit initialization (default)
```

## Key Design Decisions

1. **FFN interior masking**: Masks applied inside FeedForward on d_ff intermediate. d_model is constant. Only FFN params are reduced per sub-model.
2. **Shuffled init**: Randomly permuted ramp (same value distribution as prefix, different neuron order). FFN neurons are symmetric, so init order shouldn't matter.
3. **tau/hard as JAX array args**: Passed through pmap as replicated scalars, not Python globals (globals are captured at compile time and never update).
4. **Mask optimizer host-side**: `mask_opt_state` stored as plain numpy, not replicated across devices.
5. **Eval uses learned masks**: `forward_with_aux` accepts `mrl_ffn_masks` to evaluate with the learned neuron selection, not hardcoded prefix.
6. **Spread penalty**: `−λ · mean(var(logits))` encourages clear on/off decisions. λ=0.01 matches matformer-olmo R6.

## Results

All experiments use identical config matching the prefix baseline in `matryoshka_results.md`:
- 87.7M params: d=512, d_ff=2048, heads=8 (kv=8), enc=8, dec=4, memory=64
- batch=32×8=256, 11038 total steps, seed=42, sparsity=50%, speech every 3
- TopK defaults: shuffled_prefix init, tau 1.0→0.2, freeze=0.6, mask_lr=3e-3, spread λ=0.01

> **Important architectural difference:** All experiments below were run on the old
> `learned-mat` branch, which applied matryoshka masking **only to normal FeedForward
> layers** (encoder local_ffn and decoder FFN). The MLPMixer FeedForwards (token_mix and
> channel_mix) were **unmasked**. The current codebase (`fixed-mrl`) applies ffn_mask to
> ALL FeedForward instances including the mixer, so these results are not directly
> comparable. The mixer masking may change optimal hyperparameters.

### E1: TopK vs Prefix Baseline (1 epoch)

| Width | d_ff | Prefix Baseline | **TopK (E1)** |
|-------|------|-----------------|---------------|
| Full | 2048 | **4.66** | 4.93 |
| 256 | 1024 | **4.66** | 4.95 |
| 128 | 512 | **4.68** | 5.03 |
| 64 | 256 | **4.80** | 5.19 |

TopK is +0.27 worse at full, +0.39 worse at d=64. The learned mask has extra overhead vs fixed prefix.

### E2: Freeze Fraction Ablation

| Width | d_ff | freeze=0.4 | freeze=0.5 | **freeze=0.6** | freeze=0.7 |
|-------|------|------------|------------|----------------|------------|
| Full | 2048 | 5.08 | 4.98 | **4.93** | 4.94 |
| 256 | 1024 | 5.14 | 5.01 | **4.95** | 4.96 |
| 128 | 512 | 5.25 | 5.12 | **5.03** | 5.07 |
| 64 | 256 | 5.43 | 5.29 | **5.19** | 5.24 |

freeze=0.6 is best overall. Too little freeze (0.4) hurts most. 0.7 is slightly worse than 0.6 (less learning time).

### E3: Tau Schedule Ablation

| Width | d_ff | tau 0.5→0.1 | **tau 1.0→0.2 (default)** | tau 2.0→0.3 |
|-------|------|-------------|---------------------------|-------------|
| Full | 2048 | 4.79 | 4.93 | **4.76** |
| 256 | 1024 | 4.82 | 4.95 | **4.78** |
| 128 | 512 | 4.93 | 5.03 | **4.88** |
| 64 | 256 | 5.09 | 5.19 | **5.02** |

Both alternatives beat the default! tau 2.0→0.3 (softer start, softer end) is best. tau 0.5→0.1 (sharper) is also strong. The default 1.0→0.2 is the worst of the three.

### E4: Mask LR Ablation

| Width | d_ff | lr=1e-3 | **lr=3e-3 (default)** | lr=1e-2 |
|-------|------|---------|-----------------------|---------|
| Full | 2048 | **4.80** | 4.93 | 5.10 |
| 256 | 1024 | **4.85** | 4.95 | 5.13 |
| 128 | 512 | **4.97** | 5.03 | 5.26 |
| 64 | 256 | **5.12** | 5.19 | 5.43 |

Lower LR (1e-3) is best. Higher LR (1e-2) hurts significantly. Default 3e-3 is middle.

### E5: Multi-Epoch (2 epochs)

| Width | d_ff | TopK 1 epoch | **TopK 2 epochs** |
|-------|------|--------------|--------------------|
| Full | 2048 | 4.93 | **4.36** |
| 256 | 1024 | 4.95 | **4.37** |
| 128 | 512 | 5.03 | **4.44** |
| 64 | 256 | 5.19 | **4.55** |

2 epochs dramatically improves everything. The 2-epoch d=64 (4.55) beats the 1-epoch prefix baseline d=64 (4.80).

### Summary: Best Single-Epoch Config

Based on the ablations, the optimal single-variable changes from default are:
- **tau 2.0→0.3** (E3): biggest win, −0.17 at d=64
- **lr 1e-3** (E4): −0.07 at d=64
- **freeze 0.6** (E2): already the default, confirmed best

A combined run with tau 2.0→0.3 + lr 1e-3 could potentially close the gap to prefix baseline further.

### C1-C5: Hyperparameter Combinations

All use combined best base: tau 2.0→0.3 + lr 1e-3 + freeze 0.6 + shuffled_prefix init.

| Width | d_ff | C1 (base) | C2 (f=0.7) | C3 (f=0.5) | **C4 (λ=0)** | C5 (λ=0.005) |
|-------|------|-----------|------------|------------|--------------|--------------|
| Full | 2048 | 5.03 | 5.10 | 4.99 | **4.82** | 5.03 |
| 256 | 1024 | 5.10 | 5.13 | 5.03 | **4.84** | 5.06 |
| 128 | 512 | 5.23 | 5.24 | 5.14 | **4.96** | 5.19 |
| 64 | 256 | 5.40 | 5.42 | 5.32 | **5.12** | 5.35 |

**Key finding: Removing the spread penalty (C4, λ=0) is the single biggest improvement.** C4 achieves 4.82 full / 5.12 d=64, beating all previous single-epoch topk runs and approaching the prefix baseline (4.66 / 4.80). The spread penalty was actively hurting — it pushes logits apart but interferes with the tau annealing schedule.

C1 (combined best with default spread) underperforms the individual E3/E4 ablations — the combination doesn't stack. This is because the spread penalty (λ=0.01) interacts poorly with tau 2.0→0.3.

Freeze=0.5 (C3) beats 0.6 (C1) and 0.7 (C2) when combined with slow tau, suggesting the softer tau schedule benefits from more learning time.

### Summary: Best Overall Single-Epoch TopK Config

| Width | d_ff | Prefix Baseline | Best TopK (C4) | Gap |
|-------|------|-----------------|----------------|-----|
| Full | 2048 | **4.66** | 4.82 | +0.16 |
| 256 | 1024 | **4.66** | 4.84 | +0.18 |
| 128 | 512 | **4.68** | 4.96 | +0.28 |
| 64 | 256 | **4.80** | 5.12 | +0.32 |

Best topk config: `--mrl-tau-start 2.0 --mrl-tau-end 0.3 --mrl-freeze-frac 0.6 --mrl-mask-lr 0.001 --mrl-spread-lambda 0.0`

> **Current equivalent:** `--mat-method topk --mat-tau-start 2.0 --mat-tau-end 0.3 --mat-freeze-frac 0.6 --mat-mask-lr 0.001 --mat-spread-lambda 0.0`

TopK is still ~0.16-0.32 behind prefix at 1 epoch, but 2-epoch topk (E5: 4.55 d=64) already beats 1-epoch prefix (4.80 d=64).

---

## Full-Matryoshka Experiments (with mixer masking)

All experiments below apply `ffn_mask` to ALL FeedForward layers including MLPMixer
token_mix and channel_mix. Baselines from `matryoshka_results.md`:
- **Prefix baseline** (full mat): 4.82 / 4.82 / 4.85 / 4.97 (full/2x/4x/8x)
- **No-matryoshka**: 4.52 (full only, no sub-models)

### R1–R5: Re-validation of Prior Findings

All use seed=42, 1 epoch, 88.5M params. Base topk defaults: tau 0.5→0.1, lr 3e-3,
λ=0.01, warmup 15%, freeze 20%, normal init.

| Width | d_ff | Prefix | R1 (default) | R2 (C4) | R3 (tau 0.5→0.1) | R4 (lr 3e-3) | R5 (λ=0.01) |
|-------|------|--------|-------------|---------|-------------------|-------------|-------------|
| Full | 2048 | 4.82 | 4.69 | 4.72 | 4.70 | 4.72 | 4.74 |
| 2x | 1024 | 4.82 | 4.94 | 4.76 | 4.75 | 4.74 | 4.78 |
| 4x | 512 | 4.85 | 5.40 | 4.92 | 4.88 | 4.85 | 4.96 |
| 8x | 256 | 4.97 | 5.81 | 5.17 | 5.05 | **5.01** | 5.14 |

**R1** = default topk (tau 0.5→0.1, lr 3e-3, λ=0.01, f=0.2, normal init).
**R2** = old C4 config (tau 2.0→0.3, lr 1e-3, λ=0, f=0.6).
**R3** = R2 but tau 0.5→0.1 (sharper). **R4** = R3 but lr 3e-3. **R5** = R4 but λ=0.01.

**Key finding shifts from old experiments (without mixer masking):**
- **Tau 0.5→0.1 (sharper) is now better** than 2.0→0.3 (softer). Reversed from before.
- **lr 3e-3 is now slightly better** than 1e-3 at sub-models. Reversed from before.
- **Spread penalty (λ=0.01) still hurts.** Confirmed.
- **Best non-saliency 1-epoch config**: tau 0.5→0.1, lr 3e-3, λ=0, f=0.6 (R4).

### N1–N3: Saliency Scale Sweep

Saliency init with best R-series config (tau 0.5→0.1, lr 3e-3, λ=0, f=0.6, 10% warmup).

| Width | d_ff | R4 (normal init) | N1 (scale=1.0) | N2 (scale=0.5) | N3 (scale=2.0) |
|-------|------|-----------------|----------------|----------------|----------------|
| Full | 2048 | 4.72 | **4.62** | 4.72 | 4.64 |
| 2x | 1024 | 4.74 | **4.63** | 4.74 | 4.64 |
| 4x | 512 | 4.85 | **4.72** | 4.78 | 4.73 |
| 8x | 256 | 5.01 | **4.88** | 4.94 | **4.88** |

**Saliency init is transformative.** Scale=1.0 and 2.0 are essentially tied; scale=0.5
is too weak. All saliency runs beat prefix at every width.

### N7: Multi-Epoch (2 epochs, non-saliency)

| Width | d_ff | R4 (1 epoch) | N7 (2 epochs) |
|-------|------|-------------|--------------|
| Full | 2048 | 4.72 | **4.49** |
| 2x | 1024 | 4.74 | **4.50** |
| 4x | 512 | 4.85 | **4.63** |
| 8x | 256 | 5.01 | **4.76** |

### S1–S10: Saliency Deep Dive

All use: tau 0.5→0.1, lr 3e-3, λ=0, f=0.6, saliency scale=1.0, seed=42.

| ID | Experiment | Full | 2x | 4x | 8x |
|----|-----------|------|-----|-----|-----|
| S1 | Saliency-only (f=1.0, 10% warmup) | 4.67 | 4.67 | 4.73 | 4.88 |
| S2 | Saliency + freeze=0.5, 10% warmup | 4.61 | 4.63 | 4.73 | 4.93 |
| S3 | Saliency + Gumbel, 10% warmup | 4.66 | 4.66 | 4.72 | 4.86 |
| S4 | Saliency, **20% warmup** | 4.60 | 4.61 | 4.69 | 4.85 |
| S5 | Saliency, **30% warmup** | 4.60 | 4.61 | 4.68 | 4.84 |
| **S6** | **Saliency, 40% warmup** | **4.59** | **4.60** | **4.67** | **4.84** |
| S7 | Saliency, 50% warmup | 4.61 | 4.62 | 4.71 | 4.89 |
| S8 | Saliency + Gumbel, 40% warmup | 4.63 | 4.64 | 4.71 | 4.88 |
| S9 | Saliency-only (f=1.0), 40% warmup | **4.59** | **4.60** | **4.67** | **4.84** |
| **S10** | **Saliency, 40% warmup, 2 epochs** | **4.38** | **4.39** | **4.45** | **4.60** |

### Summary: Best Configs

| Config | Full | 2x | 4x | 8x | Notes |
|--------|------|-----|-----|-----|-------|
| No matryoshka | 4.52 | - | - | - | No sub-models |
| Prefix (static) | 4.82 | 4.82 | 4.85 | 4.97 | Simple, no learning |
| **Best topk 1ep (S9)** | **4.59** | **4.60** | **4.67** | **4.84** | Saliency-only, 40% warmup, no mask learning |
| Best topk 2ep (S10) | 4.38 | 4.39 | 4.45 | 4.60 | S9 config at 2 epochs |

**Optimal config (S9 — saliency-only)**: `--mat-method topk --mat-init-mode saliency --mat-saliency-scale 1.0 --mat-warmup-frac 0.4 --mat-freeze-frac 1.0`

S9 is the recommended default: saliency picks the neurons during 40% warmup, then hard masks are frozen for the remaining 60%. No mask optimizer, no tau annealing — simplest and best. Mask learning (S6, freeze=0.6) achieves identical results with more complexity.

### Experiments Still Needed

#### Export Verification

| ID | Experiment | Details |
|----|-----------|---------|
| N9 | **Export correctness** | Extract learned hard masks, slice FFN weights, verify: loads, PPL matches masked eval, size matches `_estimate_mrl_params`, works for all 3 widths (256, 128, 64) |
| N10 | **Export + quantization** | INT4 quantization on exported sub-models (smaller d_ff may interact with group_size=32 alignment) |

## Key Lessons Learned

### From old experiments (without mixer masking)
1. **Spread penalty hurts.** λ=0 was the single biggest improvement (C4). The penalty fights the tau schedule.
2. **Multi-epoch is transformative.** 2 epochs closes the gap to prefix baseline entirely.
3. **Combinations don't always stack.** C1 (tau+lr combined) was worse than individual ablations due to spread penalty interaction.

### New findings (with mixer masking)
4. **Saliency init is the most important hyperparameter.** It alone provides +0.1–0.13 PPL improvement over normal init at every width.
5. **Warmup fraction matters more than expected.** 40% warmup is optimal — gives the saliency estimator enough data to rank neurons accurately. Below 20% is noticeably worse; above 50% wastes too much training time on full-model-only.
6. **Mask learning adds almost nothing with good saliency.** S9 (saliency-only, freeze=1.0) matches S6 (saliency + learning). The gradient-importance ranking is already near-optimal.
7. **Gumbel noise hurts.** Per-item noise during soft phase adds variance without benefit. Saliency provides enough exploration.
8. **Tau and LR priors reversed with mixer masking.** Sharper tau (0.5→0.1) and higher LR (3e-3) now beat their softer/lower counterparts. More masked parameters may need faster convergence.
9. **Spread penalty still hurts.** Confirmed across all configs.
10. **2-epoch saliency (S10) is the best model overall.** 4.38 full PPL beats the no-matryoshka baseline (4.52) by 0.14, while providing 3 additional sub-models down to 8x compression.

## Historical Notes

Earlier iterations (on the `learned-mat-old` branch) applied masks to d_model at the output logit projection, not inside FFN. This was fundamentally wrong — sub-models shared all internal computation. Evidence: shuffled_prefix gave different results than prefix with output-only masking (d=128: 4.63 vs 4.55 PPL), proving the model exploited contiguous prefix structure rather than learning genuine sub-networks. Moving to FFN interior masking fixed this.
