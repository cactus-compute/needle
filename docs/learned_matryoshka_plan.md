# Learned Matryoshka (TopK FFN Masking)

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

## CLI Args

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

## Experiments Still Needed

### Hyperparameter Combinations (1 epoch each)

1. **Combined best**: tau 2.0→0.3 + lr 1e-3 + freeze 0.6 — stack the two best single-variable winners
2. **Combined best + freeze 0.7**: tau 2.0→0.3 shifts more learning into early steps, so longer freeze may now work
3. **Combined best + freeze 0.5**: test the other direction
4. **No spread penalty**: combined best config but λ=0 — spread penalty may hurt with softer tau
5. **Lower spread**: combined best but λ=0.005 — half the current penalty

### Saliency Initialization

Saliency init uses a **10% warmup phase** (full-model-only, no MRL) to accumulate per-FFN-neuron gradient importance via `||grad(down_proj)[:, j]||²` EMA across all layers. At the warmup→learning transition, these scores are converted to mask logits via quantile normalization. The **freeze phase is 50%** (shorter than the 60% default for shuffled, since saliency gives a better starting point and we want more learning time to refine).

**Saliency logit conversion** requires a temperature/scale parameter (`--mrl-saliency-scale`) to control how strongly the initial ranking influences mask selection:
- High scale (e.g., 2.0): strong prior — top-saliency neurons are almost certainly selected, bottom ones almost certainly pruned. Less room for the optimizer to override.
- Low scale (e.g., 0.5): weak prior — all neurons start near the sigmoid boundary, optimizer has full freedom. Saliency just provides a slight bias.
- The conversion: `logits[j] = scale * (1 - 2 * rank[j] / (d_ff - 1))` where `rank[j]` is the saliency rank (0 = most important).

6. **Saliency init (scale=1.0)**: 10% warmup, 50% freeze, tau 2.0→0.3, lr 1e-3, saliency-scale 1.0. Baseline saliency config.
7. **Saliency init (scale=0.5)**: Same but weaker prior — more optimizer freedom.
8. **Saliency init (scale=2.0)**: Same but stronger prior — trust the saliency ranking more.
9. **Saliency init + no freeze**: Saliency scale=1.0 but freeze=0.0 — test if saliency-initialized masks are good enough to stay soft throughout.
10. **Saliency init + 60% freeze**: Saliency scale=1.0 but freeze=0.6 — compare against the 50% default.
11. **Saliency vs shuffled at 2 epochs**: Both with combined best config. Does the saliency advantage persist or wash out?

### Export Verification

12. **Export correctness**: For a topk-trained checkpoint, extract the learned hard masks, gather the active FFN neuron indices, and slice gate_proj/up_proj/down_proj to produce a smaller model. Verify:
    - Exported model loads and runs inference without errors
    - Exported model's PPL matches the eval-time masked PPL (should be identical)
    - Exported model size matches `_estimate_mrl_params` predictions
    - Export works for all 3 MRL dims (256, 128, 64)
13. **Export + quantization**: Verify INT4 quantization works on exported sub-models (smaller d_ff may interact with group_size=32 alignment)

## Historical Notes

Earlier iterations (on the `learned-mat` branch) applied masks to d_model at the output logit projection, not inside FFN. This was fundamentally wrong — sub-models shared all internal computation. Evidence: shuffled_prefix gave different results than prefix with output-only masking (d=128: 4.63 vs 4.55 PPL), proving the model exploited contiguous prefix structure rather than learning genuine sub-networks. Moving to FFN interior masking fixed this.
