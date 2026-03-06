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

### TopK (shuffled init, this branch, 75.6M params, 1378 steps)

| Width | d_ff | TopK Shuffled |
|-------|------|---------------|
| Full | 2048 | 9.62 |
| 256 | 1024 | 8.98 |
| 128 | 512 | 8.43 |
| 64 | 256 | 8.86 |

> Note: Different config from baseline (75.6M vs 87.7M params, batch=256 vs 256, 1378 vs 11038 steps). Needs rerun with matching config for fair comparison.

## Experiments Still Needed

1. **Apples-to-apples comparison**: TopK shuffled vs prefix baseline with identical config (87.7M params, 8 enc / 4 dec, batch 256, 1 epoch, same sparsity/speech settings)
2. **Ablation: freeze fraction**: Test 0.4, 0.5, 0.6, 0.7
3. **Ablation: tau schedule**: Test different start/end temperatures
4. **Ablation: mask LR**: Test 1e-3, 3e-3, 1e-2
5. **Multi-epoch**: Does topk improve relative to prefix with longer training?
6. **Export**: Verify learned neuron selection exports correctly (gather active neurons from gate/up/down projections)
7. **Saliency init**: Initialize mask logits from accumulated gradient importance during warmup phase

## Historical Notes

Earlier iterations (on the `learned-mat` branch) applied masks to d_model at the output logit projection, not inside FFN. This was fundamentally wrong — sub-models shared all internal computation. Evidence: shuffled_prefix gave different results than prefix with output-only masking (d=128: 4.63 vs 4.55 PPL), proving the model exploited contiguous prefix structure rather than learning genuine sub-networks. Moving to FFN interior masking fixed this.
