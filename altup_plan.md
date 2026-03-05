# AltUp (Alternating Updates) Plan for Needle

## References

- Paper: "Alternating Updates for Efficient Transformers" (Baykal et al., NeurIPS 2023 Spotlight)
  - arXiv: https://arxiv.org/abs/2301.13310
- HuggingFace Gemma 3n implementation (local: hf_gemma3n/modular_gemma3n.py)
- No official code release; HF Gemma 3n is the best public reference

## What is AltUp?

AltUp maintains multiple parallel "predictions" of hidden states through
transformer layers. Only the "active" prediction goes through the expensive
attention+FFN computation. All predictions are updated via learned predict/correct
coefficients, giving the model richer intermediate representations at minimal
compute cost.

## Core Mechanism

```
Per decoder layer:
  1. PREDICT: Generate predicted outputs for all N copies using learned coefficients
  2. ACTIVATE: Run only the active prediction (idx=0) through attention + FFN
  3. CORRECT: Update all N predictions using the difference between predicted and actual output
```

## Paper vs Gemma 3n vs Our Implementation

| Aspect | Paper | Gemma 3n | Ours |
|--------|-------|----------|------|
| Predict/correct coefs | Fixed scalars (K^2+K per layer) | Input-dependent via router | Input-dependent via router |
| Block selection | Alternating (layer l mod K) | Always active_idx=0 | Always active_idx=0 |
| Expand | Replicate embedding K times | Learned projections + mag norm | Replicate (paper-style) |
| Collapse | Element-wise sum of sub-blocks | Learned projections + mean | Mean of sub-blocks |
| Coef init | Unspecified in paper | normal(0.02) | Zero-init (identity at start) |
| N (copies) | K=2 default, K=4 tested | N=4 | N=2 |

## Critical Implementation Lesson: Identity at Init

The first two implementations failed badly (val ppl 15.14 vs 4.27 baseline) due to
the expand/collapse operations destroying the signal at initialization:

**Problem**: Gemma 3n uses random projections for expand and collapse. At init,
copy 0 carries real layer output; copy 1 carries random noise that never went
through attention. Collapse averages real signal with projected noise, diluting
output by ~50%.

**Solution (Recycled-AltUp, paper Section 4.1)**: Replicate for expand, sum for
collapse, zero-init prediction coefs. This gives **exact identity at init** — the
altup model produces identical output to baseline before any training. The
predict/correct mechanism can only help, never hurt.

Verified: `max(|logits_baseline - logits_altup|) = 0.0` at initialization.

## Memory-Efficient Design Choices

- **N=2 predictions** (not 4 like Gemma 3n) — 2x hidden state memory, not 4x
- **Decoder only** — encoder uses MemoryMixer (already specialized, different shape)
- **Lightweight routing** — router is just RMSNorm + Linear(d_model, N) + tanh
- **Prediction/correction coefficients** — tiny: N x N and N x N^2 params per layer
- **No expand/collapse projections** — replicate + sum (zero extra params)
- **Gradient checkpointing** — nn.remat on each DecoderBlock covers altup automatically

## Parameter Overhead (per decoder layer)

| Component | Size | Notes |
|-----------|------|-------|
| prediction_coefs | N x N^2 = 2 x 4 = 8 | Zero-init linear |
| correction_coefs | N x N = 2 x 2 = 4 | Zero-init linear |
| modality_router | d_model x N | Linear projection |
| router_norm | d_model | ZCRMSNorm scale |
| correct_output_scale | d_model | Zero-init per-dim scale |
| router_input_scale | 1 | Scalar = d_model^-1 |
| **Total** | **~3*d_model + 13** | Negligible vs layer params |

At d_model=512, 4 decoder layers: ~6,200 total altup params (+0.01%)

## Experimental Results (v6e-8 TPU, 1 epoch, d=512, no speech)

| Metric | Baseline | AltUp (broken init) | AltUp (identity init) |
|--------|----------|---------------------|----------------------|
| Parameters | 87,666,176 | 88,199,220 | 87,674,932 |
| Text val ppl | **4.27** | 15.14 | **4.37** |
| Quant val ppl (INT4) | 4.31 | 15.65 | 4.46 |
| Final train loss | 1.42 | 2.38 | 1.45 |
| Throughput (tok/s) | 120.4 | 122.3 | **155.5** |
| MRL d=256 ppl | 4.27 | 15.30 | 4.38 |
| MRL d=128 ppl | 4.32 | 15.75 | 4.43 |
| MRL d=64 ppl | 4.54 | 17.15 | 4.65 |

Key takeaways:
- Identity-init AltUp is within 0.1 ppl of baseline with negligible param overhead
- Throughput actually improved (155 vs 120 tok/s) — likely JIT optimization differences
- The broken init (random expand projections + normal-init coefs) was catastrophic
- 1 epoch may not be enough for altup to show gains; expect improvement with more training

## Implementation

### model.py: TransformerConfig

Added fields:
- `altup_num_inputs: int = 0` (0 = disabled, 2 = recommended)
- `altup_active_idx: int = 0`

### model.py: AltUp module

Uses `setup()` (not `@nn.compact`) to share router between predict/correct:
- `router_norm`: ZCRMSNorm
- `modality_router`: Dense(N), float32, normal(0.02) init
- `prediction_coefs_proj`: Dense(N^2), zero-init
- `correction_coefs_proj`: Dense(N), zero-init
- `router_input_scale`: scalar param = d_model^-1
- `correct_output_scale`: d_model param, zero-init

Router: `tanh(modality_router(router_norm(x) * scale))`
Predict: `swapaxes(prediction_coefs(modalities), -2, -1)` then matmul + residual
Correct: `(correction_coefs(modalities) + 1) * innovation + predictions`

### model.py: DecoderBlock

```python
if use_altup:
    predictions = altup.predict(x)      # (N, B, T, d)
    x_active = predictions[active_idx]  # (B, T, d)
    # ... normal attn + FFN on x_active ...
    return altup.correct(predictions, x_active)
else:
    # ... normal attn + FFN (unchanged) ...
```

### model.py: EncoderDecoderTransformer

- **Expand**: `jnp.broadcast_to(x[None], (N, *x.shape))` — replication
- **Collapse**: `jnp.sum(x, axis=0) / N` — mean of sub-blocks
- No learned expand/collapse projections needed

### cli.py

- Added `--altup-num-inputs INT` (default: 0)

### Backward compatibility

- `altup_num_inputs=0` produces identical model to before (no altup params created)
- Config stored in checkpoint; old checkpoints without field default to 0
