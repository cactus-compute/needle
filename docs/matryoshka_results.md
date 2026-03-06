# FFN Interior Matryoshka — Results

## Approach

Heterogeneous batch MRL: a per-batch-item prefix mask zeros out FFN neurons beyond the target width. Single forward-backward pass, single optimizer step. All MRL widths trained simultaneously.

- **Full width**: d_ff = 2048 (all neurons active)
- **MRL 256**: d_ff = 1024 (first 1024 neurons)
- **MRL 128**: d_ff = 512 (first 512 neurons)
- **MRL 64**: d_ff = 256 (first 256 neurons)

d_model, attention heads, embeddings, and mixer are unchanged across all widths.

Two modes:
- **Unique input** (default): each batch item is a unique sample, assigned a fixed width based on its position in the batch. Full data diversity per epoch.
- **Shared input** (`--mrl-shared-input`): each unique sample is repeated across all widths. Smaller unique batch but every sample trains every width. Epoch = 4x more steps to see all data.

## Training Config

All runs use the same architecture and hyperparameters:

| Setting | Value |
|---|---|
| Parameters | 87,666,176 |
| d_model | 512 |
| d_ff | 2048 |
| Heads (KV) | 8 (8) |
| Layers | 8 enc / 4 dec |
| Memory slots | 64 |
| Activation | dReLU |
| Dtype | bfloat16 |
| Devices | 8 (TPU v6e-8) |
| Batch | 32 x 8 = 256 |
| LR (Adam) | 0.0003 x 8 = 0.0024 |
| LR (Muon) | 0.02 -> 0.0566 |
| Sparsity | 50% target (gradual cubic, group=32) |
| Speech | every 3 text steps |

## Results (1 epoch, val PPL)

| Model | Main (output-only slice) | FFN unique input | FFN shared input |
|---|---|---|---|
| Full (d_ff=2048) | **4.52** | 4.66 | 5.11 |
| MRL 256 (d_ff=1024) | - | **4.66** | 5.11 |
| MRL 128 (d_ff=512) | - | **4.68** | 5.11 |
| MRL 64 (d_ff=256) | - | **4.80** | 5.20 |
| Quant (INT4 g32) | **4.59** | 4.75 | 5.20 |

> Main branch MRL PPLs omitted — output-only slice MRL does not constrain internal computation, so sub-model PPLs are misleading.

| Metric | Main | FFN unique | FFN shared |
|---|---|---|---|
| MRL method | Output-only slice | FFN interior mask | FFN interior mask |
| Total steps | 11,038 | 11,038 | 44,156 |
| Unique items/batch | 32/dev | 32/dev | 8/dev x 4 |
| Speed | ~19.3 it/s | 11.8 it/s | 20.9 it/s |
| Wall time | ~10 min | ~16 min | ~36 min |
| Avg train loss | 4.43 | 4.42 | 3.12 |
| Final train loss | 1.51 | 1.53 | 1.66 |
| Text val PPL | **4.52** | 4.66 | 5.11 |
| Quant val PPL | **4.59** | 4.75 | 5.20 |
| Weight sparsity | 50.02% | 50.01% | 68.28% |
| Speech val PPL | 87.26 | 92.59 | 78.64 |
| Speech WER | 1.0000 | 0.9695 | 0.9618 |
| Throughput | 135.7 tok/s | 46.3 tok/s | 156.8 tok/s |

## Analysis

**Main branch full-model PPL (4.52) is the best**, as expected — it doesn't spend any capacity on sub-model training. FFN unique input (4.66) pays a +0.14 PPL cost for genuinely training 4 widths simultaneously.

**FFN unique input beats shared input** (4.66 vs 5.11). Both see the same total unique data per epoch, but unique input has 4x data diversity per step (32 unique items vs 8 repeated 4x).

**Shared input over-prunes** (68% vs 50% target) because the pruning schedule fractions apply to 4x more total steps, creating a wider absolute pruning window.

**FFN interior MRL sub-models are tight.** With unique input, the d_ff=256 sub-model (1/8 of FFN, 68.7M params) is only +0.14 PPL worse than full. This is a genuine sub-network — unlike main's output-only approach which shares all internal computation.

**Main is faster** (~19 it/s) because it runs one full forward + cheap logit slicing. FFN unique input (11.8 it/s) does one forward with heterogeneous masking but can't exploit XLA batch optimizations as well. FFN shared input (20.9 it/s) is fastest per-step due to repeated inputs.

### Recommendation

Use **unique input** (default, no flag) for best quality. The +0.14 PPL cost over main buys genuinely independent sub-models that can be exported at reduced FFN width.

## Training Speed

| Approach | it/s | Notes |
|---|---|---|
| Main branch (output-only slice) | ~19.3 | 1 forward + logit slicing (no true sub-models) |
| FFN unique input | 11.8 | Single forward, heterogeneous batch |
| FFN shared input | 20.9 | Single forward, repeated inputs (XLA optimizes) |

## Generation Samples (Unique Input)

```
<story>[Once upon a time] where gan the vessel was calculated with a sensible light
<story>[The little dog] and the big boots and buntline furnished with a very little red buttons and si
<story>[She was very happy because] but where if you ever see her again i know she has a very special thing i'm go
```

## Notes

- All runs are 1-epoch on TinyStories + LibriSpeech. Multi-epoch runs would improve all metrics.
- Speech WER is poor across all methods — expected with interleaved training and limited speech data.
- The heterogeneous batch approach uses the same memory as a single full-width forward (~1x). No overhead from MRL.
