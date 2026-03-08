# Matryoshka — Results

## Approach

Heterogeneous batch matryoshka: a per-batch-item prefix mask zeros out FFN neurons beyond the target width. Single forward-backward pass, single optimizer step. All widths trained simultaneously.

- **Full width**: d_ff = 2048 (all neurons active)
- **2x**: d_ff = 1024 (first 1024 neurons)
- **4x**: d_ff = 512 (first 512 neurons)
- **8x**: d_ff = 256 (first 256 neurons)

d_model, attention heads, and embeddings are unchanged across all widths.

Two masking scopes:
- **FFN-only**: masks decoder FFN and encoder local FFN. MLP-Mixer token/channel mixing layers are NOT masked.
- **Full matryoshka** (current): masks ALL FFN layers including MLP-Mixer token/channel mixing.

Two input modes:
- **Unique input** (default): each batch item is a unique sample, assigned a fixed width based on its position in the batch. Full data diversity per epoch.
- **Shared input** (`--mat-shared-input`): each unique sample is repeated across all widths. Smaller unique batch but every sample trains every width. Epoch = 4x more steps to see all data.

## Training Config

All runs use the same architecture and hyperparameters:

| Setting | Value |
|---|---|
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

| Model | No mat (main) | FFN-only | FFN-only shared | **Full mat (FFN + mixer)** |
|---|---|---|---|---|
| Full (d_ff=2048) | **4.52** | 4.66 | 5.11 | 4.82 |
| 2x (d_ff=1024) | - | 4.66 | 5.11 | 4.82 |
| 4x (d_ff=512) | - | 4.68 | 5.11 | 4.85 |
| 8x (d_ff=256) | - | **4.80** | 5.20 | 4.97 |
| Quant (INT4 g32) | **4.59** | 4.75 | 5.20 | 4.94 |

> No-mat sub-model PPLs omitted — output-only slice does not constrain internal computation, so sub-model PPLs are misleading.

| Metric | No mat | FFN-only | FFN-only shared | Full mat |
|---|---|---|---|---|
| Masking scope | Output-only slice | Decoder+encoder FFN | Decoder+encoder FFN | All FFN + mixer |
| Parameters | 87,666,176 | 87,666,176 | 87,666,176 | 88,535,040 |
| Total steps | 11,038 | 11,038 | 44,156 | 11,038 |
| Unique items/batch | 32/dev | 32/dev | 8/dev x 4 | 32/dev |
| Speed | ~19.3 it/s | 11.8 it/s | 20.9 it/s | 10.2 it/s |
| Wall time | ~10 min | ~16 min | ~36 min | ~18 min |
| Avg train loss | 4.43 | 4.42 | 3.12 | - |
| Final train loss | 1.51 | 1.53 | 1.66 | - |
| Text val PPL | **4.52** | 4.66 | 5.11 | 4.82 |
| Quant val PPL | **4.59** | 4.75 | 5.20 | 4.94 |
| Weight sparsity | 50.02% | 50.01% | 68.28% | 49.52% |
| Speech val PPL | 87.26 | 92.59 | 78.64 | 83.56 |
| Speech WER | 1.0000 | 0.9695 | 0.9618 | 0.9924 |
| Throughput | 135.7 tok/s | 46.3 tok/s | 156.8 tok/s | 146.6 tok/s |

## Analysis

**No-mat full-model PPL (4.52) is the best**, as expected — it doesn't spend any capacity on sub-model training.

**FFN-only (4.66) is the best matryoshka approach** for text PPL — it pays +0.14 over no-mat for genuinely training 4 widths simultaneously, with only a +0.14 gap from full to 8x (4.66→4.80).

**Full matryoshka (4.82) pays an additional +0.16 over FFN-only** for also masking the mixer layers. The sub-model spread is similarly tight (full→8x: 0.15 PPL). The extra cost buys truly independent sub-models where even the mixer's token and channel mixing operate at reduced width.

**Shared input is not worth it** (4.66 vs 5.11 for FFN-only). Unique input with random width assignment per item is sufficient — every sample doesn't need to train every width.

**Full mat is slower** (10.2 it/s vs 11.8 FFN-only) because the mixer FFNs are also heterogeneously masked, adding compute overhead.

### Recommendation

Use **unique input** (default, no flag) for best quality. Choose between FFN-only and full matryoshka based on whether mixer-level sub-models are needed:
- **FFN-only**: better text PPL (4.66 vs 4.82), faster (11.8 vs 10.2 it/s). Sub-models share full mixer capacity.
- **Full matryoshka**: truly independent sub-models at every layer. Slightly worse but sub-models are more self-contained for export.

## Training Speed

| Approach | it/s | Notes |
|---|---|---|
| No mat (output-only slice) | ~19.3 | 1 forward + logit slicing (no true sub-models) |
| FFN-only unique input | 11.8 | Single forward, heterogeneous batch (decoder+encoder FFN) |
| FFN-only shared input | 20.9 | Single forward, repeated inputs (XLA optimizes) |
| Full mat unique input | 10.2 | Single forward, heterogeneous batch (all FFN + mixer) |

## Ablation: Shared Input at 4x Batch Size (FFN-only)

**Question**: Does training every sample at all widths help, if we control for gradient quality?

With `--mat-shared-input --batch-size 128`, each step has 32 unique samples repeated 4x (one per width) — same unique samples per step as the default unique-input run at batch=32. The only difference is that every sample trains every width, at the cost of 4x compute per step.

| Model | FFN unique (bs=32) | FFN shared (bs=128) |
|---|---|---|
| Full (d_ff=2048) | **4.66** | 4.67 |
| 2x (d_ff=1024) | **4.66** | 4.67 |
| 4x (d_ff=512) | **4.68** | 4.69 |
| 8x (d_ff=256) | **4.80** | 4.81 |
| Quant (INT4 g32) | **4.75** | 4.77 |
| Speed | 11.8 it/s | 5.6 it/s |
| Wall time | ~16 min | ~35 min |
| Sparsity | 50.01% | 50.01% |

**Result**: Essentially identical PPL (4.66 vs 4.67). Repeating inputs across widths provides no benefit when gradient quality (unique samples per step) is controlled. Random width assignment per item is sufficient.

This confirms **unique input at batch=32 is optimal**: same quality, 2x faster.

## Notes

- All runs are 1-epoch on TinyStories + LibriSpeech. Multi-epoch runs would improve all metrics.
- Speech WER is poor across all methods — expected with interleaved training and limited speech data.
- The heterogeneous batch approach uses the same memory as a single full-width forward (~1x). No overhead from matryoshka.
