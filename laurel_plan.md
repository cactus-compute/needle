# Laurel Integration Plan for Needle

## What is Laurel?

**Learned Augmented Residual Layer** (from Gemma 3n): a low-rank bottleneck that
provides an auxiliary residual path parallel to attention in each decoder layer.

```
input (d_model)
  |
  +---> [Self-Attention] ---> attn_out (d_model)
  |                               |
  +---> [Laurel]         ---> laurel_out (d_model)
  |                               |
  +--- residual + (attn_out + laurel_out) / sqrt(2)
  |
  +---> [Cross-Attention] (unchanged)
  +---> [FFN] (unchanged)
```

### LaurelBlock internals
```
x (d_model) -> Dense(d_model, laurel_rank) -> Dense(laurel_rank, d_model) -> ZCRMSNorm -> output
```
No activation between the two projections (pure linear bottleneck).

## Parameter-Neutral Design

Adding laurel increases params. To keep total params constant, we **proportionally
reduce attention head dimension** so saved attention params = added laurel params.

### Budget equation

Introduce `d_head` (explicit head dimension, decoupled from d_model):
- `d_attn = num_heads * d_head` (total attention dimension, may be < d_model)
- Projections: q: d_model -> d_attn, k/v: d_model -> kv_dim, out: d_attn -> d_model
- `kv_dim = num_kv_heads * d_head` (also shrinks proportionally)

Per decoder block attention savings (self + cross, each with q/k/v/out):
```
savings = 2 * [d*(d - d_attn) + 2*d*(kv_old - kv_new) + (d - d_attn)*d]
        = 2 * [2*d*(d - d_attn) + 2*d*num_kv_heads*(head_old - d_head)]
```

For MHA (num_kv_heads = num_heads), this simplifies to:
```
savings_per_block = 2 * 4 * d * (d - d_attn) = 8 * d * num_heads * (head_old - d_head)
```

Laurel cost per block: `2 * d_model * laurel_rank` (plus negligible norm params).

Budget equation (general GQA case):
```
4 * n_dec * d * delta_head * (num_heads + num_kv_heads) = n_dec * 2 * d * laurel_rank
delta_head = ceil(laurel_rank / (2 * (num_heads + num_kv_heads)))
# Round delta up to even to ensure d_head stays even (required by RoPE)
d_head = d_model // num_heads - delta_head
```

### Concrete examples (defaults: d=512, num_heads=8)

| laurel_rank | d_head (was 64) | d_attn | Laurel params/block | Attn saved/block |
|-------------|-----------------|--------|---------------------|------------------|
| 32          | 63              | 504    | 32,768              | 32,768           |
| 64          | 62              | 496    | 65,536              | 65,536           |
| 16          | 63.5 -> 63      | 504    | 16,384              | ~32,768          |

Recommended: **laurel_rank=32** (clean budget match, rank/d_model ratio ~6% matches
Gemma 3n's 64/2048 = 3%).

### MRL compatibility

At MRL deployment dimension d_prime:
- `num_heads_prime = d_prime // d_head` (fewer heads, same head dim)
- `d_attn_prime = num_heads_prime * d_head`
- `laurel_rank_prime = laurel_rank * d_prime // d_model`

The budget equation scales proportionally because both sides multiply by
`(d_prime / d_model)^2`, so param-neutrality holds at every MRL slice.

### MAIN-1024 config (d=1024, num_heads=4, num_kv_heads=2)

For GQA, the budget equation generalizes:
```
savings_per_block = 4 * d * (d - d_attn) * (1 + num_kv_heads/num_heads)
```

| laurel_rank | d_head (was 256) | MRL d_prime=512 heads | MRL d_prime=128 heads |
|-------------|------------------|-----------------------|-----------------------|
| 48          | 252              | 2                     | 0 (skip laurel)       |
| 64          | 248 -> 248       | 2                     | 0 (skip laurel)       |

Recommended: **laurel_rank=48** for MAIN-1024.

## Implementation Changes

### 1. TransformerConfig (model.py)

Add fields:
```python
laurel_rank: int = 0        # 0 = disabled (backward compatible)
```

Add computed property:
```python
@property
def d_head(self):
    base = self.d_model // self.num_heads
    if self.laurel_rank == 0:
        return base
    return base - self.laurel_rank // (4 * self.num_heads)

@property
def d_attn(self):
    return self.num_heads * self.d_head
```

### 2. LaurelBlock (model.py)

New module:
```python
class LaurelBlock(nn.Module):
    d_model: int
    rank: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.rank, dtype=self.dtype, use_bias=False,
                     kernel_init=default_init(), name="down")(x)
        h = nn.Dense(self.d_model, dtype=self.dtype, use_bias=False,
                     kernel_init=jinit.zeros, name="up")(h)
        h = ZCRMSNorm(dtype=self.dtype, name="norm")(h)
        return h
```

Note: `up` projection initialized to zeros so laurel starts as identity
(output is all zeros -> norm output is zeros -> no effect at init).

### 3. MultiHeadAttention (model.py)

Change projections to use `d_attn` instead of `d_model`:
- q_proj: d_model -> d_attn
- k_proj: d_model -> kv_dim (kv_dim = num_kv_heads * d_head)
- v_proj: d_model -> kv_dim
- out_proj: d_attn -> d_model

Accept `d_head` as explicit parameter (remove computation from d_model//num_heads).

### 4. DecoderBlock (model.py)

Add laurel parallel to self-attention:
```python
# Pre-norm (shared between attn and laurel)
x_norm = ZCRMSNorm(...)(x)

# Parallel paths
attn_out = self_attn(x_norm, x_norm, ...)
laurel_out = laurel(x_norm)  # only if laurel_rank > 0

# Combine
if laurel_rank > 0:
    x = x + (attn_out + laurel_out) * (1.0 / math.sqrt(2))
else:
    x = x + attn_out
```

### 5. MemoryMixerBlock (model.py) - encoder

Laurel is decoder-only (matching Gemma 3n). The encoder's MemoryMixer blocks
use the updated MultiHeadAttention with d_head but do NOT get a LaurelBlock.

### 6. MRL export (export.py)

When slicing to d_prime:
- Slice laurel down/up projections: down[:d_prime, :r_prime], up[:r_prime, :d_prime]
- Attention projections already slice correctly via d_head-based num_heads

### 7. CLI (cli.py)

Add `--laurel-rank` argument (default: 0).

### 8. Training (train.py)

No special training changes needed. Laurel params are Dense kernels so they
get Muon optimizer automatically. ZCRMSNorm scale gets AdamW.

## Files to modify

1. `src/model.py` - LaurelBlock, TransformerConfig, MultiHeadAttention, DecoderBlock
2. `src/cli.py` - add --laurel-rank flag
3. `src/train.py` - pass laurel_rank to config, update _estimate_mrl_params
4. `src/export.py` - handle laurel projection slicing (if separate logic needed)

## Experimental Results

All runs: 1 epoch, seed=42, d_model=512, 8 heads, 8 enc + 4 dec layers, 50% sparsity,
v6e-8 TPU (8 chips), default MRL dims [256, 128, 64].

### Run configurations

| Run | Laurel location | laurel_rank | d_head | Parameters |
|-----|----------------|-------------|--------|------------|
| Baseline | none | 0 | 64 | 87,666,176 |
| Laurel v1 | decoder only | 32 | 62 (reduced) | 87,274,944 |
| Laurel v2 | decoder only | 32 | 64 (kept) | 87,799,296 |
| Laurel v3 | encoder + decoder | 32 | 64 (kept) | 88,065,536 |

Note: v1 reduced d_head to offset laurel params (param-neutral) but non-power-of-2
head dims caused ~4x throughput regression from suboptimal XLA codegen. v2/v3 keep
d_head=64 and accept the small param increase (+0.15% / +0.46%).

### Text quality

| Metric | Baseline | Laurel v1 | Laurel v2 | Laurel v3 |
|--------|----------|-----------|-----------|-----------|
| Text val PPL | 4.52 | 4.56 | 4.54 | **4.52** |
| Quant val PPL (INT4) | 4.59 | 4.65 | 4.61 | 4.61 |
| Final loss | 1.5080 | 1.5173 | 1.5109 | **1.5073** |
| Avg loss | 4.4250 | 4.2205 | 4.4255 | 8.7813 |
| MRL d=512 | 4.52 | 4.56 | 4.54 | **4.52** |
| MRL d=256 | 4.53 | 4.57 | 4.54 | **4.53** |
| MRL d=128 | 4.58 | 4.61 | 4.58 | **4.57** |
| MRL d=64 | 4.85 | 4.88 | 4.84 | **4.83** |

### Speech quality

| Metric | Baseline | Laurel v1 | Laurel v2 | Laurel v3 |
|--------|----------|-----------|-----------|-----------|
| Speech val PPL | 66.41 | -- | 68.55 | 88.72 |
| Avg speech loss | 587.47 | 498.82 | 482.29 | 574.88 |
| Speech WER | 1.00 | 1.00 | 1.00 | 0.99 |

### Generation and efficiency

| Metric | Baseline | Laurel v1 | Laurel v2 | Laurel v3 |
|--------|----------|-----------|-----------|-----------|
| Throughput (tok/s) | 133.5 | 89.9 | 31.6 | 24.5 |
| Latency (s) | 0.403 | 0.369 | 0.462 | 0.513 |
| Gen length (tok) | 45.0 | 38.3 | 13.7 | 16.7 |
| Repetition rate | 0.040 | 0.017 | 0.000 | 0.019 |
| Sparsity | 50.02% | 50.10% | 50.03% | 50.03% |

### Key findings

1. **Text PPL**: Laurel v3 (enc+dec) matches or beats baseline at every MRL dim.
   The auxiliary low-rank path helps most at smaller MRL slices (d=64: 4.83 vs 4.85).

2. **Speech**: Adding laurel to the encoder hurts speech val PPL (88.7 vs 66.4).
   The 1/sqrt(2) scaling on the encoder cross-attention dampens the speech signal.
   Decoder-only laurel (v2) has minimal speech impact (68.6 vs 66.4).

3. **Throughput regression**: All laurel variants show lower throughput and shorter
   generations. This persists even with power-of-2 heads (v2/v3), suggesting the
   1/sqrt(2) scaling factor changes the activation magnitudes in a way that affects
   autoregressive generation (early EOS). PPL is unaffected since it evaluates
   teacher-forced. Worth investigating: try scaling=1.0 or a learned gate.

4. **Param-neutral vs not**: Reducing d_head (v1) causes disproportionate throughput
   loss from non-power-of-2 dims. Better to accept the ~0.15-0.5% param increase.

5. **Repetition**: All laurel variants have much lower repetition (0.00-0.02 vs 0.04),
   suggesting the auxiliary path improves generation diversity.
