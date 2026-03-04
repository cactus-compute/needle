# ReLU² vs dReLU Experiment

**Setup:** param-matched at 173M, 50% sparsity, 1 epoch TinyStories, 8x TPU v6e

**Key change:** Replace gated dReLU (3 projections, d_ff=4x) with non-gated ReLU² (2 projections, d_ff=6x)

## Results

**ReLU² vs dReLU (baseline) — param-matched at 173M, 50% sparsity**

**Perplexity (lower is better):**
- Val PPL: 4.42 → **4.38** (-0.9%)
- Quant val PPL (INT4 g32): 4.48 → **4.41** (-1.6%)
- Train PPL: 4.37 → **4.27** (-2.3%)

**MRL sub-models (val PPL):**
- d=1024: 4.42 → **4.38** (-0.9%)
- d=512: 4.42 → **4.39** (-0.7%)
- d=256: 4.44 → **4.41** (-0.7%)
- d=128: 4.54 → **4.51** (-0.7%)
- d=64: 4.82 → **4.79** (-0.6%)

**Speed:**
- Throughput: 223.0 → **259.5 tok/s** (+16.4%)
- Latency: 0.230s → 0.230s (same)

**Generation quality:**
- Repetition rate: **0.046** → 0.101 (higher = worse)
- Gen length: 52.7 → 52.3 tok (same)

**Takeaway:** ReLU² improves perplexity at every scale and is 16% faster (one fewer projection per FFN), but shows higher bigram repetition in generations.

## Generation Samples

### dReLU (baseline)
- **"Once upon a time"** → , I want to take you all on an adventure!" answered the other. So, the two friends went on a trip t
- **"The little dog"** → hiss was safe and sound. " played, little dog!" said his friend, Lily the cat, " like Max, we should
- **"She was very happy because"** → she loves to play with it. But then, she saw a dog outside. The dog was scared of the dog. The dog

### ReLU²
- **"Once upon a time"** → with the big, round, ball. The owner of the park was a vendor. He was tall and full of things to do.
- **"The little dog"** → the dog, the dog and the cat, and the dog wag its tail. The dog was happy to see the friends. The mo
- **"She was very happy because"** → . he moved a little bit. It was a tunnel! The tunnel was dark and scary. there were many tunnels in

## Raw Results

### dReLU (baseline)

```
  Avg loss            67.6071
  Final loss           1.4744
  Train ppl              4.37
  Val ppl                4.42
  Quant val ppl          4.48  (INT4 g32)
  Sparsity             49.98%  (86,527,622/173,121,536)

  MRL exportable sub-models:
     dim     val ppl        params  heads
    1024        4.42   173,121,536      4  (full)
     512        4.42    52,953,088      2
     256        4.44    14,417,920      1
     128        4.54     4,456,448      1
      64        4.82     1,540,096      1

  Throughput          223.0 tok/s
  Latency              0.230s
  Gen length           52.7 tok
  Repetition            0.046
```

### ReLU²

```
  Avg loss            57.7758
  Final loss           1.4514
  Train ppl              4.27
  Val ppl                4.38
  Quant val ppl          4.41  (INT4 g32)
  Sparsity             49.98%  (86,533,433/173,121,536)

  MRL exportable sub-models:
     dim     val ppl        params  heads
    1024        4.38   173,121,536      4  (full)
     512        4.39    51,118,080      2
     256        4.41    14,024,704      1
     128        4.51     4,390,912      1
      64        4.79     1,540,096      1

  Throughput          259.5 tok/s
  Latency              0.230s
  Gen length           52.3 tok
  Repetition            0.101
```
