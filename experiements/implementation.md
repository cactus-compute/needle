# Query-CTC Non-Autoregressive Decoding for Needle

Reference: NARVL (arXiv 2403.02249) — Non-autoregressive Sequence-to-Sequence
Vision-Language Models, adapted from VLM to pure text generation.

Goal: Replace autoregressive token-by-token decoding with a single parallel
forward pass through learnable query tokens, trained with Query-CTC loss.
Target ~5x inference speedup on TinyStories generation.

## Background

Standard needle decoding: generate T tokens sequentially, each conditioned on
all previous tokens via causal mask. Cost = O(T) decoder forward passes.

NAR decoding: feed N learnable query embeddings into the decoder (no causal
mask), predict all output positions in one forward pass. CTC loss handles the
alignment between N queries and T target tokens (N >= T). The collapse
operation B removes blanks and deduplicates consecutive tokens to recover the
target. Cost = O(1) decoder forward passes.

Key difference between Q-CTC and standard CTC: standard CTC marginalizes over
alignment paths from encoder frames to output. Q-CTC marginalizes over paths
from *learnable query indices* to output tokens. The query embeddings
themselves are learned parameters optimized alongside model weights.

For deeper, more thorough reference at Q-CTC, look at the copies in this repo of
the paper this is derived from, `2304.pdf`, and the copy of their codebase `NARVL`.

## Architecture Changes

### 1. TransformerConfig additions (`model.py`)

```python
@dataclass
class TransformerConfig:
    ...
    # NAR fields (ignored by AR path)
    num_queries: int = 0          # 0 = AR mode, >0 = NAR mode
    blank_token_id: int = 8192    # appended to vocab for CTC blank
```

When `num_queries > 0`, the model operates in NAR mode. The blank token extends
the vocab by 1 (8192 -> 8193) for CTC alignment. The embedding table grows by
one row. The blank token is never part of the final output — it's only used
during CTC loss computation and is removed during collapse decoding.

### 2. Learnable Query Embeddings (`model.py`)

Add to `EncoderDecoderTransformer.setup()`:

```python
if self.config.num_queries > 0:
    self.query_embed = self.param(
        "query_embed",
        jinit.normal(stddev=0.02),
        (self.config.num_queries, self.config.d_model),
    )
```

These are a `(num_queries, d_model)` parameter — not an `nn.Embed` lookup
table, just a raw learned tensor. Each "query" is a fixed position in the
decoder input sequence. During forward, they are broadcast to batch size.

### 3. NAR forward path (`model.py`)

New method `forward_nar()` on `EncoderDecoderTransformer`:

```python
def forward_nar(self, src, src_mask=None):
    # Encode as normal
    encoder_out = self.encode(src, src_mask=src_mask)

    # Expand queries to batch: (1, N, d) -> (B, N, d)
    B = src.shape[0]
    queries = jnp.broadcast_to(
        self.query_embed[None] * self.embed_scale,
        (B, self.config.num_queries, self.config.d_model),
    )

    # Decoder with NO causal mask (bidirectional self-attention)
    rope = self._rope(self.config.num_queries)
    x = self.decoder(queries, encoder_out, self_mask=None, cross_mask=None, rope=rope)

    # Project to vocab+blank (8193 classes)
    x_f32 = x.astype(jnp.float32)
    emb = self.embedding.embedding  # (8192, d)

    # Blank token has its own learned projection vector
    blank_proj = ... # (1, d) — separate param or last row of extended embedding
    full_emb = jnp.concatenate([emb, blank_proj], axis=0)  # (8193, d)
    logits = x_f32 @ full_emb.T  # (B, N, 8193)
    return logits, encoder_out
```

Key differences from AR `forward()`:
- Input to decoder = learned queries, NOT target token embeddings
- No causal mask — all queries attend to each other bidirectionally
- Output vocab includes blank token (8193 classes)
- Single forward pass produces all N positions simultaneously

### 4. Blank token projection (`model.py`)

Two options for the blank token's embedding vector:

**Option A (simpler):** Extend `nn.Embed` from `vocab_size` to `vocab_size + 1`.
The extra row is the blank embedding. Tied output projection uses the full
(8193, d) table. Downside: AR path now has a dead row.

**Option B (cleaner):** Keep embedding at 8192. Add a separate learned parameter
`blank_embed = self.param("blank_embed", jinit.normal(0.02), (1, d_model))`
and concatenate it at projection time. AR path is completely unchanged.

Recommend **Option B** — zero impact on existing AR training/checkpoints.

## Training Changes

### 5. CTC loss function (`train.py`)

```python
def ctc_loss(logits, targets, blank_id, pad_id):
    """
    logits:  (B, N, V+1) — decoder output over queries
    targets: (B, T)       — ground truth token IDs (padded)

    Returns scalar loss.
    """
    B, N, V = logits.shape
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # (B, N, V+1)

    # Transpose to (N, B, V+1) for CTC convention (time, batch, classes)
    log_probs = jnp.transpose(log_probs, (1, 0, 2))

    # Input lengths: all queries are valid (no padding in query sequence)
    input_lengths = jnp.full((B,), N, dtype=jnp.int32)

    # Target lengths: count non-pad tokens
    target_lengths = jnp.sum(targets != pad_id, axis=1).astype(jnp.int32)

    # Use optax.ctc_loss (or jax-native implementation)
    loss = optax.ctc_loss(log_probs, input_lengths, targets, target_lengths,
                          blank_id=blank_id)

    return jnp.mean(loss)
```

Notes:
- `optax.ctc_loss` expects `(T, B, C)` log-probabilities — transpose from our
  `(B, N, C)` output.
- `input_lengths` = `num_queries` for every sample (queries are never padded).
- `target_lengths` = number of non-PAD tokens in each target sequence.
- The loss marginalizes over all valid alignments between N query positions and
  T target tokens, allowing blanks and repetitions.

### 6. NAR training step (`train.py`)

New `_nar_train_step` parallel to existing `_train_step`:

```python
def _nar_train_step(state, ema_params, src, tgt_out):
    """Single NAR training step — no tgt_in needed."""
    pad_id = 0
    blank_id = state.apply_fn.config.blank_token_id  # 8192
    ema_decay = 0.999

    def loss_fn(params):
        src_mask = make_padding_mask(src, pad_id)
        logits, _ = state.apply_fn(
            {"params": params}, src, src_mask=src_mask,
            method="forward_nar",
        )
        loss = ctc_loss(logits, tgt_out, blank_id, pad_id)
        # z-loss for stability
        z_loss = 1e-4 * jnp.mean(jax.nn.logsumexp(logits, axis=-1) ** 2)
        return loss + z_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    state = state.apply_gradients(grads=grads)
    ema_params = jax.tree.map(
        lambda e, p: ema_decay * e + (1 - ema_decay) * p,
        ema_params, state.params,
    )
    return state, ema_params, loss
```

Key difference: no `tgt_in` needed — the decoder input is always the learned
queries, not shifted targets. Only `src` and `tgt_out` (targets for CTC
alignment) are required.

### 7. Data pipeline (`data.py`)

NAR training uses the same encoder-decoder pairs. The only difference:
- `dec_inputs` (shifted targets with BOS prefix) is unused — queries replace it
- `dec_targets` is used directly as CTC targets
- Target sequences should NOT have BOS prepended (CTC aligns raw token content)

Add a flag or separate data prep function:
```python
def prepare_nar_pairs(ds, tokenizer, max_enc_len, max_dec_len):
    """Like prepare_encoder_decoder_pairs but targets have no BOS prefix."""
    # enc_inputs: same as before
    # dec_targets: raw token sequence (no BOS), padded to max_dec_len
    ...
```

### 8. CLI changes (`cli.py`)

Add NAR-specific arguments to the `train` subparser:

```
--nar                       Enable NAR training mode
--num-queries INT           Number of learnable query tokens (default: 0 = AR)
--nar-checkpoint PATH       AR teacher checkpoint for knowledge distillation
--kd-weight FLOAT           KD loss weight (default: 1.0)
```

And a new `run` mode or flag for NAR inference:

```
needle run --checkpoint ckpt.pkl --nar --prompts "Once upon a time"
```

## Inference Changes

### 9. NAR generation (`run.py`)

```python
def generate_nar(model, params, tokenizer, prompt, seed=0):
    """One-shot parallel generation via CTC collapse."""
    enc_tokens = tokenizer.encode(prompt)
    enc_input = jnp.array([enc_tokens])
    src_mask = make_padding_mask(enc_input, tokenizer.pad_token_id)

    # Single forward pass — all tokens at once
    logits, _ = model.apply(
        {"params": params}, enc_input, src_mask=src_mask,
        method="forward_nar",
    )  # (1, N, V+1)

    # Greedy decode: argmax at each query position
    pred_tokens = jnp.argmax(logits[0], axis=-1)  # (N,)

    # CTC collapse: remove blanks, deduplicate consecutive tokens
    collapsed = ctc_collapse(pred_tokens, blank_id=8192)

    # Remove trailing PAD/EOS if present
    text = tokenizer.decode(collapsed)
    return text


def ctc_collapse(tokens, blank_id):
    """Remove blanks and merge consecutive duplicate tokens."""
    result = []
    prev = -1
    for t in tokens:
        t = int(t)
        if t == blank_id:
            prev = -1       # blank resets dedup so same token can appear again
            continue
        if t == prev:
            continue
        result.append(t)
        prev = t
    return result
```

This is the core speedup: instead of T sequential decode steps, we do 1 encode
+ 1 decode. For 128-token sequences this is theoretically ~64-128x fewer decode
calls (encoder is the same in both cases).

## Optional: Knowledge Distillation

### 10. KD from AR teacher (`train.py`)

The NARVL paper shows KD is critical for quality on longer sequences. The idea:
train a standard AR model first, then use its output distribution as soft
targets for the NAR model.

```python
def kd_loss(nar_logits, teacher_logits, temperature=2.0):
    """KL divergence between NAR and AR teacher distributions."""
    # teacher_logits: (B, T, V) from AR model — need to align with queries
    # nar_logits: (B, N, V+1) — drop blank dim for KD, keep only vocab V
    nar_probs = jax.nn.log_softmax(nar_logits[..., :V] / temperature, axis=-1)
    teacher_probs = jax.nn.softmax(teacher_logits / temperature, axis=-1)
    # ... alignment between N queries and T teacher positions via CTC-like matching
```

This is complex because the NAR queries don't have 1:1 correspondence with AR
positions. Two approaches:
- **Simple:** Run AR teacher, take argmax to get hard targets, use those as CTC
  targets (pseudo-labeling). The teacher's outputs are more deterministic than
  ground truth, reducing the alignment difficulty.
- **Full KD:** Align query positions to teacher positions via the CTC forward
  algorithm, then compute KL at aligned positions. More principled but harder.

Recommend starting with **pseudo-labeling** (simple) for the first experiment.

## Deployment & Execution

All training runs happen on the TPU VM, not locally. The workflow:

```
local development → sync to VM → run on VM → sync results back
```

### Sync scripts

```bash
# Push code changes to the VM (excludes checkpoints, caches, venvs)
./sync_code_to_vm.sh <HOST>          # default HOST=test-run

# Pull checkpoints/results back from the VM
./sync_results_from_vm.sh <HOST>     # default HOST=test-run
```

### Execution sequence

Every code change must be synced and tested on the VM before being considered
done. The three-stage validation pipeline before real experiments:

**Stage 1 — VM smoke test with `--toy`**
```bash
# Sync code, then on the VM:
./sync_code_to_vm.sh
ssh test-run
cd ~/needle && source ./setup
needle train --toy --nar --num-queries 40 --epochs 1
```
This uses the toy config (2 layers each, 128 seq len, 10k samples). Should
complete in a few minutes on the VM. Validates:
- CTC loss computes and backprops without NaN/inf
- Query embeddings receive gradients
- NAR generation produces tokens (even if garbage at 1 epoch)
- No shape mismatches or JAX compilation errors

**Stage 2 — VM short validation run (subset of data, full model config)**
```bash
needle train --nar --num-queries 80 --epochs 1 --max-samples 100000 \
             --eval-every 100 --wandb
```
Full model config but only 100k samples, 1 epoch. Validates:
- CTC loss converges (loss should decrease over steps)
- Multi-device pmap works correctly with NAR step
- Memory fits on the TPU (query embeddings + bidirectional attention use more
  memory than AR since there's no causal mask sparsity)
- wandb logging works for NAR metrics

**Stage 3 — Full experiment runs**
Only proceed after stages 1-2 pass cleanly. See experiment phases below.

After each experiment, pull results back:
```bash
./sync_results_from_vm.sh
```

## Experiment Plan

### Phase 1: Minimal viable NAR (validate CTC training works)

```bash
# On the VM:
needle train --toy --nar --num-queries 40 --epochs 3 --wandb
```

- Toy config: 2 enc + 2 dec layers, 128 seq len, 10k samples
- 40 queries for 128-token max target (need N > T, so ~1.3x overprovisioning
  since targets average ~40-60 tokens after 30% split)
- Compare against AR toy baseline on val perplexity
- Metric: does CTC loss converge? Does collapse decoding produce coherent text?

### Phase 2: Query count sweep

```bash
# On the VM:
for nq in 20 40 60 80 128; do
    needle train --toy --nar --num-queries $nq --epochs 3 --seed 42 --wandb
done
```

- Find sweet spot: too few queries = can't represent long targets, too many =
  harder alignment + diminishing returns (NARVL Table 10 showed this)
- Measure: val perplexity, generation quality, inference latency

### Phase 3: Full-scale NAR + KD

```bash
# On the VM:
# Step 1: Train AR teacher (or use existing checkpoint if available)
needle train --wandb --epochs 3 --seed 42

# Step 2: Train NAR with KD from AR teacher
needle train --nar --num-queries 80 --nar-checkpoint checkpoints/ar_teacher.pkl \
             --kd-weight 1.0 --wandb --epochs 3 --seed 42
```

- Full MAIN-1024 config
- Compare: AR vs NAR vs NAR+KD on val ppl, benchmark scores, throughput
- Pull checkpoints back: `./sync_results_from_vm.sh`

### Phase 4: Integration with compression axes

- MRL: forward_nar should support mrl_dims (slice queries at smaller d')
- Pruning: same gradual magnitude pruning applies to NAR params
- QAT: same fake INT4 quantization applies
- Test full compression grid on NAR model

## Evaluation Strategy

### What existing evals CAN'T be used for NAR

- **Perplexity (val ppl, WikiText-2 ppl):** Requires AR factorization —
  P(y_t | y_{<t}, x). NAR predicts all positions jointly, so per-token
  cross-entropy is undefined. CTC loss is a marginalization over alignments,
  not a per-position probability.
- **evaluate.py benchmarks (LAMBADA, HellaSwag, ARC-Easy):** All score
  candidates by AR next-token or sequence log-probability. NAR can't compute
  "probability of token X given prior context" since it generates everything at
  once. These benchmarks fundamentally assume autoregressive factorization.

### What to measure instead

The core comparison: give AR and NAR the same encoder inputs, generate from
both, measure quality of outputs against ground truth decoder targets.

**1. BLEU-4 against ground truth (primary quality metric)**

The most direct apples-to-apples comparison. For N val samples:
- Encode the source with both AR and NAR models
- Generate continuations from both
- Compute corpus BLEU-4 against the ground truth decoder target

```python
def evaluate_nar_bleu(model, params, tokenizer, enc_inputs, dec_targets,
                      num_samples=1000, blank_id=8192, pad_id=0):
    """BLEU-4 of NAR generations against ground truth targets."""
    from collections import Counter

    references = []
    hypotheses = []

    for i in range(min(num_samples, len(enc_inputs))):
        src = jnp.array([enc_inputs[i]])
        src_mask = make_padding_mask(src, pad_id)

        logits, _ = model.apply(
            {"params": params}, src, src_mask=src_mask,
            method="forward_nar",
        )
        pred_tokens = jnp.argmax(logits[0], axis=-1)
        hyp = ctc_collapse(pred_tokens, blank_id)

        # Ground truth: strip padding
        ref = [int(t) for t in dec_targets[i] if t != pad_id]

        references.append(ref)
        hypotheses.append(hyp)

    return corpus_bleu(references, hypotheses)
```

This works for AR too — just use the AR generate function instead.

**2. Val CTC loss (NAR training objective)**

Direct training loss on the val set. Comparable across NAR runs (different
query counts, with/without KD) but NOT comparable to AR val perplexity.

```python
def compute_nar_val_loss(model, params, enc_inputs, dec_targets, batch_size,
                         blank_id, pad_id):
    """Average CTC loss on validation set."""
    total_loss, total_samples = 0.0, 0
    for src, _, tgt_out in get_batches(enc_inputs, ..., dec_targets, batch_size):
        src_mask = make_padding_mask(src, pad_id)
        logits, _ = model.apply({"params": params}, src, src_mask=src_mask,
                                method="forward_nar")
        loss = ctc_loss(logits, tgt_out, blank_id, pad_id)
        total_loss += float(loss) * src.shape[0]
        total_samples += src.shape[0]
    return total_loss / total_samples
```

**3. Generation length accuracy**

Does the model produce roughly the right number of tokens? Measures whether
the model learned proper blank/content distribution.

```python
def length_accuracy(hypotheses, references):
    """Ratio of predicted length to reference length."""
    ratios = [len(h) / max(len(r), 1) for h, r in zip(hypotheses, references)]
    return {
        "mean_length_ratio": np.mean(ratios),     # ideal = 1.0
        "std_length_ratio": np.std(ratios),
        "too_short_pct": np.mean([r < 0.5 for r in ratios]) * 100,
        "too_long_pct": np.mean([r > 1.5 for r in ratios]) * 100,
    }
```

Too many blanks → output too short. Too few blanks → repetition/garbled.

**4. Distinct-n (generation diversity)**

Fraction of unique n-grams in generated text. Catches mode collapse and
repetition more precisely than bigram repetition rate alone.

```python
def distinct_n(texts, n=2):
    """Fraction of unique n-grams across all generated texts."""
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        all_ngrams.extend(zip(*[tokens[i:] for i in range(n)]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)
```

Report Distinct-1 (unigram diversity) and Distinct-2 (bigram diversity).

**5. Inference throughput**

Single forward pass timing — the whole point of NAR. Measure:
- Encoder time (same for both AR and NAR)
- Decoder time: 1 forward pass (NAR) vs T sequential passes (AR)
- Total wall-clock time for full generation including collapse

```python
def measure_nar_throughput(model, params, tokenizer, num_runs=10,
                           prompt="Once upon a time"):
    # Warmup + compile
    ...
    for run in range(num_runs):
        t0 = time.perf_counter()
        # 1 encode + 1 decode + collapse
        logits, _ = model.apply(...)
        pred_tokens = jnp.argmax(logits[0], axis=-1)
        collapsed = ctc_collapse(pred_tokens, blank_id)
        jax.block_until_ready(pred_tokens)
        elapsed = time.perf_counter() - t0
        ...
```

**6. Manual inspection of generations**

Non-automated but essential. Print 10-20 NAR generations side-by-side with AR
generations from the same prompts. Look for:
- Coherence: does the text make sense as a story?
- Repetition: repeated phrases or words?
- Mode mixing: fragments from different stories merged?
- Blank distribution: are blanks clustered at the end (good) or scattered (bad)?

### Summary: metrics table

| Metric | AR baseline | NAR | NAR+KD | Comparable? |
|--------|-------------|-----|--------|-------------|
| Val loss (CE for AR, CTC for NAR) | ? | ? | ? | No (different losses) |
| BLEU-4 vs ground truth | ? | ? | ? | Yes (primary) |
| Distinct-1 | ? | ? | ? | Yes |
| Distinct-2 | ? | ? | ? | Yes |
| Mean length ratio | ? | ? | ? | Yes |
| Inference latency (ms) | ? | ? | ? | Yes |
| Tokens/sec | ? | ? | ? | Yes |
| Bigram repetition rate | ? | ? | ? | Yes |
| Generation coherence (manual) | ? | ? | ? | Yes |

## Known Risks

1. **Coherence on long text.** NARVL tested on short outputs (2-20 tokens).
   TinyStories targets are 40-80 tokens. Conditional independence between query
   positions may cause repetition, contradiction, or mode mixing (fragments from
   multiple valid continuations merged into one output). The paper explicitly
   notes this as their main failure mode.

2. **CTC loss convergence.** Q-CTC is harder to optimize than CE. NARVL started
   from pretrained OFA weights. We're training from scratch, so early training
   may be unstable. May need careful LR tuning, warmup, or curriculum (start
   with short targets, gradually increase).

3. **Query count vs sequence length.** Need num_queries >= max_target_length for
   CTC to have valid alignments. But too many queries wastes compute. The 30%
   encoder-decoder split means targets average ~70% of story length.

4. **Blank token overhead.** Adding 1 token to vocab is negligible, but the
   model must learn when to output blank vs real tokens. In speech CTC this
   comes naturally (silence = blank). In text there's no obvious "blank" analog,
   so the model must learn this purely from the loss signal.

## Handling Long Sequences (Sliding Window Data Strategy)

CTC requires `num_queries >= target_length`. Rather than overprovisioning
queries (which wastes O(N²) attention compute) or simply discarding long
stories, we use a **sliding window** approach that generates multiple
encoder-decoder pairs per story, each with targets capped to `num_queries`.

### Phase A: Truncation baseline

Targets longer than `num_queries` are simply truncated. This gets training
working but means the model never sees the later portions of long stories.
Data pipeline: `prepare_nar_pairs(..., max_target_len=num_queries)`.

CTC loss uses `zero_infinity` clamping: infeasible samples (if any slip through)
are capped at 1e4 instead of returning ~1e5 * seq_len.

### Phase B: Sliding window pairs (`prepare_nar_sliding_pairs`)

For each story, instead of one fixed 30/70 split, generate **multiple pairs**
with sliding encoder/decoder windows:

```
Story: [t0, t1, t2, ..., t127]    (128 tokens)
num_queries = 40

Pair 1: enc = [t0 .. t31]    tgt = [t32 .. t63]     (32 in, 32 out)
Pair 2: enc = [t0 .. t63]    tgt = [t64 .. t95]     (64 in, 32 out)
Pair 3: enc = [t0 .. t95]    tgt = [t96 .. t127]    (96 in, 32 out)
```

Every pair has `len(tgt) <= num_queries`, so CTC always has a feasible
alignment. The model sees the **entire story** across multiple pairs, and
learns to continue from varying context lengths.

Implementation in `data.py`:

```python
def prepare_nar_sliding_pairs(ds, tokenizer, max_enc_len, max_dec_len,
                               num_queries, stride=None):
    """Generate multiple enc/dec pairs per story using sliding windows.

    For each story, slide a window that produces pairs where:
      - enc = tokens[0 : split_point]
      - tgt = tokens[split_point : split_point + num_queries]
    Split points advance by `stride` tokens (default: num_queries // 2
    for 50% overlap between consecutive target windows).
    """
    if stride is None:
        stride = max(1, num_queries // 2)

    # For each tokenized story:
    #   min_enc = 2  (need at least some encoder context)
    #   first split_point = min_enc
    #   advance split_point by stride each iteration
    #   stop when split_point >= len(tokens)
    #   each pair: enc = tokens[:split_point], tgt = tokens[split_point:split_point+num_queries]
    ...
```

Benefits:
- **Full dataset coverage**: every token in every story appears as a target in
  at least one pair — no information is discarded
- **Variable context lengths**: the model sees both short and long encoder
  inputs, making it robust at inference time
- **More training data**: a 128-token story with stride=20 and num_queries=40
  yields ~5 pairs instead of 1
- **Natural curriculum**: shorter encoder inputs (early pairs) are easier;
  longer ones provide more context

CLI: `--nar-sliding` flag to enable. Default stride is `num_queries // 2`.
`--nar-stride INT` to override.

### Inference with sliding windows

At inference time, the model still does a single NAR forward pass:
1. Encode the full prompt
2. Decode all `num_queries` positions in parallel
3. CTC collapse to get output tokens

If longer output is needed, chain multiple passes:
```
Pass 1: encode(prompt) → NAR decode → collapse → output_1
Pass 2: encode(prompt + output_1) → NAR decode → collapse → output_2
Pass 3: encode(prompt + output_1 + output_2) → ...
```

This is still faster than AR since each pass produces `num_queries` tokens
in O(1) decoder calls rather than generating them one at a time.

## Stabilizing CTC Training

CTC training from scratch is notoriously difficult. NARVL explicitly notes
"model initialization weights are non-trivial for training Q-CTC loss" — they
started from pretrained OFA. The following fixes are ordered by priority.

### Fix 1 (critical): Initialize encoder+decoder from AR checkpoint

The single biggest improvement. Load a trained AR checkpoint's encoder and
decoder weights, only randomly initialize `query_embed` and `blank_embed`.
The encoder already produces meaningful representations, the decoder already
knows how to cross-attend. CTC only needs to learn the query→token alignment.

```bash
needle train --nar --num-queries 80 --nar-checkpoint checkpoints/ar_baseline.pkl \
             --wandb --epochs 3
```

Implementation: in `train()`, when `--nar-checkpoint` is provided, load the
AR checkpoint params and copy all matching keys (encoder, decoder, embedding)
into the NAR model's initial params. `query_embed` and `blank_embed` keep
their random init since they don't exist in the AR checkpoint.

### Fix 2 (critical): Disable QAT during NAR training

The current `_nar_train_step` runs `_quantize_params()` on every forward pass.
Fake INT4 quantization adds noise that CE can tolerate but CTC cannot — CTC's
optimization landscape is already much harder (sparse gradients from the
forward-backward algorithm). Quantization noise prevents convergence.

Fix: pass raw params in the NAR train step, not quantized params.

```python
# BEFORE (broken):
logits, _ = state.apply_fn(
    {"params": _quantize_params(params, group_size=_GROUP_SIZE)}, ...)

# AFTER (fixed):
logits, _ = state.apply_fn({"params": params}, ...)
```

QAT can be re-enabled later once CTC converges without it.

### Fix 3: Differentiate queries at initialization

`jinit.normal(stddev=0.02)` gives N queries that are nearly identical noise.
The decoder sees N indistinguishable inputs → all positions produce nearly
identical logit distributions → CTC gets ~uniform probability across all
alignment paths → vanishingly small gradients.

Use orthogonal or sinusoidal init so queries are maximally distinct from step 1:

```python
# Option A: Orthogonal — queries start maximally differentiated
self.query_embed = self.param(
    "query_embed", jinit.orthogonal(),
    (self.config.num_queries, self.config.d_model),
)

# Option B: Sinusoidal positional patterns
positions = jnp.arange(num_queries)
freqs = 1.0 / (10000 ** (jnp.arange(0, d_model, 2) / d_model))
angles = positions[:, None] * freqs[None, :]
init = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
```

### Fix 4: Separate higher LR for query embeddings

Query embeddings need to learn much faster than the pretrained encoder/decoder.
Add a third optimizer group with ~10x the AdamW LR:

```python
def _param_labels(params):
    def _label(path, leaf):
        name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
        if name in ("query_embed", "blank_embed"):
            return "query"       # high LR AdamW
        if name == "kernel" and leaf.ndim == 2:
            return "muon"
        return "adam"
    return jax.tree_util.tree_map_with_path(_label, params)
```

### Fix 5: Disable Muon for NAR (AdamW only)

Newton-Schulz orthogonalization was tuned for CE gradients. CTC gradients have
very different structure (sparse, dominated by a few high-probability paths).
Orthogonalizing these can destroy the useful gradient signal.

Use pure AdamW for all NAR parameters:

```python
if nar_mode:
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(adam_schedule, b2=0.95, weight_decay=0.01),
    )
```

Re-add Muon once basic convergence is confirmed.

### Fix 6: Temperature scaling on CTC logits

Soften distributions early in training so CTC can spread probability across
multiple alignment paths instead of collapsing to one bad path:

```python
def ctc_loss(logits, targets, blank_id, pad_id, temperature=1.0):
    if temperature != 1.0:
        logits = logits / temperature
    ...
```

Start at `temperature=2.0`, anneal to 1.0 over training.

### Fix 7: Longer warmup for CTC

CTC needs gentler warmup than CE due to more erratic early gradients.
Increase from 5% to 15-20% of total steps:

```python
warmup_ratio = 0.15 if nar_mode else args.warmup_ratio
```

### Prioritized implementation order

1. Fix 1 + Fix 2 — **do first**, these alone should unblock convergence
2. Fix 3 + Fix 5 — if loss still plateaus after fix 1+2
3. Fix 4 + Fix 6 + Fix 7 — fine-tuning for faster/better convergence

## File Change Summary

| File | Change |
|------|--------|
| `src/model.py` | Add `num_queries`, `blank_token_id` to config. Add `query_embed` + `blank_embed` params. Add `forward_nar()` method. |
| `src/train.py` | Add `ctc_loss()`. Add `_nar_train_step()`. Branch on `--nar` flag in `train()`. Add NAR val loss. |
| `src/data.py` | Add `prepare_nar_pairs()` (targets without BOS). |
| `src/run.py` | Add `generate_nar()` + `ctc_collapse()`. Branch on `--nar` flag. |
| `src/test.py` | Add NAR throughput + perplexity benchmarks. |
| `src/cli.py` | Add `--nar`, `--num-queries`, `--nar-checkpoint`, `--kd-weight` args. |
