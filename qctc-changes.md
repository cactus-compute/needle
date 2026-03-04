# Query-CTC Implementation: Code Changes Walkthrough

This documents every code change made to implement Query-CTC (Q-CTC) non-autoregressive decoding in Needle, and why each piece is needed.

Reference: NARVL (arXiv 2403.02249) — adapted from VLM to pure text generation.

**Core idea**: Instead of generating tokens one-by-one (AR), feed N learnable query embeddings into the decoder in parallel. CTC loss aligns the N query outputs to the T target tokens (N >= T). At inference, argmax + CTC collapse (remove blanks, deduplicate consecutive tokens) recovers the output in a single forward pass.

---

## 1. Model Changes (`src/model.py`)

### 1a. Config additions

```python
num_queries: int = 0          # 0 = AR mode, >0 = NAR mode
blank_token_id: int = 8192    # appended to vocab for CTC blank
query_init: str = "normal"    # "normal" or "orthogonal"
```

**Why**: `num_queries` is the master switch — when 0 the model is AR, when >0 it's NAR. CTC requires a blank token (represents "no output at this position"), so we extend the vocab by 1. `query_init` lets us experiment with orthogonal init to encourage queries to specialize to different positions.

### 1b. Learnable parameters in `EncoderDecoderTransformer.setup()`

```python
if self.config.num_queries > 0:
    self.query_embed = self.param("query_embed", ...)   # (num_queries, d_model)
    self.blank_embed = self.param("blank_embed", ...)    # (1, d_model)
```

**Why**: `query_embed` replaces the shifted-target input that AR uses — these are the "questions" the decoder answers in parallel. `blank_embed` is a separate learned embedding for the CTC blank token (can't reuse a vocab embedding since blank isn't a real token). It gets concatenated to the embedding matrix to form a (vocab+1, d_model) projection.

### 1c. `forward_nar()` method

```python
def forward_nar(self, src, src_mask=None):
    encoder_out = self.encode(src, src_mask=src_mask)
    queries = query_embed[None] * embed_scale  # broadcast to (B, N, d_model)
    x = self.decoder(queries, encoder_out, self_mask=None, ...)  # bidirectional!
    full_emb = concat(vocab_embedding, blank_embed)  # (vocab+1, d_model)
    logits = x @ full_emb.T  # (B, N, vocab+1)
    return logits, encoder_out
```

**Why**: This is the NAR forward pass. Key differences from AR:
- **No causal mask** (`self_mask=None`): queries attend to each other bidirectionally, since all positions are predicted simultaneously. This is critical — causal masking would cripple NAR because position i can't see positions i+1..N.
- **No target input**: queries replace the shifted target sequence. The decoder cross-attends to encoder output the same as AR.
- **Blank projection**: logits include the blank token so CTC can assign "no output" to some query positions.
- **Float32 logits**: upcast for numerical stability in CTC's log-sum-exp.

---

## 2. Data Changes (`src/data.py`)

### 2a. `prepare_nar_pairs()`

**Why**: NAR targets differ from AR targets:
- **No BOS prefix**: AR targets are `[BOS, t1, t2, ...]` (shifted right for teacher forcing). CTC aligns against raw token sequences — BOS would be an extra token CTC has to model.
- **`max_target_len` filtering**: CTC requires `target_len <= num_queries`. Samples violating this produce infinite loss. We cap targets to `num_queries // 2` (2x overprovisioning gives CTC room for blanks and repeated tokens in its alignment paths).

### 2b. `prepare_nar_sliding_pairs()`

**Why**: The basic split (30% encoder / 70% decoder) wastes data — long stories get truncated and short targets underutilize queries. Sliding window generates multiple (enc, tgt) pairs per story by advancing a window. Every token appears as a target in at least one pair, giving much better data coverage. This is important because CTC with short targets needs lots of diverse examples.

### 2c. `get_nar_batches()`

**Why**: NAR batches are `(enc, tgt)` — no `dec_inputs` needed since there's no teacher forcing. Just a simpler 2-tuple batch iterator.

---

## 3. Training Changes (`src/train.py`)

### 3a. Model initialization

```python
if config.num_queries > 0:
    variables = model.init(..., method="forward_nar")
```

**Why**: NAR model has different parameters (query_embed, blank_embed) and doesn't need a target input for init. Using `method="forward_nar"` ensures Flax traces the NAR path and creates all necessary parameters.

### 3b. `adamw_only` optimizer option

**Why**: Muon (Newton-Schulz orthogonalization optimizer) is designed for large 2D+ weight matrices. The query embeddings are a 2D matrix but semantically different — they're positional embeddings, not weight matrices. Muon can destabilize query learning. Having an AdamW-only path lets us experiment with a more conservative optimizer for NAR.

### 3c. `ctc_loss()` function

```python
def ctc_loss(logits, targets, blank_id, pad_id):
    loss = optax.ctc_loss(logits, logit_paddings, targets, label_paddings, blank_id=blank_id)
    loss = jnp.where(jnp.isfinite(loss), loss, 0.0)  # zero_infinity
    loss = jnp.minimum(loss, 1e4)  # cap extreme values
```

**Why**: Wraps `optax.ctc_loss` with two critical additions:
- **zero_infinity**: When `target_len > num_queries`, CTC returns infinity (no valid alignment exists). We clamp these to 0, matching PyTorch's `zero_infinity=True` used by NARVL. Without this, a single infeasible sample blows up the entire batch.
- **1e4 cap**: Even finite CTC losses can be extremely large early in training. Capping prevents gradient explosions.
- **Averaging over valid samples only**: Infeasible (zeroed) samples don't count toward the denominator.

### 3d. `_nar_train_step()`

```python
def _nar_train_step(state, ema_params, src, tgt_out):
    loss = ctc_loss(logits, tgt_out, blank_id, pad_id)
    z_loss = 1e-4 * jnp.mean(jax.nn.logsumexp(logits, axis=-1) ** 2)
    return loss + z_loss
```

**Why**: Separate train step because:
- **No tgt_in or causal_mask args**: NAR doesn't use teacher forcing.
- **z-loss regularizer**: Encourages logits to stay small, preventing the model from becoming overconfident on blanks early in training (a known CTC failure mode where the model outputs all blanks).

### 3e. AR checkpoint initialization

```python
if nar_mode and nar_checkpoint:
    # Copy matching params from AR checkpoint; query_embed/blank_embed keep random init
    merged_params = _merge_ar_into_nar(state.params, ar_params)
```

**Why**: The encoder, decoder, and embedding weights are identical between AR and NAR architectures. Initializing from a trained AR checkpoint gives NAR a huge head start — it only needs to learn the query embeddings and blank embedding from scratch. The merge function matches parameters by path and shape, so NAR-only parameters (query_embed, blank_embed) keep their random initialization.

### 3f. Disabled MRL/QAT for NAR

**Why**: MRL (Matryoshka dimension slicing) and QAT (quantization-aware training) are orthogonal compression axes designed for the AR model. NAR is an experimental decoding strategy — adding MRL/QAT simultaneously would confound results. These can be added later once NAR training is stable.

### 3g. Validation metrics

**Why**: NAR reports CTC loss directly instead of perplexity. CTC loss isn't cross-entropy per token, so `exp(loss)` doesn't give a meaningful perplexity. We track raw CTC loss as `val_ctc` in tqdm and wandb.

### 3h. Higher warmup ratio (15% vs default)

**Why**: CTC loss landscapes are notoriously unstable early in training. Longer warmup helps the model learn basic blank/non-blank discrimination before the learning rate ramps up.

### 3i. End-of-epoch evaluation additions

BLEU-4, distinct-1/2, and length ratio metrics are computed at each epoch end for both AR and NAR models.

**Why**: CTC loss alone doesn't tell you if the model generates coherent text. BLEU-4 measures n-gram overlap with ground truth (cross-comparable between AR and NAR). Distinct-n measures vocabulary diversity (CTC models can degenerate into repetitive output). Length ratio catches over/under-generation.

---

## 4. Inference Changes (`src/run.py`)

### 4a. `ctc_collapse()`

```python
def ctc_collapse(tokens, blank_id):
    # Remove blanks, merge consecutive duplicates
```

**Why**: CTC's output is a sequence of length N (num_queries) with blanks and possible repeated tokens. The collapse operation `B` recovers the actual output: remove all blank tokens, then merge consecutive identical tokens. For example: `[a, a, BLANK, b, b, b, c]` → `[a, b, c]`. The blank reset (`prev = -1`) is important — it allows the same token to appear non-consecutively (e.g., `[a, BLANK, a]` → `[a, a]`).

### 4b. `generate_nar()` — chained multi-pass generation

```python
def generate_nar(model, params, tokenizer, prompt, max_passes=10):
    for pass_idx in range(max_passes):
        context = enc_tokens + all_generated
        logits, _ = nar_forward(enc_input, src_mask)
        collapsed = ctc_collapse(argmax(logits), blank_id)
        all_generated.extend(collapsed)
        if hit_eos: break
```

**Why**: A single NAR forward pass produces at most `num_queries` tokens (after collapse, usually fewer). For longer generation, we chain passes: each pass's output is appended to the encoder context for the next pass. This is analogous to how NARVL generates captions — iterative refinement with growing context. Stops on EOS or when a pass produces no new tokens (model thinks it's done).

### 4c. Auto-detect NAR mode

```python
nar_mode = getattr(args, "nar", False) or config.num_queries > 0
```

**Why**: If the checkpoint was trained with `num_queries > 0`, it's a NAR model — automatically use NAR inference without requiring `--nar` flag.

---

## 5. Evaluation Changes (`src/test.py`)

### 5a. `compute_nar_ctc_loss()`

**Why**: CTC loss replaces perplexity as the primary NAR quality metric. Computes average CTC loss over validation data with the same zero_infinity clamping as training.

### 5b. `measure_nar_throughput()`

**Why**: Measures single-pass NAR inference speed. This is the key speedup metric — NAR should be ~Nx faster than AR (where N is the average AR generation length), since it produces all tokens in one decoder pass instead of one pass per token.

### 5c. `benchmark_nar_generation_quality()`

**Why**: Uses `generate_nar()` (chained passes) to produce text samples, then measures the same quality metrics as AR (length, repetition, distinct-n). Enables direct comparison.

### 5d. `evaluate_bleu4()` — cross-comparable quality metric

**Why**: BLEU-4 is the main quality metric that works for both AR and NAR. Given the same encoder input, it compares generated output against ground truth decoder targets. NAR eval is batched (all queries forward at once), AR eval generates one-at-a-time. Reports BLEU score + length ratio (ideal = 1.0).

### 5e. `distinct_n()`

**Why**: Measures the fraction of unique n-grams in generated text. CTC models are prone to mode collapse (generating the same few tokens repeatedly), so distinct-1 and distinct-2 are important health checks.

### 5f. `corpus_bleu4()`

**Why**: Self-contained BLEU-4 implementation (no nltk dependency). Computes corpus-level BLEU with brevity penalty across all reference/hypothesis pairs.

---

## 6. Benchmark Changes (`src/evaluate.py`)

### 6a. `eval_nar_wikitext2()`

**Why**: Standard WikiText-2 perplexity doesn't apply to NAR models. This evaluates CTC loss on WikiText-2 test split instead — split each passage into encoder/decoder portions, run NAR forward, compute CTC loss. Separate benchmark registry (`NAR_BENCHMARKS`) so `needle evaluate` automatically picks the right metrics.

### 6b. Auto-routing in `evaluate.main()`

```python
available = NAR_BENCHMARKS if nar_mode else BENCHMARKS
```

**Why**: NAR models can't run AR benchmarks (LAMBADA, HellaSwag, ARC-Easy rely on next-token probabilities). The routing ensures only compatible benchmarks run.

---

## 7. CLI Changes (`src/cli.py`)

New flags:
| Flag | Purpose |
|------|---------|
| `--nar` | Enable NAR training mode |
| `--num-queries N` | Number of learnable query tokens |
| `--nar-checkpoint PATH` | AR teacher checkpoint for weight initialization |
| `--kd-weight FLOAT` | Knowledge distillation weight (for future use) |
| `--query-init {normal,orthogonal}` | Query embedding initialization |
| `--adamw-only` | Use AdamW instead of Muon+AdamW |
| `--nar-sliding` | Sliding window data preparation |
| `--nar-stride N` | Stride for sliding window |

---

## 8. Experiment Script (`run_nar_experiments.sh`)

Sweeps 4 experimental axes from a common baseline (`--nar --num-queries 20 --nar-checkpoint AR_CKPT`):
1. **Higher LR** (3e-3 vs default 3e-4) — CTC may need more aggressive optimization
2. **Orthogonal query init** — encourage position specialization
3. **AdamW only** — test if Muon hurts query learning
4. **Sliding window pairs** — better data coverage

Each run saves to a separate checkpoint directory for comparison.
