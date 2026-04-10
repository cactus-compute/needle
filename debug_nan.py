"""Debug why train.py produces NaN from step 0. Run on a TPU worker."""
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.model import (
    EncoderDecoderTransformer, TransformerConfig,
    make_packing_mask, make_causal_packing_mask, make_cross_packing_mask,
)
from src.data import load_prepared_data

print("=" * 60)
print("Loading checkpoint...")
with open("checkpoints/needle_base.pkl", "rb") as f:
    ckpt = pickle.load(f)
config = TransformerConfig(**ckpt["config"])
params = jax.tree.map(jnp.array, ckpt["params"])
print(f"  config: d={config.d_model} heads={config.num_heads} layers={config.num_encoder_layers}/{config.num_decoder_layers}")
print(f"  dtype={config.dtype} no_ff={config.no_feedforward}")

print("Loading data...")
data = load_prepared_data("train", mmap=True)
bs = 4
src = jnp.asarray(data["packed_enc"][:bs], dtype=jnp.int32)
tgt_in = jnp.asarray(data["packed_dec_in"][:bs], dtype=jnp.int32)
tgt_out = jnp.asarray(data["packed_dec_tgt"][:bs], dtype=jnp.int32)
enc_seg = jnp.asarray(data["packed_enc_seg"][:bs], dtype=jnp.int32)
dec_seg = jnp.asarray(data["packed_dec_seg"][:bs], dtype=jnp.int32)
lm = jnp.asarray(data["packed_loss"][:bs], dtype=jnp.int32)

print(f"  src: shape={src.shape} range=[{int(src.min())}, {int(src.max())}]")
print(f"  tgt_in: shape={tgt_in.shape} range=[{int(tgt_in.min())}, {int(tgt_in.max())}]")
print(f"  tgt_out: shape={tgt_out.shape} range=[{int(tgt_out.min())}, {int(tgt_out.max())}]")
print(f"  enc_seg: range=[{int(enc_seg.min())}, {int(enc_seg.max())}] nnz_rows={int((enc_seg.sum(axis=-1) > 0).sum())}/{bs}")
print(f"  dec_seg: range=[{int(dec_seg.min())}, {int(dec_seg.max())}] nnz_rows={int((dec_seg.sum(axis=-1) > 0).sum())}/{bs}")
print(f"  lm: range=[{int(lm.min())}, {int(lm.max())}]")

print("=" * 60)
print("Checking params for NaN/Inf...")
any_bad = False
for path, leaf in jax.tree_util.tree_leaves_with_path(params):
    n_bad = int(jnp.sum(~jnp.isfinite(leaf)))
    if n_bad:
        any_bad = True
        print(f"  BAD {path}: {n_bad} nonfinite")
if not any_bad:
    print("  all params finite")

print("=" * 60)
print("Building masks...")
src_mask = make_packing_mask(enc_seg)
tgt_mask = make_causal_packing_mask(dec_seg)
cross_mask = make_cross_packing_mask(enc_seg, dec_seg)
print(f"  src_mask: {src_mask.shape} true_frac={float(src_mask.mean()):.3f}")
print(f"  tgt_mask: {tgt_mask.shape} true_frac={float(tgt_mask.mean()):.3f}")
print(f"  cross_mask: {cross_mask.shape} true_frac={float(cross_mask.mean()):.3f}")

# Check if any attention row is all-False
def all_false_rows(m):
    # m: (B, 1, T_q, T_k)
    return int((~m.any(axis=-1)).sum())
print(f"  src_mask all-False rows: {all_false_rows(src_mask)}")
print(f"  tgt_mask all-False rows: {all_false_rows(tgt_mask)}")
print(f"  cross_mask all-False rows: {all_false_rows(cross_mask)}")

print("=" * 60)
print("Running __call__ forward (no dropout)...")
model = EncoderDecoderTransformer(config)
logits = model.apply({"params": params}, src, tgt_in,
                     src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)
print(f"  logits: shape={logits.shape} dtype={logits.dtype}")
print(f"  NaN={int(jnp.sum(jnp.isnan(logits)))} Inf={int(jnp.sum(jnp.isinf(logits)))}")
print(f"  min={float(jnp.min(logits)):+.3g} max={float(jnp.max(logits)):+.3g}")

logits_f32 = logits.astype(jnp.float32)
ce = optax.softmax_cross_entropy_with_integer_labels(logits_f32, tgt_out)
print(f"  CE: NaN={int(jnp.sum(jnp.isnan(ce)))} min={float(jnp.min(ce)):+.3g} max={float(jnp.max(ce)):+.3g}")
padding = (dec_seg > 0).astype(jnp.float32)
num_tok = jnp.sum(padding)
ce_loss = jnp.sum(ce * padding) / jnp.maximum(num_tok, 1.0)
print(f"  masked CE loss: {float(ce_loss):.4f} (num_tok={float(num_tok):.0f})")

# Check z_loss
lse = jax.nn.logsumexp(logits_f32, axis=-1)
print(f"  logsumexp: NaN={int(jnp.sum(jnp.isnan(lse)))} min={float(jnp.min(lse)):+.3g} max={float(jnp.max(lse)):+.3g}")
z_loss = 1e-4 * jnp.mean(lse ** 2)
print(f"  z_loss: {float(z_loss):.4f}")

print("=" * 60)
print("Running forward_masked (with dropout)...")
ffn_mask = jnp.ones((bs, config.d_ff), dtype=config.jax_dtype)
logits2, _ = model.apply(
    {"params": params}, src, tgt_in,
    src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask,
    ffn_mask=ffn_mask, deterministic=False,
    method="forward_masked", rngs={"dropout": jax.random.PRNGKey(0)},
)
print(f"  logits: NaN={int(jnp.sum(jnp.isnan(logits2)))} Inf={int(jnp.sum(jnp.isinf(logits2)))}")
print(f"  min={float(jnp.min(logits2)):+.3g} max={float(jnp.max(logits2)):+.3g}")

ce2 = optax.softmax_cross_entropy_with_integer_labels(logits2.astype(jnp.float32), tgt_out)
print(f"  CE: NaN={int(jnp.sum(jnp.isnan(ce2)))} min={float(jnp.min(ce2)):+.3g} max={float(jnp.max(ce2)):+.3g}")
