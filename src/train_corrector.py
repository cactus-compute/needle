"""Stage-2 correction fine-tuning.

Reads the JSONL produced by src.mine_corrections, tokenizes on-the-fly,
and fine-tunes a SEPARATE corrector model initialized from an SFT
checkpoint. The base SFT ckpt is not modified. Inference at deploy time
runs both models in sequence (base -> draft -> corrector).

Encoder input:  [query, <tools>, tools_json, <tool_call>, draft_json]
Decoder input:  [EOS, <tool_call>, gold_answer]
Decoder target: [<tool_call>, gold_answer, EOS]

The encoder uses <tool_call> (already in the tokenizer vocab) as the
tools->draft delimiter — no tokenizer changes required.

Eval mirrors train.py exactly: same stratified single/multi-call sampling
(50 per tool-count bucket, rng seed 42), same tool-call metrics, same
display samples. At init we run the frozen base SFT ckpt once over the
full val set and cache the drafts. Every subsequent eval step just
re-runs the CORRECTOR on those cached drafts — fast.

The eval report prints both `draft` (base only, baseline) and
`corrected` (base + corrector) side by side, so the improvement or
regression per bucket is a direct read.

Usage:
    python -m src.train_corrector \\
        --base-checkpoint checkpoints/needle.pkl \\
        --corrector-init checkpoints/needle.pkl \\
        --data data/corrections.jsonl \\
        --epochs 1 --lr 1e-4 --batch-size 64 \\
        --wandb
"""

import argparse
import json
import math
import os
import pickle
import queue
import threading
import time
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from .data import (
    _token_classes_for_answer, get_tokenizer, load_tool_calls,
)
from .distributed import (
    _replicate, _unreplicate, shard_batch, _upload_checkpoint,
)
from .model import (
    EncoderDecoderTransformer, TransformerConfig,
    make_causal_mask, make_padding_mask,
)
from .optim import create_train_state, _wsd_schedule
from .run import generate_batch, load_checkpoint
from .tokenizer import EOS_ID, PAD_ID, TOOL_CALL_ID, TOOLS_ID


_LOSS_WEIGHT_MAP = jnp.array([1.0, 2.0, 4.0, 1.5], dtype=jnp.float32)


# ───────────────────────── filter: unrecoverable hallucinations ─────────────────────────

import re as _re

_ID_WITH_DIGIT = _re.compile(r"[a-zA-Z]_[a-zA-Z0-9]*\d")
_OPAQUE_CODE = _re.compile(r"^[A-Za-z0-9_-]+$")


def _looks_id_like(value):
    """Detect ID-shaped values: emails, URLs, paths, coded names, compound IDs."""
    if not isinstance(value, str) or len(value) < 4:
        return False
    if "@" in value:
        left, _, right = value.rpartition("@")
        if left and "." in right:
            return True
    if value.startswith(("http://", "https://", "www.")):
        return True
    if "/" in value or "\\" in value:
        return True
    if _ID_WITH_DIGIT.search(value):
        return True
    if len(value) > 15 and _OPAQUE_CODE.match(value) and any(c.isdigit() for c in value):
        return True
    if value.count("_") >= 2 and len(value) >= 10:
        return True
    return False


def _value_in_query(value, query):
    if not isinstance(value, str):
        return True
    return value.lower() in query.lower()


def _gold_has_unrecoverable(query, gold_text):
    """True if gold contains any ID-shaped arg value that isn't in the query."""
    try:
        gold_calls = json.loads(gold_text)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(gold_calls, list):
        return False
    for call in gold_calls:
        if not isinstance(call, dict):
            continue
        args = call.get("arguments", {}) or {}
        if not isinstance(args, dict):
            continue
        for _, value in args.items():
            if _looks_id_like(value) and not _value_in_query(value, query):
                return True
    return False


# ───────────────────────── encoder format ─────────────────────────

def _build_correction_encoder_input(tokenizer, query, tools, draft, max_enc_len,
                                    query_cap=256, draft_min=128):
    """Build [query, <tools>, tools_json, <tool_call>, draft_json] up to max_enc_len.

    Truncation policy: query gets up to `query_cap` tokens, then tools and
    draft share the remaining budget with tools getting priority but
    always leaving at least `draft_min` tokens for the draft.
    """
    q_toks = tokenizer.encode(query)[:query_cap]
    t_toks = tokenizer.encode(tools)
    d_toks = tokenizer.encode(draft)

    budget = max_enc_len - len(q_toks) - 2  # two separator tokens
    tools_cap = max(0, budget - draft_min)
    t_toks = t_toks[:min(len(t_toks), tools_cap)]
    budget -= len(t_toks)
    d_toks = d_toks[:max(0, budget)]
    return q_toks + [TOOLS_ID] + t_toks + [TOOL_CALL_ID] + d_toks


def _tokenize_example(tokenizer, sp_model, row, max_enc_len, max_dec_len):
    """Build padded (enc, dec_in, dec_tgt, loss_mask) arrays for one row.

    Returns None if the row is filtered (unrecoverable hallucination) or the
    decoder target doesn't fit in max_dec_len.
    """
    # Retroactive cleanup: drafts from pre-fix mining runs have a <tool_call>
    # prefix that would re-encode to a duplicate TOOL_CALL_ID token. Strip it.
    draft = row.get("draft", "")
    if isinstance(draft, str) and draft.startswith("<tool_call>"):
        draft = draft[len("<tool_call>"):]

    # Drop rows where the gold fabricates an ID-shaped value that isn't in the
    # query. Pass-through correct rows are kept unconditionally — they teach
    # "don't change a correct answer" even if the gold has an ID-looking value.
    if row.get("category") != "correct":
        if _gold_has_unrecoverable(row.get("query", ""), row.get("answers", "")):
            return None

    enc_tokens = _build_correction_encoder_input(
        tokenizer, row["query"], row["tools"], draft, max_enc_len)

    answer = row["answers"]
    a_toks = tokenizer.encode(answer)
    if 2 + len(a_toks) + 1 > max_dec_len:
        return None

    dec_in_tokens = [EOS_ID, TOOL_CALL_ID] + list(a_toks)
    dec_tgt_tokens = [TOOL_CALL_ID] + list(a_toks) + [EOS_ID]
    loss_classes_full = np.zeros(len(dec_tgt_tokens), dtype=np.int8)
    token_classes = _token_classes_for_answer(answer, a_toks, sp_model)
    loss_classes_full[1:1 + len(token_classes)] = token_classes

    enc = np.full(max_enc_len, PAD_ID, dtype=np.int32)
    enc[:len(enc_tokens)] = enc_tokens
    dec_in = np.full(max_dec_len, PAD_ID, dtype=np.int32)
    dec_in[:len(dec_in_tokens)] = dec_in_tokens
    dec_tgt = np.full(max_dec_len, PAD_ID, dtype=np.int32)
    dec_tgt[:len(dec_tgt_tokens)] = dec_tgt_tokens
    loss = np.zeros(max_dec_len, dtype=np.int32)
    loss[:len(loss_classes_full)] = loss_classes_full
    return enc, dec_in, dec_tgt, loss


# ───────────────────────── streaming loader ─────────────────────────

def _read_jsonl_shuffled(path, seed):
    """Generator over (row_dict) in shuffled order. Loads full index into memory."""
    # Build line-offset index for random access
    offsets = []
    with open(path, "rb") as f:
        pos = 0
        for line in f:
            offsets.append(pos)
            pos += len(line)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(offsets))
    with open(path, "rb") as f:
        for idx in order:
            f.seek(offsets[idx])
            line = f.readline()
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _stream_batches(tokenizer, sp_model, path, batch_size,
                     max_enc_len, max_dec_len, seed):
    """Yield (enc, dec_in, dec_tgt, loss_mask) numpy batches from a JSONL file."""
    enc_batch = np.full((batch_size, max_enc_len), PAD_ID, dtype=np.int32)
    dec_in_batch = np.full((batch_size, max_dec_len), PAD_ID, dtype=np.int32)
    dec_tgt_batch = np.full((batch_size, max_dec_len), PAD_ID, dtype=np.int32)
    loss_batch = np.zeros((batch_size, max_dec_len), dtype=np.int32)
    idx = 0

    for row in _read_jsonl_shuffled(path, seed):
        result = _tokenize_example(tokenizer, sp_model, row,
                                    max_enc_len, max_dec_len)
        if result is None:
            continue
        enc, di, dt, lm = result
        enc_batch[idx] = enc
        dec_in_batch[idx] = di
        dec_tgt_batch[idx] = dt
        loss_batch[idx] = lm
        idx += 1
        if idx == batch_size:
            yield (enc_batch.copy(), dec_in_batch.copy(),
                   dec_tgt_batch.copy(), loss_batch.copy())
            enc_batch[:] = PAD_ID
            dec_in_batch[:] = PAD_ID
            dec_tgt_batch[:] = PAD_ID
            loss_batch[:] = 0
            idx = 0


class _PrefetchStream:
    def __init__(self, gen_fn, prefetch=4):
        self._queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._produce, args=(gen_fn,), daemon=True)
        self._thread.start()

    def _produce(self, gen_fn):
        try:
            for batch in gen_fn():
                if self._stop.is_set():
                    return
                self._queue.put(batch)
            self._queue.put(None)
        except Exception as e:
            self._queue.put(e)

    def __iter__(self): return self
    def __next__(self):
        item = self._queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self._stop.set()


# ───────────────────────── train step ─────────────────────────

def _train_step(state, src, tgt_in, tgt_out, loss_mask, rng):
    """CE + z_loss with token-class weighting. Mirrors train.py."""

    def loss_fn(params):
        src_mask = make_padding_mask(src, PAD_ID)
        tgt_mask = make_causal_mask(tgt_in.shape[1]) & make_padding_mask(tgt_in, PAD_ID)
        logits = state.apply_fn(
            {"params": params}, src, tgt_in,
            src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=src_mask,
        )
        logits_f32 = logits.astype(jnp.float32)
        token_weights = _LOSS_WEIGHT_MAP[loss_mask]
        padding_mask = (tgt_out != PAD_ID).astype(jnp.float32)
        mask = token_weights * padding_mask
        num_tokens = jnp.maximum(jnp.sum(padding_mask), 1.0)
        ce = jnp.sum(
            optax.softmax_cross_entropy_with_integer_labels(logits_f32, tgt_out) * mask
        ) / num_tokens
        z_loss = 1e-4 * jnp.mean(jax.nn.logsumexp(logits_f32, axis=-1) ** 2)
        return ce + z_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    return state, loss, grad_norm


# ───────────────────────── eval: stratified sampling ─────────────────────────

def _stratified_val_pools(val_ds, per_bucket, seed):
    """Return (single_examples, multi_examples) sampled identically to train.py.

    Same bucketing by min(len(tools), 10), same rng math.
    """
    pool_names = ["single", "multi"]
    buckets = {n: {t: [] for t in range(11)} for n in pool_names}

    for k in range(len(val_ds)):
        ex = val_ds[k]
        try:
            answers = json.loads(ex["answers"])
        except (ValueError, TypeError):
            continue
        if not answers or not isinstance(answers, list):
            continue
        call_type = "single" if len(answers) == 1 else "multi"
        try:
            nc = min(len(json.loads(ex["tools"])), 10)
        except (ValueError, TypeError):
            nc = 0
        buckets[call_type][nc].append(k)

    rng = np.random.RandomState(seed)
    out = {}
    for name in pool_names:
        pool = []
        for t in range(11):
            b = np.array(buckets[name][t])
            if len(b) > 0:
                rng.shuffle(b)
                pool.extend(b[:per_bucket].tolist())
        rng.shuffle(pool)
        out[name] = [val_ds[k] for k in pool]
    return out["single"], out["multi"]


# ───────────────────────── eval: metrics ─────────────────────────

_PC_KEYS = ("n", "exact", "name_tp", "name_fp", "name_fn",
            "call_tp", "call_fp", "call_fn", "parse_err")


def _call_key(c):
    if not isinstance(c, dict):
        return None
    return json.dumps({"name": c.get("name"), "arguments": c.get("arguments")},
                      sort_keys=True)


def _score_pool(examples, preds):
    """Compute tool-call metrics for one pool. Mirrors train.py._eval_pool."""
    m_n = m_exact = m_parse_err = 0
    m_name_tp = m_name_fp = m_name_fn = 0
    m_call_tp = m_call_fp = m_call_fn = 0
    m_args_correct = m_args_total = 0
    m_halluc = m_total_pred_params = 0
    m_missing = m_total_ref_params = 0
    m_correct_values = m_matched_params = 0
    m_per_count = {t: {k: 0 for k in _PC_KEYS} for t in range(11)}

    for ex, pred_text in zip(examples, preds):
        try:
            tool_defs = json.loads(ex["tools"])
            num_tools = min(len(tool_defs), 10)
        except (ValueError, TypeError):
            tool_defs = []
            num_tools = 0
        pc = m_per_count[num_tools]

        ref_text = ex["answers"].strip()
        pred_text = pred_text.strip()
        ref_is_empty = ref_text in ("", "[]")
        pred_is_empty = pred_text in ("", "[]")

        try:
            ref_calls = json.loads(ref_text) if not ref_is_empty else []
        except (ValueError, TypeError):
            ref_calls = []
        try:
            pred_calls = json.loads(pred_text) if not pred_is_empty else []
            if not isinstance(pred_calls, list):
                pred_calls = [pred_calls] if isinstance(pred_calls, dict) else []
        except (ValueError, TypeError):
            m_parse_err += 1
            pc["parse_err"] += 1
            pred_calls = []

        m_n += 1
        pc["n"] += 1
        if ref_is_empty and pred_is_empty:
            m_exact += 1
            pc["exact"] += 1
        elif not ref_is_empty and not pred_is_empty:
            r = sorted([_call_key(c) for c in ref_calls if _call_key(c)])
            p = sorted([_call_key(c) for c in pred_calls if _call_key(c)])
            if r == p and len(r) == len(ref_calls) and len(p) == len(pred_calls):
                m_exact += 1
                pc["exact"] += 1

        ref_names = {c["name"] for c in ref_calls if isinstance(c, dict) and "name" in c}
        pred_names = {c["name"] for c in pred_calls if isinstance(c, dict) and "name" in c}
        m_name_tp += len(pred_names & ref_names); pc["name_tp"] += len(pred_names & ref_names)
        m_name_fp += len(pred_names - ref_names); pc["name_fp"] += len(pred_names - ref_names)
        m_name_fn += len(ref_names - pred_names); pc["name_fn"] += len(ref_names - pred_names)

        rk = {_call_key(c) for c in ref_calls} - {None}
        pk = {_call_key(c) for c in pred_calls} - {None}
        m_call_tp += len(pk & rk); pc["call_tp"] += len(pk & rk)
        m_call_fp += len(pk - rk); pc["call_fp"] += len(pk - rk)
        m_call_fn += len(rk - pk); pc["call_fn"] += len(rk - pk)

        ref_by_name = {}
        for c in ref_calls:
            if isinstance(c, dict) and "name" in c:
                ref_by_name.setdefault(c["name"], []).append(c.get("arguments", {}))
        for c in pred_calls:
            if isinstance(c, dict) and "name" in c and c["name"] in ref_by_name:
                m_args_total += 1
                pa = json.dumps(c.get("arguments", {}), sort_keys=True)
                if any(pa == json.dumps(ra, sort_keys=True) for ra in ref_by_name[c["name"]]):
                    m_args_correct += 1

        tool_param_map = {}
        for t in tool_defs:
            if isinstance(t, dict) and "name" in t:
                tool_param_map[t["name"]] = set((t.get("parameters") or {}).keys())
        for c in pred_calls:
            if not isinstance(c, dict) or "name" not in c:
                continue
            cname = c["name"]
            if cname not in tool_param_map:
                continue
            pred_keys = set((c.get("arguments") or {}).keys())
            m_total_pred_params += len(pred_keys)
            m_halluc += len(pred_keys - tool_param_map[cname])
            if cname in ref_by_name:
                ref_args = ref_by_name[cname][0]
                ref_keys = set((ref_args if isinstance(ref_args, dict) else {}).keys())
                m_total_ref_params += len(ref_keys)
                m_missing += len(ref_keys - pred_keys)
                matched = pred_keys & ref_keys
                m_matched_params += len(matched)
                for k in matched:
                    if json.dumps(c.get("arguments", {})[k], sort_keys=True) == \
                       json.dumps(ref_args[k], sort_keys=True):
                        m_correct_values += 1

    if m_n == 0:
        return {}, m_per_count
    np_ = m_name_tp + m_name_fp
    nr_ = m_name_tp + m_name_fn
    cp_ = m_call_tp + m_call_fp
    cr_ = m_call_tp + m_call_fn
    return {
        "n": m_n,
        "parse_rate": 1.0 - m_parse_err / m_n,
        "exact_match": m_exact / m_n,
        "name_f1": 2 * m_name_tp / max(np_ + nr_, 1),
        "call_f1": 2 * m_call_tp / max(cp_ + cr_, 1),
        "args_acc": m_args_correct / max(m_args_total, 1),
        "param_haluc": m_halluc / max(m_total_pred_params, 1),
        "param_miss": m_missing / max(m_total_ref_params, 1),
        "value_acc": m_correct_values / max(m_matched_params, 1),
    }, m_per_count


def _print_metrics(label, metrics, pc):
    if not metrics:
        return
    print(f"  ─── {label} ({metrics['n']} samples) ──")
    for k, key in [("JSON parse", "parse_rate"), ("Name F1", "name_f1"),
                    ("Param haluc", "param_haluc"), ("Param miss", "param_miss"),
                    ("Value acc", "value_acc"), ("Args acc", "args_acc"),
                    ("Call F1", "call_f1"), ("Exact match", "exact_match")]:
        print(f"  {k:<14} {metrics[key]:>10.1%}")
    has_any = any(pc[t]["n"] > 0 for t in range(11))
    if has_any:
        print(f"  {'#tools':>6}  {'n':>4}  {'name_f1':>8}  {'call_f1':>8}  {'exact':>6}")
        for t in range(11):
            d = pc[t]
            if d["n"] == 0:
                continue
            np_ = d["name_tp"] + d["name_fp"]; nr_ = d["name_tp"] + d["name_fn"]
            cp_ = d["call_tp"] + d["call_fp"]; cr_ = d["call_tp"] + d["call_fn"]
            nf1 = 2 * d["name_tp"] / max(np_ + nr_, 1)
            cf1 = 2 * d["call_tp"] / max(cp_ + cr_, 1)
            ex = d["exact"] / d["n"]
            print(f"  {t:>6}  {d['n']:>4}  {nf1:>7.1%}  {cf1:>7.1%}  {ex:>5.1%}")


def _print_comparison(label, draft_m, corr_m):
    """Print side-by-side draft vs corrected metrics."""
    if not draft_m or not corr_m:
        return
    print(f"\n  ─── {label} — draft vs corrected ───")
    print(f"  {'metric':<14} {'draft':>8} {'corr':>8} {'Δ':>8}")
    for k, key in [("JSON parse", "parse_rate"), ("Name F1", "name_f1"),
                    ("Param haluc", "param_haluc"), ("Param miss", "param_miss"),
                    ("Value acc", "value_acc"), ("Args acc", "args_acc"),
                    ("Call F1", "call_f1"), ("Exact match", "exact_match")]:
        d, c = draft_m[key], corr_m[key]
        sign = "+" if c >= d else ""
        print(f"  {k:<14} {d:>8.1%} {c:>8.1%} {sign}{c-d:>7.1%}")


# ───────────────────────── corrector generation ─────────────────────────

def _generate_corrector_batch(model, params, tokenizer, queries, tools_list,
                               drafts, max_gen_len, max_enc_len,
                               batch_size=32, desc="corrector-gen"):
    """Greedy generate with encoder=[query, <tools>, tools, <tool_call>, draft]."""
    from .run import _get_decode_fn
    pad_id = PAD_ID
    eos_id = EOS_ID
    decode_fn = _get_decode_fn(model, max_gen_len)

    all_preds = []
    n_total = len(queries)
    pbar = tqdm(total=n_total, desc=desc, unit="ex", leave=False)
    for start in range(0, n_total, batch_size):
        chunk_q = queries[start:start + batch_size]
        chunk_t = tools_list[start:start + batch_size]
        chunk_d = drafts[start:start + batch_size]
        B = len(chunk_q)

        enc_token_lists = [
            _build_correction_encoder_input(tokenizer, q, t, d, max_enc_len)
            for q, t, d in zip(chunk_q, chunk_t, chunk_d)
        ]
        max_enc = max(len(toks) for toks in enc_token_lists)
        enc_input = np.full((B, max_enc), pad_id, dtype=np.int32)
        for i, toks in enumerate(enc_token_lists):
            enc_input[i, :len(toks)] = toks
        enc_input = jnp.array(enc_input)
        src_mask = make_padding_mask(enc_input, pad_id)

        encoder_out, enc_mask = model.apply(
            {"params": params}, enc_input, src_mask=src_mask, method="encode")

        dec_buffer = np.full((B, max_gen_len), pad_id, dtype=np.int32)
        dec_buffer[:, 0] = eos_id
        dec_buffer = jnp.array(dec_buffer)

        finished = [False] * B
        gen_tokens = [[] for _ in range(B)]
        logits = decode_fn(params, dec_buffer, encoder_out, enc_mask)

        for pos in range(max_gen_len - 1):
            for i in range(B):
                if finished[i]:
                    continue
                next_token = int(jnp.argmax(logits[i, pos]))
                if next_token == eos_id:
                    finished[i] = True
                    continue
                gen_tokens[i].append(next_token)
                dec_buffer = dec_buffer.at[i, pos + 1].set(next_token)
            if all(finished):
                break
            logits = decode_fn(params, dec_buffer, encoder_out, enc_mask)

        for i in range(B):
            text = tokenizer.decode(gen_tokens[i])
            if text.startswith("<tool_call>"):
                text = text[len("<tool_call>"):]
            all_preds.append(text)
        pbar.update(B)
    pbar.close()
    return all_preds


# ───────────────────────── main ─────────────────────────

def train_corrector(args):
    if os.path.exists("/dev/accel0"):
        jax.distributed.initialize()

    num_devices = jax.local_device_count()
    num_hosts = jax.process_count()
    host_id = jax.process_index()
    is_main = host_id == 0

    use_wandb = args.wandb and is_main
    if use_wandb:
        import wandb
        if wandb.run is None:
            wandb.init(project="needle-v1", name=args.name, config=vars(args))

    print(f"[1/5] Devices: {num_devices} local, {jax.device_count()} total")

    print(f"[2/5] Loading tokenizer and SP model ...")
    tokenizer = get_tokenizer()
    sp_model = tokenizer.sp  # SentencePiece processor for class computation

    # ─── load base (frozen, for drafts) and corrector init ───
    print(f"[3/5] Loading base SFT checkpoint: {args.base_checkpoint}")
    base_params, base_config = load_checkpoint(args.base_checkpoint)
    base_model = EncoderDecoderTransformer(base_config)

    print(f"       Loading corrector init: {args.corrector_init}")
    corr_params, corr_config = load_checkpoint(args.corrector_init)
    corr_model = EncoderDecoderTransformer(corr_config)
    param_count = sum(x.size for x in jax.tree.leaves(corr_params))
    print(f"       Corrector parameters: {param_count:,}")

    # ─── sample val pools (train.py-compatible) and cache base drafts ───
    print(f"[4/5] Sampling stratified val pools and caching base drafts ...")
    val_ds = load_tool_calls("val")
    single_pool, multi_pool = _stratified_val_pools(
        val_ds, per_bucket=args.tc_per_bucket, seed=args.eval_seed)
    print(f"       single-call: {len(single_pool)}, multi-call: {len(multi_pool)}")

    def _gen_drafts(pool, label, chunk=64):
        queries = [ex["query"] for ex in pool]
        tools_list = [ex["tools"] for ex in pool]
        print(f"       Generating {label} drafts from base model ...")
        t0 = time.perf_counter()
        params_j = jax.tree.map(jnp.array, base_params)
        drafts = []
        pbar = tqdm(total=len(queries), desc=f"base-draft-{label}",
                    unit="ex", leave=False)
        for i in range(0, len(queries), chunk):
            drafts.extend(generate_batch(
                base_model, params_j, tokenizer,
                queries[i:i + chunk], tools_list[i:i + chunk],
                max_gen_len=args.max_gen_len, max_enc_len=args.max_enc_len,
                constrained=False))
            pbar.update(min(chunk, len(queries) - i))
        pbar.close()
        print(f"       {label}: {len(drafts)} drafts in {time.perf_counter()-t0:.1f}s")
        return drafts

    single_drafts = _gen_drafts(single_pool, "single") if is_main else []
    multi_drafts = _gen_drafts(multi_pool, "multi") if is_main else []

    # Free base model weights — no longer needed
    del base_params, base_model
    import gc; gc.collect()

    # ─── estimate total steps for schedule ───
    print(f"[5/5] Building optimizer + pmap ...")
    with open(args.data) as f:
        n_rows = sum(1 for _ in f)
    effective_batch = args.batch_size * num_devices
    global_batch = effective_batch * num_hosts
    steps_per_epoch = n_rows // global_batch
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.05))
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    # ─── build state ───
    scaled_lr = args.lr * jax.device_count()
    muon_lr = 0.02 * math.sqrt(jax.device_count())
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, corr_config, scaled_lr, muon_lr,
                                total_steps, warmup, decay_ratio=0.05)

    # Load corrector init weights into state
    corr_params_cast = jax.tree.map(
        lambda ref, loaded: jnp.asarray(loaded, dtype=ref.dtype),
        state.params, jax.tree.map(jnp.array, corr_params))
    state = state.replace(params=corr_params_cast)
    del corr_params, corr_params_cast

    state = _replicate(state)
    p_train_step = jax.pmap(_train_step, axis_name="batch", donate_argnums=(0,))

    adam_schedule = _wsd_schedule(scaled_lr, total_steps, warmup)

    if is_main:
        print(f"\n  ─────────────────────────────────────")
        print(f"  Corrector fine-tune")
        print(f"  ─────────────────────────────────────")
        print(f"  Train rows      {n_rows:>12,}")
        print(f"  Steps/epoch     {steps_per_epoch:>12,}")
        print(f"  Total steps     {total_steps:>12,}")
        print(f"  Epochs          {args.epochs:>12}")
        print(f"  Batch           {args.batch_size:>4} x {jax.device_count()} = {global_batch}")
        print(f"  LR              {args.lr:>7} x {jax.device_count()} = {scaled_lr}")
        print(f"  Encoder max     {args.max_enc_len:>12}")
        print(f"  Decoder max     {args.max_dec_len:>12}")
        print(f"  ─────────────────────────────────────\n")

    # ─── baseline eval (draft-only metrics) ───
    baseline_single_m = baseline_multi_m = None
    baseline_single_pc = baseline_multi_pc = None
    if is_main:
        print("Baseline eval (draft-only, no correction) ...")
        baseline_single_m, baseline_single_pc = _score_pool(single_pool, single_drafts)
        baseline_multi_m, baseline_multi_pc = _score_pool(multi_pool, multi_drafts)
        _print_metrics("BASELINE Single-Call", baseline_single_m, baseline_single_pc)
        _print_metrics("BASELINE Multi-Call", baseline_multi_m, baseline_multi_pc)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    global_step = 0
    best_call_f1 = -1.0
    best_path = None

    def _run_eval():
        nonlocal best_call_f1, best_path
        if not is_main:
            return
        eval_params = _unreplicate(state).params
        preds_single = _generate_corrector_batch(
            corr_model, eval_params, tokenizer,
            [ex["query"] for ex in single_pool],
            [ex["tools"] for ex in single_pool],
            single_drafts,
            max_gen_len=args.max_gen_len, max_enc_len=args.max_enc_len,
            desc=f"eval-step{global_step}-single")
        preds_multi = _generate_corrector_batch(
            corr_model, eval_params, tokenizer,
            [ex["query"] for ex in multi_pool],
            [ex["tools"] for ex in multi_pool],
            multi_drafts,
            max_gen_len=args.max_gen_len, max_enc_len=args.max_enc_len,
            desc=f"eval-step{global_step}-multi")
        single_m, single_pc = _score_pool(single_pool, preds_single)
        multi_m, multi_pc = _score_pool(multi_pool, preds_multi)
        _print_metrics("CORR Single-Call", single_m, single_pc)
        _print_metrics("CORR Multi-Call", multi_m, multi_pc)
        _print_comparison("Single-Call", baseline_single_m, single_m)
        _print_comparison("Multi-Call", baseline_multi_m, multi_m)

        cur_f1 = single_m.get("call_f1", -1)
        if cur_f1 > best_call_f1:
            best_call_f1 = cur_f1
            best_path = os.path.join(args.checkpoint_dir, "needle_corrector_best.pkl")
            params_np = jax.tree.map(lambda x: np.array(x).astype(np.float16), eval_params)
            with open(best_path, "wb") as f:
                pickle.dump({"params": params_np, "config": corr_config.__dict__,
                             "step": global_step}, f)
            print(f"  ** New best single call_f1={best_call_f1:.1%} -> {best_path}")
            _upload_checkpoint(best_path)

        if use_wandb:
            import wandb
            log = {"corrector/step": global_step,
                   "best/single_call_f1": best_call_f1}
            for scope, m in (("single", single_m), ("multi", multi_m)):
                for k in ("parse_rate", "exact_match", "name_f1", "call_f1",
                          "args_acc", "value_acc", "param_haluc", "param_miss"):
                    log[f"val/{scope}_{k}"] = m.get(k, 0)
            for scope, m in (("single", baseline_single_m), ("multi", baseline_multi_m)):
                for k in ("call_f1", "value_acc", "exact_match"):
                    log[f"baseline/{scope}_{k}"] = m.get(k, 0)
            wandb.log(log)

    # ─── training loop ───
    for epoch in range(args.epochs):
        stream = _PrefetchStream(
            lambda: _stream_batches(tokenizer, sp_model, args.data, global_batch,
                                     args.max_enc_len, args.max_dec_len,
                                     seed=args.seed + epoch),
            prefetch=4)
        pbar = tqdm(total=steps_per_epoch, desc=f"Corrector epoch {epoch+1}/{args.epochs}",
                    disable=not is_main)
        losses = []
        t0 = time.perf_counter()
        try:
            for batch in stream:
                if args.max_steps and global_step >= args.max_steps:
                    break
                src, tgt_in, tgt_out, lm = batch
                host_slice = slice(host_id * effective_batch,
                                    (host_id + 1) * effective_batch)
                src = src[host_slice]; tgt_in = tgt_in[host_slice]
                tgt_out = tgt_out[host_slice]; lm = lm[host_slice]

                src_b = shard_batch(src, num_devices)
                tgt_in_b = shard_batch(tgt_in, num_devices)
                tgt_out_b = shard_batch(tgt_out, num_devices)
                lm_b = shard_batch(lm, num_devices)

                rng, step_rng = jax.random.split(rng)
                rngs = jax.random.split(step_rng, num_devices)

                state, loss, gnorm = p_train_step(
                    state, src_b, tgt_in_b, tgt_out_b, lm_b, rngs)
                loss_val = float(loss.addressable_shards[0].data[0])
                gnorm_val = float(gnorm.addressable_shards[0].data[0])
                losses.append(loss_val)
                global_step += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss_val:.4f}",
                                  gnorm=f"{gnorm_val:.2f}",
                                  lr=f"{float(adam_schedule(global_step)):.2e}")
                if use_wandb:
                    import wandb
                    wandb.log({"corrector/loss": loss_val,
                               "corrector/grad_norm": gnorm_val,
                               "corrector/lr": float(adam_schedule(global_step)),
                               "corrector/step": global_step})

        finally:
            stream.close()
            pbar.close()
        _run_eval()

    # Final save (last eval already ran at end of final epoch)
    if is_main:
        final_path = os.path.join(args.checkpoint_dir, "needle_corrector.pkl")
        final_params = _unreplicate(state).params
        params_np = jax.tree.map(lambda x: np.array(x).astype(np.float16), final_params)
        with open(final_path, "wb") as f:
            pickle.dump({"params": params_np, "config": corr_config.__dict__,
                         "step": global_step}, f)
        print(f"\nFinal corrector: {final_path}")
        if best_path:
            print(f"Best corrector:  {best_path}  (call_f1={best_call_f1:.1%})")
        _upload_checkpoint(final_path)

    if use_wandb:
        import wandb
        wandb.finish()
    if num_hosts > 1:
        jax.experimental.multihost_utils.sync_global_devices("corrector_done")


def parse_args():
    p = argparse.ArgumentParser(description="Stage-2 correction fine-tuning")
    p.add_argument("--base-checkpoint", type=str, default="checkpoints/needle.pkl",
                   help="Frozen SFT checkpoint used to generate drafts for eval")
    p.add_argument("--corrector-init", type=str, default="checkpoints/needle.pkl",
                   help="Checkpoint to initialize the corrector from (default: same as base)")
    p.add_argument("--data", type=str, default="data/corrections.jsonl")
    p.add_argument("--name", type=str, default="corrector")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-enc-len", type=int, default=1024)
    p.add_argument("--max-dec-len", type=int, default=512)
    p.add_argument("--max-gen-len", type=int, default=256)
    p.add_argument("--tc-per-bucket", type=int, default=50,
                   help="Eval samples per tool-count bucket (train.py default)")
    p.add_argument("--eval-every", type=int, default=500,
                   help="Run full stratified eval every N steps")
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train_corrector(parse_args())
