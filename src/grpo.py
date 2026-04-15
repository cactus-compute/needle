"""GRPO fine-tuning from an SFT checkpoint.

For each prompt we sample K completions from the current policy, score them
with a programmatic reward (exact call-set match against the reference answer),
normalize rewards within the group to get advantages, and take a policy
gradient step with a KL penalty against a frozen reference (the SFT ckpt).

No KV cache, no constrained decoding, no reward model.

Usage:
    needle grpo --checkpoint checkpoints/needle.pkl --wandb
"""

import json
import math
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import tqdm

from .data import (
    load_tool_calls, load_prepared_data, get_batches, to_snake_case,
)
from .distributed import (
    _replicate, _unreplicate, _upload_checkpoint,
)

_HF_SFT_REPO = "Cactus-Compute/needle"


def _download_sft_checkpoint(path_or_name):
    """Return a local path to the SFT checkpoint, downloading from HF if needed.

    The SFT checkpoint lives in Cactus-Compute/needle, not the generic
    checkpoints repo that _download_checkpoint targets.
    """
    if os.path.exists(path_or_name):
        return path_or_name
    from huggingface_hub import hf_hub_download
    local_dir = os.path.dirname(path_or_name) or "checkpoints"
    os.makedirs(local_dir, exist_ok=True)
    print(f"  Downloading {os.path.basename(path_or_name)} "
          f"from {_HF_SFT_REPO}...", flush=True)
    local_path = hf_hub_download(
        repo_id=_HF_SFT_REPO,
        filename=os.path.basename(path_or_name),
        repo_type="model",
        local_dir=local_dir,
    )
    print(f"  Downloaded to {local_path}", flush=True)
    return local_path
from .model import (
    EncoderDecoderTransformer, TransformerConfig,
    make_causal_mask, make_padding_mask,
    make_packing_mask, make_causal_packing_mask, make_cross_packing_mask,
)
from .optim import _wsd_schedule
from .run import _build_encoder_input
from .tokenizer import EOS_ID, PAD_ID, get_tokenizer


def _call_key_set(calls):
    """Canonicalize a list of tool calls: snake_case names, sorted."""
    keys = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        name = to_snake_case(c.get("name", ""))
        args = c.get("arguments", {}) or {}
        keys.append(json.dumps({"name": name, "arguments": args}, sort_keys=True))
    return sorted(keys)


def _score(pred_text, ref_answers):
    """Binary reward: 1.0 if pred call-set == ref call-set (order-invariant)."""
    if pred_text.startswith("<tool_call>"):
        pred_text = pred_text[len("<tool_call>"):]
    try:
        pred_calls = json.loads(pred_text)
        if isinstance(pred_calls, dict):
            pred_calls = [pred_calls]
        elif not isinstance(pred_calls, list):
            return 0.0
    except (json.JSONDecodeError, TypeError):
        return 0.0
    try:
        ref_calls = json.loads(ref_answers)
        if not isinstance(ref_calls, list):
            ref_calls = []
    except (json.JSONDecodeError, TypeError):
        return 0.0
    return 1.0 if _call_key_set(pred_calls) == _call_key_set(ref_calls) else 0.0


def _build_prompts(tokenizer, batch_examples, max_enc_len):
    """Tokenize and pad encoder inputs for a batch of prompts."""
    enc_lists = [
        _build_encoder_input(tokenizer, ex["query"], ex["tools"], max_enc_len)
        for ex in batch_examples
    ]
    B = len(enc_lists)
    enc = np.full((B, max_enc_len), PAD_ID, dtype=np.int32)
    for i, toks in enumerate(enc_lists):
        enc[i, :len(toks)] = toks
    return enc


def _make_fns(model, max_gen_len):
    """Build pmapped encode / sample-step / log-prob functions."""

    @jax.pmap
    def p_encode(params, enc_input):
        src_mask = make_padding_mask(enc_input, PAD_ID)
        return model.apply({"params": params}, enc_input,
                           src_mask=src_mask, method="encode")

    @jax.pmap
    def p_sample_step(params, dec_buffer, encoder_out, cross_mask,
                      rng, step_idx, finished):
        sm = make_causal_mask(max_gen_len)
        logits = model.apply(
            {"params": params}, dec_buffer, encoder_out,
            self_mask=sm, cross_mask=cross_mask, method="decode",
        )
        next_logits = jax.lax.dynamic_index_in_dim(
            logits, step_idx, axis=1, keepdims=False,
        )
        next_tokens = jax.random.categorical(rng, next_logits, axis=-1)
        next_tokens = jnp.where(finished, PAD_ID, next_tokens)
        new_finished = finished | (next_tokens == EOS_ID)
        return next_tokens, new_finished

    @jax.pmap
    def p_log_probs(params, enc_input, dec_in, dec_tgt):
        src_mask = make_padding_mask(enc_input, PAD_ID)
        tm = make_causal_mask(dec_in.shape[-1]) & make_padding_mask(dec_in, PAD_ID)
        logits = model.apply({"params": params}, enc_input, dec_in,
                             src_mask=src_mask, tgt_mask=tm)
        lp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        return jnp.take_along_axis(lp, dec_tgt[..., None], axis=-1).squeeze(-1)

    return p_encode, p_sample_step, p_log_probs


def _make_grpo_step(model, beta):
    """Build pmapped GRPO gradient step."""

    def loss_fn(params, enc_input, dec_in, dec_tgt, advantages,
                response_mask, ref_logp):
        src_mask = make_padding_mask(enc_input, PAD_ID)
        tm = make_causal_mask(dec_in.shape[-1]) & make_padding_mask(dec_in, PAD_ID)
        logits = model.apply({"params": params}, enc_input, dec_in,
                             src_mask=src_mask, tgt_mask=tm)
        lp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        pol_logp = jnp.take_along_axis(lp, dec_tgt[..., None], axis=-1).squeeze(-1)

        denom = jnp.maximum(response_mask.sum(), 1.0)
        pg = -(advantages[:, None] * pol_logp * response_mask).sum() / denom

        # Schulman k3 KL estimator: unbiased, always non-negative.
        log_ratio = ref_logp - pol_logp
        kl_tok = (jnp.exp(log_ratio) - log_ratio - 1.0) * response_mask
        kl = kl_tok.sum() / denom

        return pg + beta * kl, (pg, kl)

    def step(state, enc_input, dec_in, dec_tgt, advantages,
             response_mask, ref_logp):
        (loss, (pg, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, enc_input, dec_in, dec_tgt, advantages,
            response_mask, ref_logp,
        )
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss = jax.lax.pmean(loss, axis_name="batch")
        pg = jax.lax.pmean(pg, axis_name="batch")
        kl = jax.lax.pmean(kl, axis_name="batch")
        state = state.apply_gradients(grads=grads)
        return state, loss, pg, kl

    return jax.pmap(step, axis_name="batch", donate_argnums=(0,))


def _sample_completions(p_encode, p_sample_step, r_params, enc_sharded,
                         num_devices, max_gen_len, rng):
    """Sample one completion per input row, in parallel across devices.

    enc_sharded: (num_devices, per_device, max_enc_len) already on-device.
    Returns numpy array (num_devices, per_device, max_gen_len): each row is
    [EOS, sampled..., EOS, PAD...] or [EOS, sampled...] if EOS wasn't emitted.
    """
    encoder_out, cross_mask = p_encode(r_params, enc_sharded)
    per_device = enc_sharded.shape[1]

    dec_buffer = np.full((num_devices, per_device, max_gen_len), PAD_ID, dtype=np.int32)
    dec_buffer[:, :, 0] = EOS_ID
    dec_buffer = jnp.asarray(dec_buffer)
    finished = jnp.zeros((num_devices, per_device), dtype=jnp.bool_)

    for step in range(max_gen_len - 1):
        rng, sub = jax.random.split(rng)
        sub_per = jax.random.split(sub, num_devices)
        step_idx = jnp.full((num_devices,), step, dtype=jnp.int32)
        next_tokens, finished = p_sample_step(
            r_params, dec_buffer, encoder_out, cross_mask,
            sub_per, step_idx, finished,
        )
        dec_buffer = dec_buffer.at[:, :, step + 1].set(next_tokens)

    return np.asarray(dec_buffer), rng


def _make_val_loss_fn(model):
    """Teacher-forced packed val loss (matches train.py)."""

    @jax.jit
    def fn(params, src, dec_in, dec_tgt, enc_seg, dec_seg):
        src_mask = make_packing_mask(enc_seg)
        tgt_mask = make_causal_packing_mask(dec_seg)
        cross_mask = make_cross_packing_mask(enc_seg, dec_seg)
        logits = model.apply({"params": params}, src, dec_in,
                             src_mask=src_mask, tgt_mask=tgt_mask,
                             cross_mask=cross_mask)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), dec_tgt)
        pmask = (dec_seg > 0).astype(jnp.float32)
        return jnp.sum(loss * pmask), jnp.sum(pmask)

    return fn


def _compute_val_ppl(val_loss_fn, params, val_data, batch_size):
    total_loss = 0.0
    total_toks = 0.0
    for vb in get_batches(
        val_data["packed_enc"], val_data["packed_dec_in"], val_data["packed_dec_tgt"],
        batch_size, shuffle=False,
        loss_mask=val_data["packed_loss"],
        enc_seg_ids=val_data["packed_enc_seg"],
        dec_seg_ids=val_data["packed_dec_seg"],
    ):
        src, di, dt, _lm, es, ds = vb
        vl, vt = val_loss_fn(
            params,
            jnp.asarray(src, dtype=jnp.int32),
            jnp.asarray(di, dtype=jnp.int32),
            jnp.asarray(dt, dtype=jnp.int32),
            jnp.asarray(es, dtype=jnp.int32),
            jnp.asarray(ds, dtype=jnp.int32),
        )
        total_loss += float(vl)
        total_toks += float(vt)
    return math.exp(min(total_loss / max(total_toks, 1), 20))


def _stratified_val_slice(val_ds, per_bucket):
    """Pick up to per_bucket single-call + per_bucket multi-call examples."""
    single, multi = [], []
    for ex in val_ds:
        try:
            calls = json.loads(ex["answers"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(calls, list) or not calls:
            continue
        (single if len(calls) == 1 else multi).append(ex)
        if len(single) >= per_bucket and len(multi) >= per_bucket:
            break
    return single[:per_bucket], multi[:per_bucket]


def _run_tool_call_eval(model, params, tokenizer, val_ds, per_bucket,
                         max_enc_len, max_gen_len):
    """Stratified single/multi-call benchmark with display samples.

    Mirrors train.py's end-of-epoch tool-call eval. Uses greedy decode WITHOUT
    constrained decoding — this reveals whether the underlying model is
    actually learning to produce correct tool names, rather than leaning on
    the grammar. `constrained=True` is the right choice at deployment; for
    training telemetry we want to see raw model behavior.
    """
    from .eval import benchmark_tool_calls

    single, multi = _stratified_val_slice(val_ds, per_bucket)
    results = {}
    for label, examples in (("single", single), ("multi", multi)):
        if not examples:
            continue
        tc = benchmark_tool_calls(
            model, params, tokenizer, num_samples=len(examples),
            max_gen_len=max_gen_len, max_enc_len=max_enc_len,
            constrained=False, ds=examples,
        )
        results[label] = {
            "name_f1": tc["name_f1"],
            "call_f1": tc["call_f1"],
            "exact_match": tc["exact_match"],
            "args_acc": tc["args_acc"],
            "value_acc": tc["value_acc"],
            "param_haluc": tc["param_haluc"],
            "param_miss": tc["param_miss"],
            "json_parse": tc["json_parse_rate"],
            "samples": tc["samples"],
            "n": len(examples),
        }
    return results


def _print_eval_block(step, val_ppl, tc_results, pbar):
    """Mirror the end-of-epoch print layout from train.py."""
    pbar.write(f"\n  ─────── step {step} eval ───────")
    if val_ppl is not None:
        pbar.write(f"  val_ppl        {val_ppl:>10.3f}")
    for label, m in tc_results.items():
        pbar.write(f"  ─── {label} ({m['n']} samples, unconstrained) ───")
        pbar.write(f"  JSON parse     {m['json_parse']:>10.1%}")
        pbar.write(f"  Name F1        {m['name_f1']:>10.1%}")
        pbar.write(f"  Call F1        {m['call_f1']:>10.1%}")
        pbar.write(f"  Exact match    {m['exact_match']:>10.1%}")
        pbar.write(f"  Args acc       {m['args_acc']:>10.1%}")
        pbar.write(f"  Value acc      {m['value_acc']:>10.1%}")
        pbar.write(f"  Param haluc    {m['param_haluc']:>10.1%}")
        pbar.write(f"  Param miss     {m['param_miss']:>10.1%}")
    # Show up to 3 display samples from the "single" pool
    single = tc_results.get("single") or tc_results.get("multi")
    if single and single.get("samples"):
        pbar.write(f"  ─── samples ───")
        for q, ref, pred in single["samples"][:3]:
            pbar.write(f"  Q: {q}")
            pbar.write(f"  R: {ref}")
            pbar.write(f"  P: {pred}")
            pbar.write("")


def _decode_sample(tokenizer, token_row):
    """Decode one sampled row to text, stopping at the first EOS after pos 0."""
    sampled = token_row[1:]
    eos_positions = np.where(sampled == EOS_ID)[0]
    end = int(eos_positions[0]) if len(eos_positions) > 0 else len(sampled)
    ids = [int(t) for t in sampled[:end] if t != PAD_ID]
    return tokenizer.decode(ids)


def grpo(args):
    num_devices = jax.local_device_count()
    num_hosts = jax.process_count()
    host_id = jax.process_index()
    total_devices = jax.device_count()
    is_main = host_id == 0

    use_wandb = getattr(args, "wandb", False) and is_main
    if use_wandb:
        import wandb
        if wandb.run is None:
            wandb.init(project="needle-v1", name=args.name, config=vars(args))

    print(f"\n[1/4] Devices: {num_devices} local, {total_devices} total "
          f"across {num_hosts} host(s)", flush=True)

    print(f"\n[2/4] Loading checkpoint {args.checkpoint} ...", flush=True)
    ckpt_path = _download_sft_checkpoint(args.checkpoint)
    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)
    config_dict = {k: v for k, v in data["config"].items()
                   if k in TransformerConfig.__dataclass_fields__}
    config = TransformerConfig(**config_dict)
    model = EncoderDecoderTransformer(config)

    dt = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    params = jax.tree.map(lambda x: jnp.array(x, dtype=dt), data["params"])
    ref_params = jax.tree.map(lambda x: jnp.array(x, dtype=dt), data["params"])

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"      params: {param_count:,}  "
          f"(d={config.d_model}, heads={config.num_heads}, "
          f"layers={config.num_encoder_layers}/{config.num_decoder_layers})")

    print(f"\n[3/4] Building optimizer + pmap ...", flush=True)
    total_steps = args.max_steps
    warmup = max(1, int(total_steps * 0.05))
    scaled_lr = args.lr * total_devices
    schedule = _wsd_schedule(scaled_lr, total_steps, warmup, decay_ratio=0.1)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, b2=0.95, weight_decay=0.0),
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    del params

    r_state = _replicate(state)
    r_ref_params = _replicate(ref_params)
    del state, ref_params

    p_encode, p_sample_step, p_log_probs = _make_fns(model, args.max_gen_len)
    p_grpo_step = _make_grpo_step(model, args.beta)

    print(f"\n[4/4] Loading tokenizer + data ...", flush=True)
    tokenizer = get_tokenizer()
    train_ds = load_tool_calls("train")
    n_examples = len(train_ds)
    print(f"      train: {n_examples:,} examples")

    val_data = None
    val_ds = None
    val_loss_fn = None
    if is_main:
        try:
            val_data = load_prepared_data("val", mmap=True)
            print(f"      val (packed): {len(val_data['packed_enc']):,} bins")
        except FileNotFoundError:
            print("      WARN: no packed val data — skipping val PPL "
                  "(run `needle tokenize` to generate)")
        try:
            val_ds = load_tool_calls("val")
            print(f"      val (tool-call): {len(val_ds):,} examples")
        except Exception as e:
            print(f"      WARN: could not load tool-call val set: {e}")
        val_loss_fn = _make_val_loss_fn(model)

    B = args.batch_size
    K = args.k
    BK = B * K
    if BK % num_devices != 0:
        raise ValueError(f"B*K ({BK}) must be divisible by num_devices ({num_devices})")
    per_device = BK // num_devices

    if is_main:
        print(f"\n  ─────────────────────────────────────")
        print(f"  GRPO")
        print(f"  ─────────────────────────────────────")
        print(f"  B (prompts/step)    {B}")
        print(f"  K (samples/prompt)  {K}")
        print(f"  B*K                 {BK}  ({per_device}/device)")
        print(f"  LR                  {args.lr} -> {scaled_lr}")
        print(f"  beta (KL)           {args.beta}")
        print(f"  max_steps           {args.max_steps}")
        print(f"  max_enc_len         {args.max_enc_len}")
        print(f"  max_gen_len         {args.max_gen_len}")
        print(f"  ─────────────────────────────────────\n")

    rng = jax.random.PRNGKey(args.seed)
    perm = np.random.RandomState(args.seed).permutation(n_examples)
    cursor = 0

    best_call_f1 = -1.0

    pbar = tqdm(total=args.max_steps, disable=not is_main, desc="GRPO")

    for step in range(args.max_steps):
        if cursor + B > len(perm):
            perm = np.random.RandomState(args.seed + step).permutation(n_examples)
            cursor = 0
        batch_idx = perm[cursor:cursor + B]
        cursor += B
        batch = [train_ds[int(i)] for i in batch_idx]

        # Build encoder inputs, replicate K times per prompt, shard across devices.
        enc_np = _build_prompts(tokenizer, batch, args.max_enc_len)
        enc_rep = np.repeat(enc_np, K, axis=0)
        enc_sharded = jnp.asarray(
            enc_rep.reshape(num_devices, per_device, args.max_enc_len)
        )

        # Sample K completions per prompt under the current policy.
        t0 = time.perf_counter()
        dec_buffer_np, rng = _sample_completions(
            p_encode, p_sample_step, r_state.params, enc_sharded,
            num_devices, args.max_gen_len, rng,
        )
        sample_time = time.perf_counter() - t0
        dec_buffer_flat = dec_buffer_np.reshape(BK, args.max_gen_len)

        # Score each completion.
        rewards = np.zeros(BK, dtype=np.float32)
        for i in range(BK):
            prompt_idx = i // K
            text = _decode_sample(tokenizer, dec_buffer_flat[i])
            rewards[i] = _score(text, batch[prompt_idx]["answers"])

        # Group-relative advantages: normalize within each prompt's K samples.
        rewards_grp = rewards.reshape(B, K)
        mean = rewards_grp.mean(axis=1, keepdims=True)
        std = rewards_grp.std(axis=1, keepdims=True)
        advantages = ((rewards_grp - mean) / (std + 1e-4)).reshape(BK).astype(np.float32)

        # Teacher-forced shapes: dec_in = buffer[:-1], dec_tgt = buffer[1:].
        dec_in_np = dec_buffer_flat[:, :-1].astype(np.int32)
        dec_tgt_np = dec_buffer_flat[:, 1:].astype(np.int32)
        response_mask_np = (dec_tgt_np != PAD_ID).astype(np.float32)
        T_resp = dec_in_np.shape[1]

        dec_in_sh = jnp.asarray(dec_in_np.reshape(num_devices, per_device, T_resp))
        dec_tgt_sh = jnp.asarray(dec_tgt_np.reshape(num_devices, per_device, T_resp))
        resp_sh = jnp.asarray(response_mask_np.reshape(num_devices, per_device, T_resp))
        adv_sh = jnp.asarray(advantages.reshape(num_devices, per_device))

        ref_logp_sh = p_log_probs(r_ref_params, enc_sharded, dec_in_sh, dec_tgt_sh)

        r_state, loss, pg, kl = p_grpo_step(
            r_state, enc_sharded, dec_in_sh, dec_tgt_sh,
            adv_sh, resp_sh, ref_logp_sh,
        )

        loss_val = float(loss.addressable_shards[0].data[0])
        pg_val = float(pg.addressable_shards[0].data[0])
        kl_val = float(kl.addressable_shards[0].data[0])
        mean_r = float(rewards.mean())
        std_r = float(rewards.std())

        pbar.update(1)
        pbar.set_postfix(
            loss=f"{loss_val:+.4f}", pg=f"{pg_val:+.4f}",
            kl=f"{kl_val:.4f}", R=f"{mean_r:.3f}", sT=f"{sample_time:.1f}s",
        )

        if use_wandb:
            import wandb
            wandb.log({
                "grpo/loss": loss_val,
                "grpo/pg": pg_val,
                "grpo/kl": kl_val,
                "grpo/reward_mean": mean_r,
                "grpo/reward_std": std_r,
                "grpo/sample_time_s": sample_time,
                "grpo/step": step,
            })

        # Full eval — val PPL + stratified tool-call benchmark + samples.
        # Mirrors the end-of-epoch eval in train.py, run every eval_every steps.
        if is_main and (step + 1) % args.eval_every == 0:
            eval_params = _unreplicate(r_state).params

            val_ppl = None
            if val_data is not None:
                val_ppl = _compute_val_ppl(
                    val_loss_fn, eval_params, val_data, args.eval_batch_size,
                )

            tc_results = {}
            if val_ds is not None:
                tc_results = _run_tool_call_eval(
                    model, eval_params, tokenizer, val_ds,
                    per_bucket=args.tool_call_samples // 2,
                    max_enc_len=args.max_enc_len,
                    max_gen_len=args.max_gen_len,
                )

            _print_eval_block(step + 1, val_ppl, tc_results, pbar)

            # Best-checkpoint tracking: match train.py, use single-call F1.
            cur_f1 = tc_results.get("single", {}).get("call_f1", -1.0)
            is_new_best = cur_f1 > best_call_f1
            if is_new_best and cur_f1 >= 0:
                best_call_f1 = cur_f1
                _save_grpo_checkpoint(
                    r_state, config, args.checkpoint_dir, step + 1,
                    filename="needle_grpo_best.pkl",
                )
                pbar.write(f"  ** New best single call_f1={best_call_f1:.1%}")

            if use_wandb:
                import wandb
                log = {"grpo/step": step + 1, "val/best_call_f1": best_call_f1}
                if val_ppl is not None:
                    log["val/text_ppl"] = val_ppl
                for label, metrics in tc_results.items():
                    for k, v in metrics.items():
                        if k in ("n", "samples"):
                            continue
                        log[f"val/{label}_{k}"] = v
                wandb.log(log)

            del eval_params

    pbar.close()

    if is_main:
        best_path = os.path.join(args.checkpoint_dir, "needle_grpo_best.pkl")
        if best_call_f1 >= 0:
            print(f"\nGRPO complete. Best single call_f1={best_call_f1:.1%} -> {best_path}")
        else:
            print(f"\nGRPO complete. No eval ran — no best checkpoint saved.")

    if use_wandb:
        import wandb
        wandb.finish()

    if num_hosts > 1:
        jax.experimental.multihost_utils.sync_global_devices("grpo_done")


def _save_grpo_checkpoint(r_state, config, checkpoint_dir, step,
                           filename="needle_grpo.pkl"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    params = _unreplicate(r_state).params
    params_np = jax.tree.map(lambda x: np.array(x).astype(np.float16), params)
    ckpt_path = os.path.join(checkpoint_dir, filename)
    with open(ckpt_path, "wb") as f:
        pickle.dump({"params": params_np, "config": config.__dict__,
                     "grpo_step": step}, f)
    size_mb = sum(x.nbytes for x in jax.tree.leaves(params_np)) / 1e6
    print(f"\n  [step {step}] Saved {ckpt_path} ({size_mb:.1f} MB)", flush=True)
    _upload_checkpoint(ckpt_path)
    return ckpt_path
