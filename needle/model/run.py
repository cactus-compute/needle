import argparse
import json as _json
import pickle
import re as _re
import sys

import jax
import jax.numpy as jnp
import numpy as np

from ..dataset.dataset import get_tokenizer, to_snake_case, DEFAULT_MAX_ENC_LEN, DEFAULT_MAX_GEN_LEN
from .architecture import (
    SimpleAttentionNetwork,
    TransformerConfig,
    make_causal_mask,
    make_padding_mask,
    precompute_rope_freqs,
)


def normalize_tools(tools_json):
    """Normalize tool names in a tools JSON string to snake_case.

    Returns (normalized_json, name_map) where name_map maps
    snake_name -> original_name for reverse mapping.
    """
    try:
        tools = _json.loads(tools_json)
    except (_json.JSONDecodeError, TypeError):
        return tools_json, {}
    name_map = {}
    for t in tools:
        if isinstance(t, dict) and "name" in t:
            orig = t["name"]
            snake = to_snake_case(orig)
            name_map[snake] = orig
            t["name"] = snake
    return _json.dumps(tools, separators=(",", ":")), name_map


def restore_tool_names(pred_text, name_map):
    """Replace snake_case tool names in model output back to original names."""
    if not name_map:
        return pred_text
    try:
        calls = _json.loads(pred_text)
    except (_json.JSONDecodeError, TypeError):
        # Fallback: string-level replacement, longest names first
        for snake, orig in sorted(name_map.items(), key=lambda x: len(x[0]), reverse=True):
            pred_text = pred_text.replace(snake, orig)
        return pred_text
    if isinstance(calls, list):
        for c in calls:
            if isinstance(c, dict) and "name" in c:
                c["name"] = name_map.get(c["name"], c["name"])
    elif isinstance(calls, dict) and "name" in calls:
        calls["name"] = name_map.get(calls["name"], calls["name"])
    return _json.dumps(calls, separators=(",", ":"))


_decode_fn_cache = {}


def _get_decode_fn(model, max_gen_len):
    """Return a JIT-compiled decode function, cached by (model, max_gen_len).

    params is an explicit argument (not closed over) so the same compiled
    function can be reused across calls with different params.
    """
    key = (id(model), max_gen_len)
    if key not in _decode_fn_cache:
        tgt_mask = make_causal_mask(max_gen_len)

        @jax.jit
        def decode_step(params, dec_buffer, encoder_out, cross_mask):
            return model.apply(
                {"params": params}, dec_buffer, encoder_out,
                self_mask=tgt_mask, cross_mask=cross_mask, method="decode",
            )

        _decode_fn_cache[key] = decode_step
    return _decode_fn_cache[key]


_step_fn_cache = {}


def _default_host_cross_cache():
    """Default cross-attention caching policy for the host-driven decode loop.

    The static cross-attention K/V cache regressed the GPU host-loop path (the
    per-token host sync dominates and the prefill/gather overhead isn't hidden),
    but helped on CPU. So default it off on GPU and on elsewhere. Callers can
    override explicitly via use_cross_cache=True/False.
    """
    try:
        return jax.default_backend() != "gpu"
    except Exception:
        return True


def _get_step_fn(model, cache_len, use_cross_cache=True):
    """Return a JIT-compiled single-token KV-cached decode step.

    The compiled function takes the current cache and returns (logits, new_cache).
    cache_len is baked in (static) so the cache buffers have a fixed shape.
    use_cross_cache controls whether the static cross-attention K/V are reused
    (True) or re-projected every step (False); it is part of the cache key since
    it changes the compiled graph and cache structure.
    """
    key = (id(model), cache_len, bool(use_cross_cache))
    if key not in _step_fn_cache:

        @jax.jit
        def step_fn(params, cache, tgt_token, encoder_out, cross_mask, self_mask, cos_p, sin_p, cache_pos):
            # cross_prefill=False: reuse the cross-attention K/V cached at prefill.
            logits, mutated = model.apply(
                {"params": params, "cache": cache},
                tgt_token, encoder_out, self_mask, cross_mask, (cos_p, sin_p),
                cache_pos, cache_len, False, use_cross_cache,
                method="decode_step", mutable=["cache"],
            )
            return logits, mutated["cache"]

        _step_fn_cache[key] = step_fn
    return _step_fn_cache[key]


def _init_cache_step(model, params, cache_len, tgt_token, encoder_out, cross_mask, self_mask, cos_p, sin_p, cache_pos, use_cross_cache=True):
    """First (prefill) decode step: allocates the KV cache (including the static
    cross-attention K/V when use_cross_cache is True) and returns (logits, cache)."""
    logits, mutated = model.apply(
        {"params": params},
        tgt_token, encoder_out, self_mask, cross_mask, (cos_p, sin_p),
        cache_pos, cache_len, True, use_cross_cache,
        method="decode_step", mutable=["cache"],
    )
    return logits, mutated["cache"]


def _rope_tables(config, cache_len):
    head_dim = config.d_model // config.num_heads
    return precompute_rope_freqs(head_dim, cache_len, config.rope_theta)


_prefill_fn_cache = {}
_ondevice_fn_cache = {}


def _get_prefill_fn(model, cache_len):
    """Return a jitted prefill step (cross_prefill=True).

    Allocates the KV cache (self-attention buffers + static cross-attention K/V),
    runs the first decode step, and returns (next_token (B,), cache). The static
    cross K/V stored here are reused by the on-device loop without recomputation.
    """
    key = (id(model), cache_len)
    if key not in _prefill_fn_cache:

        @jax.jit
        def prefill_fn(params, first_token, encoder_out, cross_mask, cos0, sin0, eos_id):
            self_mask = (jnp.arange(cache_len) <= 0)[None, None, None, :]
            logits, mutated = model.apply(
                {"params": params},
                first_token, encoder_out, self_mask, cross_mask, (cos0, sin0),
                jnp.array(0, jnp.int32), cache_len, True, True,
                method="decode_step", mutable=["cache"],
            )
            nxt = jnp.argmax(logits, axis=-1).astype(jnp.int32)  # (B,)
            return nxt, mutated["cache"]

        _prefill_fn_cache[key] = prefill_fn
    return _prefill_fn_cache[key]


def _get_ondevice_fn(model, cache_len):
    """Return a jitted, fully on-device greedy decode loop (no per-token host sync).

    Runs the whole autoregressive loop inside lax.while_loop with early stop when
    every sequence has emitted EOS. Assumes the cache has been prefilled (so cross
    K/V are already stored); each step reuses them (cross_prefill=False). Greedy
    argmax only (no constrained decoding, which requires host-side automaton
    state). Returns (gen_tokens, lengths).
    """
    key = (id(model), cache_len)
    if key not in _ondevice_fn_cache:
        max_steps = cache_len - 1

        @jax.jit
        def loop_fn(params, cache, tok0, encoder_out, cross_mask, cos_full, sin_full, eos_id):
            B = tok0.shape[0]
            # tok0 is the token produced by the prefill step (stored at column 0).
            gen0 = jnp.zeros((B, max_steps), jnp.int32).at[:, 0].set(tok0)
            fin0 = tok0 == eos_id
            len0 = jnp.where(fin0, 0, max_steps)

            def cond(state):
                pos, cur, cache, gen, fin, length = state
                return jnp.logical_and(pos < max_steps, jnp.logical_not(jnp.all(fin)))

            def body(state):
                pos, cur, cache, gen, fin, length = state
                cos_p = jax.lax.dynamic_slice_in_dim(cos_full, pos, 1, axis=0)
                sin_p = jax.lax.dynamic_slice_in_dim(sin_full, pos, 1, axis=0)
                self_mask = (jnp.arange(cache_len) <= pos)[None, None, None, :]
                logits, mutated = model.apply(
                    {"params": params, "cache": cache},
                    cur, encoder_out, self_mask, cross_mask, (cos_p, sin_p),
                    pos, cache_len, False, True,
                    method="decode_step", mutable=["cache"],
                )
                nxt = jnp.argmax(logits, axis=-1).astype(jnp.int32)  # (B,)
                is_eos = nxt == eos_id
                newly = jnp.logical_and(jnp.logical_not(fin), is_eos)
                # Record token count before the first EOS (EOS itself is not emitted).
                length = jnp.where(newly, pos, length)
                gen = gen.at[:, pos].set(nxt)
                fin = jnp.logical_or(fin, is_eos)
                return (pos + 1, nxt[:, None], mutated["cache"], gen, fin, length)

            state = (jnp.array(1, jnp.int32), tok0[:, None], cache, gen0, fin0, len0)
            _, _, _, gen, _, length = jax.lax.while_loop(cond, body, state)
            return gen, length

        _ondevice_fn_cache[key] = loop_fn
    return _ondevice_fn_cache[key]


def load_checkpoint(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    params = jax.tree.map(lambda x: jnp.array(x, dtype=jnp.bfloat16), data["params"])
    config = TransformerConfig(**data["config"])
    return params, config


def _build_encoder_input(tokenizer, query, tools, max_enc_len=DEFAULT_MAX_ENC_LEN):
    """Build encoder input: [query..., <tools>, tools...] truncated to max_enc_len."""
    tools_sep_id = tokenizer.tools_token_id
    q_toks = tokenizer.encode(query)
    t_toks = tokenizer.encode(tools)

    max_query = max_enc_len - 2
    if len(q_toks) > max_query:
        q_toks = q_toks[:max_query]
    remaining = max_enc_len - len(q_toks) - 1
    t_toks = t_toks[:remaining]
    return q_toks + [tools_sep_id] + t_toks


def generate(model, params, tokenizer, query, tools="[]", max_gen_len=DEFAULT_MAX_GEN_LEN, max_enc_len=DEFAULT_MAX_ENC_LEN, seed=0, stream=True, task_token_id=None, normalize=True, constrained=True, use_cache=True, use_cross_cache=None):
    """Generate tool-call output.

    Encoder: [query_tokens..., <tools>, tools_tokens...] truncated to max_enc_len.
    Decoder: prefilled with [EOS], model predicts <tool_call> first, then answer tokens.

    use_cache=True uses the incremental KV-cached decode path (single-token steps);
    use_cache=False uses the original full-buffer re-decode path.
    use_cross_cache toggles the static cross-attention K/V cache on the host loop
    (None=auto: off on GPU, on elsewhere); the on-device path always uses it.
    """
    if use_cache and not constrained and not stream:
        # Fully on-device single-token loop (no per-token host sync).
        return generate_batch_ondevice(
            model, params, tokenizer, [query], [tools], max_gen_len=max_gen_len,
            max_enc_len=max_enc_len, normalize=normalize,
        )[0]
    if use_cache:
        return generate_cached(
            model, params, tokenizer, query, tools=tools, max_gen_len=max_gen_len,
            max_enc_len=max_enc_len, seed=seed, stream=stream, task_token_id=task_token_id,
            normalize=normalize, constrained=constrained, use_cross_cache=use_cross_cache,
        )

    name_map = {}
    if normalize:
        tools, name_map = normalize_tools(tools)

    enc_tokens = _build_encoder_input(tokenizer, query, tools, max_enc_len)
    enc_input = jnp.array([enc_tokens])

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    src_mask = make_padding_mask(enc_input, pad_id)
    encoder_out, enc_mask = model.apply(
        {"params": params}, enc_input, src_mask=src_mask, method="encode"
    )

    dec_buffer = jnp.full((1, max_gen_len), pad_id, dtype=jnp.int32)
    dec_buffer = dec_buffer.at[0, 0].set(eos_id)

    decode_fn = _get_decode_fn(model, max_gen_len)

    generated_tokens = []

    constrained_decoder = None
    if constrained:
        from .constrained import build_constrained_decoder
        constrained_decoder = build_constrained_decoder([tools], tokenizer)

    if stream:
        sys.stdout.write(f"\n")
        sys.stdout.flush()

    logits = decode_fn(params, dec_buffer, encoder_out, enc_mask)

    for i in range(0, max_gen_len - 1):
        next_logits = logits[0, i]

        if constrained_decoder and constrained_decoder.is_active(0):
            logits_np = np.array(next_logits)
            logits_np = constrained_decoder.constrain_logits(logits_np, 0)
            next_token = int(np.argmax(logits_np))
        else:
            next_token = int(jnp.argmax(next_logits))

        if constrained_decoder:
            constrained_decoder.update(0, next_token)

        if next_token == eos_id:
            break

        generated_tokens.append(next_token)
        dec_buffer = dec_buffer.at[0, i + 1].set(next_token)

        if stream:
            sys.stdout.write(tokenizer.decode([next_token]))
            sys.stdout.flush()

        logits = decode_fn(params, dec_buffer, encoder_out, enc_mask)

    if stream:
        sys.stdout.write("\n")

    result = tokenizer.decode(generated_tokens)
    # Strip leading <tool_call> token text from output if present
    if result.startswith("<tool_call>"):
        result = result[len("<tool_call>"):]
    if normalize and name_map:
        result = restore_tool_names(result, name_map)
    return result


def generate_cached(model, params, tokenizer, query, tools="[]", max_gen_len=DEFAULT_MAX_GEN_LEN, max_enc_len=DEFAULT_MAX_ENC_LEN, seed=0, stream=True, task_token_id=None, normalize=True, constrained=True, use_cross_cache=None):
    """Incremental KV-cached single-example generation. Token-for-token equivalent
    to generate(..., use_cache=False) but O(T^2) instead of O(T^3) work.

    use_cross_cache=None auto-selects (off on GPU, on elsewhere); pass True/False
    to force the static cross-attention K/V cache on or off."""
    if use_cross_cache is None:
        use_cross_cache = _default_host_cross_cache()
    name_map = {}
    if normalize:
        tools, name_map = normalize_tools(tools)

    enc_tokens = _build_encoder_input(tokenizer, query, tools, max_enc_len)
    enc_input = jnp.array([enc_tokens])

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    src_mask = make_padding_mask(enc_input, pad_id)
    encoder_out, enc_mask = model.apply(
        {"params": params}, enc_input, src_mask=src_mask, method="encode"
    )

    cache_len = max_gen_len
    cos_full, sin_full = _rope_tables(model.config, cache_len)
    step_fn = _get_step_fn(model, cache_len, use_cross_cache)

    constrained_decoder = None
    if constrained:
        from .constrained import build_constrained_decoder
        constrained_decoder = build_constrained_decoder([tools], tokenizer)

    if stream:
        sys.stdout.write("\n")
        sys.stdout.flush()

    generated_tokens = []
    current_token = jnp.array([[eos_id]], dtype=jnp.int32)
    cache = None

    for i in range(0, max_gen_len - 1):
        cos_p = cos_full[i:i + 1]
        sin_p = sin_full[i:i + 1]
        self_mask = (jnp.arange(cache_len) <= i)[None, None, None, :]
        pos = jnp.array(i, dtype=jnp.int32)

        if cache is None:
            logits, cache = _init_cache_step(
                model, params, cache_len, current_token, encoder_out, enc_mask,
                self_mask, cos_p, sin_p, pos, use_cross_cache,
            )
        else:
            logits, cache = step_fn(
                params, cache, current_token, encoder_out, enc_mask,
                self_mask, cos_p, sin_p, pos,
            )

        next_logits = logits[0]

        if constrained_decoder and constrained_decoder.is_active(0):
            logits_np = np.array(next_logits)
            logits_np = constrained_decoder.constrain_logits(logits_np, 0)
            next_token = int(np.argmax(logits_np))
        else:
            next_token = int(jnp.argmax(next_logits))

        if constrained_decoder:
            constrained_decoder.update(0, next_token)

        if next_token == eos_id:
            break

        generated_tokens.append(next_token)
        current_token = jnp.array([[next_token]], dtype=jnp.int32)

        if stream:
            sys.stdout.write(tokenizer.decode([next_token]))
            sys.stdout.flush()

    if stream:
        sys.stdout.write("\n")

    result = tokenizer.decode(generated_tokens)
    if result.startswith("<tool_call>"):
        result = result[len("<tool_call>"):]
    if normalize and name_map:
        result = restore_tool_names(result, name_map)
    return result


def generate_batch(model, params, tokenizer, queries, tools_list, max_gen_len=DEFAULT_MAX_GEN_LEN, max_enc_len=DEFAULT_MAX_ENC_LEN, normalize=True, constrained=True, use_cache=True, use_cross_cache=None):
    """Batch-generate tool-call outputs for multiple examples at once.

    Encoder: [query_tokens..., <tools>, tools_tokens...] per example, truncated to max_enc_len.
    Decoder: prefilled with [EOS], model predicts <tool_call> first, then answer tokens.

    Returns a list of decoded strings, one per example.

    use_cross_cache toggles the static cross-attention K/V cache on the host loop
    (None=auto: off on GPU, on elsewhere); the on-device path always uses it.
    """
    if use_cache and not constrained:
        # Fully on-device loop (no per-token host sync). Constrained decoding
        # cannot use this path (host-side automaton state), so it falls through
        # to the host-driven cached loop below.
        return generate_batch_ondevice(
            model, params, tokenizer, queries, tools_list, max_gen_len=max_gen_len,
            max_enc_len=max_enc_len, normalize=normalize,
        )
    if use_cache:
        return generate_batch_cached(
            model, params, tokenizer, queries, tools_list, max_gen_len=max_gen_len,
            max_enc_len=max_enc_len, normalize=normalize, constrained=constrained,
            use_cross_cache=use_cross_cache,
        )

    name_maps = []
    if normalize:
        normed_tools = []
        for t in tools_list:
            nt, nm = normalize_tools(t)
            normed_tools.append(nt)
            name_maps.append(nm)
        tools_list = normed_tools

    B = len(queries)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    # --- Encode: [query..., <tools>, tools...] per example, truncated ---
    enc_token_lists = []
    for q, t in zip(queries, tools_list):
        enc_token_lists.append(_build_encoder_input(tokenizer, q, t, max_enc_len))
    max_enc = max(len(toks) for toks in enc_token_lists)
    enc_input = np.full((B, max_enc), pad_id, dtype=np.int32)
    for i, toks in enumerate(enc_token_lists):
        enc_input[i, :len(toks)] = toks
    enc_input = jnp.array(enc_input)
    src_mask = make_padding_mask(enc_input, pad_id)

    encoder_out, enc_mask = model.apply(
        {"params": params}, enc_input, src_mask=src_mask, method="encode"
    )

    dec_buffer = np.full((B, max_gen_len), pad_id, dtype=np.int32)
    dec_buffer[:, 0] = eos_id
    dec_buffer = jnp.array(dec_buffer)

    decode_fn = _get_decode_fn(model, max_gen_len)

    finished = [False] * B
    gen_tokens = [[] for _ in range(B)]

    constrained_decoder = None
    if constrained:
        from .constrained import build_constrained_decoder
        constrained_decoder = build_constrained_decoder(tools_list, tokenizer)

    logits = decode_fn(params, dec_buffer, encoder_out, enc_mask)

    for pos in range(0, max_gen_len - 1):
        for i in range(B):
            if finished[i]:
                continue
            if constrained_decoder and constrained_decoder.is_active(i):
                logits_np = np.array(logits[i, pos])
                logits_np = constrained_decoder.constrain_logits(logits_np, i)
                next_token = int(np.argmax(logits_np))
            else:
                next_token = int(jnp.argmax(logits[i, pos]))
            if constrained_decoder:
                constrained_decoder.update(i, next_token)
            if next_token == eos_id:
                finished[i] = True
                continue
            gen_tokens[i].append(next_token)
            dec_buffer = dec_buffer.at[i, pos + 1].set(next_token)

        if all(finished):
            break

        logits = decode_fn(params, dec_buffer, encoder_out, enc_mask)

    results = []
    for i in range(B):
        text = tokenizer.decode(gen_tokens[i])
        if text.startswith("<tool_call>"):
            text = text[len("<tool_call>"):]
        results.append(text)
    if normalize and name_maps:
        results = [restore_tool_names(r, nm) for r, nm in zip(results, name_maps)]
    return results


def generate_batch_cached(model, params, tokenizer, queries, tools_list, max_gen_len=DEFAULT_MAX_GEN_LEN, max_enc_len=DEFAULT_MAX_ENC_LEN, normalize=True, constrained=True, use_cross_cache=None):
    """Incremental KV-cached batched generation. Token-for-token equivalent to
    generate_batch(..., use_cache=False).

    use_cross_cache=None auto-selects (off on GPU, on elsewhere); pass True/False
    to force the static cross-attention K/V cache on or off."""
    if use_cross_cache is None:
        use_cross_cache = _default_host_cross_cache()
    name_maps = []
    if normalize:
        normed_tools = []
        for t in tools_list:
            nt, nm = normalize_tools(t)
            normed_tools.append(nt)
            name_maps.append(nm)
        tools_list = normed_tools

    B = len(queries)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    enc_token_lists = []
    for q, t in zip(queries, tools_list):
        enc_token_lists.append(_build_encoder_input(tokenizer, q, t, max_enc_len))
    max_enc = max(len(toks) for toks in enc_token_lists)
    enc_input = np.full((B, max_enc), pad_id, dtype=np.int32)
    for i, toks in enumerate(enc_token_lists):
        enc_input[i, :len(toks)] = toks
    enc_input = jnp.array(enc_input)
    src_mask = make_padding_mask(enc_input, pad_id)

    encoder_out, enc_mask = model.apply(
        {"params": params}, enc_input, src_mask=src_mask, method="encode"
    )

    cache_len = max_gen_len
    cos_full, sin_full = _rope_tables(model.config, cache_len)
    step_fn = _get_step_fn(model, cache_len, use_cross_cache)

    constrained_decoder = None
    if constrained:
        from .constrained import build_constrained_decoder
        constrained_decoder = build_constrained_decoder(tools_list, tokenizer)

    finished = [False] * B
    gen_tokens = [[] for _ in range(B)]
    current_token = np.full((B, 1), eos_id, dtype=np.int32)
    cache = None

    for pos_i in range(0, max_gen_len - 1):
        cos_p = cos_full[pos_i:pos_i + 1]
        sin_p = sin_full[pos_i:pos_i + 1]
        self_mask = (jnp.arange(cache_len) <= pos_i)[None, None, None, :]
        pos = jnp.array(pos_i, dtype=jnp.int32)
        tok_in = jnp.array(current_token)

        if cache is None:
            logits, cache = _init_cache_step(
                model, params, cache_len, tok_in, encoder_out, enc_mask,
                self_mask, cos_p, sin_p, pos, use_cross_cache,
            )
        else:
            logits, cache = step_fn(
                params, cache, tok_in, encoder_out, enc_mask,
                self_mask, cos_p, sin_p, pos,
            )

        for i in range(B):
            if finished[i]:
                continue
            if constrained_decoder and constrained_decoder.is_active(i):
                logits_np = np.array(logits[i])
                logits_np = constrained_decoder.constrain_logits(logits_np, i)
                next_token = int(np.argmax(logits_np))
            else:
                next_token = int(jnp.argmax(logits[i]))
            if constrained_decoder:
                constrained_decoder.update(i, next_token)
            if next_token == eos_id:
                finished[i] = True
                continue
            gen_tokens[i].append(next_token)
            current_token[i, 0] = next_token

        if all(finished):
            break

    results = []
    for i in range(B):
        text = tokenizer.decode(gen_tokens[i])
        if text.startswith("<tool_call>"):
            text = text[len("<tool_call>"):]
        results.append(text)
    if normalize and name_maps:
        results = [restore_tool_names(r, nm) for r, nm in zip(results, name_maps)]
    return results


def generate_batch_ondevice(model, params, tokenizer, queries, tools_list, max_gen_len=DEFAULT_MAX_GEN_LEN, max_enc_len=DEFAULT_MAX_ENC_LEN, normalize=True):
    """Fully on-device KV-cached batched generation (greedy, unconstrained).

    The autoregressive loop runs inside a single jitted lax.while_loop, so there
    is no per-token host sync (only one device->host transfer at the end). This
    removes the launch-bound overhead that hurts small-batch GPU decoding.
    Token-for-token equivalent to generate_batch(..., use_cache=True) with
    constrained=False.
    """
    name_maps = []
    if normalize:
        normed_tools = []
        for t in tools_list:
            nt, nm = normalize_tools(t)
            normed_tools.append(nt)
            name_maps.append(nm)
        tools_list = normed_tools

    B = len(queries)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    enc_token_lists = [_build_encoder_input(tokenizer, q, t, max_enc_len) for q, t in zip(queries, tools_list)]
    max_enc = max(len(toks) for toks in enc_token_lists)
    enc_input = np.full((B, max_enc), pad_id, dtype=np.int32)
    for i, toks in enumerate(enc_token_lists):
        enc_input[i, :len(toks)] = toks
    enc_input = jnp.array(enc_input)
    src_mask = make_padding_mask(enc_input, pad_id)

    encoder_out, enc_mask = model.apply(
        {"params": params}, enc_input, src_mask=src_mask, method="encode"
    )

    cache_len = max_gen_len
    cos_full, sin_full = _rope_tables(model.config, cache_len)
    first_token = jnp.full((B, 1), eos_id, dtype=jnp.int32)
    eos = jnp.array(eos_id, dtype=jnp.int32)

    # Prefill step: allocates the cache and projects the static cross-attention
    # K/V once. The on-device loop then reuses them on every subsequent step.
    prefill_fn = _get_prefill_fn(model, cache_len)
    loop_fn = _get_ondevice_fn(model, cache_len)

    tok0, cache = prefill_fn(params, first_token, encoder_out, enc_mask, cos_full[0:1], sin_full[0:1], eos)
    gen, length = loop_fn(params, cache, tok0, encoder_out, enc_mask, cos_full, sin_full, eos)
    gen = np.array(gen)          # single device -> host transfer
    length = np.array(length)

    results = []
    for i in range(B):
        toks = gen[i, :int(length[i])].tolist()
        text = tokenizer.decode(toks)
        if text.startswith("<tool_call>"):
            text = text[len("<tool_call>"):]
        results.append(text)
    if normalize and name_maps:
        results = [restore_tool_names(r, nm) for r, nm in zip(results, name_maps)]
    return results


def main(args):
    print(f"Loading checkpoint: {args.checkpoint}")
    params, config = load_checkpoint(args.checkpoint)

    model = SimpleAttentionNetwork(config)
    tokenizer = get_tokenizer()

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameters: {param_count:,}")

    use_constrained = not getattr(args, "no_constrained", False)

    query = getattr(args, "query", None)
    tools = getattr(args, "tools", None) or "[]"

    if query:
        queries = [(query, tools)]
    else:
        queries = [
            ('What is the weather in San Francisco?', '[{"name": "get_weather", "description": "Get current weather for a city.", "parameters": {"location": {"type": "string", "description": "City name.", "required": true}}}]'),
            ('Send an email to john@example.com saying hello', '[{"name": "send_email", "description": "Send an email to a recipient.", "parameters": {"to": {"type": "string", "description": "The recipient email address.", "required": true}, "body": {"type": "string", "description": "The email body text.", "required": true}}}]'),
            ('Get the current stock price of AAPL', '[{"name": "get_stock_price", "description": "Get the current stock price.", "parameters": {"symbol": {"type": "string", "description": "Ticker symbol.", "required": true}}}]'),
        ]

    for i, (q, t) in enumerate(queries):
        print(f"\nQuery: {q}")
        print(f"Tools: {t[:80]}{'...' if len(t) > 80 else ''}")
        generate(
            model,
            params,
            tokenizer,
            q,
            tools=t,
            max_gen_len=args.max_len,
            seed=args.seed + i,
            stream=True,
            constrained=use_constrained,
        )


def encode_for_retrieval(model, params, tokenizer, texts, max_len=256, batch_size=64):
    """Encode texts into contrastive embeddings for retrieval. Returns (N, contrastive_dim) numpy array."""
    pad_id = tokenizer.pad_token_id
    all_embs = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        token_lists = [tokenizer.encode(t)[:max_len] for t in batch_texts]
        max_t = max(len(toks) for toks in token_lists)
        tokens = np.full((len(batch_texts), max_t), pad_id, dtype=np.int32)
        for i, toks in enumerate(token_lists):
            tokens[i, :len(toks)] = toks
        embs = model.apply(
            {"params": params}, jnp.array(tokens),
            deterministic=True, method="encode_contrastive",
        )
        all_embs.append(np.array(embs))

    return np.concatenate(all_embs, axis=0)


def retrieve_tools(model, params, tokenizer, query, tool_descriptions, top_k=5, max_len=256):
    """Retrieve top-k tools by cosine similarity to query.

    Args:
        query: query string
        tool_descriptions: list of tool description strings (one per tool)
        top_k: number of results to return

    Returns:
        list of (index, score) tuples sorted by descending similarity
    """
    q_emb = encode_for_retrieval(model, params, tokenizer, [query], max_len=max_len)
    t_emb = encode_for_retrieval(model, params, tokenizer, tool_descriptions, max_len=max_len)
    scores = (q_emb @ t_emb.T)[0]  # (N_tools,)
    top_indices = np.argsort(-scores)[:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate tool calls with trained transformer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch3.pkl")
    parser.add_argument("--query", type=str, default=None, help="Query text for tool-call generation")
    parser.add_argument("--tools", type=str, default=None, help="Tools JSON for tool-call generation")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
