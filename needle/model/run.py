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


def generate(model, params, tokenizer, query, tools="[]", max_gen_len=DEFAULT_MAX_GEN_LEN, max_enc_len=DEFAULT_MAX_ENC_LEN, seed=0, stream=True, task_token_id=None, normalize=True, constrained=True):
    """Generate tool-call output.

    Encoder: [query_tokens..., <tools>, tools_tokens...] truncated to max_enc_len.
    Decoder: prefilled with [EOS], model predicts <tool_call> first, then answer tokens.
    """
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


def generate_batch(model, params, tokenizer, queries, tools_list, max_gen_len=DEFAULT_MAX_GEN_LEN, max_enc_len=DEFAULT_MAX_ENC_LEN, normalize=True, constrained=True):
    """Batch-generate tool-call outputs for multiple examples at once.

    Encoder: [query_tokens..., <tools>, tools_tokens...] per example, truncated to max_enc_len.
    Decoder: prefilled with [EOS], model predicts <tool_call> first, then answer tokens.

    Returns a list of decoded strings, one per example.
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
