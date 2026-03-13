import argparse
import json as _json
import pickle
import re as _re
import sys

import jax
import jax.numpy as jnp
import numpy as np

from .data import get_tokenizer, compute_mel_spectrogram, to_snake_case, DEFAULT_MAX_ENC_LEN, DEFAULT_MAX_GEN_LEN
from .model import (
    EncoderDecoderTransformer,
    TransformerConfig,
    make_causal_mask,
    make_padding_mask,
    make_mel_padding_mask,
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
    params = jax.tree.map(jnp.array, data["params"])
    config = TransformerConfig(**data["config"])
    return params, config


def _build_encoder_input(tokenizer, query, tools, max_enc_len=DEFAULT_MAX_ENC_LEN):
    """Build encoder input: [query..., <tools>, tools...] truncated to max_enc_len.

    Empty tools ([]) are replaced with the <defer> token.
    """
    tools_sep_id = tokenizer.tools_token_id
    defer_id = tokenizer.defer_token_id
    q_toks = tokenizer.encode(query)

    # Use <defer> token for empty tools
    try:
        parsed = _json.loads(tools)
        is_empty = isinstance(parsed, list) and len(parsed) == 0
    except (ValueError, TypeError):
        is_empty = False

    if is_empty:
        t_toks = [defer_id]
    else:
        t_toks = tokenizer.encode(tools)

    max_query = max_enc_len - 2
    if len(q_toks) > max_query:
        q_toks = q_toks[:max_query]
    remaining = max_enc_len - len(q_toks) - 1
    t_toks = t_toks[:remaining]
    return q_toks + [tools_sep_id] + t_toks


def generate(model, params, tokenizer, query, tools="[]", max_gen_len=DEFAULT_MAX_GEN_LEN, max_enc_len=DEFAULT_MAX_ENC_LEN, seed=0, stream=True, task_token_id=None, normalize=True):
    """Generate tool-call output.

    Encoder: [query_tokens..., <tools>, tools_tokens...] truncated to max_enc_len.
    Decoder: prefilled with [EOS, <tool_call>], then greedy decode.
    """
    name_map = {}
    if normalize:
        tools, name_map = normalize_tools(tools)

    enc_tokens = _build_encoder_input(tokenizer, query, tools, max_enc_len)
    enc_input = jnp.array([enc_tokens])

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    tool_call_id = task_token_id if task_token_id is not None else tokenizer.tool_call_token_id

    src_mask = make_padding_mask(enc_input, pad_id)
    encoder_out, enc_mask = model.apply(
        {"params": params}, enc_input, src_mask=src_mask, method="encode"
    )

    prefix = [eos_id, tool_call_id]
    prefix_len = 2

    dec_buffer = jnp.full((1, max_gen_len), pad_id, dtype=jnp.int32)
    dec_buffer = dec_buffer.at[0, 0].set(eos_id)
    dec_buffer = dec_buffer.at[0, 1].set(tool_call_id)

    decode_fn = _get_decode_fn(model, max_gen_len)

    generated_tokens = []

    if stream:
        sys.stdout.write(f"\n")
        sys.stdout.flush()

    logits = decode_fn(params, dec_buffer, encoder_out, enc_mask)

    for i in range(prefix_len - 1, max_gen_len - 1):
        next_logits = logits[0, i]
        next_token = int(jnp.argmax(next_logits))

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
    if normalize and name_map:
        result = restore_tool_names(result, name_map)
    return result


def generate_batch(model, params, tokenizer, queries, tools_list, max_gen_len=DEFAULT_MAX_GEN_LEN, max_enc_len=DEFAULT_MAX_ENC_LEN, normalize=True):
    """Batch-generate tool-call outputs for multiple examples at once.

    Encoder: [query_tokens..., <tools>, tools_tokens...] per example, truncated to max_enc_len.
    Decoder: uniform prefix [EOS, <tool_call>] for all examples.

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
    tool_call_id = tokenizer.tool_call_token_id

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

    # --- Decoder prefix: uniform [EOS, <tool_call>] (prefix_len=2) ---
    prefix_len = 2
    dec_buffer = np.full((B, max_gen_len), pad_id, dtype=np.int32)
    dec_buffer[:, 0] = eos_id
    dec_buffer[:, 1] = tool_call_id
    dec_buffer = jnp.array(dec_buffer)

    decode_fn = _get_decode_fn(model, max_gen_len)

    # --- Autoregressive decoding ---
    finished = [False] * B
    gen_tokens = [[] for _ in range(B)]

    logits = decode_fn(params, dec_buffer, encoder_out, enc_mask)

    for pos in range(prefix_len - 1, max_gen_len - 1):
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

    results = [tokenizer.decode(toks) for toks in gen_tokens]
    if normalize and name_maps:
        results = [restore_tool_names(r, nm) for r, nm in zip(results, name_maps)]
    return results


def load_audio(path, target_sr=16000):
    """Load an audio file and resample to target_sr. Returns (audio_array, sr)."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        from scipy.signal import resample
        num_samples = int(len(audio) * target_sr / sr)
        audio = resample(audio, num_samples).astype(np.float32)
        sr = target_sr
    return audio, sr


def generate_from_audio(model, params, tokenizer, audio_array, sr=16000, tools="[]", max_gen_len=DEFAULT_MAX_GEN_LEN, seed=0, stream=True):
    """Generate tool-call output from audio using the speech encoder pathway.

    mel -> encode_speech -> decoder [BOS, <tool_call>, tools_tokens...] -> greedy decode.
    Same structure as generate() but uses speech encoder + mel padding mask.
    """
    n_mels = model.config.n_mels
    mel = compute_mel_spectrogram(audio_array, sr=sr, n_mels=n_mels)
    mel_input = jnp.array(mel)[None, :, :]  # (1, T_mel, n_mels)

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    tool_call_id = tokenizer.tool_call_token_id

    src_mask = make_mel_padding_mask(mel_input)
    encoder_out, enc_mask = model.apply(
        {"params": params}, mel_input, src_mask=src_mask, deterministic=True, method="encode_speech"
    )

    tools_tokens = tokenizer.encode(tools)
    prefix = [eos_id, tool_call_id] + tools_tokens
    prefix_len = min(len(prefix), max_gen_len)

    dec_buffer = jnp.full((1, max_gen_len), pad_id, dtype=jnp.int32)
    for j, tok in enumerate(prefix[:max_gen_len]):
        dec_buffer = dec_buffer.at[0, j].set(tok)

    decode_fn = _get_decode_fn(model, max_gen_len)

    generated_tokens = []

    if stream:
        sys.stdout.write("\n")
        sys.stdout.flush()

    logits = decode_fn(params, dec_buffer, encoder_out, enc_mask)

    for i in range(prefix_len - 1, max_gen_len - 1):
        next_logits = logits[0, i]
        next_token = int(jnp.argmax(next_logits))

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

    return tokenizer.decode(generated_tokens)


def main(args):
    print(f"Loading checkpoint: {args.checkpoint}")
    params, config = load_checkpoint(args.checkpoint)

    model = EncoderDecoderTransformer(config)
    tokenizer = get_tokenizer()

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameters: {param_count:,}")

    audio_files = getattr(args, "audio", None)
    if audio_files:
        tools = getattr(args, "tools", None) or "[]"
        for i, audio_path in enumerate(audio_files):
            print(f"\nVoice-to-tool-call: {audio_path}")
            print(f"Tools: {tools[:80]}{'...' if len(tools) > 80 else ''}")
            audio, sr = load_audio(audio_path)
            generate_from_audio(
                model,
                params,
                tokenizer,
                audio,
                sr=sr,
                tools=tools,
                max_gen_len=args.max_len,
                seed=args.seed + i,
                stream=True,
            )
        return

    query = getattr(args, "query", None)
    tools = getattr(args, "tools", None) or "[]"

    if query:
        queries = [(query, tools)]
    else:
        queries = [
            ('What is the weather in San Francisco?', '[{"name": "get_weather", "parameters": {"location": "string"}}]'),
            ('Send an email to john@example.com saying hello', '[{"name": "send_email", "parameters": {"to": "string", "body": "string"}}]'),
            ('Get the current stock price of AAPL', '[{"name": "get_stock_price", "parameters": {"symbol": "string"}}]'),
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
    parser.add_argument("--audio", type=str, nargs="*", help="Audio file paths for voice-to-tool-call")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
