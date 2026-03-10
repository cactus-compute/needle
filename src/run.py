import argparse
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

from .data import get_tokenizer, compute_mel_spectrogram
from .model import (
    EncoderDecoderTransformer,
    TransformerConfig,
    make_causal_mask,
    make_padding_mask,
    make_mel_padding_mask,
)


_decode_fn_cache = {}


def _get_decode_fn(model, max_gen_len, ffn_mask=None):
    """Return a JIT-compiled decode function, cached by (model, max_gen_len, ffn_mask).

    params is an explicit argument (not closed over) so the same compiled
    function can be reused across calls with different params.
    """
    key = (id(model), max_gen_len, id(ffn_mask) if ffn_mask is not None else None)
    if key not in _decode_fn_cache:
        tgt_mask = make_causal_mask(max_gen_len)

        @jax.jit
        def decode_step(params, dec_buffer, encoder_out):
            return model.apply(
                {"params": params}, dec_buffer, encoder_out,
                self_mask=tgt_mask, ffn_mask=ffn_mask, method="decode",
            )

        _decode_fn_cache[key] = decode_step
    return _decode_fn_cache[key]


def load_checkpoint(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    params = jax.tree.map(jnp.array, data["params"])
    config = TransformerConfig(**data["config"])
    ffn_mask = _build_ffn_mask(data, config) if "mat_mask_indices" in data else None
    return params, config, ffn_mask


def _build_ffn_mask(data, config):
    """Build (n_blocks, 1, d_ff) binary FFN mask from exported topk mask indices."""
    indices = data["mat_mask_indices"]
    n_blocks = len(indices)
    mask = np.zeros((n_blocks, 1, config.d_ff), dtype=np.float32)
    for b, idx in enumerate(indices):
        mask[b, 0, idx] = 1.0
    enc_mask = jnp.array(mask[:config.num_encoder_layers])
    dec_mask = jnp.array(mask[config.num_encoder_layers:])
    return {"encoder": enc_mask, "decoder": dec_mask}


def generate(model, params, tokenizer, query, tools="[]", max_gen_len=512, seed=0, stream=True, task_token_id=None, ffn_mask=None):
    """Generate tool-call output.

    Encoder: query only.
    Decoder: prefilled with [BOS, <tool_call>, tools_tokens...], then greedy decode.
    """
    enc_tokens = tokenizer.encode(query)
    enc_input = jnp.array([enc_tokens])

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    tool_call_id = task_token_id if task_token_id is not None else tokenizer.tool_call_token_id

    enc_ffn = ffn_mask["encoder"] if ffn_mask else None
    dec_ffn = ffn_mask["decoder"] if ffn_mask else None

    src_mask = make_padding_mask(enc_input, pad_id)
    encoder_out = model.apply(
        {"params": params}, enc_input, src_mask=src_mask, ffn_mask=enc_ffn, method="encode"
    )

    tools_tokens = tokenizer.encode(tools)
    prefix = [eos_id, tool_call_id] + tools_tokens
    prefix_len = min(len(prefix), max_gen_len)

    dec_buffer = jnp.full((1, max_gen_len), pad_id, dtype=jnp.int32)
    for j, tok in enumerate(prefix[:max_gen_len]):
        dec_buffer = dec_buffer.at[0, j].set(tok)

    decode_fn = _get_decode_fn(model, max_gen_len, ffn_mask=dec_ffn)

    generated_tokens = []

    if stream:
        sys.stdout.write(f"\n")
        sys.stdout.flush()

    logits = decode_fn(params, dec_buffer, encoder_out)

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

        logits = decode_fn(params, dec_buffer, encoder_out)

    if stream:
        sys.stdout.write("\n")

    return tokenizer.decode(generated_tokens)


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


def generate_from_audio(model, params, tokenizer, audio_array, sr=16000, tools="[]", max_gen_len=512, seed=0, stream=True, ffn_mask=None):
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

    enc_ffn = ffn_mask["encoder"] if ffn_mask else None
    dec_ffn = ffn_mask["decoder"] if ffn_mask else None

    src_mask = make_mel_padding_mask(mel_input)
    encoder_out = model.apply(
        {"params": params}, mel_input, src_mask=src_mask, ffn_mask=enc_ffn, deterministic=True, method="encode_speech"
    )

    tools_tokens = tokenizer.encode(tools)
    prefix = [eos_id, tool_call_id] + tools_tokens
    prefix_len = min(len(prefix), max_gen_len)

    dec_buffer = jnp.full((1, max_gen_len), pad_id, dtype=jnp.int32)
    for j, tok in enumerate(prefix[:max_gen_len]):
        dec_buffer = dec_buffer.at[0, j].set(tok)

    decode_fn = _get_decode_fn(model, max_gen_len, ffn_mask=dec_ffn)

    generated_tokens = []

    if stream:
        sys.stdout.write("\n")
        sys.stdout.flush()

    logits = decode_fn(params, dec_buffer, encoder_out)

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

        logits = decode_fn(params, dec_buffer, encoder_out)

    if stream:
        sys.stdout.write("\n")

    return tokenizer.decode(generated_tokens)


def main(args):
    print(f"Loading checkpoint: {args.checkpoint}")
    params, config, ffn_mask = load_checkpoint(args.checkpoint)

    model = EncoderDecoderTransformer(config)
    tokenizer = get_tokenizer()

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameters: {param_count:,}")
    if ffn_mask:
        print(f"TopK sub-model: per-layer FFN masking active")

    audio_files = getattr(args, "audio", None)
    if audio_files:
        tools = getattr(args, "tools", None) or "[]"
        for i, audio_path in enumerate(audio_files):
            print(f"\nVoice-to-tool-call: {audio_path}")
            print(f"Tools: {tools[:80]}{'...' if len(tools) > 80 else ''}")
            audio, sr = load_audio(audio_path)
            generate_from_audio(
                model, params, tokenizer, audio,
                sr=sr, tools=tools,
                max_gen_len=args.max_len,
                seed=args.seed + i,
                stream=True, ffn_mask=ffn_mask,
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
            model, params, tokenizer, q,
            tools=t, max_gen_len=args.max_len,
            seed=args.seed + i, stream=True, ffn_mask=ffn_mask,
        )


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
