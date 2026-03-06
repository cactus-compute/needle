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


def load_checkpoint(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    params = jax.tree.map(jnp.array, data["params"])
    config = TransformerConfig(**data["config"])
    return params, config


def generate(model, params, tokenizer, prompt, max_gen_len=128, temperature=0.8, seed=0, stream=True):
    enc_tokens = tokenizer.encode(prompt)
    enc_input = jnp.array([enc_tokens])

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    src_mask = make_padding_mask(enc_input, pad_id)
    encoder_out = model.apply(
        {"params": params}, enc_input, src_mask=src_mask, method="encode"
    )

    # Fixed-size buffer so JAX compiles once
    tgt_mask = make_causal_mask(max_gen_len)
    dec_buffer = jnp.full((1, max_gen_len), pad_id, dtype=jnp.int32)
    dec_buffer = dec_buffer.at[0, 0].set(eos_id)

    @jax.jit
    def decode_step(dec_buffer, encoder_out, src_mask):
        logits = model.apply(
            {"params": params},
            dec_buffer,
            encoder_out,
            self_mask=tgt_mask,
            cross_mask=src_mask,
                       method="decode",
        )
        return logits

    rng = jax.random.PRNGKey(seed)
    generated_tokens = []

    if stream:
        sys.stdout.write(f"\n{prompt}")
        sys.stdout.flush()

    # Compile once on first call
    logits = decode_step(dec_buffer, encoder_out, src_mask)

    for i in range(max_gen_len - 1):
        next_logits = logits[0, i] / temperature
        rng, sample_rng = jax.random.split(rng)
        next_token = jax.random.categorical(sample_rng, next_logits).item()

        if next_token == eos_id:
            break

        generated_tokens.append(next_token)
        dec_buffer = dec_buffer.at[0, i + 1].set(next_token)

        if stream:
            sys.stdout.write(tokenizer.decode([next_token]))
            sys.stdout.flush()

        logits = decode_step(dec_buffer, encoder_out, src_mask)

    if stream:
        sys.stdout.write("\n")

    return tokenizer.decode(generated_tokens)


def load_audio(path, target_sr=16000):
    """Load an audio file and resample to target_sr. Returns (audio_array, sr)."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32")
    # Convert stereo to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Resample if needed
    if sr != target_sr:
        from scipy.signal import resample
        num_samples = int(len(audio) * target_sr / sr)
        audio = resample(audio, num_samples).astype(np.float32)
        sr = target_sr
    return audio, sr


def transcribe(model, params, tokenizer, audio_array, sr=16000, max_gen_len=128, temperature=0.8, seed=0, stream=True):
    """Transcribe audio using the speech encoder pathway."""
    n_mels = model.config.n_mels
    mel = compute_mel_spectrogram(audio_array, sr=sr, n_mels=n_mels)
    mel_input = jnp.array(mel)[None, :, :]  # (1, T_mel, n_mels)

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    transcribe_id = tokenizer.transcribe_token_id

    src_mask = make_mel_padding_mask(mel_input)
    encoder_out = model.apply(
        {"params": params}, mel_input, src_mask=src_mask, deterministic=True, method="encode_speech"
    )

    tgt_mask = make_causal_mask(max_gen_len)
    dec_buffer = jnp.full((1, max_gen_len), pad_id, dtype=jnp.int32)
    dec_buffer = dec_buffer.at[0, 0].set(eos_id)  # BOS = EOS in this tokenizer
    dec_buffer = dec_buffer.at[0, 1].set(transcribe_id)

    @jax.jit
    def decode_step(dec_buffer, encoder_out, src_mask):
        logits = model.apply(
            {"params": params},
            dec_buffer,
            encoder_out,
            self_mask=tgt_mask,
            cross_mask=src_mask,
            method="decode",
        )
        return logits

    rng = jax.random.PRNGKey(seed)
    generated_tokens = []

    if stream:
        sys.stdout.write("\n")
        sys.stdout.flush()

    logits = decode_step(dec_buffer, encoder_out, src_mask)

    # Start generating from position 2 (after BOS + <transcribe>)
    for i in range(1, max_gen_len - 1):
        next_logits = logits[0, i] / temperature
        rng, sample_rng = jax.random.split(rng)
        next_token = jax.random.categorical(sample_rng, next_logits).item()

        if next_token == eos_id:
            break

        generated_tokens.append(next_token)
        dec_buffer = dec_buffer.at[0, i + 1].set(next_token)

        if stream:
            sys.stdout.write(tokenizer.decode([next_token]))
            sys.stdout.flush()

        logits = decode_step(dec_buffer, encoder_out, src_mask)

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

    # --- Audio transcription mode ---
    audio_files = getattr(args, "audio", None)
    if audio_files:
        for i, audio_path in enumerate(audio_files):
            print(f"\nTranscribing: {audio_path}")
            audio, sr = load_audio(audio_path)
            transcribe(
                model,
                params,
                tokenizer,
                audio,
                sr=sr,
                max_gen_len=args.max_len,
                temperature=args.temperature,
                seed=args.seed + i,
                stream=True,
            )
        return

    # --- Text generation mode ---
    prompts = args.prompts or ["Once upon a time", "The little dog", "She was very happy because"]

    for i, prompt in enumerate(prompts):
        generate(
            model,
            params,
            tokenizer,
            prompt,
            max_gen_len=args.max_len,
            temperature=args.temperature,
            seed=args.seed + i,
            stream=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate stories with trained transformer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch3.pkl")
    parser.add_argument("--prompts", type=str, nargs="*", help="Prompts to continue")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
