#!/usr/bin/env python3
"""Unified dataset builder: text + TTS audio -> HuggingFace Dataset on GCS.

Combines the text pipeline (tools_data.py) with TTS audio generation into
one script that produces a single HF Dataset with an Audio column.

Usage:
    python scripts/build_dataset.py --max-samples 1000 --workers 16
    python scripts/build_dataset.py --max-samples 1000 --gcs-upload gs://cactus-dataset/tool_calls
"""

import argparse
import io
import logging
import os
import random
import subprocess
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import Audio, Dataset, load_from_disk
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools_data import load_and_combine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VOICES = [
    ("en-US", "en-US-Journey-D", "MALE", False, False),
    ("en-US", "en-US-Journey-F", "FEMALE", False, False),
    ("en-US", "en-US-Journey-O", "FEMALE", False, False),
    ("en-GB", "en-GB-Wavenet-B", "MALE", True, False),
    ("en-GB", "en-GB-Wavenet-A", "FEMALE", True, False),
    ("en-AU", "en-AU-Wavenet-B", "MALE", True, False),
    ("en-AU", "en-AU-Wavenet-C", "FEMALE", True, False),
    ("en-US", "en-US-Wavenet-D", "MALE", True, False),
    ("en-IN", "en-IN-Wavenet-A", "FEMALE", True, False),
    ("en-IN", "en-IN-Wavenet-B", "MALE", True, False),
    ("en-IN", "en-IN-Wavenet-C", "MALE", True, False),
    ("en-GB", "en-GB-Wavenet-D", "MALE", True, False),
    ("en-AU", "en-AU-Wavenet-D", "MALE", True, False),
    ("en-US", "en-US-Wavenet-E", "FEMALE", True, True),
    ("en-GB", "en-GB-Wavenet-A", "FEMALE", True, True),
    ("en-IN", "en-IN-Wavenet-A", "FEMALE", True, True),
]


def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Convert raw LINEAR16 PCM bytes to a WAV byte buffer."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _synthesize_one(client, idx, query, max_retries=5):
    """Synthesize TTS audio for a single query via GCP Cloud TTS.

    Selects a voice from VOICES based on idx, applies randomized prosody
    for configurable voices, and retries on transient API errors.

    Returns:
        Tuple of (idx, wav_bytes, voice_name, error_msg).
        wav_bytes is None on failure; error_msg is None on success.
    """
    from google.cloud import texttospeech
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

    lang, voice_name, ssml_gender, configurable, child = VOICES[idx % len(VOICES)]
    gender = getattr(texttospeech.SsmlVoiceGender, ssml_gender)

    synthesis_input = texttospeech.SynthesisInput(text=query)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=lang,
        name=voice_name,
        ssml_gender=gender,
    )

    if configurable:
        rng = random.Random(idx)
        if child:
            pitch = rng.uniform(4.0, 8.0)
            rate = rng.uniform(1.0, 1.15)
        else:
            pitch = rng.uniform(-3.0, 3.0)
            rate = rng.uniform(0.9, 1.1)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            pitch=pitch,
            speaking_rate=rate,
            volume_gain_db=rng.uniform(-1.5, 1.5),
        )
    else:
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
        )

    for attempt in range(max_retries):
        try:
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config,
            )
            wav_bytes = _pcm_to_wav_bytes(response.audio_content, 16000)
            return idx, wav_bytes, voice_name, None
        except (ResourceExhausted, ServiceUnavailable):
            wait = min(2 ** attempt, 30)
            time.sleep(wait)
        except Exception as e:
            return idx, None, voice_name, str(e)

    return idx, None, voice_name, "max retries exceeded"


def _load_text_dataset(output_path, max_samples):
    """Load the text dataset and optionally resume from an existing build.

    Returns:
        Tuple of (combined_dataset, existing_dataset_or_None).
    """
    existing_ds = None
    if os.path.exists(output_path):
        logger.info(f"Found existing dataset at {output_path}, loading for resume...")
        existing_ds = load_from_disk(output_path)
        logger.info(f"  Existing dataset: {len(existing_ds)} rows")

    logger.info("Loading and combining text datasets...")
    combined = load_and_combine()

    if max_samples:
        combined = combined.select(range(min(max_samples, len(combined))))
    logger.info(f"Text dataset: {len(combined)} rows")

    return combined, existing_ds


def _find_rows_needing_tts(combined, existing_ds):
    """Identify which rows lack audio by inspecting the raw Arrow column.

    Accesses the underlying Arrow table directly to avoid triggering
    HF audio decoding (which requires torchcodec).

    Returns:
        Set of row indices that need TTS synthesis.
    """
    n = len(combined)
    needs_tts = set(range(n))

    if existing_ds is not None and len(existing_ds) == n and "audio" in existing_ds.column_names:
        audio_column = existing_ds.data.column("audio")
        for i in range(n):
            if audio_column[i].as_py() is not None:
                needs_tts.discard(i)
        logger.info(f"  {n - len(needs_tts)} rows already have audio, {len(needs_tts)} need TTS")

    return needs_tts


def _recover_existing_audio(existing_ds, n, needs_tts):
    """Extract audio bytes and voice names from a previously saved dataset.

    Reads raw Arrow data to avoid triggering audio decoding.

    Returns:
        Tuple of (audio_data list, voice_names list).
    """
    audio_data = [None] * n
    voice_names = [""] * n

    if existing_ds is not None and len(existing_ds) == n and "audio" in existing_ds.column_names:
        audio_column = existing_ds.data.column("audio")
        voice_col = existing_ds["voice"] if "voice" in existing_ds.column_names else None
        for i in range(n):
            if i not in needs_tts:
                raw = audio_column[i].as_py()
                if isinstance(raw, dict) and raw.get("bytes"):
                    audio_data[i] = raw["bytes"]
                elif isinstance(raw, dict) and raw.get("path"):
                    with open(raw["path"], "rb") as f:
                        audio_data[i] = f.read()
                if voice_col is not None:
                    voice_names[i] = voice_col[i] or ""

    return audio_data, voice_names


def _synthesize_missing(combined, needs_tts, audio_data, voice_names, workers, rate_limit):
    """Run TTS synthesis for all rows that lack audio.

    Uses a thread pool with rate-limited submission to stay within
    GCP Cloud TTS API quotas.
    """
    if not needs_tts:
        return

    from google.cloud import texttospeech

    queries_to_synth = [(i, combined[i]["query"]) for i in sorted(needs_tts)]
    logger.info(f"Synthesizing TTS for {len(queries_to_synth)} rows with {workers} workers...")

    def _worker(item):
        idx, query = item
        client = texttospeech.TextToSpeechClient()
        return _synthesize_one(client, idx, query)

    success, failed = 0, 0
    delay = 1.0 / rate_limit if rate_limit > 0 else 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for q in queries_to_synth:
            futures[pool.submit(_worker, q)] = q
            time.sleep(delay)
        with tqdm(total=len(futures), desc="TTS") as pbar:
            for future in as_completed(futures):
                idx, wav_bytes, voice_name, err = future.result()
                if wav_bytes is not None:
                    audio_data[idx] = wav_bytes
                    voice_names[idx] = voice_name
                    success += 1
                else:
                    failed += 1
                    logger.warning(f"Failed idx={idx}: {err}")
                pbar.update(1)

    logger.info(f"TTS done: {success} generated, {failed} failed")


def _backfill_voice_names(voice_names):
    """Assign default voice names to any rows that don't have one."""
    for i in range(len(voice_names)):
        if not voice_names[i]:
            _, voice_name, _, _, _ = VOICES[i % len(VOICES)]
            voice_names[i] = voice_name


def _build_and_save(combined, audio_data, voice_names, output_path):
    """Assemble the unified HF Dataset and save to disk.

    Returns:
        The saved Dataset.
    """
    n = len(combined)
    logger.info("Building unified HF Dataset...")
    ds_dict = {
        "query": [combined[i]["query"] for i in range(n)],
        "audio": audio_data,
        "tools": [combined[i]["tools"] for i in range(n)],
        "answers": [combined[i]["answers"] for i in range(n)],
        "source": [combined[i]["source"] for i in range(n)],
        "voice": voice_names,
    }

    unified = Dataset.from_dict(ds_dict)
    unified = unified.cast_column("audio", Audio(sampling_rate=16000))

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    unified.save_to_disk(output_path)
    logger.info(f"Saved unified dataset to {output_path}/ ({len(unified)} rows)")
    return unified


def _train_and_save_tokenizer(max_samples):
    """Train a SentencePiece tokenizer on the unified dataset."""
    from src.data import train_tokenizer, TOKENIZER_PREFIX
    logger.info("Training tokenizer on unified dataset...")
    model_path = train_tokenizer(max_samples=max_samples)
    logger.info(f"Tokenizer ready at {model_path}")


def _upload_to_gcs(output_path, gcs_upload):
    """Upload the dataset and tokenizer to GCS via gcloud CLI."""
    from src.data import TOKENIZER_PREFIX

    logger.info(f"Uploading dataset to {gcs_upload}...")
    result = subprocess.run(
        ["gcloud", "storage", "cp", "-r", output_path, gcs_upload],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        logger.info(f"Dataset upload complete: {gcs_upload}")
    else:
        logger.error(f"Dataset upload failed: {result.stderr}")
        sys.exit(1)

    gcs_bucket = gcs_upload.rsplit("/", 1)[0]
    gcs_tokenizer_path = f"{gcs_bucket}/tokenizer/"
    logger.info(f"Uploading tokenizer to {gcs_tokenizer_path}...")
    tokenizer_dir = os.path.dirname(TOKENIZER_PREFIX)
    result = subprocess.run(
        ["gcloud", "storage", "cp", "-r", tokenizer_dir, gcs_tokenizer_path],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        logger.info(f"Tokenizer upload complete: {gcs_tokenizer_path}")
    else:
        logger.error(f"Tokenizer upload failed: {result.stderr}")
        sys.exit(1)


def main():
    """Entry point for the unified dataset build pipeline."""
    parser = argparse.ArgumentParser(description="Build unified tool-call dataset with TTS audio")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--workers", type=int, default=16, help="TTS concurrent workers (default: 16)")
    parser.add_argument("--output", type=str, default="data/tool_calls_unified", help="Output directory")
    parser.add_argument("--rate-limit", type=float, default=100.0,
                        help="Max TTS requests per second (default: 100)")
    parser.add_argument("--gcs-upload", type=str, default=None,
                        help="GCS path to upload dataset (e.g. gs://cactus-dataset/tool_calls)")
    args = parser.parse_args()

    combined, existing_ds = _load_text_dataset(args.output, args.max_samples)
    needs_tts = _find_rows_needing_tts(combined, existing_ds)
    audio_data, voice_names = _recover_existing_audio(existing_ds, len(combined), needs_tts)
    _synthesize_missing(combined, needs_tts, audio_data, voice_names, args.workers, args.rate_limit)
    _backfill_voice_names(voice_names)
    _build_and_save(combined, audio_data, voice_names, args.output)
    _train_and_save_tokenizer(args.max_samples)

    if args.gcs_upload:
        _upload_to_gcs(args.output, args.gcs_upload)


if __name__ == "__main__":
    main()
