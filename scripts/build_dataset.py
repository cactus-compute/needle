#!/usr/bin/env python3
"""Unified dataset builder: text + TTS audio -> HuggingFace Dataset on GCS.

Combines the text pipeline (tools_data.py) with TTS audio generation into
one script that produces a single HF Dataset with an Audio column.

When --gcs-upload is provided, processes in batches: each batch is synthesized,
saved as a shard, uploaded to GCS, and deleted locally — keeping disk usage bounded.

Usage:
    python scripts/build_dataset.py --max-samples 1000 --workers 16
    python scripts/build_dataset.py --gcs-upload gs://cactus-dataset/tool_calls --batch-size 500
"""

import argparse
import io
import logging
import os
import random
import shutil
import subprocess
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor

from datasets import Audio, Dataset, load_from_disk
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools_data import load_and_combine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_VOICE_TEMPLATES = [
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

_VOICE_TYPES = ["Wavenet", "Neural2"]

VOICES = _VOICE_TEMPLATES


def _get_voice(idx):
    """Return (lang, voice_name, gender_str, configurable, child) for a sample.

    Cycles through voice types (Wavenet->Neural2) via idx % 2,
    then picks a voice template via (idx // 3) % len(_VOICE_TEMPLATES).
    """
    vtype = _VOICE_TYPES[idx % len(_VOICE_TYPES)]
    template = _VOICE_TEMPLATES[(idx // 3) % len(_VOICE_TEMPLATES)]
    lang, base_name, gender, configurable, child = template
    voice_name = base_name.replace("Wavenet", vtype)
    return lang, voice_name, gender, configurable, child


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

    Returns:
        Tuple of (idx, wav_bytes, voice_name, error_msg).
        wav_bytes is None on failure; error_msg is None on success.
    """
    from google.cloud import texttospeech
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

    lang, voice_name, ssml_gender, configurable, child = _get_voice(idx)
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

def _synthesize_rows(batch_ds, audio_data, voice_names, workers, rate_limit,
                     cache_dir, global_offset=0):
    """Run TTS synthesis for rows where audio_data[i] is None.

    Uses global_offset + local_index for voice selection so that voice
    assignment is deterministic regardless of batch boundaries.
    WAV files are written to cache_dir/{local_idx}.wav.
    """
    needs = [i for i in range(len(batch_ds)) if audio_data[i] is None]
    if not needs:
        return

    from google.cloud import texttospeech

    os.makedirs(cache_dir, exist_ok=True)

    items = [(i, global_offset + i, batch_ds[i]["query"]) for i in needs]
    logger.info(f"Synthesizing TTS for {len(items)} rows with {workers} workers...")

    def _worker(item):
        local_idx, global_idx, query = item
        client = texttospeech.TextToSpeechClient()
        _, wav_bytes, voice_name, err = _synthesize_one(client, global_idx, query)
        return local_idx, wav_bytes, voice_name, err

    success, failed = 0, 0
    delay = 1.0 / rate_limit if rate_limit > 0 else 0
    total = len(items)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        submitted = 0
        qi = iter(items)

        for _ in range(min(workers * 2, total)):
            q = next(qi)
            pending[pool.submit(_worker, q)] = q
            submitted += 1

        with tqdm(total=total, desc="TTS") as pbar:
            while pending:
                done_set = {f for f in list(pending) if f.done()}
                if not done_set:
                    time.sleep(0.05)
                    continue
                for fut in done_set:
                    local_idx, wav_bytes, voice_name, err = fut.result()
                    if wav_bytes is not None:
                        wav_path = os.path.join(cache_dir, f"{local_idx}.wav")
                        with open(wav_path, "wb") as f:
                            f.write(wav_bytes)
                        audio_data[local_idx] = wav_path
                        voice_names[local_idx] = voice_name
                        success += 1
                    else:
                        failed += 1
                        logger.warning(f"Failed idx={global_offset + local_idx}: {err}")
                    pbar.update(1)
                    del pending[fut]

                    if submitted < total:
                        q = next(qi)
                        pending[pool.submit(_worker, q)] = q
                        submitted += 1
                        time.sleep(delay)

    logger.info(f"TTS done: {success} generated, {failed} failed")


def _backfill_voice_names(voice_names, global_offset=0):
    """Assign default voice names to any rows that don't have one."""
    for i in range(len(voice_names)):
        if not voice_names[i]:
            _, voice_name, _, _, _ = VOICES[(global_offset + i) % len(VOICES)]
            voice_names[i] = voice_name

def _build_arrow(batch_ds, audio_data, voice_names, arrow_path):
    """Build an HF Dataset from a batch and write a single .arrow file.

    Uses save_to_disk into a temp dir so that Audio columns are properly
    serialized (WAV bytes embedded), then moves the generated .arrow file.
    """
    import tempfile

    n = len(batch_ds)
    audio_column = []
    for i in range(n):
        path = audio_data[i]
        if path is not None and os.path.exists(path):
            audio_column.append(path)
        else:
            audio_column.append(None)

    ds_dict = {
        "query": [batch_ds[i]["query"] for i in range(n)],
        "audio": audio_column,
        "tools": [batch_ds[i]["tools"] for i in range(n)],
        "answers": [batch_ds[i]["answers"] for i in range(n)],
        "source": [batch_ds[i]["source"] for i in range(n)],
        "voice": voice_names[:n],
    }

    ds = Dataset.from_dict(ds_dict)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    with tempfile.TemporaryDirectory() as tmpdir:
        ds.save_to_disk(tmpdir)
        arrow_files = sorted(f for f in os.listdir(tmpdir) if f.endswith(".arrow"))
        if len(arrow_files) == 1:
            shutil.move(os.path.join(tmpdir, arrow_files[0]), arrow_path)
        else:
            import pyarrow as pa
            tables = []
            for af in arrow_files:
                with open(os.path.join(tmpdir, af), "rb") as fh:
                    reader = pa.ipc.open_stream(fh)
                    tables.append(reader.read_all())
            merged = pa.concat_tables(tables)
            with pa.OSFile(arrow_path, "wb") as sink:
                writer = pa.ipc.new_stream(sink, merged.schema)
                writer.write_table(merged)
                writer.close()

    logger.info(f"Built {arrow_path} ({n} rows)")
    info = ds.info
    del ds, ds_dict, audio_column
    return info


def _arrow_exists_on_gcs(gcs_base, arrow_name):
    """Check if an arrow file already exists on GCS."""
    gcs_path = gcs_base + "/" + arrow_name
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_path],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and result.stdout.strip()


def _upload_file_to_gcs(local_path, gcs_dest):
    """Upload a single file to GCS."""
    logger.info(f"Uploading {os.path.basename(local_path)} -> {gcs_dest}")
    result = subprocess.run(
        ["gcloud", "storage", "cp", local_path, gcs_dest],
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Upload failed for {local_path}")
        sys.exit(1)


def _upload_metadata_to_gcs(gcs_base, num_batches, dataset_info):
    """Generate and upload state.json + dataset_info.json so load_from_disk works."""
    import json
    import tempfile

    arrow_files = [
        {"filename": f"data-{i:05d}-of-{num_batches:05d}.arrow"}
        for i in range(num_batches)
    ]

    state = {
        "_data_files": arrow_files,
        "_fingerprint": "batch_upload",
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": None,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "state.json")
        with open(state_path, "w") as f:
            json.dump(state, f)
        _upload_file_to_gcs(state_path, gcs_base + "/state.json")

        info_path = os.path.join(tmpdir, "dataset_info.json")
        dataset_info.write_to_directory(tmpdir)
        _upload_file_to_gcs(info_path, gcs_base + "/dataset_info.json")


def _process_batched(combined, batch_size, workers, rate_limit, output_path, gcs_upload):
    """Process dataset in batches: one arrow per batch, flat on GCS.

    GCS layout:
        gs://bucket/path/data-00000-of-00020.arrow
        gs://bucket/path/data-00001-of-00020.arrow
        ...
        gs://bucket/path/state.json
        gs://bucket/path/dataset_info.json

    load_from_disk("gs://bucket/path") works directly.
    """
    n = len(combined)
    num_batches = (n + batch_size - 1) // batch_size
    total_rows = 0
    dataset_info = None

    logger.info(f"Processing {n} rows in {num_batches} batches of {batch_size}...")

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)
        arrow_name = f"data-{batch_idx:05d}-of-{num_batches:05d}.arrow"

        # Resume: skip arrows already on GCS
        if _arrow_exists_on_gcs(gcs_upload, arrow_name):
            logger.info(f"{arrow_name} already on GCS, skipping (rows {start}-{end - 1})")
            total_rows += (end - start)
            continue

        logger.info(f"=== Batch {batch_idx + 1}/{num_batches}: {arrow_name} (rows {start}-{end - 1}) ===")
        batch = combined.select(range(start, end))
        batch_len = len(batch)

        audio_data = [None] * batch_len
        voice_names = [""] * batch_len
        cache_dir = os.path.join(output_path, f".cache_{batch_idx:05d}")

        existing_cache = os.path.join(output_path, ".audio_cache")
        if os.path.isdir(existing_cache):
            recovered = 0
            for i in range(batch_len):
                wav_path = os.path.join(existing_cache, f"{start + i}.wav")
                if os.path.exists(wav_path):
                    audio_data[i] = wav_path
                    recovered += 1
            if recovered:
                logger.info(f"  Recovered {recovered}/{batch_len} WAVs from existing cache")

        _synthesize_rows(batch, audio_data, voice_names, workers, rate_limit,
                         cache_dir, global_offset=start)
        _backfill_voice_names(voice_names, global_offset=start)

        os.makedirs(output_path, exist_ok=True)
        arrow_path = os.path.join(output_path, arrow_name)
        info = _build_arrow(batch, audio_data, voice_names, arrow_path)
        if dataset_info is None:
            dataset_info = info

        _upload_file_to_gcs(arrow_path, gcs_upload + "/" + arrow_name)
        os.remove(arrow_path)
        shutil.rmtree(cache_dir, ignore_errors=True)

        existing_cache = os.path.join(output_path, ".audio_cache")
        if os.path.isdir(existing_cache):
            for i in range(batch_len):
                wav_path = os.path.join(existing_cache, f"{start + i}.wav")
                if os.path.exists(wav_path):
                    os.remove(wav_path)

        total_rows += batch_len
        logger.info(f"Progress: {total_rows}/{n} rows uploaded")

    if dataset_info is not None:
        _upload_metadata_to_gcs(gcs_upload, num_batches, dataset_info)

    return total_rows


def _load_text_dataset(output_path, max_samples):
    """Load the text dataset and optionally resume from an existing build."""
    existing_ds = None
    if os.path.exists(output_path):
        try:
            logger.info(f"Found existing dataset at {output_path}, loading for resume...")
            existing_ds = load_from_disk(output_path)
            logger.info(f"  Existing dataset: {len(existing_ds)} rows")
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"Could not load existing dataset (will use audio cache only): {e}")

    logger.info("Loading and combining text datasets...")
    combined = load_and_combine()

    if max_samples:
        combined = combined.select(range(min(max_samples, len(combined))))
    logger.info(f"Text dataset: {len(combined)} rows")

    return combined, existing_ds


def _find_rows_needing_tts(combined, existing_ds, output_path):
    """Identify which rows lack audio."""
    n = len(combined)
    needs_tts = set(range(n))

    cache_dir = os.path.join(output_path, ".audio_cache")
    if os.path.isdir(cache_dir):
        cached = 0
        for fname in os.listdir(cache_dir):
            if fname.endswith(".wav"):
                try:
                    idx = int(fname[:-4])
                    if idx < n:
                        needs_tts.discard(idx)
                        cached += 1
                except ValueError:
                    pass
        if cached:
            logger.info(f"  {cached} rows found in audio cache")

    if existing_ds is not None and len(existing_ds) == n and "audio" in existing_ds.column_names:
        audio_column = existing_ds.data.column("audio")
        for i in range(n):
            if i in needs_tts and audio_column[i].as_py() is not None:
                needs_tts.discard(i)

    logger.info(f"  {n - len(needs_tts)} rows already have audio, {len(needs_tts)} need TTS")
    return needs_tts


def _recover_existing_audio(existing_ds, n, needs_tts, output_path):
    """Recover audio file paths from the cache dir and existing dataset."""
    audio_data = [None] * n
    voice_names = [""] * n

    cache_dir = os.path.join(output_path, ".audio_cache")
    if os.path.isdir(cache_dir):
        for i in range(n):
            if i not in needs_tts:
                wav_path = os.path.join(cache_dir, f"{i}.wav")
                if os.path.exists(wav_path):
                    audio_data[i] = wav_path

    if existing_ds is not None and len(existing_ds) == n and "audio" in existing_ds.column_names:
        audio_column = existing_ds.data.column("audio")
        voice_col = existing_ds["voice"] if "voice" in existing_ds.column_names else None
        for i in range(n):
            if i not in needs_tts and audio_data[i] is None:
                raw = audio_column[i].as_py()
                if isinstance(raw, dict) and raw.get("bytes"):
                    wav_path = os.path.join(cache_dir, f"{i}.wav")
                    os.makedirs(cache_dir, exist_ok=True)
                    with open(wav_path, "wb") as f:
                        f.write(raw["bytes"])
                    audio_data[i] = wav_path
                elif isinstance(raw, dict) and raw.get("path") and os.path.exists(raw["path"]):
                    audio_data[i] = raw["path"]
            if voice_col is not None and i not in needs_tts:
                voice_names[i] = voice_col[i] or ""

    return audio_data, voice_names


def _process_local(combined, workers, rate_limit, output_path):
    """Original local-only processing: synthesize all, build one dataset."""
    existing_ds = None
    if os.path.exists(output_path):
        try:
            existing_ds = load_from_disk(output_path)
        except Exception:
            pass

    needs_tts = _find_rows_needing_tts(combined, existing_ds, output_path)
    audio_data, voice_names = _recover_existing_audio(
        existing_ds, len(combined), needs_tts, output_path
    )

    cache_dir = os.path.join(output_path, ".audio_cache")
    _synthesize_rows(combined, audio_data, voice_names, workers, rate_limit, cache_dir)
    _backfill_voice_names(voice_names)

    n = len(combined)
    os.makedirs(output_path, exist_ok=True)
    arrow_path = os.path.join(output_path, "data-00000-of-00001.arrow")
    info = _build_arrow(combined, audio_data, voice_names, arrow_path)

    # Write metadata so load_from_disk works
    import json
    state = {
        "_data_files": [{"filename": "data-00000-of-00001.arrow"}],
        "_fingerprint": "local_build",
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": None,
    }
    with open(os.path.join(output_path, "state.json"), "w") as f:
        json.dump(state, f)
    info.write_to_directory(output_path)
    logger.info(f"Saved {n} rows to {output_path}/")
    return n

def main():
    parser = argparse.ArgumentParser(description="Build unified tool-call dataset with TTS audio")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--workers", type=int, default=16, help="TTS concurrent workers (default: 16)")
    parser.add_argument("--output", type=str, default="data/tool_calls_unified", help="Output directory")
    parser.add_argument("--rate-limit", type=float, default=30.0,
                        help="Max TTS requests per second (default: 30, quota is 2x1000/min)")
    parser.add_argument("--gcs-upload", type=str, default=None,
                        help="GCS path to upload dataset (e.g. gs://cactus-dataset/tool_calls)")
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="Rows per batch when using --gcs-upload (default: 500)")
    args = parser.parse_args()

    output_path = args.output

    logger.info("Loading and combining text datasets...")
    combined = load_and_combine()
    if args.max_samples:
        combined = combined.select(range(min(args.max_samples, len(combined))))
    logger.info(f"Text dataset: {len(combined)} rows")

    if args.gcs_upload:
        total_rows = _process_batched(
            combined, args.batch_size, args.workers, args.rate_limit,
            output_path, args.gcs_upload,
        )
    else:
        total_rows = _process_local(combined, args.workers, args.rate_limit, output_path)

    logger.info(f"Pipeline complete: {total_rows} rows")


if __name__ == "__main__":
    main()
