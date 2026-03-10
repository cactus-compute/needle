#!/usr/bin/env python3
"""Build an English-only Emilia audio dataset with mel features + CSV metadata.

This script is designed to live alongside `scripts/build_dataset.py` and
`scripts/tools_data.py`, and follows the same CLI/script style.

Pipeline:
1. Stream `amphion/Emilia-Dataset` from Hugging Face.
2. Keep language-matching rows (default: English only).
3. Decode audio, resample, compute log-mel features.
4. Upload audio + mel `.npy` files to GCS.
5. Write/upload CSV metadata manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import mimetypes
import os
import re
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("Missing dependency 'numpy'. Install with: pip install numpy") from exc

try:
    from datasets import Audio, load_dataset
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency 'datasets'. Install with: pip install datasets"
    ) from exc

try:
    from google.cloud import storage
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency 'google-cloud-storage'. Install with: pip install google-cloud-storage"
    ) from exc

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency 'soundfile'. Install with: pip install soundfile"
    ) from exc

try:
    from scipy import signal
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("Missing dependency 'scipy'. Install with: pip install scipy") from exc


# Keep script-local imports consistent with scripts/build_dataset.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from tools_data import DATASETS as TOOL_DATASETS
except Exception:  # pragma: no cover - optional metadata only
    TOOL_DATASETS = {}

try:
    from build_dataset import VOICES as BUILD_DATASET_VOICES
except Exception:  # pragma: no cover - optional metadata only
    BUILD_DATASET_VOICES = []


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class WorkResult:
    ok: bool
    row: dict[str, Any] | None
    error: str | None = None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_meta(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        text = raw.decode("utf-8", errors="replace")
    elif isinstance(raw, str):
        text = raw
    else:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_language(meta: dict[str, Any]) -> str:
    for key in ("language", "lang", "language_code"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_transcript(meta: dict[str, Any]) -> str:
    for key in ("text", "transcript", "normalized_text", "sentence", "content"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _matches_language(language: str, pattern: re.Pattern[str]) -> bool:
    lang = language.strip()
    if not lang:
        return False
    return bool(pattern.search(lang))


def _sanitize_id(value: str | None, fallback_seed: str) -> str:
    if value is None or value == "":
        return hashlib.sha1(fallback_seed.encode("utf-8")).hexdigest()[:20]
    cleaned = re.sub(r"[^a-zA-Z0-9._/-]+", "_", str(value))
    cleaned = cleaned.strip("._/")
    if not cleaned:
        return hashlib.sha1(fallback_seed.encode("utf-8")).hexdigest()[:20]
    return cleaned.replace("/", "_")


def _belongs_to_shard(source_id: str, num_shards: int, shard_index: int) -> bool:
    if num_shards <= 1:
        return True
    digest = hashlib.sha1(source_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_shards == shard_index


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _make_mel_filter(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    if fmax <= fmin:
        raise ValueError(f"Invalid mel range: fmin={fmin}, fmax={fmax}")

    fft_bins = n_fft // 2 + 1
    mel_min = _hz_to_mel(np.array([fmin], dtype=np.float64))[0]
    mel_max = _hz_to_mel(np.array([fmax], dtype=np.float64))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float64)
    hz_points = _mel_to_hz(mel_points)

    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int32)
    bins = np.clip(bins, 0, fft_bins - 1)

    fb = np.zeros((n_mels, fft_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left = int(bins[m - 1])
        center = int(bins[m])
        right = int(bins[m + 1])

        if center <= left:
            center = min(left + 1, fft_bins - 1)
        if right <= center:
            right = min(center + 1, fft_bins)

        if center > left:
            fb[m - 1, left:center] = np.linspace(0.0, 1.0, center - left, endpoint=False)
        if right > center:
            fb[m - 1, center:right] = np.linspace(1.0, 0.0, right - center, endpoint=False)

    width_hz = np.maximum(hz_points[2:] - hz_points[:-2], 1e-8)
    fb *= (2.0 / width_hz).astype(np.float32)[:, None]
    return fb


def _guess_audio_ext(path_hint: str | None, default_ext: str = ".mp3") -> str:
    if not path_hint:
        return default_ext
    ext = Path(path_hint).suffix.lower()
    if not ext or len(ext) > 8:
        return default_ext
    return ext


def _decode_audio(audio_field: Any) -> tuple[np.ndarray, int, bytes, str | None]:
    raw_bytes: bytes | None = None
    path_hint: str | None = None

    if isinstance(audio_field, dict):
        raw = audio_field.get("bytes")
        if raw is not None:
            raw_bytes = bytes(raw)
        path_hint = audio_field.get("path")
        if raw_bytes is None and path_hint and os.path.exists(path_hint):
            with open(path_hint, "rb") as f:
                raw_bytes = f.read()
    elif isinstance(audio_field, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(audio_field)
    elif isinstance(audio_field, str):
        path_hint = audio_field
        if os.path.exists(audio_field):
            with open(audio_field, "rb") as f:
                raw_bytes = f.read()

    if raw_bytes is None:
        raise ValueError("Could not extract audio bytes from sample.")

    audio_array, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    if audio_array.ndim == 2:
        audio_array = np.mean(audio_array, axis=1)
    if audio_array.ndim != 1:
        raise ValueError(f"Expected mono waveform, got shape={audio_array.shape}")

    return np.asarray(audio_array, dtype=np.float32), int(sample_rate), raw_bytes, path_hint


def _resample_if_needed(waveform: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return waveform.astype(np.float32, copy=False)
    gcd = math.gcd(src_rate, dst_rate)
    up = dst_rate // gcd
    down = src_rate // gcd
    return signal.resample_poly(waveform, up, down).astype(np.float32, copy=False)


def _compute_log_mel(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    mel_filter: np.ndarray,
) -> np.ndarray:
    if waveform.size < win_length:
        waveform = np.pad(waveform, (0, win_length - waveform.size))

    _, _, spec = signal.stft(
        waveform,
        fs=sample_rate,
        window="hann",
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
    )
    power = np.abs(spec).astype(np.float32) ** 2
    mel = np.matmul(mel_filter, power)
    return np.log(np.maximum(mel, 1e-10)).astype(np.float32)


def _upload_string(bucket: storage.Bucket, blob_path: str, payload: bytes, content_type: str) -> str:
    bucket.blob(blob_path).upload_from_string(payload, content_type=content_type)
    return f"gs://{bucket.name}/{blob_path}"


def _upload_file(bucket: storage.Bucket, blob_path: str, filename: str, content_type: str) -> str:
    bucket.blob(blob_path).upload_from_filename(filename, content_type=content_type)
    return f"gs://{bucket.name}/{blob_path}"


def _process_one(
    sample: dict[str, Any],
    source_index: int,
    bucket: storage.Bucket | None,
    prefix: str,
    split: str,
    target_sample_rate: int,
    n_mels: int,
    win_length: int,
    hop_length: int,
    n_fft: int,
    mel_filter: np.ndarray,
    upload_audio: bool,
    dry_run: bool,
) -> WorkResult:
    try:
        meta = _normalize_meta(sample.get("json"))
        language = _extract_language(meta)
        transcript = _extract_transcript(meta)
        duration_meta = _safe_float(meta.get("duration"))

        source_key = sample.get("__key__")
        source_url = sample.get("__url__")
        source_id = _sanitize_id(
            source_key,
            fallback_seed=f"emilia:{split}:{source_index}:{transcript[:40]}",
        )
        item_id = f"{source_id}-{source_index:09d}"

        waveform, src_sr, audio_bytes, path_hint = _decode_audio(sample.get("mp3"))
        waveform = _resample_if_needed(waveform, src_rate=src_sr, dst_rate=target_sample_rate)
        mel = _compute_log_mel(
            waveform=waveform,
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            mel_filter=mel_filter,
        )

        mel_buf = io.BytesIO()
        np.save(mel_buf, mel.astype(np.float16), allow_pickle=False)
        mel_payload = mel_buf.getvalue()

        audio_ext = _guess_audio_ext(path_hint, default_ext=".mp3")
        audio_ct = mimetypes.guess_type(f"file{audio_ext}")[0] or "application/octet-stream"

        mel_blob = f"{prefix}/emilia-large/{split}/mels/{item_id}.npy"
        audio_blob = f"{prefix}/emilia-large/{split}/audio/{item_id}{audio_ext}"

        if dry_run:
            mel_uri = f"dry-run://{mel_blob}"
            audio_uri = f"dry-run://{audio_blob}" if upload_audio else ""
        else:
            if bucket is None:
                raise RuntimeError("GCS bucket is required when not using --dry-run.")
            mel_uri = _upload_string(
                bucket=bucket,
                blob_path=mel_blob,
                payload=mel_payload,
                content_type="application/octet-stream",
            )
            audio_uri = ""
            if upload_audio:
                audio_uri = _upload_string(
                    bucket=bucket,
                    blob_path=audio_blob,
                    payload=audio_bytes,
                    content_type=audio_ct,
                )

        row = {
            "item_id": item_id,
            "split": split,
            "language": language,
            "transcript": transcript,
            "speaker": str(meta.get("speaker") or ""),
            "source_key": str(source_key or ""),
            "source_url": str(source_url or ""),
            "meta_duration_s": duration_meta,
            "audio_duration_s": round(float(len(waveform)) / float(target_sample_rate), 6),
            "source_sample_rate_hz": src_sr,
            "target_sample_rate_hz": target_sample_rate,
            "n_mels": n_mels,
            "mel_frames": int(mel.shape[1]),
            "num_audio_samples": int(len(waveform)),
            "dnsmos": _safe_float(meta.get("dnsmos")),
            "phone_count": meta.get("phone_count"),
            "audio_gcs_uri": audio_uri,
            "mel_gcs_uri": mel_uri,
        }
        return WorkResult(ok=True, row=row)
    except Exception as exc:
        return WorkResult(ok=False, row=None, error=str(exc))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Emilia English audio dataset with mel features and metadata CSV."
    )
    parser.add_argument("--bucket", help="Destination GCS bucket.")
    parser.add_argument("--project", default=None, help="Optional GCP project.")
    parser.add_argument(
        "--prefix",
        default="speech_datasets",
        help="GCS key prefix under the bucket.",
    )
    parser.add_argument("--split", default="train", help="Dataset split (default: train).")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN).",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=1_000_000,
        help="Number of successful samples to write (default: 1,000,000).",
    )
    parser.add_argument(
        "--language-regex",
        default=r"^(en($|[-_])|eng$|english$)",
        help="Case-insensitive regex for allowed languages (default: English only).",
    )
    parser.add_argument(
        "--max-source-rows",
        type=int,
        default=None,
        help="Optional cap on streamed source rows for smoke tests.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, max(4, os.cpu_count() or 4)),
        help="Thread workers (default: min(16, cpu_count)).",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="Max pending futures (default: workers*4).",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate.")
    parser.add_argument("--n-mels", type=int, default=80, help="Number of mel bins.")
    parser.add_argument("--win-ms", type=float, default=25.0, help="STFT window length in ms.")
    parser.add_argument("--hop-ms", type=float, default=10.0, help="STFT hop length in ms.")
    parser.add_argument("--fmin", type=float, default=0.0, help="Mel min frequency.")
    parser.add_argument("--fmax", type=float, default=None, help="Mel max frequency.")
    parser.add_argument(
        "--upload-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload source audio objects to GCS (default: true).",
    )
    parser.add_argument("--csv-name", default=None, help="Output CSV filename in GCS.")
    parser.add_argument("--local-csv-path", default=None, help="Optional local CSV output path.")
    parser.add_argument("--num-shards", type=int, default=1, help="Logical number of shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index for this process.")
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use HF streaming mode (default: true).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and write local CSV without GCS uploads.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log progress every N successful samples.",
    )
    return parser.parse_args()


def _drain_one_completed(
    futures: set[Future[WorkResult]],
    csv_writer: csv.DictWriter,
    csv_fp,
    counters: dict[str, int],
    log_every: int,
    started_at: float,
) -> None:
    done, _ = wait(futures, return_when=FIRST_COMPLETED)
    for future in done:
        futures.discard(future)
        counters["completed"] += 1
        try:
            result = future.result()
        except Exception as exc:  # pragma: no cover - defensive guard
            counters["failed"] += 1
            if counters["failed"] <= 10:
                logger.warning("Worker failure: %s", exc)
            continue

        if not result.ok or result.row is None:
            counters["failed"] += 1
            if counters["failed"] <= 10 and result.error:
                logger.warning("Sample failed: %s", result.error)
            continue

        csv_writer.writerow(result.row)
        counters["success"] += 1

        if counters["success"] % max(1, log_every) == 0:
            csv_fp.flush()
            elapsed = max(0.001, time.time() - started_at)
            logger.info(
                "Progress success=%s failed=%s completed=%s rate=%.2f samples/s",
                counters["success"],
                counters["failed"],
                counters["completed"],
                counters["success"] / elapsed,
            )


def main() -> None:
    args = _parse_args()

    if args.target_samples < 1:
        raise SystemExit("--target-samples must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, num_shards)")
    if not args.dry_run and not args.bucket:
        raise SystemExit("--bucket is required unless --dry-run is set")

    sample_rate = int(args.sample_rate)
    language_re = re.compile(args.language_regex, flags=re.IGNORECASE)
    win_length = max(1, int(round(sample_rate * (args.win_ms / 1000.0))))
    hop_length = max(1, int(round(sample_rate * (args.hop_ms / 1000.0))))
    n_fft = 1
    while n_fft < win_length:
        n_fft *= 2

    fmax = float(args.fmax) if args.fmax is not None else float(sample_rate) / 2.0
    mel_filter = _make_mel_filter(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=int(args.n_mels),
        fmin=float(args.fmin),
        fmax=fmax,
    )
    max_in_flight = args.max_in_flight if args.max_in_flight is not None else args.workers * 4
    max_in_flight = max(args.workers, max_in_flight)

    if args.csv_name:
        csv_name = args.csv_name
    elif args.num_shards > 1:
        csv_name = f"metadata-shard-{args.shard_index:03d}-of-{args.num_shards:03d}.csv"
    else:
        csv_name = "metadata.csv"

    if args.local_csv_path:
        local_csv_path = args.local_csv_path
        os.makedirs(os.path.dirname(os.path.abspath(local_csv_path)), exist_ok=True)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="audio_data_")
        local_csv_path = os.path.join(tmp_dir, csv_name)

    bucket: storage.Bucket | None = None
    if not args.dry_run:
        client = storage.Client(project=args.project) if args.project else storage.Client()
        bucket = client.bucket(args.bucket)

    logger.info(
        "Starting Emilia ingest split=%s target=%s shard=%s/%s workers=%s dry_run=%s",
        args.split,
        args.target_samples,
        args.shard_index,
        args.num_shards,
        args.workers,
        args.dry_run,
    )
    logger.info(
        "Mel config sr=%s n_mels=%s win_ms=%s hop_ms=%s n_fft=%s",
        sample_rate,
        args.n_mels,
        args.win_ms,
        args.hop_ms,
        n_fft,
    )

    ds_kwargs: dict[str, Any] = {
        "path": "amphion/Emilia-Dataset",
        "split": args.split,
        "streaming": args.streaming,
    }
    if args.hf_token:
        ds_kwargs["token"] = args.hf_token

    ds = load_dataset(**ds_kwargs)
    ds = ds.cast_column("mp3", Audio(decode=False))

    fieldnames = [
        "item_id",
        "split",
        "language",
        "transcript",
        "speaker",
        "source_key",
        "source_url",
        "meta_duration_s",
        "audio_duration_s",
        "source_sample_rate_hz",
        "target_sample_rate_hz",
        "n_mels",
        "mel_frames",
        "num_audio_samples",
        "dnsmos",
        "phone_count",
        "audio_gcs_uri",
        "mel_gcs_uri",
    ]

    counters = {
        "source_rows_seen": 0,
        "language_rows_seen": 0,
        "submitted": 0,
        "completed": 0,
        "success": 0,
        "failed": 0,
    }
    started_at = time.time()
    futures: set[Future[WorkResult]] = set()

    with open(local_csv_path, "w", newline="", encoding="utf-8") as csv_fp:
        writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for source_index, sample in enumerate(ds):
                counters["source_rows_seen"] += 1
                if args.max_source_rows is not None and counters["source_rows_seen"] >= args.max_source_rows:
                    logger.info("Reached --max-source-rows=%s; stopping source stream.", args.max_source_rows)
                    break
                if counters["success"] >= args.target_samples:
                    break

                meta = _normalize_meta(sample.get("json"))
                language = _extract_language(meta)
                if not _matches_language(language, language_re):
                    continue
                counters["language_rows_seen"] += 1

                source_key = str(sample.get("__key__") or f"idx-{source_index}")
                if not _belongs_to_shard(source_key, args.num_shards, args.shard_index):
                    continue

                while counters["success"] + len(futures) >= args.target_samples and futures:
                    _drain_one_completed(
                        futures=futures,
                        csv_writer=writer,
                        csv_fp=csv_fp,
                        counters=counters,
                        log_every=args.log_every,
                        started_at=started_at,
                    )
                    if counters["success"] >= args.target_samples:
                        break
                if counters["success"] >= args.target_samples:
                    break

                while len(futures) >= max_in_flight:
                    _drain_one_completed(
                        futures=futures,
                        csv_writer=writer,
                        csv_fp=csv_fp,
                        counters=counters,
                        log_every=args.log_every,
                        started_at=started_at,
                    )
                    if counters["success"] >= args.target_samples:
                        break
                if counters["success"] >= args.target_samples:
                    break

                future = pool.submit(
                    _process_one,
                    sample=sample,
                    source_index=source_index,
                    bucket=bucket,
                    prefix=args.prefix.strip("/"),
                    split=args.split,
                    target_sample_rate=sample_rate,
                    n_mels=int(args.n_mels),
                    win_length=win_length,
                    hop_length=hop_length,
                    n_fft=n_fft,
                    mel_filter=mel_filter,
                    upload_audio=bool(args.upload_audio),
                    dry_run=bool(args.dry_run),
                )
                futures.add(future)
                counters["submitted"] += 1

            while futures:
                _drain_one_completed(
                    futures=futures,
                    csv_writer=writer,
                    csv_fp=csv_fp,
                    counters=counters,
                    log_every=args.log_every,
                    started_at=started_at,
                )

        csv_fp.flush()

    elapsed = max(0.001, time.time() - started_at)
    summary = {
        "dataset": "amphion/Emilia-Dataset",
        "split": args.split,
        "target_samples": args.target_samples,
        "language_regex": args.language_regex,
        "num_success": counters["success"],
        "num_failed": counters["failed"],
        "num_submitted": counters["submitted"],
        "num_source_rows_seen": counters["source_rows_seen"],
        "num_language_rows_seen": counters["language_rows_seen"],
        "workers": args.workers,
        "max_in_flight": max_in_flight,
        "sample_rate_hz": sample_rate,
        "n_mels": int(args.n_mels),
        "win_ms": float(args.win_ms),
        "hop_ms": float(args.hop_ms),
        "fmin_hz": float(args.fmin),
        "fmax_hz": fmax,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "upload_audio": bool(args.upload_audio),
        "elapsed_s": round(elapsed, 3),
        "items_per_s": round(counters["success"] / elapsed, 4),
        "local_csv_path": local_csv_path,
        "local_csv_size_bytes": os.path.getsize(local_csv_path),
        "dry_run": bool(args.dry_run),
        "tools_data_sources": TOOL_DATASETS,
        "build_dataset_voice_count": len(BUILD_DATASET_VOICES),
        "completed_at_unix": int(time.time()),
    }

    if args.dry_run:
        logger.info("Dry run complete: %s", json.dumps(summary, indent=2))
        return

    if bucket is None:  # pragma: no cover - defensive guard
        raise RuntimeError("GCS bucket is unexpectedly unavailable.")

    csv_blob_path = f"{args.prefix.strip('/')}/emilia-large/{args.split}/{csv_name}"
    csv_uri = _upload_file(
        bucket=bucket,
        blob_path=csv_blob_path,
        filename=local_csv_path,
        content_type="text/csv",
    )
    summary["csv_gcs_uri"] = csv_uri

    summary_blob_path = (
        f"{args.prefix.strip('/')}/emilia-large/{args.split}/summary-shard-{args.shard_index:03d}.json"
    )
    summary_uri = _upload_string(
        bucket=bucket,
        blob_path=summary_blob_path,
        payload=json.dumps(summary, indent=2).encode("utf-8"),
        content_type="application/json",
    )
    summary["summary_gcs_uri"] = summary_uri

    logger.info(
        "Completed ingestion: success=%s failed=%s csv=%s summary=%s",
        counters["success"],
        counters["failed"],
        csv_uri,
        summary_uri,
    )


if __name__ == "__main__":
    main()
