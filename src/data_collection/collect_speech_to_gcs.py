#!/usr/bin/env python3
"""Stream speech datasets from Hugging Face and upload audio + metadata to GCS.

Supported datasets:
  - Emilia-Large repo: amphion/Emilia-Dataset
  - SPGISpeech: kensho/spgispeech
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import logging
import mimetypes
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Avoid hanging on shutdown due to hf-xet background transport in some environments.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

try:
    from datasets import Audio, load_dataset
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        f"Could not import 'datasets' with interpreter '{sys.executable}': {exc}\n"
        "Install into this interpreter with:\n"
        "  python -m pip install datasets"
    ) from exc

try:
    from google.cloud import storage
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        f"Could not import 'google-cloud-storage' with interpreter '{sys.executable}': {exc}\n"
        "Install into this interpreter with:\n"
        "  python -m pip install google-cloud-storage"
    ) from exc

try:
    import soundfile as sf
except ImportError:  # Optional fallback only when decoded arrays are encountered.
    sf = None

import numpy as np


LOGGER = logging.getLogger("collect_speech_to_gcs")

import csv


METADATA_CSV_FIELDS = [
    "dataset",
    "hf_dataset",
    "hf_config",
    "split",
    "item_id",
    "source_id",
    "source_url",
    "gcs_audio_uri",
    "gcs_mel_uri",
    "transcript",
    "language",
    "duration_s",
    "timestamp_start_s",
    "timestamp_end_s",
    "timestamp_segments_json",
    "audio_ext",
    "audio_content_type",
    "audio_num_bytes",
    "mel_shape",
    "mel_dtype",
    "mel_format",
    "mel_num_bytes",
    "gcs_metadata_csv_uri",
    "ingested_at_unix",
]


@dataclass(frozen=True)
class DatasetSplit:
    logical_name: str
    hf_dataset: str
    hf_config: str | None
    split: str
    streaming: bool


@dataclass(frozen=True)
class MelConfig:
    enabled: bool
    preset: str
    sample_rate: int
    n_mels: int
    n_fft: int
    win_length: int
    hop_length: int
    fmin: float
    fmax: float
    log_base: str
    whisper_clamp_and_scale: bool
    max_frames: int | None
    file_format: str
    dtype: str


class ManifestWriter:
    """Write sharded JSONL.gz manifests and upload each closed shard to GCS."""

    def __init__(
        self,
        bucket: storage.Bucket,
        bucket_prefix: str,
        dataset_name: str,
        split: str,
        shard_size: int,
    ) -> None:
        self.bucket = bucket
        self.bucket_prefix = bucket_prefix.strip("/")
        self.dataset_name = dataset_name
        self.split = split
        self.shard_size = max(1, shard_size)

        self._tmp_dir = tempfile.TemporaryDirectory(prefix="manifest_")
        self._file = None
        self._path = None
        self._rows_in_shard = 0
        self._shard_idx = 0
        self._manifest_uris: list[str] = []

    @property
    def manifest_uris(self) -> list[str]:
        return list(self._manifest_uris)

    def _open_next(self) -> None:
        filename = f"manifest-{self._shard_idx:06d}.jsonl.gz"
        path = os.path.join(self._tmp_dir.name, filename)
        self._path = path
        self._file = gzip.open(path, "wt", encoding="utf-8")
        self._rows_in_shard = 0

    def _upload_current(self) -> None:
        if self._file is None or self._path is None:
            return
        self._file.close()
        filename = os.path.basename(self._path)
        blob_path = (
            f"{self.bucket_prefix}/{self.dataset_name}/{self.split}/manifests/{filename}"
        )
        self.bucket.blob(blob_path).upload_from_filename(
            self._path, content_type="application/gzip"
        )
        self._manifest_uris.append(f"gs://{self.bucket.name}/{blob_path}")
        os.remove(self._path)
        self._file = None
        self._path = None
        self._shard_idx += 1

    def write(self, row: dict[str, Any]) -> None:
        if self._file is None:
            self._open_next()

        if self._rows_in_shard >= self.shard_size:
            self._upload_current()
            self._open_next()

        self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._rows_in_shard += 1

    def close(self) -> None:
        if self._file is not None:
            self._upload_current()
        self._tmp_dir.cleanup()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    if audio.ndim == 2:
        # Handle (channels, time) and (time, channels).
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            return np.mean(audio, axis=0).astype(np.float32)
        return np.mean(audio, axis=1).astype(np.float32)
    return audio.reshape(-1).astype(np.float32)


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    try:
        from scipy.signal import resample
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for resampling audio. Install with: python -m pip install scipy"
        ) from exc

    target_len = int(round(len(audio) * float(dst_sr) / float(src_sr)))
    if target_len <= 0:
        return np.zeros((1,), dtype=np.float32)
    return resample(audio, target_len).astype(np.float32)


def _decode_audio_for_mel(audio_field: Any, target_sr: int) -> tuple[np.ndarray, int]:
    if sf is None:
        raise RuntimeError("soundfile is required to decode audio for mel extraction.")

    if isinstance(audio_field, (bytes, bytearray, memoryview)):
        audio, sr = sf.read(io.BytesIO(bytes(audio_field)), dtype="float32")
        audio = _mono(np.asarray(audio))
        return _resample(audio, sr, target_sr), target_sr

    if isinstance(audio_field, str):
        audio, sr = sf.read(audio_field, dtype="float32")
        audio = _mono(np.asarray(audio))
        return _resample(audio, sr, target_sr), target_sr

    if isinstance(audio_field, dict):
        raw_bytes = audio_field.get("bytes")
        raw_path = audio_field.get("path")
        raw_array = audio_field.get("array")
        raw_sr = audio_field.get("sampling_rate")

        if raw_bytes is not None:
            audio, sr = sf.read(io.BytesIO(bytes(raw_bytes)), dtype="float32")
            audio = _mono(np.asarray(audio))
            return _resample(audio, sr, target_sr), target_sr

        if raw_path and os.path.exists(raw_path):
            audio, sr = sf.read(raw_path, dtype="float32")
            audio = _mono(np.asarray(audio))
            return _resample(audio, sr, target_sr), target_sr

        if raw_array is not None:
            arr = _mono(np.asarray(raw_array, dtype=np.float32))
            sr = int(raw_sr) if raw_sr else target_sr
            return _resample(arr, sr, target_sr), target_sr

    # datasets>=4 may provide torchcodec AudioDecoder when decode=True.
    if hasattr(audio_field, "get_all_samples"):
        samples = audio_field.get_all_samples()
        data = samples.data
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()
        elif hasattr(data, "numpy"):
            data = data.numpy()
        arr = _mono(np.asarray(data, dtype=np.float32))
        sr = int(getattr(samples, "sample_rate", target_sr))
        return _resample(arr, sr, target_sr), target_sr

    raise ValueError(f"Unsupported audio field for mel decode: {type(audio_field)}")


def _mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    fmax = min(float(fmax), sr / 2.0)
    fmin = max(0.0, float(fmin))

    mel_low = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_high = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(np.int32)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left = bins[i]
        center = bins[i + 1]
        right = bins[i + 2]

        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1

        for k in range(left, min(center, fb.shape[1])):
            fb[i, k] = (k - left) / max(center - left, 1)
        for k in range(center, min(right, fb.shape[1])):
            fb[i, k] = (right - k) / max(right - center, 1)
    return fb


def _compute_mel_spectrogram(audio: np.ndarray, cfg: MelConfig) -> np.ndarray:
    try:
        from scipy.fft import rfft
        from scipy.signal import windows
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for mel extraction. Install with: python -m pip install scipy"
        ) from exc

    if audio.size < cfg.win_length:
        pad = cfg.win_length - audio.size
        audio = np.pad(audio, (0, pad))

    window = windows.hann(cfg.win_length, sym=False).astype(np.float32)
    n_frames = 1 + max(0, (audio.size - cfg.win_length) // cfg.hop_length)
    if n_frames <= 0:
        return np.zeros((1, cfg.n_mels), dtype=np.float32)

    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, cfg.win_length),
        strides=(audio.strides[0] * cfg.hop_length, audio.strides[0]),
    ).copy()
    frames *= window[None, :]

    if cfg.n_fft > cfg.win_length:
        frames = np.pad(frames, ((0, 0), (0, cfg.n_fft - cfg.win_length)))
    elif cfg.n_fft < cfg.win_length:
        frames = frames[:, : cfg.n_fft]

    power = np.abs(rfft(frames, n=cfg.n_fft, axis=-1)) ** 2
    fb = _mel_filterbank(
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    mel = power @ fb.T
    mel = np.maximum(mel, 1e-10)

    if cfg.log_base == "log10":
        mel = np.log10(mel)
    else:
        mel = np.log(mel)

    if cfg.whisper_clamp_and_scale:
        mel = np.maximum(mel, mel.max() - 8.0)
        mel = (mel + 4.0) / 4.0

    if cfg.max_frames is not None and cfg.max_frames > 0:
        mel = mel[: cfg.max_frames]

    return mel.astype(np.float32)


def _upload_mel_array(
    bucket: storage.Bucket,
    blob_path: str,
    mel: np.ndarray,
    file_format: str,
    dtype: str,
) -> dict[str, Any]:
    arr = mel.astype(np.float16 if dtype == "float16" else np.float32)
    buff = io.BytesIO()

    if file_format == "npz":
        np.savez_compressed(buff, mel=arr)
        content_type = "application/x-npz"
    else:
        np.save(buff, arr)
        content_type = "application/octet-stream"

    payload = buff.getvalue()
    bucket.blob(blob_path).upload_from_string(payload, content_type=content_type)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "format": file_format,
        "num_bytes": len(payload),
    }


def _extract_timestamps(normalized: dict[str, Any]) -> dict[str, Any]:
    meta = normalized.get("metadata")
    if not isinstance(meta, dict):
        meta = {}

    raw = normalized.get("raw_fields")
    if not isinstance(raw, dict):
        raw = {}

    start = _safe_float(
        _first_non_empty(meta, ("start", "start_s", "start_time", "offset"))
    )
    if start is None:
        start = _safe_float(
            _first_non_empty(raw, ("start", "start_s", "start_time", "offset"))
        )

    end = _safe_float(_first_non_empty(meta, ("end", "end_s", "end_time", "stop")))
    if end is None:
        end = _safe_float(_first_non_empty(raw, ("end", "end_s", "end_time", "stop")))

    duration = normalized.get("duration_s")
    if start is None and duration is not None:
        start = 0.0
    if end is None and start is not None and duration is not None:
        end = float(start) + float(duration)

    segments = _first_non_empty(
        meta, ("segments", "timestamps", "word_timestamps", "words")
    )
    if segments is None:
        segments = _first_non_empty(
            raw, ("segments", "timestamps", "word_timestamps", "words")
        )

    if segments is None and start is not None and end is not None:
        segments = _fallback_transcript_segments(
            transcript=normalized.get("transcript"),
            start_s=float(start),
            end_s=float(end),
        )

    return {
        "start_s": start,
        "end_s": end,
        "segments": _json_safe(segments),
    }


def _fallback_transcript_segments(
    transcript: str | None, start_s: float, end_s: float
) -> list[dict[str, Any]]:
    """Create approximate segment timestamps from transcript text.

    This is only used when source datasets don't provide segment-level timing.
    """
    text = (transcript or "").strip()
    if not text:
        return [{"start_s": start_s, "end_s": end_s, "text": None}]

    # Split by punctuation first.
    chunks = [c.strip() for c in re.split(r"(?<=[\.\!\?;,:])\s+", text) if c.strip()]
    if not chunks:
        chunks = [text]

    # Further split very long chunks into smaller word groups.
    refined: list[str] = []
    for chunk in chunks:
        words = chunk.split()
        if len(words) > 18:
            step = 8
            for i in range(0, len(words), step):
                piece = " ".join(words[i : i + step]).strip()
                if piece:
                    refined.append(piece)
        else:
            refined.append(chunk)
    chunks = refined or [text]

    total_dur = max(0.0, float(end_s) - float(start_s))
    if total_dur <= 0 or len(chunks) == 1:
        return [{"start_s": start_s, "end_s": end_s, "text": " ".join(chunks)}]

    weights = [max(1, len(c)) for c in chunks]
    total_w = float(sum(weights))
    current = float(start_s)
    out: list[dict[str, Any]] = []

    for i, (chunk, w) in enumerate(zip(chunks, weights)):
        if i == len(chunks) - 1:
            nxt = float(end_s)
        else:
            nxt = current + total_dur * (float(w) / total_w)
        out.append(
            {
                "start_s": round(current, 3),
                "end_s": round(nxt, 3),
                "text": chunk,
            }
        )
        current = nxt

    out[-1]["end_s"] = round(float(end_s), 3)
    return out


def _record_to_csv_row(record: dict[str, Any]) -> dict[str, Any]:
    ts = record.get("timestamps") or {}
    audio = record.get("audio") or {}
    mel = record.get("mel") or {}
    return {
        "dataset": record.get("dataset"),
        "hf_dataset": record.get("hf_dataset"),
        "hf_config": record.get("hf_config"),
        "split": record.get("split"),
        "item_id": record.get("item_id"),
        "source_id": record.get("source_id"),
        "source_url": record.get("source_url"),
        "gcs_audio_uri": record.get("gcs_audio_uri"),
        "gcs_mel_uri": record.get("gcs_mel_uri"),
        "transcript": record.get("transcript"),
        "language": record.get("language"),
        "duration_s": record.get("duration_s"),
        "timestamp_start_s": ts.get("start_s"),
        "timestamp_end_s": ts.get("end_s"),
        "timestamp_segments_json": (
            json.dumps(ts.get("segments"), ensure_ascii=False)
            if ts.get("segments") is not None
            else None
        ),
        "audio_ext": audio.get("ext"),
        "audio_content_type": audio.get("content_type"),
        "audio_num_bytes": audio.get("num_bytes"),
        "mel_shape": (
            "x".join(str(x) for x in mel.get("shape"))
            if isinstance(mel.get("shape"), list)
            else None
        ),
        "mel_dtype": mel.get("dtype"),
        "mel_format": mel.get("format"),
        "mel_num_bytes": mel.get("num_bytes"),
        "gcs_metadata_csv_uri": record.get("gcs_metadata_csv_uri"),
        "ingested_at_unix": record.get("ingested_at_unix"),
    }


def _csv_row_to_text(row: dict[str, Any]) -> str:
    buff = io.StringIO()
    writer = csv.DictWriter(buff, fieldnames=METADATA_CSV_FIELDS)
    writer.writeheader()
    writer.writerow(row)
    return buff.getvalue()


def _upload_metadata_csv(
    bucket: storage.Bucket,
    blob_path: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    text = _csv_row_to_text(row)
    payload = text.encode("utf-8")
    bucket.blob(blob_path).upload_from_string(payload, content_type="text/csv")
    return {
        "uri": f"gs://{bucket.name}/{blob_path}",
        "num_bytes": len(payload),
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"_type": "bytes", "num_bytes": len(value)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        return {"_type": "array", "shape": list(shape), "dtype": str(dtype)}
    return str(value)


def _first_non_empty(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = d.get(key)
        if value not in (None, "", []):
            return value
    return None


def _parse_emilia_json(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, bytes):
        text = payload.decode("utf-8", errors="replace")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw_json": text}
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {"raw_json": payload}
    return {"raw_json": str(payload)}


def _normalize_sample(dataset_name: str, sample: dict[str, Any]) -> dict[str, Any]:
    if dataset_name == "emilia-large":
        parsed_meta = _parse_emilia_json(sample.get("json"))
        transcript = _first_non_empty(
            parsed_meta,
            ("text", "transcript", "normalized_text", "sentence", "content"),
        )
        language = _first_non_empty(parsed_meta, ("language", "lang", "language_code"))
        duration = _safe_float(
            _first_non_empty(
                parsed_meta,
                ("duration", "duration_s", "duration_sec", "audio_duration"),
            )
        )
        source_id = sample.get("__key__")
        audio_field = sample.get("mp3") or sample.get("audio")
        source_url = sample.get("__url__")
        raw_fields = {
            k: _json_safe(v)
            for k, v in sample.items()
            if k not in ("mp3", "audio", "json")
        }
        return {
            "source_id": source_id,
            "transcript": transcript,
            "language": language,
            "duration_s": duration,
            "audio_field": audio_field,
            "metadata": _json_safe(parsed_meta),
            "raw_fields": raw_fields,
            "source_url": source_url,
            "preferred_ext": ".mp3",
        }

    if dataset_name == "spgi-speech":
        transcript = sample.get("transcript") or sample.get("text")
        source_id = sample.get("wav_filename") or sample.get("id")
        duration = _safe_float(sample.get("duration"))
        audio_field = sample.get("audio")
        raw_fields = {
            k: _json_safe(v)
            for k, v in sample.items()
            if k not in ("audio", "array")
        }
        return {
            "source_id": source_id,
            "transcript": transcript,
            "language": "en",
            "duration_s": duration,
            "audio_field": audio_field,
            "metadata": {"wav_filesize": sample.get("wav_filesize")},
            "raw_fields": raw_fields,
            "source_url": None,
            "preferred_ext": ".wav",
        }

    raise ValueError(f"Unsupported dataset_name={dataset_name}")


def _normalize_audio_ext(path_or_name: str | None, default_ext: str) -> str:
    if not path_or_name:
        return default_ext
    ext = Path(path_or_name).suffix.lower()
    if not ext:
        return default_ext
    if len(ext) > 8:
        return default_ext
    return ext


def _sanitize_id(value: str | None, fallback_seed: str) -> str:
    if value is None or value == "":
        return hashlib.sha1(fallback_seed.encode("utf-8")).hexdigest()[:20]
    cleaned = re.sub(r"[^a-zA-Z0-9._/-]+", "_", str(value))
    cleaned = cleaned.strip("._/")
    if not cleaned:
        return hashlib.sha1(fallback_seed.encode("utf-8")).hexdigest()[:20]
    return cleaned.replace("/", "_")


def _audio_field_ext_hint(audio_field: Any, default_ext: str) -> str:
    if isinstance(audio_field, str):
        return _normalize_audio_ext(audio_field, default_ext)
    if isinstance(audio_field, dict):
        return _normalize_audio_ext(audio_field.get("path"), default_ext)
    return default_ext


def _upload_blob_from_audio_field(
    bucket: storage.Bucket,
    blob_path: str,
    audio_field: Any,
    preferred_ext: str,
) -> dict[str, Any]:
    """Upload audio data from common HuggingFace audio representations."""

    blob = bucket.blob(blob_path)

    if isinstance(audio_field, (bytes, bytearray, memoryview)):
        payload = bytes(audio_field)
        ext = preferred_ext
        content_type = mimetypes.guess_type(f"file{ext}")[0] or "application/octet-stream"
        blob.upload_from_string(payload, content_type=content_type)
        return {"ext": ext, "content_type": content_type, "num_bytes": len(payload)}

    if isinstance(audio_field, str):
        ext = _normalize_audio_ext(audio_field, preferred_ext)
        content_type = mimetypes.guess_type(audio_field)[0] or "application/octet-stream"
        if not os.path.exists(audio_field):
            raise FileNotFoundError(f"Audio file does not exist: {audio_field}")
        blob.upload_from_filename(audio_field, content_type=content_type)
        return {
            "ext": ext,
            "content_type": content_type,
            "num_bytes": os.path.getsize(audio_field),
        }

    if isinstance(audio_field, dict):
        bytes_payload = audio_field.get("bytes")
        path = audio_field.get("path")
        array = audio_field.get("array")
        sampling_rate = audio_field.get("sampling_rate")

        if bytes_payload is not None:
            payload = bytes(bytes_payload)
            ext = _normalize_audio_ext(path, preferred_ext)
            content_type = (
                mimetypes.guess_type(path or f"file{ext}")[0]
                or "application/octet-stream"
            )
            blob.upload_from_string(payload, content_type=content_type)
            return {
                "ext": ext,
                "content_type": content_type,
                "num_bytes": len(payload),
                "sampling_rate": sampling_rate,
            }

        if path and os.path.exists(path):
            ext = _normalize_audio_ext(path, preferred_ext)
            content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            blob.upload_from_filename(path, content_type=content_type)
            return {
                "ext": ext,
                "content_type": content_type,
                "num_bytes": os.path.getsize(path),
                "sampling_rate": sampling_rate,
            }

        if array is not None:
            if sf is None:
                raise RuntimeError(
                    "Encountered decoded audio arrays but 'soundfile' is not installed."
                )
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                sf.write(tmp_path, array, sampling_rate or 16000)
                blob.upload_from_filename(tmp_path, content_type="audio/wav")
                return {
                    "ext": ".wav",
                    "content_type": "audio/wav",
                    "num_bytes": os.path.getsize(tmp_path),
                    "sampling_rate": sampling_rate,
                }
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    raise ValueError(f"Unsupported audio field type: {type(audio_field)}")


def _upload_json(bucket: storage.Bucket, blob_path: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    bucket.blob(blob_path).upload_from_string(data, content_type="application/json")
    return f"gs://{bucket.name}/{blob_path}"


def _load_split(dataset_split: DatasetSplit, hf_token: str | None):
    kwargs: dict[str, Any] = {
        "path": dataset_split.hf_dataset,
        "split": dataset_split.split,
        "streaming": dataset_split.streaming,
    }
    if dataset_split.hf_config:
        kwargs["name"] = dataset_split.hf_config
    if hf_token:
        kwargs["token"] = hf_token

    ds = load_dataset(**kwargs)

    # For datasets with an Audio feature, keep decode disabled to avoid large in-memory arrays.
    if dataset_split.logical_name == "emilia-large":
        try:
            ds = ds.cast_column("mp3", Audio(decode=False))
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.warning("Could not cast Emilia mp3 column with decode=False: %s", exc)

    if dataset_split.logical_name == "spgi-speech":
        try:
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.warning("Could not cast SPGI audio column with decode=False: %s", exc)

    return ds


def _iter_dataset_to_gcs(
    bucket: storage.Bucket,
    dataset_split: DatasetSplit,
    prefix: str,
    max_samples: int | None,
    manifest_shard_size: int,
    log_every: int,
    hf_token: str | None,
    mel_cfg: MelConfig,
) -> dict[str, Any]:
    start = time.time()
    ds = _load_split(dataset_split, hf_token=hf_token)

    writer = ManifestWriter(
        bucket=bucket,
        bucket_prefix=prefix,
        dataset_name=dataset_split.logical_name,
        split=dataset_split.split,
        shard_size=manifest_shard_size,
    )

    uploaded = 0
    failed = 0
    bytes_uploaded = 0
    durations_found = 0
    duration_total = 0.0
    mel_uploaded = 0
    mel_failed = 0
    mel_bytes = 0
    metadata_csv_uploaded = 0
    metadata_csv_failed = 0
    metadata_csv_bytes = 0

    metadata_csv_prefix = (
        f"gs://{bucket.name}/{prefix}/{dataset_split.logical_name}/{dataset_split.split}/metadata_csv/"
    )

    try:
        for idx, sample in enumerate(ds):
            if max_samples is not None and idx >= max_samples:
                break

            normalized = _normalize_sample(dataset_split.logical_name, sample)
            fallback_seed = (
                f"{dataset_split.logical_name}:{dataset_split.split}:{idx}:"
                f"{normalized.get('transcript') or ''}"
            )
            source_id = _sanitize_id(normalized["source_id"], fallback_seed=fallback_seed)
            item_id = f"{source_id}-{idx:09d}"
            audio_ext = _audio_field_ext_hint(
                normalized["audio_field"], normalized["preferred_ext"]
            )
            audio_blob_path = (
                f"{prefix}/{dataset_split.logical_name}/{dataset_split.split}/audio/{item_id}{audio_ext}"
            )

            try:
                audio_info = _upload_blob_from_audio_field(
                    bucket=bucket,
                    blob_path=audio_blob_path,
                    audio_field=normalized["audio_field"],
                    preferred_ext=audio_ext,
                )
            except Exception as exc:
                failed += 1
                LOGGER.warning(
                    "Skipping sample idx=%s (%s/%s): %s",
                    idx,
                    dataset_split.logical_name,
                    dataset_split.split,
                    exc,
                )
                continue

            bytes_uploaded += int(audio_info.get("num_bytes", 0))
            duration_s = normalized.get("duration_s")
            decoded_audio = None
            decoded_sr = None
            if duration_s is None or mel_cfg.enabled:
                try:
                    decoded_audio, decoded_sr = _decode_audio_for_mel(
                        normalized["audio_field"], mel_cfg.sample_rate
                    )
                except Exception as exc:
                    if duration_s is None:
                        LOGGER.warning(
                            "Failed audio decode for duration idx=%s (%s/%s): %s",
                            idx,
                            dataset_split.logical_name,
                            dataset_split.split,
                            exc,
                        )

            if duration_s is None and decoded_audio is not None and decoded_sr:
                duration_s = float(len(decoded_audio)) / float(decoded_sr)
                normalized["duration_s"] = duration_s

            if duration_s is not None:
                durations_found += 1
                duration_total += float(duration_s)

            timestamps = _extract_timestamps(normalized)
            mel_uri = None
            mel_info = None
            if mel_cfg.enabled:
                try:
                    if decoded_audio is None:
                        decoded_audio, _ = _decode_audio_for_mel(
                            normalized["audio_field"], mel_cfg.sample_rate
                        )
                    mel = _compute_mel_spectrogram(decoded_audio, mel_cfg)
                    mel_ext = "npz" if mel_cfg.file_format == "npz" else "npy"
                    mel_blob_path = (
                        f"{prefix}/{dataset_split.logical_name}/{dataset_split.split}/mel/{item_id}.{mel_ext}"
                    )
                    mel_info = _upload_mel_array(
                        bucket=bucket,
                        blob_path=mel_blob_path,
                        mel=mel,
                        file_format=mel_cfg.file_format,
                        dtype=mel_cfg.dtype,
                    )
                    mel_uri = f"gs://{bucket.name}/{mel_blob_path}"
                    mel_uploaded += 1
                    mel_bytes += int(mel_info.get("num_bytes", 0))
                except Exception as exc:
                    mel_failed += 1
                    LOGGER.warning(
                        "Failed mel extraction idx=%s (%s/%s): %s",
                        idx,
                        dataset_split.logical_name,
                        dataset_split.split,
                        exc,
                    )

            record = {
                "dataset": dataset_split.logical_name,
                "hf_dataset": dataset_split.hf_dataset,
                "hf_config": dataset_split.hf_config,
                "split": dataset_split.split,
                "item_id": item_id,
                "source_id": source_id,
                "gcs_audio_uri": f"gs://{bucket.name}/{audio_blob_path}",
                "transcript": normalized.get("transcript"),
                "language": normalized.get("language"),
                "duration_s": duration_s,
                "timestamps": timestamps,
                "audio": audio_info,
                "gcs_mel_uri": mel_uri,
                "mel": mel_info,
                "source_url": normalized.get("source_url"),
                "metadata": normalized.get("metadata"),
                "raw_fields": normalized.get("raw_fields"),
                "gcs_metadata_csv_uri": None,
                "ingested_at_unix": int(time.time()),
            }

            csv_blob_path = (
                f"{prefix}/{dataset_split.logical_name}/{dataset_split.split}/metadata_csv/{item_id}.csv"
            )
            csv_uri = f"gs://{bucket.name}/{csv_blob_path}"
            record["gcs_metadata_csv_uri"] = csv_uri
            csv_row = _record_to_csv_row(record)
            try:
                csv_upload = _upload_metadata_csv(
                    bucket=bucket,
                    blob_path=csv_blob_path,
                    row=csv_row,
                )
                metadata_csv_uploaded += 1
                metadata_csv_bytes += int(csv_upload["num_bytes"])
            except Exception as exc:
                metadata_csv_failed += 1
                record["gcs_metadata_csv_uri"] = None
                LOGGER.warning(
                    "Failed metadata CSV upload idx=%s (%s/%s): %s",
                    idx,
                    dataset_split.logical_name,
                    dataset_split.split,
                    exc,
                )

            writer.write(record)
            uploaded += 1

            if uploaded > 0 and uploaded % log_every == 0:
                LOGGER.info(
                    "Uploaded %s samples for %s/%s",
                    uploaded,
                    dataset_split.logical_name,
                    dataset_split.split,
                )
    finally:
        writer.close()

    elapsed_s = max(0.001, time.time() - start)
    summary = {
        "dataset": dataset_split.logical_name,
        "hf_dataset": dataset_split.hf_dataset,
        "hf_config": dataset_split.hf_config,
        "split": dataset_split.split,
        "streaming": dataset_split.streaming,
        "num_uploaded": uploaded,
        "num_failed": failed,
        "bytes_uploaded": bytes_uploaded,
        "bytes_uploaded_gb": round(bytes_uploaded / (1024 ** 3), 4),
        "duration_s_total": round(duration_total, 3),
        "duration_h_total": round(duration_total / 3600.0, 3),
        "duration_s_counted_items": durations_found,
        "mel_enabled": mel_cfg.enabled,
        "mel_preset": mel_cfg.preset,
        "mel_bins": mel_cfg.n_mels,
        "mel_uploaded": mel_uploaded,
        "mel_failed": mel_failed,
        "mel_bytes": mel_bytes,
        "mel_bytes_gb": round(mel_bytes / (1024 ** 3), 4),
        "metadata_csv_prefix": metadata_csv_prefix,
        "metadata_csv_uploaded": metadata_csv_uploaded,
        "metadata_csv_failed": metadata_csv_failed,
        "metadata_csv_bytes": metadata_csv_bytes,
        "metadata_csv_bytes_gb": round(metadata_csv_bytes / (1024 ** 3), 4),
        "elapsed_s": round(elapsed_s, 3),
        "items_per_s": round(uploaded / elapsed_s, 3),
        "manifest_uris": writer.manifest_uris,
        "bucket": bucket.name,
        "prefix": prefix,
    }

    summary_blob = (
        f"{prefix}/{dataset_split.logical_name}/{dataset_split.split}/summary.json"
    )
    summary_uri = _upload_json(bucket, summary_blob, summary)
    summary["summary_uri"] = summary_uri
    return summary


def _build_work_items(args: argparse.Namespace) -> list[DatasetSplit]:
    items: list[DatasetSplit] = []
    selected = set(args.datasets)

    if "emilia-large" in selected:
        for split in args.emilia_splits:
            items.append(
                DatasetSplit(
                    logical_name="emilia-large",
                    hf_dataset="amphion/Emilia-Dataset",
                    hf_config=None,
                    split=split,
                    streaming=args.streaming,
                )
            )

    if "spgi-speech" in selected:
        for split in args.spgi_splits:
            items.append(
                DatasetSplit(
                    logical_name="spgi-speech",
                    hf_dataset="kensho/spgispeech",
                    hf_config=args.spgi_config,
                    split=split,
                    streaming=args.streaming,
                )
            )

    return items


def _resolve_mel_config(args: argparse.Namespace) -> MelConfig:
    if not args.store_mel:
        return MelConfig(
            enabled=False,
            preset=args.mel_preset,
            sample_rate=16000,
            n_mels=args.mel_n_mels or 80,
            n_fft=400,
            win_length=400,
            hop_length=160,
            fmin=0.0,
            fmax=8000.0,
            log_base="log10",
            whisper_clamp_and_scale=False,
            max_frames=args.mel_max_frames,
            file_format=args.mel_file_format,
            dtype=args.mel_dtype,
        )

    if args.mel_preset == "whisper":
        cfg = MelConfig(
            enabled=True,
            preset="whisper",
            sample_rate=16000,
            n_mels=args.mel_n_mels or 80,
            n_fft=400,
            win_length=400,
            hop_length=160,
            fmin=0.0,
            fmax=8000.0,
            log_base="log10",
            whisper_clamp_and_scale=True,
            max_frames=args.mel_max_frames,
            file_format=args.mel_file_format,
            dtype=args.mel_dtype,
        )
        return cfg

    # "parakeet" style defaults (NeMo-friendly log-mel setup).
    return MelConfig(
        enabled=True,
        preset="parakeet",
        sample_rate=16000,
        n_mels=args.mel_n_mels or 80,
        n_fft=512,
        win_length=400,
        hop_length=160,
        fmin=0.0,
        fmax=8000.0,
        log_base="ln",
        whisper_clamp_and_scale=False,
        max_frames=args.mel_max_frames,
        file_format=args.mel_file_format,
        dtype=args.mel_dtype,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect speech datasets from HF into GCS with manifests."
    )
    parser.add_argument("--bucket", required=True, help="Destination GCS bucket name")
    parser.add_argument(
        "--project",
        default=None,
        help="Optional GCP project (uses ADC default if omitted)",
    )
    parser.add_argument(
        "--prefix",
        default="speech_datasets",
        help="GCS key prefix under the bucket",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["emilia-large", "spgi-speech"],
        choices=["emilia-large", "spgi-speech"],
        help="Datasets to ingest",
    )
    parser.add_argument(
        "--emilia-splits",
        nargs="+",
        default=["train"],
        help="Emilia split names (default: train)",
    )
    parser.add_argument(
        "--spgi-config",
        default="L",
        help="SPGISpeech config name (S, M, L, dev, test; default: L)",
    )
    parser.add_argument(
        "--spgi-splits",
        nargs="+",
        default=["train"],
        help="SPGISpeech split names for the chosen config (default: train)",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use HF streaming mode (default: true)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per split for smoke tests",
    )
    parser.add_argument(
        "--manifest-shard-size",
        type=int,
        default=100000,
        help="Rows per manifest shard before rotating",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Progress logging interval in uploaded samples",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--store-mel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute and store per-sample mel spectrograms (default: true)",
    )
    parser.add_argument(
        "--mel-preset",
        default="whisper",
        choices=["whisper", "parakeet"],
        help="Mel feature preset",
    )
    parser.add_argument(
        "--mel-n-mels",
        type=int,
        default=None,
        help="Override mel bins for selected preset (default: preset value)",
    )
    parser.add_argument(
        "--mel-max-frames",
        type=int,
        default=None,
        help="Optional max mel frames to keep per sample",
    )
    parser.add_argument(
        "--mel-file-format",
        default="npz",
        choices=["npz", "npy"],
        help="Storage format for mel arrays",
    )
    parser.add_argument(
        "--mel-dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Stored dtype for mel arrays",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    client = storage.Client(project=args.project) if args.project else storage.Client()
    bucket = client.bucket(args.bucket)
    mel_cfg = _resolve_mel_config(args)

    work_items = _build_work_items(args)
    if not work_items:
        raise SystemExit("No dataset work items selected.")

    LOGGER.info(
        "Starting ingestion for %s work item(s) to gs://%s/%s",
        len(work_items),
        args.bucket,
        args.prefix.strip("/"),
    )
    LOGGER.info(
        "Mel config: enabled=%s preset=%s n_mels=%s sample_rate=%s n_fft=%s hop=%s win=%s format=%s dtype=%s",
        mel_cfg.enabled,
        mel_cfg.preset,
        mel_cfg.n_mels,
        mel_cfg.sample_rate,
        mel_cfg.n_fft,
        mel_cfg.hop_length,
        mel_cfg.win_length,
        mel_cfg.file_format,
        mel_cfg.dtype,
    )

    summaries = []
    for item in work_items:
        LOGGER.info(
            "Ingesting dataset=%s hf=%s config=%s split=%s",
            item.logical_name,
            item.hf_dataset,
            item.hf_config,
            item.split,
        )
        summary = _iter_dataset_to_gcs(
            bucket=bucket,
            dataset_split=item,
            prefix=args.prefix.strip("/"),
            max_samples=args.max_samples,
            manifest_shard_size=args.manifest_shard_size,
            log_every=max(1, args.log_every),
            hf_token=args.hf_token,
            mel_cfg=mel_cfg,
        )
        summaries.append(summary)
        LOGGER.info(
            "Finished %s/%s: uploaded=%s failed=%s hours=%.2f summary=%s",
            item.logical_name,
            item.split,
            summary["num_uploaded"],
            summary["num_failed"],
            summary["duration_h_total"],
            summary["summary_uri"],
        )

    run_summary = {
        "bucket": args.bucket,
        "prefix": args.prefix.strip("/"),
        "completed_items": len(summaries),
        "mel_config": {
            "enabled": mel_cfg.enabled,
            "preset": mel_cfg.preset,
            "sample_rate": mel_cfg.sample_rate,
            "n_mels": mel_cfg.n_mels,
            "n_fft": mel_cfg.n_fft,
            "win_length": mel_cfg.win_length,
            "hop_length": mel_cfg.hop_length,
            "fmin": mel_cfg.fmin,
            "fmax": mel_cfg.fmax,
            "log_base": mel_cfg.log_base,
            "whisper_clamp_and_scale": mel_cfg.whisper_clamp_and_scale,
            "max_frames": mel_cfg.max_frames,
            "file_format": mel_cfg.file_format,
            "dtype": mel_cfg.dtype,
        },
        "datasets": summaries,
        "completed_at_unix": int(time.time()),
    }
    run_blob = f"{args.prefix.strip('/')}/run-summary-{int(time.time())}.json"
    run_uri = _upload_json(bucket, run_blob, run_summary)
    LOGGER.info("Run complete. Summary: %s", run_uri)


if __name__ == "__main__":
    main()
