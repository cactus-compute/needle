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
import json
import logging
import mimetypes
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
except ImportError:  # Optional fallback only when decoded arrays are encountered.
    sf = None


LOGGER = logging.getLogger("collect_speech_to_gcs")


@dataclass(frozen=True)
class DatasetSplit:
    logical_name: str
    hf_dataset: str
    hf_config: str | None
    split: str
    streaming: bool


class ManifestWriter:
    """Write sharded JSONL.gz manifests and upload each closed shard to GCS."""

    def __init__(
        self,
        bucket: storage.Bucket,
        bucket_prefix: str,
        dataset_name: str,
        split: str,
        shard_size: int,
        shard_namespace: str | None = None,
    ) -> None:
        self.bucket = bucket
        self.bucket_prefix = bucket_prefix.strip("/")
        self.dataset_name = dataset_name
        self.split = split
        self.shard_size = max(1, shard_size)
        self.shard_namespace = shard_namespace.strip("/") if shard_namespace else None

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
        base_path = f"{self.bucket_prefix}/{self.dataset_name}/{self.split}/manifests"
        if self.shard_namespace:
            blob_path = f"{base_path}/{self.shard_namespace}/{filename}"
        else:
            blob_path = f"{base_path}/{filename}"
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
    if dataset_split.logical_name == "spgi-speech":
        try:
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.warning("Could not cast SPGI audio column with decode=False: %s", exc)

    return ds


def _shard_namespace(shard_index: int, shard_count: int) -> str:
    return f"shard-{shard_index:05d}-of-{shard_count:05d}"


def _maybe_apply_worker_shard(
    ds: Any,
    dataset_split: DatasetSplit,
    shard_count: int,
    shard_index: int,
    shard_contiguous: bool,
) -> tuple[Any | None, int]:
    """Apply deterministic worker sharding to a dataset split."""
    if shard_count <= 1:
        return ds, 1

    dataset_shards = getattr(ds, "num_shards", None)
    if not isinstance(dataset_shards, int) or dataset_shards < 1:
        if not hasattr(ds, "shard"):
            raise ValueError(
                f"Cannot shard dataset {dataset_split.logical_name}/{dataset_split.split}: "
                "dataset does not expose shard() or num_shards."
            )
        return (
            ds.shard(
                num_shards=shard_count,
                index=shard_index,
                contiguous=shard_contiguous,
            ),
            shard_count,
        )

    effective_count = min(int(shard_count), int(dataset_shards))
    if effective_count != shard_count:
        LOGGER.warning(
            "Requested --shard-count=%s for %s/%s but source has %s shards; clamping to %s.",
            shard_count,
            dataset_split.logical_name,
            dataset_split.split,
            dataset_shards,
            effective_count,
        )

    if shard_index >= effective_count:
        LOGGER.info(
            "No work for worker shard-index=%s on %s/%s (effective shard count=%s).",
            shard_index,
            dataset_split.logical_name,
            dataset_split.split,
            effective_count,
        )
        return None, effective_count

    return (
        ds.shard(
            num_shards=effective_count,
            index=shard_index,
            contiguous=shard_contiguous,
        ),
        effective_count,
    )


def _iter_dataset_to_gcs(
    bucket: storage.Bucket,
    dataset_split: DatasetSplit,
    prefix: str,
    max_samples: int | None,
    manifest_shard_size: int,
    log_every: int,
    hf_token: str | None,
    worker_shard_count: int,
    worker_shard_index: int,
    worker_shard_contiguous: bool,
) -> dict[str, Any]:
    start = time.time()
    ds = _load_split(dataset_split, hf_token=hf_token)
    ds, effective_shard_count = _maybe_apply_worker_shard(
        ds=ds,
        dataset_split=dataset_split,
        shard_count=worker_shard_count,
        shard_index=worker_shard_index,
        shard_contiguous=worker_shard_contiguous,
    )
    shard_namespace = (
        _shard_namespace(worker_shard_index, effective_shard_count)
        if effective_shard_count > 1
        else None
    )

    if ds is None:
        elapsed_s = max(0.001, time.time() - start)
        summary = {
            "dataset": dataset_split.logical_name,
            "hf_dataset": dataset_split.hf_dataset,
            "hf_config": dataset_split.hf_config,
            "split": dataset_split.split,
            "streaming": dataset_split.streaming,
            "worker_shard_count_requested": worker_shard_count,
            "worker_shard_count_effective": effective_shard_count,
            "worker_shard_index": worker_shard_index,
            "worker_shard_contiguous": worker_shard_contiguous,
            "worker_shard_namespace": shard_namespace,
            "num_uploaded": 0,
            "num_failed": 0,
            "bytes_uploaded": 0,
            "bytes_uploaded_gb": 0.0,
            "duration_s_total": 0.0,
            "duration_h_total": 0.0,
            "duration_s_counted_items": 0,
            "elapsed_s": round(elapsed_s, 3),
            "items_per_s": 0.0,
            "manifest_uris": [],
            "bucket": bucket.name,
            "prefix": prefix,
            "worker_has_data": False,
        }
        if shard_namespace:
            summary_blob = (
                f"{prefix}/{dataset_split.logical_name}/{dataset_split.split}/summary/{shard_namespace}.json"
            )
        else:
            summary_blob = (
                f"{prefix}/{dataset_split.logical_name}/{dataset_split.split}/summary.json"
            )
        summary_uri = _upload_json(bucket, summary_blob, summary)
        summary["summary_uri"] = summary_uri
        return summary

    writer = ManifestWriter(
        bucket=bucket,
        bucket_prefix=prefix,
        dataset_name=dataset_split.logical_name,
        split=dataset_split.split,
        shard_size=manifest_shard_size,
        shard_namespace=shard_namespace,
    )

    uploaded = 0
    failed = 0
    bytes_uploaded = 0
    durations_found = 0
    duration_total = 0.0

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
            if effective_shard_count > 1:
                item_id = f"{source_id}-w{worker_shard_index:05d}-{idx:09d}"
            else:
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
            if duration_s is not None:
                durations_found += 1
                duration_total += float(duration_s)

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
                "audio": audio_info,
                "source_url": normalized.get("source_url"),
                "metadata": normalized.get("metadata"),
                "raw_fields": normalized.get("raw_fields"),
                "ingested_at_unix": int(time.time()),
            }
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
        "worker_shard_count_requested": worker_shard_count,
        "worker_shard_count_effective": effective_shard_count,
        "worker_shard_index": worker_shard_index,
        "worker_shard_contiguous": worker_shard_contiguous,
        "worker_shard_namespace": shard_namespace,
        "num_uploaded": uploaded,
        "num_failed": failed,
        "bytes_uploaded": bytes_uploaded,
        "bytes_uploaded_gb": round(bytes_uploaded / (1024 ** 3), 4),
        "duration_s_total": round(duration_total, 3),
        "duration_h_total": round(duration_total / 3600.0, 3),
        "duration_s_counted_items": durations_found,
        "elapsed_s": round(elapsed_s, 3),
        "items_per_s": round(uploaded / elapsed_s, 3),
        "manifest_uris": writer.manifest_uris,
        "bucket": bucket.name,
        "prefix": prefix,
        "worker_has_data": True,
    }

    if shard_namespace:
        summary_blob = (
            f"{prefix}/{dataset_split.logical_name}/{dataset_split.split}/summary/{shard_namespace}.json"
        )
    else:
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
        "--shard-count",
        type=int,
        default=1,
        help=(
            "Total worker count for split-level sharding. "
            "Each worker must use the same shard-count."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based worker index in [0, shard-count).",
    )
    parser.add_argument(
        "--shard-contiguous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use contiguous shard assignment (default: true). "
            "Set --no-shard-contiguous for round-robin shard assignment."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    if args.shard_index < 0:
        raise SystemExit("--shard-index must be >= 0")
    if args.shard_index >= args.shard_count:
        raise SystemExit("--shard-index must be < --shard-count")

    client = storage.Client(project=args.project) if args.project else storage.Client()
    bucket = client.bucket(args.bucket)

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
        "Worker shard config: shard_count=%s shard_index=%s contiguous=%s",
        args.shard_count,
        args.shard_index,
        args.shard_contiguous,
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
            worker_shard_count=args.shard_count,
            worker_shard_index=args.shard_index,
            worker_shard_contiguous=args.shard_contiguous,
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
        "worker_shard_config": {
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "shard_contiguous": args.shard_contiguous,
        },
        "datasets": summaries,
        "completed_at_unix": int(time.time()),
    }
    if args.shard_count > 1:
        run_blob = (
            f"{args.prefix.strip('/')}/run-summary-{int(time.time())}-"
            f"{_shard_namespace(args.shard_index, args.shard_count)}.json"
        )
    else:
        run_blob = f"{args.prefix.strip('/')}/run-summary-{int(time.time())}.json"
    run_uri = _upload_json(bucket, run_blob, run_summary)
    LOGGER.info("Run complete. Summary: %s", run_uri)


if __name__ == "__main__":
    main()
