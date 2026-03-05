# Data Collection

This directory contains scripts for collecting speech datasets and uploading
audio + metadata manifests to a Google Cloud Storage bucket.

## Script

- `collect_speech_to_gcs.py`
  - Sources:
    - `amphion/Emilia-Dataset` (Emilia-Large repo)
    - `kensho/spgispeech` (SPGISpeech)
  - Writes:
    - Audio files to `gs://<bucket>/<prefix>/<dataset>/<split>/audio/...`
    - Mel spectrogram files to `.../mel/...` (`.npz` by default)
    - Sharded manifests to `.../manifests/manifest-*.jsonl.gz`
    - Per-sample CSV metadata files to `.../metadata_csv/<item_id>.csv` (one file per audio)
    - Split summary to `.../summary.json`
    - Run summary to `gs://<bucket>/<prefix>/run-summary-<ts>.json`
  - Manifest records include:
    - transcript
    - language
    - duration
    - timestamps (`start_s`, `end_s`, optional `segments`)
    - `gcs_audio_uri`
    - `gcs_mel_uri` (if mel storage enabled)
    - `gcs_metadata_csv_uri` (one CSV file per audio sample)

## Auth Requirements

- Google Cloud auth via ADC (example: `gcloud auth application-default login`)
- Hugging Face token for gated datasets:
  - `export HF_TOKEN=...`

## Quick Smoke Test

```bash
python src/data_collection/collect_speech_to_gcs.py \
  --bucket YOUR_BUCKET \
  --prefix speech_datasets \
  --max-samples 100 \
  --datasets emilia-large spgi-speech \
  --mel-preset whisper
```

## Full Ingest Example

```bash
python src/data_collection/collect_speech_to_gcs.py \
  --bucket YOUR_BUCKET \
  --prefix speech_datasets \
  --datasets emilia-large spgi-speech \
  --spgi-config L \
  --emilia-splits train \
  --spgi-splits train \
  --mel-preset whisper \
  --streaming
```

## Notes

- The default mode uses Hugging Face streaming to avoid local dataset materialization.
- If you want SPGISpeech eval slices, use a different `--spgi-config` (for example
  `dev` or `test`) and select its available split names via `--spgi-splits`.
- Mel options:
  - `--mel-preset whisper|parakeet`
  - `--mel-n-mels <int>` to override mel bins
  - `--mel-file-format npz|npy`
  - `--mel-dtype float16|float32`
  - `--no-store-mel` to disable mel extraction/upload
