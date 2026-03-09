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
    - Sharded manifests to `.../manifests/manifest-*.jsonl.gz`
    - Split summary to `.../summary.json`
    - Run summary to `gs://<bucket>/<prefix>/run-summary-<ts>.json`

## Auth Requirements

- Google Cloud auth via ADC (example: `gcloud auth application-default login`)
- Hugging Face token for gated datasets:
  - `export HF_TOKEN=...`

## Quick Smoke Test

```bash
python data_collection/collect_speech_to_gcs.py \
  --bucket YOUR_BUCKET \
  --prefix speech_datasets \
  --max-samples 100 \
  --datasets emilia-large spgi-speech
```

## Full Ingest Example

```bash
python data_collection/collect_speech_to_gcs.py \
  --bucket YOUR_BUCKET \
  --prefix speech_datasets \
  --datasets emilia-large spgi-speech \
  --spgi-config L \
  --emilia-splits train \
  --spgi-splits train \
  --streaming
```

## Notes

- The default mode uses Hugging Face streaming to avoid local dataset materialization.
- If you want SPGISpeech eval slices, use a different `--spgi-config` (for example
  `dev` or `test`) and select its available split names via `--spgi-splits`.
