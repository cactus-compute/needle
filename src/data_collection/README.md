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
      - when worker sharding is enabled, manifests go under:
        `.../manifests/shard-<index>-of-<count>/manifest-*.jsonl.gz`
    - Per-sample CSV metadata files to `.../metadata_csv/<item_id>.csv` (one file per audio)
    - Split summary to `.../summary.json`
      - when worker sharding is enabled, summaries go to:
        `.../summary/shard-<index>-of-<count>.json`
    - Run summary to `gs://<bucket>/<prefix>/run-summary-<ts>.json`
  - Manifest records include:
    - transcript
    - language
    - duration
    - timestamps (`start_s`, `end_s`, `segments`)
      - if source segments are missing, collector writes approximate transcript chunks
        distributed across clip duration
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

## Parallel Full Ingest (Recommended)

Use worker sharding so each worker ingests a deterministic subset.

Example: launch 16 workers for SPGI train split on one machine:

```bash
mkdir -p logs
for i in $(seq 0 15); do
  python src/data_collection/collect_speech_to_gcs.py \
    --bucket YOUR_BUCKET \
    --project YOUR_PROJECT \
    --prefix speech_datasets \
    --datasets spgi-speech \
    --spgi-config L \
    --spgi-splits train \
    --mel-preset parakeet \
    --streaming \
    --shard-count 16 \
    --shard-index "$i" \
    > "logs/spgi_worker_${i}.log" 2>&1 &
done
wait
```

Example: single worker command (for VM/Batch jobs where each job has one index):

```bash
python src/data_collection/collect_speech_to_gcs.py \
  --bucket YOUR_BUCKET \
  --project YOUR_PROJECT \
  --prefix speech_datasets \
  --datasets emilia-large spgi-speech \
  --spgi-config L \
  --emilia-splits train \
  --spgi-splits train validation test \
  --mel-preset parakeet \
  --streaming \
  --shard-count 64 \
  --shard-index WORKER_INDEX
```

## Notes

- The default mode uses Hugging Face streaming to avoid local dataset materialization.
- Script disables HF Xet transport (`HF_HUB_DISABLE_XET=1`) by default to avoid
  occasional process hang after completion in some local environments.
- If you want SPGISpeech eval slices, use a different `--spgi-config` (for example
  `dev` or `test`) and select its available split names via `--spgi-splits`.
- Mel options:
  - `--mel-preset whisper|parakeet`
  - `--mel-n-mels <int>` to override mel bins
  - `--mel-file-format npz|npy`
  - `--mel-dtype float16|float32`
  - `--no-store-mel` to disable mel extraction/upload
- Worker sharding options:
  - `--shard-count <int>` total workers
  - `--shard-index <int>` this worker id
  - `--[no-]shard-contiguous` shard assignment strategy
- Keep `--shard-count` and `--shard-contiguous` fixed for a run. If you change
  them mid-run, use a new `--prefix` to avoid mixing naming layouts.
