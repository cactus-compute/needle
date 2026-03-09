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
      - when worker sharding is enabled: `.../manifests/shard-<index>-of-<count>/manifest-*.jsonl.gz`
    - Split summary to `.../summary.json`
      - when worker sharding is enabled: `.../summary/shard-<index>-of-<count>.json`
    - Run summary to `gs://<bucket>/<prefix>/run-summary-<ts>.json`

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
  --datasets emilia-large spgi-speech
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
  --streaming
```

## One-VM Parallel Run

Run multiple workers on one VM, each with a unique `--shard-index`.

Example: 16 workers

```bash
mkdir -p logs
for i in $(seq 0 15); do
  python src/data_collection/collect_speech_to_gcs.py \
    --bucket YOUR_BUCKET \
    --project YOUR_PROJECT \
    --prefix speech_datasets \
    --datasets emilia-large spgi-speech \
    --spgi-config L \
    --emilia-splits train \
    --spgi-splits train validation test \
    --streaming \
    --shard-count 16 \
    --shard-index "$i" \
    > "logs/ingest_worker_${i}.log" 2>&1 &
done
wait
```

## Notes

- The default mode uses Hugging Face streaming to avoid local dataset materialization.
- If you want SPGISpeech eval slices, use a different `--spgi-config` (for example
  `dev` or `test`) and select its available split names via `--spgi-splits`.
- Worker sharding options:
  - `--shard-count <int>` total workers
  - `--shard-index <int>` worker id
  - `--[no-]shard-contiguous` assignment strategy
- Keep `--shard-count` and `--shard-contiguous` fixed for a run. If you change
  either one, use a new `--prefix` to avoid mixing naming layouts.
