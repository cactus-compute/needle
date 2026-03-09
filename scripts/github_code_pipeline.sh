#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="needle-488623"
BUCKET_NAME="needle-datasets-bucket"
RAW_PREFIX="github_code"
CLEAN_PREFIX="github_code/clean"
LOCATION="US"
REDPAJAMA_DIR="third_party/openmodels.RedPajama-Data/data_prep/github"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/github_code_pipeline.sh export [options]
  bash scripts/github_code_pipeline.sh clean [options]
  bash scripts/github_code_pipeline.sh smoke [options]
  bash scripts/github_code_pipeline.sh run-all [options]

Subcommands:
  export
    Run one staged BigQuery script that scans the public GitHub tables once,
    materializes the filtered rows into a temporary table, and exports them to:
    - gs://<bucket>/<raw-prefix>/5plus_stars/<run_id>/
    - gs://<bucket>/<raw-prefix>/select_languages/<run_id>/

  clean
    Download one exported selection locally, partition the raw files, run the
    full RedPajama clean/dedup/filter pipeline across the whole selection, and
    upload the cleaned outputs back to GCS.

  smoke
    Run a small sampled export for both selections and print the output URIs.

  run-all
    Export both selections once, then clean one or both selections.

Common options:
  --bucket NAME
  --project-id ID
  --location LOCATION
  --run-id ID
  --raw-prefix PREFIX
  --clean-prefix PREFIX

Export-only options:
  --sample-percent N   Integer TABLESAMPLE percent for contents only.
  --limit N            Optional LIMIT on the staged result.
  --dry-run            Submit a BigQuery dry run instead of executing.

Clean-only options:
  --selection NAME     One of: stars, language_only
  --work-dir DIR       Local working directory. Default: /tmp/github_code_pipeline
  --num-partitions N   Number of local-dedup partitions. Default: 32
  --keep-local         Keep local work directory after completion.

Smoke defaults:
  --sample-percent 1
  --limit 2000

Run-all options:
  --clean-selection NAME  One of: stars, language_only, both. Default: stars
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_python_module() {
  local module_name="$1"

  python3 - "$module_name" <<'PY'
import importlib
import sys

module_name = sys.argv[1]
try:
    importlib.import_module(module_name)
except Exception as exc:
    print(f"missing required python module: {module_name} ({exc})", file=sys.stderr)
    raise SystemExit(1)
PY
}

timestamp_utc() {
  date -u +"%Y%m%dT%H%M%SZ"
}

filter_cpu_count() {
  python3 - <<'PY'
import multiprocessing as mp

print(max(2, mp.cpu_count()))
PY
}

warm_filter_tokenizer() {
  python3 - <<'PY'
from transformers import AutoTokenizer

AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b-deduped")
PY
}

selection_dir() {
  case "$1" in
    stars) echo "5plus_stars" ;;
    language_only) echo "select_languages" ;;
    *) echo "unknown selection: $1" >&2; exit 1 ;;
  esac
}

write_partitions() {
  local src_dir="$1"
  local parts_dir="$2"
  local requested_partitions="$3"

  if (( requested_partitions < 1 )); then
    echo "num_partitions must be >= 1" >&2
    exit 1
  fi

  shopt -s nullglob
  local raw_files=("${src_dir}"/github_*.jsonl.gz)
  if [[ "${#raw_files[@]}" -eq 0 ]]; then
    echo "no github_*.jsonl.gz files found in ${src_dir}" >&2
    exit 1
  fi

  local partition_count="$requested_partitions"
  if (( partition_count > ${#raw_files[@]} )); then
    partition_count="${#raw_files[@]}"
  fi

  local i
  for (( i=0; i<partition_count; i++ )); do
    : > "${parts_dir}/chunk_$(printf '%03d' "$i").txt"
  done

  local idx
  for idx in "${!raw_files[@]}"; do
    printf '%s\n' "${raw_files[$idx]}" >> "${parts_dir}/chunk_$(printf '%03d' "$((idx % partition_count))").txt"
  done

  echo "[clean] raw files=${#raw_files[@]}"
  echo "[clean] partitions=${partition_count}"
}

run_local_dedup_partitions() {
  local parts_dir="$1"
  local processed_dir="$2"

  shopt -s nullglob
  local partition_files=("${parts_dir}"/chunk_*.txt)
  if [[ "${#partition_files[@]}" -eq 0 ]]; then
    echo "no chunk_*.txt files found in ${parts_dir}" >&2
    exit 1
  fi

  local partition_file
  for partition_file in "${partition_files[@]}"; do
    if [[ ! -s "${partition_file}" ]]; then
      continue
    fi

    echo "[clean] local dedup + cleaning ${partition_file##*/}"
    python3 "${REDPAJAMA_DIR}/github_clean_dedup_local.py" \
      --input "${partition_file}" \
      --target_dir "${processed_dir}"
  done
}

build_base_select() {
  local sample_percent="$1"
  local limit_rows="$2"

  local contents_source='`bigquery-public-data.github_repos.contents`'
  if [[ -n "$sample_percent" ]]; then
    contents_source="${contents_source} TABLESAMPLE SYSTEM (${sample_percent} PERCENT)"
  fi

  local limit_clause=""
  if [[ -n "$limit_rows" ]]; then
    limit_clause="LIMIT ${limit_rows}"
  fi

  cat <<EOF
WITH
source_contents AS (
  SELECT id, size, content, binary, copies
  FROM ${contents_source}
),
source_files AS (
  SELECT repo_name, ref, path, mode, id, symlink_target
  FROM \`bigquery-public-data.github_repos.files\`
),
star_repos AS (
  SELECT repo_name, TRUE AS matched_stars, FALSE AS matched_language_group
  FROM \`bigquery-public-data.github_repos.sample_repos\`
  WHERE watch_count >= 5
),
lang_repos AS (
  SELECT repo_name, FALSE AS matched_stars, TRUE AS matched_language_group
  FROM \`bigquery-public-data.github_repos.languages\`, UNNEST(language) AS lang
  WHERE lang.name IN ('TypeScript', 'Scala', 'Go', 'Rust')
  GROUP BY repo_name
),
repo_filter AS (
  SELECT
    repo_name,
    LOGICAL_OR(matched_stars) AS matched_stars,
    LOGICAL_OR(matched_language_group) AS matched_language_group
  FROM (
    SELECT * FROM star_repos
    UNION ALL
    SELECT * FROM lang_repos
  )
  GROUP BY repo_name
)
SELECT
  f.repo_name,
  f.ref,
  f.path,
  f.mode,
  f.id,
  f.symlink_target,
  c.size,
  c.content,
  c.binary,
  c.copies,
  l.license,
  sr.watch_count,
  rf.matched_stars,
  rf.matched_language_group
FROM source_contents c
JOIN source_files f USING (id)
JOIN \`bigquery-public-data.github_repos.licenses\` l USING (repo_name)
JOIN repo_filter rf USING (repo_name)
LEFT JOIN \`bigquery-public-data.github_repos.sample_repos\` sr USING (repo_name)
WHERE (LOWER(l.license) LIKE 'mit%'
    OR LOWER(l.license) LIKE 'bsd%'
    OR LOWER(l.license) LIKE 'apache%')
  AND c.content IS NOT NULL
${limit_clause}
EOF
}

build_export_script() {
  local bucket="$1"
  local raw_prefix="$2"
  local run_id="$3"
  local sample_percent="$4"
  local limit_rows="$5"

  local base_select
  base_select="$(build_base_select "$sample_percent" "$limit_rows")"
  local stars_uri="gs://${bucket}/${raw_prefix%/}/5plus_stars/${run_id}/github_*.jsonl.gz"
  local languages_uri="gs://${bucket}/${raw_prefix%/}/select_languages/${run_id}/github_*.jsonl.gz"

  cat <<EOF
CREATE TEMP TABLE selected AS
${base_select}
;

EXPORT DATA OPTIONS(
  uri='${stars_uri}',
  format='JSON',
  compression='GZIP',
  overwrite=true
) AS
SELECT *
FROM selected
WHERE matched_stars
;

EXPORT DATA OPTIONS(
  uri='${languages_uri}',
  format='JSON',
  compression='GZIP',
  overwrite=true
) AS
SELECT *
FROM selected
WHERE matched_language_group AND NOT matched_stars
;
EOF
}

run_export() {
  require_cmd bq

  local run_id="$(timestamp_utc)"
  local bucket="$BUCKET_NAME"
  local raw_prefix="$RAW_PREFIX"
  local project_id="$PROJECT_ID"
  local location="$LOCATION"
  local sample_percent=""
  local limit_rows=""
  local dry_run=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --bucket) bucket="$2"; shift 2 ;;
      --raw-prefix) raw_prefix="$2"; shift 2 ;;
      --project-id) project_id="$2"; shift 2 ;;
      --location) location="$2"; shift 2 ;;
      --run-id) run_id="$2"; shift 2 ;;
      --sample-percent) sample_percent="$2"; shift 2 ;;
      --limit) limit_rows="$2"; shift 2 ;;
      --dry-run) dry_run=1; shift ;;
      *) echo "unknown export option: $1" >&2; exit 1 ;;
    esac
  done

  local export_script
  export_script="$(build_export_script "$bucket" "$raw_prefix" "$run_id" "$sample_percent" "$limit_rows")"
  local stars_uri="gs://${bucket}/${raw_prefix%/}/5plus_stars/${run_id}/"
  local languages_uri="gs://${bucket}/${raw_prefix%/}/select_languages/${run_id}/"

  echo "[export] run_id=${run_id}"
  echo "[export] stars=${stars_uri}"
  echo "[export] select_languages=${languages_uri}"

  if [[ "$dry_run" -eq 1 ]]; then
    echo "[export] dry run for a multi-statement script; BigQuery may only show partial byte estimates."
    bq query \
      --project_id="${project_id}" \
      --use_legacy_sql=false \
      --dry_run \
      --location="${location}" \
      --format=prettyjson \
      "${export_script}"
    return
  fi

  bq query \
    --project_id="${project_id}" \
    --use_legacy_sql=false \
    --location="${location}" \
    "${export_script}"
}

run_clean() {
  require_cmd gcloud
  require_cmd python3
  require_python_module transformers

  local run_id=""
  local bucket="$BUCKET_NAME"
  local raw_prefix="$RAW_PREFIX"
  local clean_prefix="$CLEAN_PREFIX"
  local selection="stars"
  local work_dir="/tmp/github_code_pipeline"
  local num_partitions=32
  local keep_local=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --bucket) bucket="$2"; shift 2 ;;
      --raw-prefix) raw_prefix="$2"; shift 2 ;;
      --clean-prefix) clean_prefix="$2"; shift 2 ;;
      --run-id) run_id="$2"; shift 2 ;;
      --selection) selection="$2"; shift 2 ;;
      --work-dir) work_dir="$2"; shift 2 ;;
      --num-partitions) num_partitions="$2"; shift 2 ;;
      --keep-local) keep_local=1; shift ;;
      --bucket-index) shift 2 ;;
      *) echo "unknown clean option: $1" >&2; exit 1 ;;
    esac
  done

  if [[ -z "$run_id" ]]; then
    echo "--run-id is required for clean" >&2
    exit 1
  fi

  echo "[clean] checking tokenizer availability"
  warm_filter_tokenizer

  local selection_path
  selection_path="$(selection_dir "$selection")"
  local raw_uri="gs://${bucket}/${raw_prefix%/}/${selection_path}/${run_id}"
  local clean_uri="gs://${bucket}/${clean_prefix%/}/${selection_path}/${run_id}"

  local local_root="${work_dir%/}/${selection_path}/${run_id}"
  local src_dir="${local_root}/src"
  local parts_dir="${local_root}/partitions"
  local processed_dir="${local_root}/processed"
  local deduped_dir="${local_root}/processed_deduped"
  local filtered_dir="${local_root}/processed_filtered"

  mkdir -p "${src_dir}" "${parts_dir}" "${processed_dir}" "${deduped_dir}" "${filtered_dir}"

  echo "[clean] downloading raw objects from ${raw_uri}"
  gcloud storage cp "${raw_uri}/github_*.jsonl.gz" "${src_dir}/"

  write_partitions "${src_dir}" "${parts_dir}" "${num_partitions}"
  run_local_dedup_partitions "${parts_dir}" "${processed_dir}"

  echo "[clean] global dedup hash pass"
  python3 "${REDPAJAMA_DIR}/github_global_dedup.py" \
    --first_step_dir "${processed_dir}" \
    --target_dir "${deduped_dir}"

  shopt -s nullglob
  local run_files=("${processed_dir}"/run_*.jsonl)
  if [[ "${#run_files[@]}" -eq 0 ]]; then
    echo "no run_*.jsonl files produced in ${processed_dir}" >&2
    exit 1
  fi

  echo "[clean] merge globally unique hashes"
  local run_file
  for run_file in "${run_files[@]}"; do
    python3 "${REDPAJAMA_DIR}/github_merge_dedup.py" \
      --first_step_dir "${processed_dir}" \
      --target_dir "${deduped_dir}" \
      --input "${run_file}"
  done

  local deduped_files=("${deduped_dir}"/deduped_*.jsonl)
  if [[ "${#deduped_files[@]}" -eq 0 ]]; then
    echo "no deduped_*.jsonl files produced in ${deduped_dir}" >&2
    exit 1
  fi

  echo "[clean] quality filter"
  local deduped_file
  local filter_cpus
  filter_cpus="$(filter_cpu_count)"
  for deduped_file in "${deduped_files[@]}"; do
    SLURM_CPUS_PER_TASK="${filter_cpus}" \
    python3 "${REDPAJAMA_DIR}/github_run_filter.py" \
      --data_file "${deduped_file}" \
      --target_dir "${filtered_dir}"
  done

  echo "[clean] uploading cleaned outputs to ${clean_uri}"
  gcloud storage cp "${filtered_dir}"/*.gz "${clean_uri}/filtered/"

  if [[ -f "${deduped_dir}/stats_deduped.jsonl" ]]; then
    gcloud storage cp "${deduped_dir}/stats_deduped.jsonl" "${clean_uri}/metadata/"
  fi

  if [[ "$keep_local" -eq 0 ]]; then
    rm -rf "${local_root}"
  fi
}

run_smoke() {
  local run_id="smoke_$(timestamp_utc)"
  local bucket="$BUCKET_NAME"
  local raw_prefix="$RAW_PREFIX"
  local project_id="$PROJECT_ID"
  local location="$LOCATION"
  local sample_percent=1
  local limit_rows=2000

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --bucket) bucket="$2"; shift 2 ;;
      --raw-prefix) raw_prefix="$2"; shift 2 ;;
      --project-id) project_id="$2"; shift 2 ;;
      --location) location="$2"; shift 2 ;;
      --run-id) run_id="$2"; shift 2 ;;
      --sample-percent) sample_percent="$2"; shift 2 ;;
      --limit) limit_rows="$2"; shift 2 ;;
      *) echo "unknown smoke option: $1" >&2; exit 1 ;;
    esac
  done

  run_export \
    --bucket "${bucket}" \
    --raw-prefix "${raw_prefix}" \
    --project-id "${project_id}" \
    --location "${location}" \
    --run-id "${run_id}" \
    --sample-percent "${sample_percent}" \
    --limit "${limit_rows}"

  echo "[smoke] complete"
  echo "[smoke] stars=gs://${bucket}/${raw_prefix%/}/5plus_stars/${run_id}/"
  echo "[smoke] select_languages=gs://${bucket}/${raw_prefix%/}/select_languages/${run_id}/"
}

run_all() {
  local run_id="$(timestamp_utc)"
  local bucket="$BUCKET_NAME"
  local raw_prefix="$RAW_PREFIX"
  local clean_prefix="$CLEAN_PREFIX"
  local project_id="$PROJECT_ID"
  local location="$LOCATION"
  local work_dir="/tmp/github_code_pipeline"
  local sample_percent=""
  local limit_rows=""
  local clean_selection="stars"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --bucket) bucket="$2"; shift 2 ;;
      --raw-prefix) raw_prefix="$2"; shift 2 ;;
      --clean-prefix) clean_prefix="$2"; shift 2 ;;
      --project-id) project_id="$2"; shift 2 ;;
      --location) location="$2"; shift 2 ;;
      --run-id) run_id="$2"; shift 2 ;;
      --work-dir) work_dir="$2"; shift 2 ;;
      --sample-percent) sample_percent="$2"; shift 2 ;;
      --limit) limit_rows="$2"; shift 2 ;;
      --clean-selection) clean_selection="$2"; shift 2 ;;
      *) echo "unknown run-all option: $1" >&2; exit 1 ;;
    esac
  done

  echo "[run-all] run_id=${run_id}"
  local export_args=(
    --bucket "${bucket}"
    --raw-prefix "${raw_prefix}"
    --project-id "${project_id}"
    --location "${location}"
    --run-id "${run_id}"
  )
  if [[ -n "${sample_percent}" ]]; then
    export_args+=(--sample-percent "${sample_percent}")
  fi
  if [[ -n "${limit_rows}" ]]; then
    export_args+=(--limit "${limit_rows}")
  fi

  run_export \
    "${export_args[@]}"

  case "${clean_selection}" in
    stars|language_only)
      run_clean \
        --bucket "${bucket}" \
        --raw-prefix "${raw_prefix}" \
        --clean-prefix "${clean_prefix}" \
        --run-id "${run_id}" \
        --selection "${clean_selection}" \
        --work-dir "${work_dir}"
      ;;
    both)
      run_clean \
        --bucket "${bucket}" \
        --raw-prefix "${raw_prefix}" \
        --clean-prefix "${clean_prefix}" \
        --run-id "${run_id}" \
        --selection stars \
        --work-dir "${work_dir}"

      run_clean \
        --bucket "${bucket}" \
        --raw-prefix "${raw_prefix}" \
        --clean-prefix "${clean_prefix}" \
        --run-id "${run_id}" \
        --selection language_only \
        --work-dir "${work_dir}"
      ;;
    *)
      echo "unknown clean selection: ${clean_selection}" >&2
      exit 1
      ;;
  esac
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  local subcommand="$1"
  shift

  case "${subcommand}" in
    export) run_export "$@" ;;
    clean) run_clean "$@" ;;
    smoke) run_smoke "$@" ;;
    run-all) run_all "$@" ;;
    -h|--help|help) usage ;;
    *) echo "unknown subcommand: ${subcommand}" >&2; usage; exit 1 ;;
  esac
}

main "$@"
