#!/usr/bin/env bash
set -euo pipefail

# Dry run using the exact query from:
# third_party/openmodels.RedPajama-Data/data_prep/github/README.md
#
# Usage:
#   bash scripts/redpajama_github_dry_run.sh

QUERY="$(cat <<'SQL'
SELECT *
FROM (SELECT *
      FROM `bigquery-public-data.github_repos.contents`
               INNER JOIN `bigquery-public-data.github_repos.files` USING (id))
         INNER JOIN `bigquery-public-data.github_repos.licenses` USING (repo_name)
         INNER JOIN `bigquery-public-data.github_repos.languages` USING (repo_name
    )
WHERE (license LIKE 'mit%'
    OR license LIKE 'bsd%'
    OR license LIKE 'apache%')
  AND content IS NOT NULL
SQL
)"

echo "[dry-run] Running BigQuery dry run in US location..."
bq query \
  --use_legacy_sql=false \
  --dry_run \
  --location=US \
  --format=prettyjson \
  "${QUERY}"

echo "[dry-run] Done. Use totalBytesProcessed from output for cost estimate."
