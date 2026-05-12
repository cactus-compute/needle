#!/bin/bash
# Launch multi-host training on all TPU workers.
# Run this from your LOCAL machine (not from the TPU VM).
#
# Usage: ./launch_train.sh [extra needle train args...]
# Example: ./launch_train.sh --epochs 3 --wandb --batch-size 64

set -euo pipefail

TPU_NAME="${TPU_NAME:-large}"
ZONE="${ZONE:-asia-northeast1-b}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"

if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
  echo "[launch] ERROR: no GCP project set. Run 'gcloud config set project <PROJECT>' or export PROJECT=..."
  exit 1
fi

TRAIN_ARGS="${@}"

echo "[launch] Distributing SSH keys across workers..."
# Generate keypair on worker 0 if not present, then distribute to all workers
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 \
  --command='[ -f ~/.ssh/id_rsa ] || ssh-keygen -t rsa -b 2048 -f ~/.ssh/id_rsa -N "" -q; cat ~/.ssh/id_rsa.pub' \
  2>/dev/null | tail -1 > /tmp/_tpu_pubkey.txt

# Add worker 0's public key to all workers' authorized_keys
PUBKEY=$(cat /tmp/_tpu_pubkey.txt)
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=all \
  --command="grep -qF '$(echo $PUBKEY | cut -d' ' -f2)' ~/.ssh/authorized_keys 2>/dev/null || echo '$PUBKEY' >> ~/.ssh/authorized_keys"

echo "[launch] Starting training on all workers..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=all \
  --command="cd ~/needle && .venv/bin/needle train $TRAIN_ARGS" \
  2>&1
