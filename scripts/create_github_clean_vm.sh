#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="needle-488623"
BUCKET_NAME="needle-datasets-bucket"
ZONE="us-central1-a"
INSTANCE_NAME="github-clean-stars"
MACHINE_TYPE="n2-highmem-16"
BOOT_DISK_GB="100"
DATA_DISK_GB="1500"
BOOT_DISK_TYPE="pd-balanced"
DATA_DISK_TYPE="pd-ssd"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
DATA_DISK_NAME="github-clean-data"
MOUNT_POINT="/mnt/disks/github-clean"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/create_github_clean_vm.sh [options]

Options:
  --project-id ID       Default: needle-488623
  --bucket NAME         Default: needle-datasets-bucket
  --zone ZONE           Default: us-central1-a
  --instance-name NAME  Default: github-clean-stars
  --machine-type TYPE   Default: n2-highmem-16
  --boot-disk-gb N      Default: 100
  --data-disk-gb N      Default: 1500
  --boot-disk-type T    Default: pd-balanced
  --data-disk-type T    Default: pd-ssd

This script:
  1. Grants the VM's default Compute Engine service account object access to the bucket.
  2. Creates a VM in the requested zone.
  3. Attaches a separate SSD data disk for pipeline scratch space.
  4. Uses a startup script to install base packages, install gcloud CLI, format the
     data disk, and mount it at /mnt/disks/github-clean.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-id) PROJECT_ID="$2"; shift 2 ;;
    --bucket) BUCKET_NAME="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --instance-name) INSTANCE_NAME="$2"; shift 2 ;;
    --machine-type) MACHINE_TYPE="$2"; shift 2 ;;
    --boot-disk-gb) BOOT_DISK_GB="$2"; shift 2 ;;
    --data-disk-gb) DATA_DISK_GB="$2"; shift 2 ;;
    --boot-disk-type) BOOT_DISK_TYPE="$2"; shift 2 ;;
    --data-disk-type) DATA_DISK_TYPE="$2"; shift 2 ;;
    -h|--help|help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

require_cmd gcloud

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "[setup] project=${PROJECT_ID}"
echo "[setup] bucket=gs://${BUCKET_NAME}"
echo "[setup] zone=${ZONE}"
echo "[setup] instance=${INSTANCE_NAME}"
echo "[setup] machine_type=${MACHINE_TYPE}"
echo "[setup] data_disk=${DATA_DISK_GB}GB ${DATA_DISK_TYPE}"

echo "[setup] granting bucket access to ${COMPUTE_SA}"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/storage.objectAdmin" \
  >/dev/null

STARTUP_SCRIPT="$(mktemp)"
cat > "${STARTUP_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y apt-transport-https ca-certificates curl git gnupg python3-pip python3-venv

if ! command -v gcloud >/dev/null 2>&1; then
  install -d -m 0755 /usr/share/keyrings
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  cat >/etc/apt/sources.list.d/google-cloud-sdk.list <<'APT'
deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main
APT
  apt-get update
  apt-get install -y google-cloud-cli
fi

DISK="/dev/disk/by-id/google-${DATA_DISK_NAME}"
MOUNT_POINT="${MOUNT_POINT}"

if ! blkid "\${DISK}" >/dev/null 2>&1; then
  mkfs.ext4 -F "\${DISK}"
fi

install -d -m 0777 "\${MOUNT_POINT}"

UUID="\$(blkid -s UUID -o value "\${DISK}")"
if ! grep -q "\${UUID}" /etc/fstab; then
  echo "UUID=\${UUID} \${MOUNT_POINT} ext4 defaults,discard,nofail 0 2" >> /etc/fstab
fi

mountpoint -q "\${MOUNT_POINT}" || mount "\${MOUNT_POINT}"
chmod 0777 "\${MOUNT_POINT}"
EOF

echo "[setup] creating VM"
gcloud compute instances create "${INSTANCE_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --maintenance-policy=MIGRATE \
  --provisioning-model=STANDARD \
  --scopes="https://www.googleapis.com/auth/cloud-platform" \
  --image-family="${IMAGE_FAMILY}" \
  --image-project="${IMAGE_PROJECT}" \
  --boot-disk-size="${BOOT_DISK_GB}GB" \
  --boot-disk-type="${BOOT_DISK_TYPE}" \
  --create-disk="device-name=${DATA_DISK_NAME},size=${DATA_DISK_GB},type=${DATA_DISK_TYPE},auto-delete=yes,mode=rw" \
  --metadata-from-file=startup-script="${STARTUP_SCRIPT}"

rm -f "${STARTUP_SCRIPT}"

cat <<EOF
[setup] VM creation submitted.

Next steps:
  1. Wait a minute for the startup script to finish.
  2. Copy your repo to the VM:
     gcloud compute scp --recurse /home/karen/needle ${INSTANCE_NAME}:~/ --zone=${ZONE} --project=${PROJECT_ID}
  3. SSH in:
     gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --project=${PROJECT_ID}
  4. On the VM, run:
     cd ~/needle
     python3 -m venv .venv
     source .venv/bin/activate
     pip install --upgrade pip
     pip install -r third_party/openmodels.RedPajama-Data/data_prep/github/github_requirements.txt
     bash scripts/github_code_pipeline.sh clean --run-id 20260309_full --selection stars --num-partitions 32 --work-dir ${MOUNT_POINT} --keep-local
EOF
