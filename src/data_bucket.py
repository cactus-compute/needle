import os
import subprocess
import sys

PROJECT = "needle-488623"
DEFAULT_BUCKET = "needle-datasets-bucket"

DATA_HELP = f"""
  data commands:
    needle data list [--prefix PREFIX] [--bucket NAME] [--recursive]
    needle data pull REMOTE_PREFIX [--dest DIR] [--bucket NAME] [--dry-run] [--delete] [--exclude REGEX]
    needle data push LOCAL_PATH [--dest-prefix PREFIX] [--bucket NAME] [--dry-run] [--delete] [--exclude REGEX]

  defaults:
    bucket: {DEFAULT_BUCKET}
    project: {PROJECT}
"""


def _run(cmd, check=True):
    print(f"[data] $ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, text=True)
    except FileNotFoundError:
        print(
            "[data] ERROR: 'gcloud' not found. Install: https://cloud.google.com/sdk/docs/install",
            file=sys.stderr,
        )
        sys.exit(1)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result


def _gs_uri(bucket, prefix):
    prefix = (prefix or "").lstrip("/")
    return f"gs://{bucket}" if not prefix else f"gs://{bucket}/{prefix}"


def _with_auth_flags(args, cmd):
    account = getattr(args, "account", None)
    impersonate = getattr(args, "impersonate_service_account", None)
    if account:
        cmd.extend(["--account", account])
    if impersonate:
        cmd.extend(["--impersonate-service-account", impersonate])
    return cmd


def data_list(args):
    uri = _gs_uri(args.bucket, args.prefix)
    cmd = ["gcloud", "storage", "ls", "--project", PROJECT]
    if args.recursive:
        cmd.append("--recursive")
    cmd.append(uri)
    cmd = _with_auth_flags(args, cmd)
    _run(cmd)


def data_pull(args):
    src_uri = _gs_uri(args.bucket, args.remote_prefix)
    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)

    cmd = [
        "gcloud", "storage", "rsync",
        src_uri, dest,
        "--recursive",
        "--project", PROJECT,
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.delete:
        cmd.append("--delete-unmatched-destination-objects")
    if args.exclude:
        cmd.extend(["--exclude", args.exclude])
    cmd = _with_auth_flags(args, cmd)
    _run(cmd)


def data_push(args):
    src = os.path.abspath(args.local_path)
    if not os.path.exists(src):
        print(f"[data] ERROR: local path does not exist: {src}", file=sys.stderr)
        sys.exit(1)

    dest_uri = _gs_uri(args.bucket, args.dest_prefix)

    if os.path.isdir(src):
        cmd = [
            "gcloud", "storage", "rsync",
            src, dest_uri,
            "--recursive",
            "--project", PROJECT,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.delete:
            cmd.append("--delete-unmatched-destination-objects")
        if args.exclude:
            cmd.extend(["--exclude", args.exclude])
    else:
        cmd = [
            "gcloud", "storage", "cp",
            src, dest_uri,
            "--project", PROJECT,
        ]
        if args.dry_run:
            print("[data] NOTE: dry-run is only supported for directory rsync; continuing with cp.")

    cmd = _with_auth_flags(args, cmd)
    _run(cmd)


def data_dispatch(args):
    actions = {
        "list": data_list,
        "pull": data_pull,
        "push": data_push,
    }
    if not args.data_action or args.data_action not in actions:
        print(DATA_HELP)
        sys.exit(0 if not args.data_action else 1)
    actions[args.data_action](args)
