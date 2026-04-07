import getpass
import os
import re
import subprocess
import sys

PROJECT = "needle-488623"

VERSION_FOR_TYPE = {
    "v6e":        "v2-alpha-tpuv6e",
}


def _resolve_version(accel_type, explicit_version):
    """Pick the right software version for the accelerator type.

    If the user explicitly passed --version, use that.
    Otherwise, match on the accelerator type prefix (e.g. 'v6e-4' → 'v6e').
    """
    if explicit_version is not None:
        return explicit_version
    for prefix, version in sorted(VERSION_FOR_TYPE.items(), key=lambda kv: -len(kv[0])):
        if accel_type.startswith(prefix):
            return version
            
    print(
        f"[tpu] WARNING: unknown accelerator type '{accel_type}', "
        f"defaulting to tpu-ubuntu2204-base. Pass --version to override.",
        file=sys.stderr,
    )
    return "tpu-ubuntu2204-base"


# Trillium (v6e) zones with quota, ordered by on-demand price (cheapest first)
# us-east/us-central/us-south/us-west: $2.70/chip/hr
# europe-west4:                        $2.97/chip/hr
# asia:                                $3.24/chip/hr
ZONES = [
    "us-east1-d",
    "us-east5-a", "us-east5-b",
    "us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f",
    "us-east4-c",
    "us-south1-a", "us-south1-b", "us-south1-c",
    "us-west1-a", "us-west1-b", "us-west1-c",
    "europe-west4-a", "europe-west4-b", "europe-west4-c",
    "asia-east1-a", "asia-east1-b", "asia-east1-c",
    "asia-northeast1-a", "asia-northeast1-b", "asia-northeast1-c",
    "asia-south1-a", "asia-south1-b", "asia-south1-c",
    "asia-southeast1-a", "asia-southeast1-b", "asia-southeast1-c",
]

TPU_HELP = """
  tpu commands:
    needle tpu create NAME [--type TYPE] [--version VER]
    needle tpu connect NAME [--zone ZONE]
    needle tpu setup NAME [--zone ZONE]    setup all workers (multi-host)
    needle tpu sync NAME [--zone ZONE]     sync code only (fast, no venv rebuild)
    needle tpu train NAME [--zone ZONE]    launch training on all workers
    needle tpu claude NAME [--zone ZONE]
    needle tpu stop NAME [--zone ZONE]
    needle tpu start NAME [--zone ZONE]
    needle tpu delete NAME [--zone ZONE]
    needle tpu list
"""


def _run(cmd, check=True, capture=False, quiet=False):
    if not quiet:
        print(f"[tpu] $ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=capture, text=True)
    except FileNotFoundError:
        print(
            "[tpu] ERROR: 'gcloud' not found. "
            "Install: https://cloud.google.com/sdk/docs/install",
            file=sys.stderr,
        )
        sys.exit(1)
    if check and result.returncode != 0:
        if capture and result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
        sys.exit(result.returncode)
    return result


def _detect_zone(name):
    print(f"[tpu] Searching for '{name}'...")
    for zone in ZONES:
        result = _run(
            ["gcloud", "compute", "tpus", "tpu-vm", "describe", name,
             "--zone", zone, "--project", PROJECT,
             "--format", "value(name)"],
            check=False, capture=True, quiet=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            print(f"[tpu] Found '{name}' in {zone}")
            return zone
    print(
        f"[tpu] ERROR: instance '{name}' not found. "
        "Run 'needle tpu list' to see instances.",
        file=sys.stderr,
    )
    sys.exit(1)


def _update_ssh_config(path, host_name, new_block):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "r") as f:
            content = f.read()
        pattern = rf"\n?Host {re.escape(host_name)}\n(?:    .*\n)*"
        content = re.sub(pattern, "", content)
    else:
        content = ""
    with open(path, "w") as f:
        f.write(content.rstrip("\n") + "\n" if content.strip() else "")
        f.write(new_block)


def _is_multihost(accel_type):
    """Check if a TPU accelerator type requires multi-host setup.

    v6e-1/4/8 are single-host. v6e-16+ are multi-host.
    """
    m = re.search(r"(\d+)$", accel_type)
    if not m:
        return False
    num_chips = int(m.group(1))
    return num_chips > 8


def _get_num_workers(name, zone):
    """Get the number of workers for a TPU VM."""
    result = _run(
        ["gcloud", "compute", "tpus", "tpu-vm", "describe", name,
         "--zone", zone, "--project", PROJECT,
         "--format", "json(networkEndpoints)"],
        check=False, capture=True, quiet=True,
    )
    import json
    try:
        data = json.loads(result.stdout)
        return len(data.get("networkEndpoints", []))
    except (json.JSONDecodeError, TypeError):
        return 1


def _run_all_workers(name, zone, command):
    """Run a command on all workers of a TPU VM."""
    return _run(
        ["gcloud", "compute", "tpus", "tpu-vm", "ssh", name,
         "--zone", zone, "--project", PROJECT,
         "--worker=all",
         "--command", command],
        check=False,
    )


def _check_tpu_health(name, zone):
    """SSH into the VM and verify /dev/accel* devices exist."""
    print(f"[tpu] Verifying TPU devices on '{name}'...")
    result = _run(
        ["gcloud", "compute", "tpus", "tpu-vm", "ssh", name,
         "--zone", zone, "--project", PROJECT,
         "--command", "ls /dev/accel* 2>/dev/null && echo TPU_OK || echo TPU_MISSING"],
        check=False, capture=True, quiet=True,
    )
    output = result.stdout.strip()
    if "TPU_OK" in output:
        devices = [l for l in output.splitlines() if l.startswith("/dev/accel")]
        print(f"[tpu] TPU healthy: {len(devices)} device(s) found")
        return True
    else:
        print(
            f"[tpu] WARNING: no /dev/accel* devices found. "
            f"The software version may not match the accelerator type.",
            file=sys.stderr,
        )
        return False


def _collect_git_config():
    """Read git user.name and user.email from local git config."""
    name = ""
    email = ""
    try:
        name = subprocess.run(
            ["git", "config", "user.name"], capture_output=True, text=True
        ).stdout.strip()
        email = subprocess.run(
            ["git", "config", "user.email"], capture_output=True, text=True
        ).stdout.strip()
    except FileNotFoundError:
        pass

    if not name or not email:
        print("[tpu] Skipping git config (user.name or user.email not set locally).")
        return None, None
    print(f"[tpu] Using git config: {name} <{email}>")
    return name, email


def _setup_git_on_instance(name, zone, git_name, git_email):
    """SSH into the VM and configure git user.name/email."""
    if not git_name or not git_email:
        return
    cmd = (
        f'git config --global user.name "{git_name}" && '
        f'git config --global user.email "{git_email}"'
    )
    print(f"[tpu] Configuring git as '{git_name} <{git_email}>'...")
    _run(
        ["gcloud", "compute", "tpus", "tpu-vm", "ssh", name,
         "--zone", zone, "--project", PROJECT,
         "--command", cmd],
        check=False, capture=True, quiet=True,
    )


def tpu_create(args):
    version = _resolve_version(args.accel_type, args.version)
    preemptible = getattr(args, "preemptible", False)
    print(f"[tpu] Accelerator: {args.accel_type}, software version: {version}"
          f"{', preemptible' if preemptible else ''}")

    git_name, git_email = _collect_git_config()

    for zone in ZONES:
        print(f"[tpu] Trying {zone}...")
        cmd = ["gcloud", "compute", "tpus", "tpu-vm", "create", args.name,
             "--zone", zone,
             "--accelerator-type", args.accel_type,
             "--version", version,
             "--project", PROJECT]
        if preemptible:
            cmd.append("--preemptible")
        result = _run(cmd,
            check=False, capture=True,
        )
        if result.returncode == 0:
            print(f"[tpu] SUCCESS: created '{args.name}' in {zone}")
            args.zone = zone
            _check_tpu_health(args.name, zone)
            if _is_multihost(args.accel_type):
                print(f"[tpu] Multi-host TPU detected ({args.accel_type})")
                _setup_git_on_instance(args.name, zone, git_name, git_email)
                tpu_setup(args)
                tpu_claude(args)
            else:
                _setup_git_on_instance(args.name, zone, git_name, git_email)
                tpu_claude(args)
                tpu_connect(args)
            return
        stderr = result.stderr.strip()
        if not stderr:
            stderr = result.stdout.strip()
        
        if stderr and any(phrase in stderr.lower() for phrase in [
            "active account selected",
            "gcloud auth login",
            "could not load the default credentials",
            "reauth",
        ]):
            print(f"[tpu] AUTH ERROR:\n{stderr}", file=sys.stderr)
            sys.exit(1)
        # extract "message" from JSON error, or fall back to first ERROR line
        msg = "unknown error"
        if stderr:
            m = re.search(r'"message":\s*"(.+?)(?<!\\)"', stderr)
            if m:
                msg = m.group(1).replace('\\"', '"')
            else:
                for line in stderr.splitlines():
                    if "ERROR" in line:
                        msg = line
                        break
                else:
                    msg = stderr.splitlines()[-1]
        print(f"[tpu] {zone}: {msg}")
    print(
        f"[tpu] ERROR: could not create '{args.name}' in any zone.",
        file=sys.stderr,
    )
    print(
        f"[tpu] Check quota: "
        f"https://console.cloud.google.com/iam-admin/quotas?project={PROJECT}",
        file=sys.stderr,
    )
    sys.exit(1)


def tpu_connect(args):
    zone = args.zone or _detect_zone(args.name)

    result = _run(
        ["gcloud", "compute", "tpus", "tpu-vm", "ssh", args.name,
         "--zone", zone, "--project", PROJECT, "--dry-run"],
        capture=True,
    )

    ip_match = re.search(
        r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
        result.stdout + result.stderr,
    )
    if not ip_match:
        print("[tpu] ERROR: could not parse IP from dry-run output.", file=sys.stderr)
        print(f"[tpu] stdout: {result.stdout}", file=sys.stderr)
        sys.exit(1)
    ip = ip_match.group(1)
    print(f"[tpu] Detected IP: {ip}")

    ssh_config_path = os.path.expanduser("~/.ssh/config")
    user = getpass.getuser()
    block = (
        f"\nHost {args.name}\n"
        f"    HostName {ip}\n"
        f"    User {user}\n"
        f"    IdentityFile ~/.ssh/google_compute_engine\n"
        f"    CheckHostIP no\n"
        f"    StrictHostKeyChecking no\n"
    )
    _update_ssh_config(ssh_config_path, args.name, block)
    print(f"[tpu] Updated {ssh_config_path} with host '{args.name}'")

    print(f"[tpu] Connecting to {args.name} (this propagates SSH keys)...")
    subprocess.run(
        ["gcloud", "compute", "tpus", "tpu-vm", "ssh", args.name,
         "--zone", zone, "--project", PROJECT],
    )


def tpu_stop(args):
    zone = args.zone or _detect_zone(args.name)
    _run(["gcloud", "compute", "tpus", "tpu-vm", "stop", args.name,
          "--zone", zone, "--project", PROJECT])
    print(f"[tpu] Stopped '{args.name}'")


def _update_ssh_config_for(args):
    """Fetch the current external IP and update ~/.ssh/config."""
    zone = args.zone
    result = _run(
        ["gcloud", "compute", "tpus", "tpu-vm", "describe", args.name,
         "--zone", zone, "--project", PROJECT,
         "--format", "get(networkEndpoints[0].accessConfig.externalIp)"],
        capture=True,
    )
    ip = result.stdout.strip()
    if not ip:
        print("[tpu] WARNING: could not get external IP, SSH config not updated.",
              file=sys.stderr)
        return
    print(f"[tpu] New IP: {ip}")
    ssh_config_path = os.path.expanduser("~/.ssh/config")
    user = getpass.getuser()
    block = (
        f"\nHost {args.name}\n"
        f"    HostName {ip}\n"
        f"    User {user}\n"
        f"    IdentityFile ~/.ssh/google_compute_engine\n"
        f"    CheckHostIP no\n"
        f"    StrictHostKeyChecking no\n"
    )
    _update_ssh_config(ssh_config_path, args.name, block)
    print(f"[tpu] Updated SSH config for '{args.name}'")


def tpu_start(args):
    zone = args.zone or _detect_zone(args.name)
    _run(["gcloud", "compute", "tpus", "tpu-vm", "start", args.name,
          "--zone", zone, "--project", PROJECT])
    print(f"[tpu] Started '{args.name}'")
    args.zone = zone
    _update_ssh_config_for(args)


def tpu_delete(args):
    zone = args.zone or _detect_zone(args.name)
    answer = input(f"[tpu] Delete '{args.name}' in {zone}? This is permanent. [y/N] ")
    if answer.lower() != "y":
        print("[tpu] Aborted.")
        return
    _run(["gcloud", "compute", "tpus", "tpu-vm", "delete", args.name,
          "--zone", zone, "--project", PROJECT, "--quiet"])
    print(f"[tpu] Deleted '{args.name}'")


def tpu_claude(args):
    zone = args.zone or _detect_zone(args.name)
    setup_script = (
        "curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && "
        "sudo apt-get install -y nodejs && "
        "sudo npm install -g @anthropic-ai/claude-code && "
        "echo '[tpu] Claude Code installed. Run: claude'"
    )
    print(f"[tpu] Installing Claude Code on '{args.name}'...")
    _run(
        ["gcloud", "compute", "tpus", "tpu-vm", "ssh", args.name,
         "--zone", zone, "--project", PROJECT,
         "--command", setup_script],
    )


def tpu_list(args):
    found = False
    for zone in ZONES:
        result = _run(
            ["gcloud", "compute", "tpus", "tpu-vm", "list",
             "--zone", zone, "--project", PROJECT],
            check=False, capture=True, quiet=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            if not found:
                header = result.stdout.strip().splitlines()[0]
                print(f"{header}  ZONE")
                found = True
            for line in result.stdout.strip().splitlines()[1:]:
                print(f"{line}  {zone}")
    if not found:
        print("[tpu] No instances found.")


def _sync_code_to_workers(name, zone, num_workers):
    """Tarball source code and SCP to all workers. Returns True on success.

    Preserves .venv, .data_cache, and checkpoints on workers if they exist.
    """
    import tempfile
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    tarball = os.path.join(tempfile.gettempdir(), "needle_sync.tar.gz")
    print(f"[tpu] Creating tarball (excluding .venv, .data_cache, .git)...")
    _run(
        ["tar", "-czf", tarball,
         "--exclude=.venv", "--exclude=.data_cache",
         "--exclude=__pycache__", "--exclude=.git",
         "--exclude=*.pyc", "--exclude=checkpoints",
         "-C", os.path.dirname(project_root),
         os.path.basename(project_root)],
    )
    tar_size = os.path.getsize(tarball)
    print(f"[tpu] Tarball: {tar_size / 1024:.0f} KB")

    print(f"[tpu] Copying to all {num_workers} workers...")
    result = _run(
        ["gcloud", "compute", "tpus", "tpu-vm", "scp",
         tarball, f"{name}:/tmp/needle_sync.tar.gz",
         "--zone", zone, "--project", PROJECT,
         "--worker=all"],
        check=False,
    )
    os.remove(tarball)
    if result.returncode != 0:
        print("[tpu] ERROR: failed to copy tarball to workers.", file=sys.stderr)
        return False

    # Extract on all workers, preserving .venv/.data_cache/checkpoints
    # 1. Move heavy dirs aside if they exist
    # 2. Remove old source code
    # 3. Extract fresh tarball
    # 4. Move heavy dirs back
    print(f"[tpu] Extracting on all {num_workers} workers...")
    extract_cmd = (
        "set -e; "
        "cd ~; "
        # Save heavy dirs if they exist
        "for d in .venv .data_cache checkpoints; do "
        "  if [ -d ~/needle/$d ]; then mv ~/needle/$d /tmp/needle_save_$d; fi; "
        "done; "
        # Remove old code and extract new
        "rm -rf ~/needle; "
        "tar -xzf /tmp/needle_sync.tar.gz -C ~; "
        "rm -f /tmp/needle_sync.tar.gz; "
        # Restore heavy dirs
        "for d in .venv .data_cache checkpoints; do "
        "  if [ -d /tmp/needle_save_$d ]; then mv /tmp/needle_save_$d ~/needle/$d; fi; "
        "done"
    )
    result = _run_all_workers(name, zone, extract_cmd)
    if result.returncode != 0:
        print("[tpu] ERROR: failed to extract on workers.", file=sys.stderr)
        return False
    return True


def tpu_setup(args):
    """Set up all workers of a multi-host TPU: sync code, install deps.

    Creates a lightweight tarball (excludes .venv, .data_cache, __pycache__,
    .git), copies it to all workers via SCP, then runs `source ./setup`
    on each worker to create a fresh venv with jax[tpu].
    """
    zone = args.zone or _detect_zone(args.name)
    num_workers = _get_num_workers(args.name, zone)

    if num_workers <= 1:
        print(f"[tpu] Single-host TPU ({num_workers} worker). Use 'needle tpu connect' instead.")
        return

    print(f"[tpu] Multi-host TPU: {num_workers} workers")

    if not _sync_code_to_workers(args.name, zone, num_workers):
        sys.exit(1)

    # Full setup: recreate venv + install jax[tpu]
    print(f"[tpu] Running full setup on all {num_workers} workers...")
    result = _run_all_workers(args.name, zone,
        "cd ~/needle && rm -rf .venv && source ./setup")
    if result.returncode != 0:
        print("[tpu] WARNING: setup failed on some workers.", file=sys.stderr)

    # Set up git config on all workers
    git_name, git_email = _collect_git_config()
    if git_name and git_email:
        _run_all_workers(args.name, zone,
            f'git config --global user.name "{git_name}" && '
            f'git config --global user.email "{git_email}"')

    print(f"[tpu] All {num_workers} workers ready.")


def tpu_sync(args):
    """Sync code to all workers without recreating venv.

    Fast alternative to `tpu setup` — just copies source code and runs
    `pip install -e .` to pick up changes. Use when the venv already exists.
    """
    zone = args.zone or _detect_zone(args.name)
    num_workers = _get_num_workers(args.name, zone)

    if num_workers <= 1:
        print(f"[tpu] Single-host TPU ({num_workers} worker). Use 'needle tpu connect' instead.")
        return

    print(f"[tpu] Multi-host TPU: {num_workers} workers (sync only)")

    if not _sync_code_to_workers(args.name, zone, num_workers):
        sys.exit(1)

    # Just reinstall the package to pick up code changes
    print(f"[tpu] Installing package on all {num_workers} workers...")
    result = _run_all_workers(args.name, zone,
        "cd ~/needle && .venv/bin/pip install -e . -q")
    if result.returncode != 0:
        print("[tpu] WARNING: pip install failed on some workers.", file=sys.stderr)

    print(f"[tpu] All {num_workers} workers synced.")


def _collect_env_exports():
    """Collect env vars to forward to remote workers (HF_TOKEN, WANDB_API_KEY, etc)."""
    exports = []
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY"):
        val = os.environ.get(var)
        if val:
            # Quote value to handle special characters in tokens
            safe_val = val.replace("'", "'\\''")
            exports.append(f"export {var}='{safe_val}'")
    return " && ".join(exports)


def tpu_train(args):
    """Launch training on all workers of a multi-host TPU."""
    zone = args.zone or _detect_zone(args.name)
    num_workers = _get_num_workers(args.name, zone)

    # Pass through any extra args after the name
    extra = getattr(args, "train_args", [])
    train_cmd = "cd ~/needle && .venv/bin/needle train"
    if extra:
        train_cmd += " " + " ".join(extra)

    # Forward auth env vars to workers
    env_exports = _collect_env_exports()
    if env_exports:
        env_count = len(env_exports.split(" && "))
        print(f"[tpu] Forwarding {env_count} env var(s) to workers")
        train_cmd = env_exports + " && " + train_cmd

    if num_workers <= 1:
        print(f"[tpu] Single-host TPU. Running training directly...")
        # Use quiet=True to avoid printing tokens, then log sanitized version
        print(f"[tpu] $ gcloud compute tpus tpu-vm ssh {args.name} --command 'needle train ...'")
        _run(
            ["gcloud", "compute", "tpus", "tpu-vm", "ssh", args.name,
             "--zone", zone, "--project", PROJECT,
             "--command", train_cmd],
            quiet=True,
        )
        return

    print(f"[tpu] Launching training on {num_workers} workers...")
    # Run with quiet=True to avoid printing tokens in command log
    result = _run(
        ["gcloud", "compute", "tpus", "tpu-vm", "ssh", args.name,
         "--zone", zone, "--project", PROJECT,
         "--worker=all",
         "--command", train_cmd],
        check=False, quiet=True,
    )
    if result.returncode != 0:
        print("[tpu] WARNING: training exited with errors on some workers.",
              file=sys.stderr)


def tpu_dispatch(args):
    actions = {
        "create": tpu_create,
        "connect": tpu_connect,
        "setup": tpu_setup,
        "sync": tpu_sync,
        "train": tpu_train,
        "claude": tpu_claude,
        "stop": tpu_stop,
        "start": tpu_start,
        "delete": tpu_delete,
        "list": tpu_list,
    }
    if not args.tpu_action or args.tpu_action not in actions:
        print(TPU_HELP)
        sys.exit(0 if not args.tpu_action else 1)
    actions[args.tpu_action](args)
