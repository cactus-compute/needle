import getpass
import os
import re
import subprocess
import sys

PROJECT = "needle-488623"

VERSION_FOR_TYPE = {
    "v2":         "tpu-ubuntu2204-base",
    "v3":         "tpu-ubuntu2204-base",
    "v4":         "tpu-ubuntu2204-base",
    "v5litepod":  "v2-alpha-tpuv5-lite",
    "v5e":        "v2-alpha-tpuv5-lite",
    "v5p":        "v2-alpha-tpuv5",
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


ZONES = [
    "us-central1-a", "us-central1-b", "us-central1-f",
    "us-east1-b", "us-east1-c", "us-east1-d",
    "us-east5-a", "us-east5-b",
    "us-south1-a", "us-south1-b",
    "us-west1-a", "us-west1-b",
    "us-west4-a",
]

TPU_HELP = """
  tpu commands:
    needle tpu create NAME [--type TYPE] [--version VER]
    needle tpu connect NAME [--zone ZONE]
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
    """Prompt for git user.name and user.email, with defaults from local git config."""
    local_name = ""
    local_email = ""
    try:
        local_name = subprocess.run(
            ["git", "config", "user.name"], capture_output=True, text=True
        ).stdout.strip()
        local_email = subprocess.run(
            ["git", "config", "user.email"], capture_output=True, text=True
        ).stdout.strip()
    except FileNotFoundError:
        pass

    prompt_name = f"  git user.name [{local_name}]: " if local_name else "  git user.name: "
    prompt_email = f"  git user.email [{local_email}]: " if local_email else "  git user.email: "

    print("[tpu] Configure git for the instance:")
    name = input(prompt_name).strip() or local_name
    email = input(prompt_email).strip() or local_email

    if not name or not email:
        print("[tpu] Skipping git config (name or email empty).")
        return None, None
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
    print(f"[tpu] Accelerator: {args.accel_type}, software version: {version}")

    git_name, git_email = _collect_git_config()

    for zone in ZONES:
        print(f"[tpu] Trying {zone}...")
        result = _run(
            ["gcloud", "compute", "tpus", "tpu-vm", "create", args.name,
             "--zone", zone,
             "--accelerator-type", args.accel_type,
             "--version", version,
             "--project", PROJECT],
            check=False, capture=True,
        )
        if result.returncode == 0:
            print(f"[tpu] SUCCESS: created '{args.name}' in {zone}")
            args.zone = zone
            _check_tpu_health(args.name, zone)
            _setup_git_on_instance(args.name, zone, git_name, git_email)
            tpu_claude(args)
            tpu_connect(args)
            return
        stderr = result.stderr.strip()
        last_line = stderr.splitlines()[-1] if stderr else "unknown error"
        print(f"[tpu] {zone}: {last_line}")
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


def tpu_start(args):
    zone = args.zone or _detect_zone(args.name)
    _run(["gcloud", "compute", "tpus", "tpu-vm", "start", args.name,
          "--zone", zone, "--project", PROJECT])
    print(f"[tpu] Started '{args.name}'")


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


def tpu_dispatch(args):
    actions = {
        "create": tpu_create,
        "connect": tpu_connect,
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
