from __future__ import annotations

import json
import os
import platform
import sys
import threading
import urllib.request
import uuid
from pathlib import Path

# Anonymous usage counts only (event name, versions, OS/arch, random install
# id) - never prompts, outputs, or paths; disclosed in the README and by the
# first-run notice. Sends run fire-and-forget on a daemon thread and must
# never raise into caller code.

ENDPOINT = os.environ.get(
    "NEEDLE_TELEMETRY_URL",
    "https://mamakztargnvtkhqjynx.supabase.co/functions/v1/telemetry",
)

_NOTICE = ("cactus-needle collects anonymous usage counts (function name, "
           "version, OS) to guide development; no prompts or outputs. "
           "Disable with NEEDLE_TELEMETRY=0.")

_lock = threading.Lock()
_anon_id: str | None = None


def _enabled() -> bool:
    if os.environ.get("NEEDLE_TELEMETRY", "1") == "0":
        return False
    if os.environ.get("DO_NOT_TRACK"):
        return False
    if os.environ.get("CI"):
        return False
    return True


def _get_anon_id() -> str:
    global _anon_id
    with _lock:
        if _anon_id:
            return _anon_id
        path = Path.home() / ".cactus_needle" / "telemetry_id"
        try:
            if path.exists():
                _anon_id = path.read_text().strip() or "ephemeral"
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                _anon_id = uuid.uuid4().hex
                path.write_text(_anon_id)
                print(_NOTICE, file=sys.stderr)
        except OSError:
            _anon_id = "ephemeral"
        return _anon_id


def _send(event: str, props: dict | None) -> None:
    try:
        from . import __version__
        from .agent.fetch import ENGINE_VERSION
        payload = json.dumps({
            "event": event,
            "anon_id": _get_anon_id(),
            "version": __version__,
            "engine": ENGINE_VERSION,
            "os": platform.system(),
            "arch": platform.machine(),
            "python": platform.python_version(),
            "props": props or {},
        }).encode("utf-8")
        req = urllib.request.Request(
            ENDPOINT, data=payload,
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=3)
    except Exception:
        pass


def track(event: str, props: dict | None = None) -> None:
    if not _enabled():
        return
    threading.Thread(target=_send, args=(event, props), daemon=True).start()
