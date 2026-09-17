import json
import os
import tempfile
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_DIR = Path(__file__).parent
_STATIC = {"/": "index.html", "/index.html": "index.html",
           "/app.js": "app.js", "/style.css": "style.css"}
_CTYPE = {".html": "text/html; charset=utf-8",
          ".js": "application/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8"}
_DOWNLOADS = Path(tempfile.mkdtemp(prefix="needle-playground-"))

#: Providers the playground offers for data synthesis. The key never leaves the
#: server: the browser picks a provider and a model, never a credential.
_PROVIDER_VIEW = {
    "orcarouter": {
        "id": "orcarouter",
        "label": "OrcaRouter",
        "methods": ["api_key", "pkce"],
        "env_key": "ORCAROUTER_API_KEY",
        "key_url": "https://www.orcarouter.ai/console/authorized-apps",
        "logo": "https://www.orcarouter.ai/orca-logo-classic.png",
    },
    "openrouter": {
        "id": "openrouter",
        "label": "OpenRouter",
        "methods": ["api_key"],
        "env_key": "OPENROUTER_API_KEY",
        "key_url": "https://openrouter.ai/keys",
    },
}


def _provider_view():
    from ..model.credentials import CredentialStore, mask_secret
    from ..model.providers import get_provider, resolve_orcarouter_origins

    store = CredentialStore()
    auth_base, _ = resolve_orcarouter_origins()
    entries = []
    for provider_id, view in _PROVIDER_VIEW.items():
        provider = get_provider(provider_id)
        record = store.get(_account_for(provider_id))
        entry = dict(view)
        entry.update({
            "default_model": provider.default_model,
            "api_base": provider.api_base,
            "auth_base": provider.auth_base or auth_base if provider_id == "orcarouter" else None,
            "connected": bool(record and not record.get("needs_reauth")),
            "connection_degraded": bool(record and record.get("needs_reauth")),
            "masked_key": mask_secret(record.get("key")) if record else None,
            "key_source": record.get("source") if record else None,
            "key_generation": record.get("generation") if record else None,
        })
        entries.append(entry)
    return entries


def _account_for(provider_id):
    return provider_id


#: Last catalog we successfully fetched, per (provider, capability). Used only
#: when a refresh fails, and always flagged degraded.
_CATALOG_CACHE = {}


def _stored_key(provider_id):
    """The server-side credential for a provider, or None. Never logged."""
    from ..model.credentials import CredentialStore
    from ..model.credentials import CredentialError, credential_for
    try:
        return credential_for(provider_id, store=CredentialStore()).api_key
    except CredentialError:
        return None


def _model_catalog(provider_id, capability="chat", input_modality=None):
    """Minimal model metadata for the browser: ids and capabilities, no secrets.

    The API key stays here. A live success is authoritative; on failure we fall
    back to the last known-good catalog and then to the verified seed, and the
    response says which so the UI can show a degraded state instead of silently
    offering a stale or unverified list.
    """
    from ..model.providers import discover_models, filter_models, get_provider

    provider = get_provider(provider_id)
    # The requested modality is part of the cache identity: a last-known-good
    # list captured for one modality must never be replayed for another.
    cache_key = (provider.id, capability, input_modality)
    result = discover_models(provider, api_key=_stored_key(provider.id),
                             capability=capability,
                             last_known_good=_CATALOG_CACHE.get(cache_key))
    if not result.degraded:
        _CATALOG_CACHE[cache_key] = result.models
    modalities = [input_modality] if input_modality else None
    models = filter_models(result.models, capability=capability,
                           input_modalities=modalities)
    return {
        "provider": provider.id,
        "api_base": provider.api_base,
        "capability": capability,
        "input_modalities": modalities or [],
        "source": result.source,
        "degraded": result.degraded,
        "error": result.error,
        "total": len(result.models),
        "models": [{"id": m["id"],
                    "name": m.get("name"),
                    "context_length": m.get("context_length"),
                    "input_modalities": (m.get("architecture") or {}).get(
                        "input_modalities") or [],
                    "supported_endpoint_types": m.get("supported_endpoint_types") or [],
                    "verified": bool(m.get("verified"))} for m in models],
    }


# --------------------------------------------------------------------------
# Connect lifecycle. One attempt at a time, identified by a monotonically
# increasing attempt number so a late reply from an old attempt can never
# overwrite the state of a newer one.
# --------------------------------------------------------------------------

_CONNECT_LOCK = threading.Lock()
_CONNECT = {
    "attempt": 0,
    "status": "idle",     # idle | pending | done | failed | cancelled
    "flow": None,
    "url": None,
    "session": None,
    "masked": None,
    "source": None,
    "scope": None,
    "error": None,
    "hint": None,
}


def _connect_snapshot_locked():
    """Build the snapshot. The caller must already hold ``_CONNECT_LOCK``."""
    state = _CONNECT
    return {
        "attempt": state["attempt"],
        "status": state["status"],
        "flow": state["flow"],
        "url": state["url"],
        "masked_key": state["masked"],
        "source": state["source"],
        "scope": state["scope"],
        "error": state["error"],
        "hint": state["hint"],
    }


def _connect_snapshot():
    with _CONNECT_LOCK:
        return _connect_snapshot_locked()


def _connect_release(attempt=None):
    """Drop any listener held by the current (or given) attempt."""
    with _CONNECT_LOCK:
        if attempt is not None and _CONNECT["attempt"] != attempt:
            return
        session, _CONNECT["session"] = _CONNECT["session"], None
    if session is not None:
        session.close()


def _connect_settle_failed(message):
    """Record a failed attempt and return the browser-safe snapshot."""
    with _CONNECT_LOCK:
        _CONNECT.update(attempt=_CONNECT["attempt"] + 1, status="failed", flow=None,
                        url=None, session=None, masked=None, source=None,
                        error=str(message), hint=None)
        return _connect_snapshot_locked()


def _connect_start(provider_id, flow, api_key=None, app_name="Needle",
                   scope="api", login_hint=None, workspace_hint=None):
    """Start a connect attempt. Returns the snapshot the browser may hold."""
    from ..model.credentials import (CredentialError, CredentialStore, begin_connect,
                                     credential_from_api_key, mask_secret)

    if api_key:
        try:
            result = credential_from_api_key(api_key, store=CredentialStore(),
                                             account=_account_for(provider_id))
        except CredentialError as exc:
            # A bad key is a normal outcome of this route, not an exception
            # escaping to the caller: report it as a failed attempt, and never
            # echo the value back.
            return _connect_settle_failed(str(exc))
        with _CONNECT_LOCK:
            _CONNECT.update(attempt=_CONNECT["attempt"] + 1, status="done",
                            flow="api_key", url=None, session=None,
                            masked=mask_secret(result.api_key), source=result.source,
                            scope=result.scope, error=None,
                            hint="key stored on the server")
            return _connect_snapshot_locked()

    session, public = begin_connect(flow=flow, app_name=app_name, scope=scope,
                                    login_hint=login_hint,
                                    workspace_hint=workspace_hint)
    with _CONNECT_LOCK:
        previous, _CONNECT["session"] = _CONNECT["session"], session
        _CONNECT.update(attempt=_CONNECT["attempt"] + 1, status="pending",
                        flow=public["flow"], url=public["url"], masked=None,
                        source="pkce", scope=public["scope"], error=None,
                        hint="waiting for approval in the browser")
        attempt = _CONNECT["attempt"]
    if previous is not None:
        previous.close()

    if public["flow"] == "loopback":
        threading.Thread(target=_connect_await_callback,
                         args=(attempt, session), daemon=True).start()
    return _connect_snapshot()


def _connect_await_callback(attempt, session):
    """Flow A: wait for the loopback redirect, then exchange."""
    _connect_finish(attempt, session)


def _connect_finish(attempt, session, code=None):
    from ..model.credentials import CredentialError, CredentialStore, finish_connect, \
        mask_secret

    try:
        result = finish_connect(session, code=code, store=CredentialStore())
    except CredentialError as exc:
        _connect_settle(attempt, status="failed", error=str(exc))
        return
    except Exception as exc:  # never let a worker thread die silently
        _connect_settle(attempt, status="failed", error=str(exc))
        return
    _connect_settle(attempt, status="done", masked=mask_secret(result.api_key),
                    scope=result.scope,
                    hint=("granted scope is %r, not %r" % (result.scope, session.scope))
                    if result.scope_downgraded else "key stored on the server")


def _connect_claim_current(session):
    """True if ``session`` is still the live attempt, so only it may be exchanged."""
    with _CONNECT_LOCK:
        return _CONNECT["session"] is session and _CONNECT["status"] == "pending"


def _connect_settle(attempt, **fields):
    """Apply a result only if this attempt is still the current one."""
    with _CONNECT_LOCK:
        if _CONNECT["attempt"] != attempt or _CONNECT["status"] == "cancelled":
            return False
        _CONNECT["session"] = None
        _CONNECT.update(**fields)
        return True


def _connect_cancel(attempt=None):
    """Release the login lock. Called on cancel, popup close and pagehide."""
    with _CONNECT_LOCK:
        if attempt is not None and _CONNECT["attempt"] != attempt:
            # A stale cancel must not touch the attempt that replaced it.
            return _connect_snapshot_locked()
        _CONNECT["attempt"] += 1
        session, _CONNECT["session"] = _CONNECT["session"], None
        _CONNECT.update(status="cancelled", flow=None, url=None, error=None,
                        hint=None)
        snapshot = _connect_snapshot_locked()
    if session is not None:
        session.close()
    return snapshot


class Engine:
    def __init__(self, weights=None):
        self.weights = weights
        self.name = os.path.basename(weights) if weights else "needle-2 (base)"
        self.lock = threading.Lock()
        self.tools_json = None
        self.agent = None

    def load(self):
        from .. import Needle
        self.agent = Needle(tools="[]", weights=self.weights)
        self.tools_json = "[]"

    def complete(self, tools_json, query):
        from .. import Needle, _lib
        with self.lock:
            if self.agent is None or tools_json != self.tools_json:
                self.agent = Needle(tools=tools_json, weights=self.weights)
                self.tools_json = tools_json
            else:
                _lib().needle_reset()
            return self.agent.complete(query)

    def reset(self):
        from .. import _lib
        with self.lock:
            if self.agent is not None:
                _lib().needle_reset()
            self.agent = None
            self.tools_json = None

    def load_weights(self, path):
        with self.lock:
            self.weights = path
            self.name = os.path.basename(path)
            self.agent = None
            self.tools_json = None
        self.load()


_FT = {"running": False, "step": "", "log": [], "checkpoint": None, "error": None}


def _log(msg):
    _FT["log"].append(msg)
    if len(_FT["log"]) > 100:
        del _FT["log"][:-100]


def _finetune_worker(tools_json, api_key, samples, engine, provider_id=None,
                     image_url=None):
    import types
    try:
        _FT.update(running=True, step="generating data", log=[], checkpoint=None, error=None)
        from ..model.finetune import generate_dataset, finetune_local, build_main, DEFAULT_BASE
        from ..model.providers import get_provider

        provider = get_provider(provider_id)
        tools = json.loads(tools_json)
        _log("inputs: text" + (" + image" if image_url else ""))
        rows = generate_dataset(
            tools, samples, api_key=api_key, provider_id=provider.id,
            input_modality="image" if image_url else None, image_url=image_url,
            progress=lambda done, total: _log(f"generated {done}/{total}"))
        data_path = str(_DOWNLOADS / "needle_playground_data.jsonl")
        with open(data_path, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

        _FT["step"] = "training"
        adapter = str(_DOWNLOADS / "needle_playground_lora.safetensors")
        finetune_local(types.SimpleNamespace(
            jsonl_path=data_path, checkpoint=None, epochs=3, batch_size=16, lr=1e-4,
            lora_rank=16, lora_alpha=32.0, max_len=1024, generate=0, model=None,
            checkpoint_dir=str(_DOWNLOADS), out=adapter), progress=_log)

        _FT["step"] = "building"
        out = str(_DOWNLOADS / "needle_tuned.cact")
        build_main(types.SimpleNamespace(checkpoint=DEFAULT_BASE, lora=adapter,
                                         out=out, upload=False, bits=None))
        engine.load_weights(out)
        _FT.update(running=False, step="done", checkpoint=os.path.basename(out))
    except Exception as exc:
        _FT.update(running=False, step="failed", error=str(exc))
        _FT["log"].append(str(exc))


class _Handler(BaseHTTPRequestHandler):
    engine = None

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, (bytes, bytearray)) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in _STATIC:
            f = _DIR / _STATIC[path]
            self._send(200, f.read_bytes(), _CTYPE[f.suffix])
        elif path == "/model":
            self._send(200, json.dumps({"name": self.engine.name}))
        elif path == "/finetune/status":
            self._send(200, json.dumps(_FT))
        elif path == "/providers":
            self._send(200, json.dumps({"providers": _provider_view()}))
        elif path == "/provider/models":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1]) \
                if "?" in self.path else {}
            provider_id = (query.get("provider") or ["orcarouter"])[0]
            capability = (query.get("capability") or ["chat"])[0]
            modality = (query.get("input_modality") or [None])[0]
            self._send(200, json.dumps(_model_catalog(
                provider_id, capability, modality)))
        elif path == "/provider/connect/status":
            self._send(200, json.dumps(_connect_snapshot()))
        elif path.startswith("/download/"):
            name = os.path.basename(path[len("/download/"):])
            f = _DOWNLOADS / name
            if f.exists():
                self._send(200, f.read_bytes(), "application/octet-stream")
            else:
                self._send(404, b"not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        try:
            if self.path == "/complete":
                body = self._json_body()
                tools = body.get("tools", [])
                tools_json = tools if isinstance(tools, str) else json.dumps(tools)
                result = self.engine.complete(tools_json, body.get("query", ""))
                self._send(200, json.dumps(result))
            elif self.path == "/reset":
                self.engine.reset()
                self._send(200, json.dumps({"ok": True}))
            elif self.path == "/load-model":
                name = os.path.basename(self.headers.get("X-Filename", "model.cact"))
                length = int(self.headers.get("Content-Length", 0))
                dest = _DOWNLOADS / name
                dest.write_bytes(self.rfile.read(length))
                self.engine.load_weights(str(dest))
                self._send(200, json.dumps({"name": self.engine.name}))
            elif self.path == "/finetune":
                if _FT["running"]:
                    self._send(200, json.dumps({"error": "a finetune is already running"}))
                    return
                body = self._json_body()
                provider_id = (body.get("provider") or "orcarouter").strip()
                api_key = (body.get("api_key") or "").strip()
                # An explicit key wins; otherwise the server-side stored
                # credential is used and the browser never sees a secret.
                if not api_key:
                    api_key = _stored_key(provider_id)
                if not api_key:
                    self._send(200, json.dumps({
                        "error": "no credential for %s: connect an account or paste "
                                 "an API key" % provider_id}))
                    return
                tools = body.get("tools", "[]")
                tools_json = tools if isinstance(tools, str) else json.dumps(tools)
                samples = int(body.get("samples", 200))
                # An image URL is this entry point's non-text modality. It is
                # only passed on for OrcaRouter, which is the provider whose
                # catalog carries the modality metadata to validate it against.
                image_url = (body.get("image_url") or "").strip() or None
                threading.Thread(target=_finetune_worker,
                                 args=(tools_json, api_key, samples, self.engine,
                                       provider_id, image_url),
                                 daemon=True).start()
                self._send(200, json.dumps({"ok": True}))
            elif self.path == "/provider/connect":
                body = self._json_body()
                provider_id = (body.get("provider") or "orcarouter").strip()
                flow = (body.get("flow") or "loopback").strip()
                self._send(200, json.dumps(_connect_start(
                    provider_id, flow,
                    api_key=(body.get("api_key") or "").strip() or None,
                    app_name=(body.get("app_name") or "Needle"),
                    scope=(body.get("scope") or "api"),
                    login_hint=body.get("login_hint"),
                    workspace_hint=body.get("workspace_hint"))))
            elif self.path == "/provider/connect/code":
                body = self._json_body()
                code = (body.get("code") or "").strip()
                if not code:
                    self._send(200, json.dumps({"error": "no code supplied"}))
                    return
                with _CONNECT_LOCK:
                    attempt, session = _CONNECT["attempt"], _CONNECT["session"]
                if session is None or not _connect_claim_current(session):
                    self._send(200, json.dumps({"error": "no connect attempt is waiting "
                                                         "for a code"}))
                    return
                threading.Thread(target=_connect_finish,
                                 args=(attempt, session, code), daemon=True).start()
                self._send(200, json.dumps(_connect_snapshot()))
            elif self.path == "/provider/connect/cancel":
                body = self._json_body()
                attempt = body.get("attempt")
                self._send(200, json.dumps(
                    _connect_cancel(int(attempt) if attempt is not None else None)))
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as exc:
            self._send(200, json.dumps({"error": str(exc)}))

    def log_message(self, *args):
        pass


def main(args):
    engine = Engine(weights=getattr(args, "weights", None))
    print("needle playground: downloading and initializing the model...", flush=True)
    engine.load()
    _Handler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"needle playground ready: http://{args.host}:{args.port}  ({engine.name})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
