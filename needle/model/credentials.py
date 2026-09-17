"""Credentials for AI providers: where a key comes from and where it is kept.

Two adapters sit on one seam. :func:`credential_from_api_key` takes a key the
user already has; :func:`connect_loopback` / :func:`connect_oob` obtain one
through OAuth 2.0 + PKCE against the OrcaRouter consent screen. Both hand back
the same :class:`CredentialResult`, and nothing downstream — the synthesis
request, the model catalog, the CLI, the playground — can tell which was used.

A PKCE exchange returns a durable OrcaRouter API key, not a refresh token.
There is no refresh grant to call, so a rejected key is a terminal
reauthentication requirement (:meth:`CredentialStore.mark_needs_reauth`) rather
than something to retry in a loop.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Optional

from .providers import (DEFAULT_AUTH_BASE, ORCAROUTER_API_KEY_ENV, authorize_url,
                        exchange_url, get_provider, resolve_orcarouter_origins,
                        valid_callback_url)

PROVIDER_ID = "orcarouter"
OAUTH_PROVIDER_ID = "orcarouter-oauth"
APP_NAME = "Needle"
DEFAULT_SCOPE = "api"
CALLBACK_PATH = "/cb"
CONNECT_TIMEOUT_SECONDS = 600
#: Credentials are stored per provider, so the account key is the provider id.
ACCOUNT_DEFAULT = PROVIDER_ID

_KEY_PATTERN = re.compile(r"sk-orca-[A-Za-z0-9_\-]+")
_BEARER_PATTERN = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]+")


def mask_secret(secret: Optional[str]) -> str:
    """A short, non-reversible label safe for logs, errors, and screenshots."""
    if not secret:
        return "(none)"
    tail = secret[-4:] if len(secret) > 4 else ""
    return "sk-orca-...%s" % tail if tail else "(set)"


def sanitize(text: str) -> str:
    """Strip anything key-shaped out of a message before it is shown or logged."""
    text = _KEY_PATTERN.sub("sk-orca-[redacted]", str(text))
    return _BEARER_PATTERN.sub(r"\1 [redacted]", text)


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def generate_verifier() -> str:
    """A fresh high-entropy PKCE verifier, from a cryptographic RNG."""
    return _b64url(secrets.token_bytes(32))


def generate_state() -> str:
    return _b64url(secrets.token_bytes(16))


def code_challenge(verifier: str) -> str:
    """``base64url(sha256(verifier))`` with no padding."""
    return _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


class CredentialError(RuntimeError):
    """A credential could not be obtained or is no longer usable.

    ``needs_reauth`` marks the failures where the user must authorize again,
    as opposed to a transient network problem they can simply retry.
    """

    def __init__(self, message, needs_reauth=False, status=None):
        super().__init__(sanitize(message))
        self.needs_reauth = needs_reauth
        self.status = status


@dataclass
class CredentialResult:
    """What both adapters produce. Downstream code reads only ``api_key``."""

    provider_id: str
    api_key: str
    source: str                       # "api_key" | "pkce"
    scope: Optional[str] = None
    account: str = ACCOUNT_DEFAULT
    user_id: Optional[str] = None
    generation: int = 0
    scope_downgraded: bool = False

    @property
    def masked(self) -> str:
        return mask_secret(self.api_key)


# --------------------------------------------------------------------------
# Storage — the project's existing cache directory, no new dependency
# --------------------------------------------------------------------------

def cache_root() -> str:
    override = os.environ.get("NEEDLE_CACHE_DIR")
    if override:
        return os.path.expanduser(override)
    return os.path.join(os.path.expanduser("~"), ".cache", "cactus-needle")


def credentials_path() -> str:
    return os.path.join(cache_root(), "orcarouter", "credentials.json")


class CredentialStore:
    """Durable store for OrcaRouter keys, with generation-safe reauth marking."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or credentials_path()

    def _read(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return {"version": 1, "accounts": {}}
        if not isinstance(data, dict) or not isinstance(data.get("accounts"), dict):
            return {"version": 1, "accounts": {}}
        return data

    def _write(self, data: dict) -> None:
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        tmp = self.path + ".tmp"
        # Create with owner-only permissions before any secret is written, so
        # the key is never readable by another user even momentarily.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def put(self, api_key, source, scope=None, account=ACCOUNT_DEFAULT,
            user_id=None) -> int:
        """Store a key and return its new credential generation."""
        data = self._read()
        previous = data["accounts"].get(account) or {}
        generation = int(previous.get("generation") or 0) + 1
        data["accounts"][account] = {
            "key": api_key,
            "source": source,
            "scope": scope,
            "generation": generation,
            "user_id": user_id if user_id is not None else previous.get("user_id"),
            "needs_reauth": False,
        }
        self._write(data)
        return generation

    def get(self, account=ACCOUNT_DEFAULT) -> Optional[dict]:
        return self._read()["accounts"].get(account)

    def clear(self, account=ACCOUNT_DEFAULT) -> bool:
        data = self._read()
        if account not in data["accounts"]:
            return False
        del data["accounts"][account]
        self._write(data)
        return True

    def mark_needs_reauth(self, account=ACCOUNT_DEFAULT, generation=None) -> bool:
        """Flag one account's exact credential generation as rejected.

        A late failure from a request made with an older generation must not
        mark the credential that replaced it. Returns whether anything changed.
        """
        data = self._read()
        entry = data["accounts"].get(account)
        if not entry:
            return False
        if generation is not None and int(entry.get("generation") or 0) != int(generation):
            return False
        if entry.get("needs_reauth"):
            return False
        entry["needs_reauth"] = True
        self._write(data)
        return True


def mark_rejected_generation(credential, store=None) -> bool:
    """Record a terminal rejection against the credential that made the request.

    This is the transition a relay ``401`` must perform, and it is generation
    safe by construction: only a credential the store itself issued carries a
    generation, so a key that came from the environment — which the store never
    saw, and which may well belong to a different account — cannot mark the
    stored record that happens to sit under the same provider id.
    """
    if credential is None or not credential.generation:
        return False
    return (store or CredentialStore()).mark_needs_reauth(
        account=credential.account or ACCOUNT_DEFAULT,
        generation=credential.generation)


# --------------------------------------------------------------------------
# Adapter 1 — a key the user already has
# --------------------------------------------------------------------------

def read_api_key(env=None, provider_id=PROVIDER_ID) -> Optional[str]:
    env = os.environ if env is None else env
    provider = get_provider(provider_id)
    value = (env.get(provider.env_key) or "").strip()
    return value or None


def credential_from_api_key(api_key, store=None, account=ACCOUNT_DEFAULT,
                            persist=True) -> CredentialResult:
    """Adapter for ``ORCAROUTER_API_KEY`` / a pasted key.

    The ``sk-orca-`` prefix is a formatting check only — it is not proof the key
    works, and no paid request is sent to make a settings form say "valid".
    """
    api_key = (api_key or "").strip()
    if not api_key:
        raise CredentialError(
            "no OrcaRouter API key: set %s or paste a key" % ORCAROUTER_API_KEY_ENV)
    if not api_key.startswith("sk-orca-"):
        raise CredentialError(
            "that does not look like an OrcaRouter API key (expected an sk-orca- prefix)")
    store = store or CredentialStore()
    generation = store.put(api_key, "api_key", scope=DEFAULT_SCOPE, account=account) \
        if persist else 0
    return CredentialResult(provider_id=PROVIDER_ID, api_key=api_key, source="api_key",
                            scope=DEFAULT_SCOPE, account=account, generation=generation)


def credential_from_env(store=None, account=ACCOUNT_DEFAULT) -> CredentialResult:
    return credential_from_api_key(read_api_key(), store=store, account=account)


# --------------------------------------------------------------------------
# Adapter 2 — OAuth 2.0 + PKCE
# --------------------------------------------------------------------------

class AuthorizationSession:
    """One PKCE attempt. The verifier stays in this object and never leaves it."""

    def __init__(self, flow="loopback", auth_base=None, callback_url=None,
                 app_name=APP_NAME, scope=DEFAULT_SCOPE, login_hint=None,
                 workspace_hint=None, prompt=None):
        self.flow = flow
        self.auth_base = auth_base or resolve_orcarouter_origins()[0]
        self.app_name = app_name
        self.scope = scope
        # Fresh cryptographic randomness for every attempt.
        self.verifier = generate_verifier()
        self.state = generate_state()
        self.challenge = code_challenge(self.verifier)
        self._server = None
        self._port = None
        self._result: Dict[str, Optional[str]] = {}
        self._event = threading.Event()

        if flow == "loopback" and not callback_url:
            callback_url = self._bind_loopback()
        elif flow == "oob":
            callback_url = "oob"
        if not callback_url:
            raise CredentialError("unknown connect flow %r" % flow)
        if flow == "loopback" and not valid_callback_url(callback_url):
            self.close()
            raise CredentialError("callback URL is not a usable loopback address")
        self.callback_url = callback_url
        self.url = authorize_url(self.auth_base, callback_url, self.challenge,
                                 self.state, app_name, scope=scope,
                                 login_hint=login_hint,
                                 workspace_hint=workspace_hint, prompt=prompt)

    # -- Flow A listener ---------------------------------------------------

    def _bind_loopback(self) -> str:
        session = self

        class _Callback(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urllib.parse.urlsplit(self.path)
                if parsed.path != CALLBACK_PATH:
                    self.send_response(404)
                    self.end_headers()
                    return
                params = urllib.parse.parse_qs(parsed.query)
                session._result = {
                    "state": (params.get("state") or [None])[0],
                    "code": (params.get("code") or [None])[0],
                    "error": (params.get("error") or [None])[0],
                }
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b"<p>Connected. You can close this tab and return to Needle.</p>")
                session._event.set()

            def log_message(self, *args):
                pass

        # Listen first so the port is known before anything is opened.
        self._server = HTTPServer(("127.0.0.1", 0), _Callback)
        self._port = self._server.server_address[1]
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()
        return "http://127.0.0.1:%d%s" % (self._port, CALLBACK_PATH)

    def wait_for_code(self, timeout=CONNECT_TIMEOUT_SECONDS) -> str:
        """Block until the browser delivers a code. Flow A only."""
        if self.flow != "loopback":
            raise CredentialError("wait_for_code is only used by the loopback flow")
        if not self._event.wait(timeout):
            raise CredentialError(
                "timed out waiting for authorization; start a new connection")
        # The state comparison happens before the code is touched, in constant
        # time. This is what stops another page dropping a code on our listener.
        received_state = self._result.get("state") or ""
        if not hmac.compare_digest(received_state, self.state):
            raise CredentialError(
                "authorization response failed the state check; nothing was exchanged")
        error = self._result.get("error")
        if error:
            raise CredentialError("authorization was refused (%s)" % sanitize(error))
        code = self._result.get("code")
        if not code:
            raise CredentialError("authorization response carried no code")
        return code

    def close(self) -> None:
        """Release the listener. Safe to call more than once."""
        server, self._server = self._server, None
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise
        self._event.set()


def exchange_code(code, verifier, auth_base=None, provider_id=PROVIDER_ID,
                  store=None, account=ACCOUNT_DEFAULT, requested_scope=DEFAULT_SCOPE,
                  opener=None, persist=True) -> CredentialResult:
    """Redeem an auth code for a durable OrcaRouter API key."""
    if not code:
        raise CredentialError("no authorization code to exchange")
    auth_base = auth_base or resolve_orcarouter_origins()[0]
    body = json.dumps({
        "code": code,
        "code_verifier": verifier,
        "code_challenge_method": "S256",
    }).encode("utf-8")
    request = urllib.request.Request(exchange_url(auth_base), data=body, headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(request, timeout=60) as response:
            payload = json.loads(response.read(1024 * 1024).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise _exchange_error(exc) from None
    except (urllib.error.URLError, OSError) as exc:
        raise CredentialError(
            "could not reach the OrcaRouter authorization service (%s); "
            "check your network and try again" % type(exc).__name__) from None
    except (ValueError, json.JSONDecodeError):
        raise CredentialError(
            "the authorization service returned a response we could not read") from None

    api_key = payload.get("key") if isinstance(payload, dict) else None
    if not api_key:
        raise CredentialError(
            "authorization succeeded but no key was returned; please try again",
            needs_reauth=True)

    # The response says what was *granted*. Believe it, not what we asked for.
    granted = payload.get("scope")
    downgraded = bool(granted) and granted != requested_scope
    user_id = str(payload.get("user_id")) if isinstance(payload, dict) and \
        payload.get("user_id") is not None else None
    store = store or CredentialStore()
    generation = store.put(api_key, "pkce", scope=granted, account=account,
                           user_id=user_id) if persist else 0
    return CredentialResult(provider_id=provider_id, api_key=api_key, source="pkce",
                            scope=granted, account=account, user_id=user_id,
                            generation=generation, scope_downgraded=downgraded)


def _exchange_error(exc) -> CredentialError:
    status = getattr(exc, "code", None)
    try:
        detail = json.loads(exc.read(64 * 1024).decode("utf-8"))
        message = detail.get("error_description") or detail.get("error") or ""
    except Exception:
        message = ""
    if status == 400:
        return CredentialError(
            "the authorization service rejected the PKCE parameters (%s)"
            % (sanitize(message) or "bad request"), status=status)
    if status == 403:
        return CredentialError(
            "the authorization code is unknown, expired, or already used, or the "
            "verifier did not match; start a new connection", needs_reauth=True,
            status=status)
    if status == 429:
        return CredentialError(
            "OrcaRouter is rate limiting new authorizations for this account; "
            "wait a little and try again, or paste an existing API key",
            status=status)
    return CredentialError(
        "authorization exchange failed with HTTP %s%s"
        % (status, (" (%s)" % sanitize(message)) if message else ""), status=status)


def begin_connect(flow="loopback", store=None, account=ACCOUNT_DEFAULT, **kwargs):
    """Start a connect attempt. Returns ``(session, public_view)``.

    The public view is what a UI may hold: it carries the URL and the attempt's
    identity but never the verifier.
    """
    session = AuthorizationSession(flow=flow, **kwargs)
    public = {
        "url": session.url,
        "callback_url": session.callback_url,
        "flow": session.flow,
        "state": session.state,
        "app_name": session.app_name,
        "scope": session.scope,
        "provider_id": OAUTH_PROVIDER_ID,
    }
    return session, public


def normalise_callback(requested, host, port) -> Optional[str]:
    """Flow B is chosen when a loopback callback cannot be received.

    A playground bound to a non-loopback host hands the browser an address on
    the *server's* loopback interface, which the browser cannot reach, so the
    out-of-band flow is used instead.
    """
    if requested:
        if not valid_callback_url(requested):
            raise CredentialError("callback URL must be https, or http on loopback")
        return requested
    if host in ("127.0.0.1", "localhost", "::1"):
        return None  # let the session bind its own ephemeral port
    return "oob"


def finish_connect(session, code=None, store=None, account=ACCOUNT_DEFAULT,
                   timeout=CONNECT_TIMEOUT_SECONDS, persist=True, opener=None) -> CredentialResult:
    """Collect the code (waiting for the loopback callback when needed) and exchange."""
    try:
        if code is None:
            code = session.wait_for_code(timeout=timeout)
    finally:
        session.close()
    return exchange_code(code, session.verifier, auth_base=session.auth_base,
                         store=store, account=account,
                         requested_scope=session.scope, persist=persist, opener=opener)


def credential_for(provider_id=None, store=None, environ=None,
                   account=None) -> CredentialResult:
    """Resolve a credential for a provider: environment first, then the store.

    Shared by the synthesis request, the model catalog and the playground, so
    none of them grows its own copy of the authentication logic. Consulting the
    store is what makes a PKCE-issued key survive a restart: it is reused until
    OrcaRouter revokes it, never re-minted on launch.
    """
    provider = get_provider(provider_id)
    env = os.environ if environ is None else environ
    value = (env.get(provider.env_key) or "").strip()
    if value:
        return CredentialResult(provider_id=provider.id, api_key=value,
                                source="api_key", account=account or provider.id)

    record = (store or CredentialStore()).get(account or provider.id)
    if record and record.get("key"):
        if record.get("needs_reauth"):
            raise CredentialError(
                "the stored %s credential was rejected and needs reauthorization; "
                "run `needle connect` or paste a new key" % provider.label,
                needs_reauth=True)
        return CredentialResult(provider_id=provider.id, api_key=record["key"],
                                source=record.get("source") or "api_key",
                                scope=record.get("scope"),
                                account=record.get("account") or account or provider.id,
                                generation=int(record.get("generation") or 0))

    raise CredentialError(
        "no API key for %s: set %s or connect an account"
        % (provider.label, provider.env_key))
