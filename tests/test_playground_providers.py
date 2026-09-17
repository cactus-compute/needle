"""Playground server tests: provider discovery, the connect lock, and lifecycle.

These run against a real ``ThreadingHTTPServer`` on an ephemeral loopback port,
so the assertions cover the actual HTTP contract the browser uses.
"""

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from needle.playground import server as pg

FAKE_KEY = "sk-orca-33333333333333333333333333333333"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("NEEDLE_CACHE_DIR", str(tmp_path))
    for name in ("ORCA_BASE_URL", "ORCA_AUTH_BASE_URL", "ORCA_API_BASE_URL",
                 "ORCAROUTER_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_URL"):
        monkeypatch.delenv(name, raising=False)
    pg._CATALOG_CACHE.clear()
    pg._CONNECT.update(attempt=0, status="idle", flow=None, url=None, session=None,
                       masked=None, source=None, scope=None, error=None, hint=None)
    yield


@pytest.fixture
def http():
    """A live playground server; returns (base_url, request_helper)."""
    httpd = pg.ThreadingHTTPServer(("127.0.0.1", 0), pg._Handler)
    pg._Handler.engine = None
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:%d" % httpd.server_address[1]

    # Bind the real opener now: catalog stubs patch ``urllib.request.urlopen``,
    # and the test's own client must not be intercepted by them.
    real_urlopen = urllib.request.urlopen

    def request(path, body=None, method=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with real_urlopen(req, timeout=10) as response:
                return response.status, json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode()

    yield base, request
    httpd.shutdown()
    httpd.server_close()


def _fake_catalog(monkeypatch, models=None, error=None):
    """Point discovery at a fixed catalog without touching the network."""
    payload = models if models is not None else [
        {"id": "vendor/chat-text", "supported_endpoint_types": ["openai"],
         "architecture": {"input_modalities": ["text"]}},
        {"id": "vendor/chat-vision", "supported_endpoint_types": ["openai"],
         "architecture": {"input_modalities": ["text", "image"]}},
        {"id": "vendor/image-gen", "supported_endpoint_types": ["image-generation"]},
    ]

    class _Response:
        def read(self, *_a):
            return json.dumps({"data": payload}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def urlopen(request, timeout=None):
        if error is not None:
            raise error
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


# ---------------------------------------------------------------------------
# Provider surface
# ---------------------------------------------------------------------------

def test_providers_endpoint_exposes_both_auth_methods_for_orcarouter(http):
    _, request = http
    status, body = request("/providers")
    assert status == 200
    by_id = {p["id"]: p for p in body["providers"]}
    assert by_id["orcarouter"]["methods"] == ["api_key", "pkce"]
    assert by_id["orcarouter"]["label"] == "OrcaRouter"
    assert by_id["orcarouter"]["api_base"] == "https://api.orcarouter.ai/v1"
    assert by_id["orcarouter"]["env_key"] == "ORCAROUTER_API_KEY"
    # OpenRouter keeps its single API-key method.
    assert by_id["openrouter"]["methods"] == ["api_key"]


def test_a_stored_key_is_reported_masked_never_in_full(http, monkeypatch):
    from needle.model.credentials import CredentialStore, credential_from_api_key
    credential_from_api_key(FAKE_KEY, store=CredentialStore())

    _, request = http
    status, body = request("/providers")
    entries = {p["id"]: p for p in body["providers"]}
    orca = entries["orcarouter"]
    assert orca["connected"] is True
    assert FAKE_KEY not in json.dumps(body)
    assert orca["masked_key"].startswith("sk-orca-")
    assert orca["key_source"] == "api_key"


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def test_model_catalog_returns_minimal_metadata_and_no_secrets(http, monkeypatch):
    from needle.model.credentials import CredentialStore, credential_from_api_key
    credential_from_api_key(FAKE_KEY, store=CredentialStore())
    _fake_catalog(monkeypatch)

    _, request = http
    status, body = request("/provider/models?provider=orcarouter&capability=chat")
    assert status == 200
    assert body["source"] == "live"
    assert body["degraded"] is False
    assert FAKE_KEY not in json.dumps(body)
    assert {m["id"] for m in body["models"]} == {"vendor/chat-text", "vendor/chat-vision"}
    # The browser gets metadata, not pricing or credentials.
    for model in body["models"]:
        assert set(model) == {"id", "name", "context_length", "input_modalities",
                              "supported_endpoint_types", "verified"}


def test_model_catalog_filters_by_capability_not_by_name(http, monkeypatch):
    _fake_catalog(monkeypatch)
    _, request = http
    _, chat = request("/provider/models?provider=orcarouter&capability=chat")
    _, image = request("/provider/models?provider=orcarouter&capability=image")
    assert {m["id"] for m in chat["models"]} == {"vendor/chat-text", "vendor/chat-vision"}
    assert {m["id"] for m in image["models"]} == {"vendor/image-gen"}


def test_model_catalog_multimodal_filter_is_fail_closed(http, monkeypatch):
    _fake_catalog(monkeypatch)
    _, request = http
    _, body = request("/provider/models?provider=orcarouter&capability=chat"
                      "&input_modality=image")
    assert {m["id"] for m in body["models"]} == {"vendor/chat-vision"}


def test_model_catalog_reports_degraded_state_when_discovery_fails(http, monkeypatch):
    _fake_catalog(monkeypatch, error=urllib.error.URLError("offline"))
    _, request = http
    status, body = request("/provider/models?provider=orcarouter&capability=chat")
    assert status == 200
    assert body["degraded"] is True
    assert body["source"] == "seed"
    assert body["error"]
    # The verified seed is offered and explicitly marked as unverified-live.
    ids = [m["id"] for m in body["models"]]
    assert ids, "a degraded catalog must not be empty"
    assert all(m["verified"] for m in body["models"])


def test_a_live_catalog_replaces_the_cached_degraded_one(http, monkeypatch):
    _fake_catalog(monkeypatch, error=urllib.error.URLError("offline"))
    _, request = http
    _, degraded = request("/provider/models?provider=orcarouter&capability=chat")
    assert degraded["degraded"] is True

    _fake_catalog(monkeypatch)
    _, live = request("/provider/models?provider=orcarouter&capability=chat")
    assert live["degraded"] is False
    assert live["source"] == "live"
    # A live success is authoritative: no seeded id survives into it.
    assert not any(m["id"].startswith("openai/gpt-5.5") for m in live["models"])


# ---------------------------------------------------------------------------
# Connect lifecycle and the login lock
# ---------------------------------------------------------------------------

def test_storing_an_api_key_through_the_connect_route(http):
    from needle.model.credentials import CredentialStore
    _, request = http
    status, body = request("/provider/connect",
                           {"provider": "orcarouter", "api_key": FAKE_KEY})
    assert status == 200
    assert body["status"] == "done"
    assert body["source"] == "api_key"
    assert FAKE_KEY not in json.dumps(body)
    assert CredentialStore().get()["key"] == FAKE_KEY


def test_an_invalid_api_key_is_rejected_without_echoing_it(http):
    _, request = http
    _, body = request("/provider/connect",
                      {"provider": "orcarouter", "api_key": "sk-or-not-orca"})
    assert body["status"] == "failed"
    assert FAKE_KEY not in json.dumps(body)


def test_starting_a_pkce_connect_exposes_a_url_and_never_the_verifier(http, monkeypatch):
    _, request = http
    status, body = request("/provider/connect",
                           {"provider": "orcarouter", "flow": "loopback"})
    assert status == 200
    assert body["status"] == "pending"
    assert body["flow"] == "loopback"
    assert body["url"].startswith("https://www.orcarouter.ai/auth?")
    parsed = urllib.parse.urlsplit(body["url"])
    params = urllib.parse.parse_qs(parsed.query)
    assert params["code_challenge_method"] == ["S256"]
    assert params["callback_url"][0].startswith("http://127.0.0.1:")
    # The verifier lives only in the server's session object.
    with pg._CONNECT_LOCK:
        verifier = pg._CONNECT["session"].verifier
    assert verifier not in body["url"]
    assert verifier not in json.dumps(body)
    pg._connect_cancel()


def test_cancel_releases_the_lock_and_a_second_login_can_start(http):
    _, request = http
    _, started = request("/provider/connect",
                         {"provider": "orcarouter", "flow": "loopback"})
    assert started["status"] == "pending"

    _, cancelled = request("/provider/connect/cancel",
                           {"attempt": started["attempt"]})
    assert cancelled["status"] == "cancelled"
    assert pg._CONNECT["session"] is None

    _, second = request("/provider/connect",
                        {"provider": "orcarouter", "flow": "loopback"})
    assert second["status"] == "pending"
    assert second["attempt"] > started["attempt"]
    pg._connect_cancel()


def test_cancelling_a_stale_attempt_does_not_kill_the_current_one(http):
    _, request = http
    _, first = request("/provider/connect", {"provider": "orcarouter", "flow": "loopback"})
    _, second = request("/provider/connect", {"provider": "orcarouter", "flow": "loopback"})
    assert second["attempt"] > first["attempt"]

    # A late cancel carrying the first attempt's id must be ignored.
    _, result = request("/provider/connect/cancel", {"attempt": first["attempt"]})
    assert result["status"] == "pending"
    assert result["attempt"] == second["attempt"]
    pg._connect_cancel()


def test_a_late_result_from_an_old_attempt_cannot_overwrite_a_new_one(http):
    _, request = http
    _, first = request("/provider/connect", {"provider": "orcarouter", "flow": "loopback"})
    _, second = request("/provider/connect", {"provider": "orcarouter", "flow": "loopback"})

    # Settle the *old* attempt: it must be a no-op.
    applied = pg._connect_settle(first["attempt"], status="done",
                                 masked="sk-orca-...dead", scope="api")
    assert applied is False
    assert pg._connect_snapshot()["attempt"] == second["attempt"]
    assert pg._connect_snapshot()["masked_key"] is None
    pg._connect_cancel()


def test_status_reports_the_authorization_hint_while_pending(http):
    _, request = http
    request("/provider/connect", {"provider": "orcarouter", "flow": "loopback"})
    _, status = request("/provider/connect/status")
    assert status["status"] == "pending"
    assert status["hint"]
    pg._connect_cancel()


def test_oob_code_submission_without_a_pending_attempt_is_refused(http):
    _, request = http
    _, body = request("/provider/connect/code", {"code": "abc"})
    assert "error" in body


def test_an_empty_oob_code_is_refused(http):
    _, request = http
    request("/provider/connect", {"provider": "orcarouter", "flow": "loopback"})
    _, body = request("/provider/connect/code", {"code": "   "})
    assert "error" in body
    pg._connect_cancel()


def test_a_denied_authorization_clears_the_pending_state(http):
    """Denial must be terminal: the lock is released and the attempt reports it."""
    _, request = http
    request("/provider/connect", {"provider": "orcarouter", "flow": "loopback"})
    with pg._CONNECT_LOCK:
        session = pg._CONNECT["session"]

    # Drive the real loopback listener with a denial, exactly as a browser would.
    denial = urllib.parse.urlencode({"error": "access_denied", "state": session.state})
    try:
        urllib.request.urlopen(session.callback_url + "?" + denial, timeout=10).read()
    except urllib.error.URLError:
        pass

    for _ in range(200):
        if pg._connect_snapshot()["status"] != "pending":
            break
        threading.Event().wait(0.05)

    snapshot = pg._connect_snapshot()
    assert snapshot["status"] == "failed"
    assert snapshot["error"]
    assert pg._CONNECT["session"] is None
    assert FAKE_KEY not in json.dumps(snapshot)


def test_a_state_mismatch_releases_the_lock_and_reports_failure(http):
    """Every terminal path must free the login lock, not just the happy one."""
    _, request = http
    request("/provider/connect", {"provider": "orcarouter", "flow": "loopback"})
    with pg._CONNECT_LOCK:
        session = pg._CONNECT["session"]

    # A callback carrying somebody else's state: refused before any exchange.
    mismatch = urllib.parse.urlencode({"code": "attacker-code", "state": "not-ours"})
    try:
        urllib.request.urlopen(session.callback_url + "?" + mismatch, timeout=10).read()
    except urllib.error.URLError:
        pass

    for _ in range(200):
        if pg._connect_snapshot()["status"] != "pending":
            break
        threading.Event().wait(0.05)

    snapshot = pg._connect_snapshot()
    assert snapshot["status"] == "failed"
    assert snapshot["error"]
    assert pg._CONNECT["session"] is None

    # And a fresh login can start without remounting anything.
    _, second = request("/provider/connect", {"provider": "orcarouter", "flow": "loopback"})
    assert second["status"] == "pending"
    pg._connect_cancel()


# ---------------------------------------------------------------------------
# Finetune route: no credential crosses the browser boundary
# ---------------------------------------------------------------------------

def test_finetune_requires_a_credential_but_never_accepts_one_inline(http, monkeypatch):
    _, request = http
    _, body = request("/finetune", {"tools": "[]", "provider": "orcarouter",
                                    "samples": 25})
    assert "error" in body
    assert "credential" in body["error"]


def test_finetune_uses_the_server_side_stored_credential(http, monkeypatch):
    from needle.model.credentials import CredentialStore, credential_from_api_key
    credential_from_api_key(FAKE_KEY, store=CredentialStore())

    started = {}
    done = threading.Event()

    def fake_worker(tools_json, api_key, samples, engine, provider_id=None,
                    image_url=None):
        started.update(api_key=api_key, provider_id=provider_id)
        done.set()

    monkeypatch.setattr(pg, "_finetune_worker", fake_worker)

    _, request = http
    _, body = request("/finetune", {"tools": "[]", "provider": "orcarouter",
                                    "samples": 25})
    assert body.get("ok") is True
    assert done.wait(10), "the finetune worker never ran"
    assert started["api_key"] == FAKE_KEY
    assert started["provider_id"] == "orcarouter"


def test_an_image_attachment_narrows_the_selector_options(http, monkeypatch):
    """What the selector is handed is the filtered list, not a superset."""
    _fake_catalog(monkeypatch)
    _, request = http
    _, text = request("/provider/models?provider=orcarouter&capability=chat")
    _, image = request("/provider/models?provider=orcarouter&capability=chat"
                       "&input_modality=image")
    assert {m["id"] for m in text["models"]} == {"vendor/chat-text", "vendor/chat-vision"}
    # Only the model that declares an image input survives.
    assert [m["id"] for m in image["models"]] == ["vendor/chat-vision"]
    assert image["input_modalities"] == ["image"]
    assert all("image" in m["input_modalities"] for m in image["models"])


def test_an_undeclared_modality_model_never_enters_the_image_selector(http, monkeypatch):
    _fake_catalog(monkeypatch, models=[
        {"id": "vendor/no-arch", "supported_endpoint_types": ["openai"]},
        {"id": "vendor/chat-vision", "supported_endpoint_types": ["openai"],
         "architecture": {"input_modalities": ["text", "image"]}},
    ])
    _, request = http
    _, body = request("/provider/models?provider=orcarouter&capability=chat"
                      "&input_modality=image")
    assert [m["id"] for m in body["models"]] == ["vendor/chat-vision"]


def test_a_last_known_good_catalog_is_not_replayed_across_modalities(http, monkeypatch):
    """A cached text-only list must never answer an image request."""
    _fake_catalog(monkeypatch)
    _, request = http
    request("/provider/models?provider=orcarouter&capability=chat")

    _fake_catalog(monkeypatch, error=urllib.error.URLError("offline"))
    _, body = request("/provider/models?provider=orcarouter&capability=chat"
                      "&input_modality=image")
    assert body["degraded"] is True
    assert body["source"] == "seed"
    # The fallback is still filtered, so nothing text-only leaks in.
    assert all("image" in m["input_modalities"] for m in body["models"])


def test_finetune_forwards_an_image_attachment_to_generation(http, monkeypatch):
    from needle.model.credentials import CredentialStore
    seen = {}

    def fake_worker(tools_json, api_key, samples, engine, provider_id=None,
                    image_url=None):
        seen.update(api_key=api_key, provider_id=provider_id, image_url=image_url)

    monkeypatch.setattr(pg, "_finetune_worker", fake_worker)
    pg._FT.update(running=False)
    CredentialStore().put(FAKE_KEY, "api_key")
    _, request = http
    status, body = request("/finetune", {
        "tools": "[]", "provider": "orcarouter", "model": "vendor/chat-vision",
        "samples": 25, "image_url": "https://example.invalid/a.png"})
    assert body.get("ok") is True
    deadline = time.time() + 5
    while not seen and time.time() < deadline:
        time.sleep(0.02)
    assert seen["image_url"] == "https://example.invalid/a.png"
    assert seen["provider_id"] == "orcarouter"
    assert seen["api_key"] == FAKE_KEY
