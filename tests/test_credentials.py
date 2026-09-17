"""Credential seam tests: the API-key adapter, the PKCE adapter, and storage.

Every fixture here is a fake key or a fake code. A real sk-orca- key is never
written to a file, a log, or an assertion message.
"""

import base64
import hashlib
import json
import threading
import urllib.error
import urllib.parse
import urllib.request

import pytest

from needle.model import credentials as creds
from needle.model import providers

FAKE_KEY = "sk-orca-00000000000000000000000000000000"
FAKE_KEY_2 = "sk-orca-11111111111111111111111111111111"


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Keep every test's credentials out of the real user cache."""
    monkeypatch.setenv("NEEDLE_CACHE_DIR", str(tmp_path))
    for name in ("ORCA_BASE_URL", "ORCA_AUTH_BASE_URL", "ORCA_API_BASE_URL",
                 "ORCAROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    yield


# ---------------------------------------------------------------------------
# API-key adapter
# ---------------------------------------------------------------------------

def test_api_key_adapter_returns_a_credential_and_persists_it():
    store = creds.CredentialStore()
    result = creds.credential_from_api_key(FAKE_KEY, store=store)
    assert result.api_key == FAKE_KEY
    assert result.source == "api_key"
    assert result.generation == 1

    stored = store.get()
    assert stored["key"] == FAKE_KEY
    assert stored["needs_reauth"] is False


def test_api_key_adapter_reads_the_environment(monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", FAKE_KEY)
    result = creds.credential_from_env(store=creds.CredentialStore())
    assert result.api_key == FAKE_KEY
    assert result.source == "api_key"


def test_api_key_adapter_rejects_absent_and_malformed_keys():
    with pytest.raises(creds.CredentialError):
        creds.credential_from_api_key("   ")
    with pytest.raises(creds.CredentialError):
        creds.credential_from_api_key("not-a-key")


def test_api_key_adapter_clear_removes_the_secret():
    store = creds.CredentialStore()
    creds.credential_from_api_key(FAKE_KEY, store=store)
    assert store.get() is not None
    assert store.clear() is True
    assert store.get() is None
    assert store.clear() is False


def test_credentials_file_is_owner_only(tmp_path):
    import os
    import stat

    path = creds.credentials_path()
    creds.credential_from_api_key(FAKE_KEY, store=creds.CredentialStore())
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, oct(mode)


def test_mask_never_reveals_the_key():
    masked = creds.mask_secret(FAKE_KEY)
    assert FAKE_KEY not in masked
    assert masked.startswith("sk-orca-")
    assert creds.mask_secret(None) == "(none)"


def test_sanitize_strips_key_shaped_text_and_bearer_headers():
    assert FAKE_KEY not in creds.sanitize("failed with " + FAKE_KEY)
    assert "sk-orca-" not in creds.sanitize("Authorization: Bearer " + FAKE_KEY)
    assert "abc123" not in creds.sanitize("Authorization: Bearer abc123")


# ---------------------------------------------------------------------------
# PKCE primitives
# ---------------------------------------------------------------------------

def _b64url_decode(value):
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def test_challenge_is_unpadded_base64url_sha256_of_the_verifier():
    verifier = creds.generate_verifier()
    challenge = creds.code_challenge(verifier)
    assert "=" not in challenge
    assert "+" not in challenge and "/" not in challenge
    assert _b64url_decode(challenge) == hashlib.sha256(verifier.encode()).digest()
    assert challenge != verifier


def test_verifier_and_state_are_fresh_per_attempt():
    verifiers = {creds.generate_verifier() for _ in range(50)}
    states = {creds.generate_state() for _ in range(50)}
    assert len(verifiers) == 50
    assert len(states) == 50
    # 32 bytes of entropy, base64url, no padding.
    assert all(len(v) == 43 for v in verifiers)
    assert all("=" not in v for v in verifiers | states)


def test_authorize_url_carries_only_the_challenge_never_the_verifier():
    auth_base, _ = providers.resolve_orcarouter_origins({})
    session = creds.AuthorizationSession(flow="oob", auth_base=auth_base)
    parsed = urllib.parse.urlsplit(session.url)
    params = urllib.parse.parse_qs(parsed.query)

    assert parsed.netloc == "www.orcarouter.ai"
    assert parsed.path == "/auth"
    assert params["callback_url"] == ["oob"]
    assert params["code_challenge"] == [session.challenge]
    assert params["code_challenge_method"] == ["S256"]
    assert params["state"] == [session.state]
    assert params["scope"] == ["api"]
    assert session.verifier not in session.url
    assert "verifier" not in session.url


def test_exchange_path_is_api_v1_auth_keys_not_v1_auth_keys():
    auth_base, _ = providers.resolve_orcarouter_origins({})
    url = providers.exchange_url(auth_base)
    assert url == "https://www.orcarouter.ai/api/v1/auth/keys"
    assert "/api/v1/auth/keys" in url
    # The classic integration mistake: the relay origin with the auth path.
    assert not url.startswith("https://api.orcarouter.ai/v1/auth/keys")


# ---------------------------------------------------------------------------
# Both adapters produce the same credential result
# ---------------------------------------------------------------------------

def _fake_exchange(monkeypatch, payload, status=200):
    """Route the exchange through a fake auth server, no network."""
    record = {}

    class _Response:
        def __init__(self, body):
            self._body = json.dumps(body).encode()

        def read(self, *_a):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def opener(request, timeout=None):
        record["url"] = request.full_url
        record["body"] = json.loads(request.data.decode())
        record["headers"] = dict(request.headers)
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "err", {},
                                         _BodyFile(json.dumps(payload).encode()))
        return _Response(payload)

    return opener, record


class _BodyFile:
    def __init__(self, data):
        self._data = data

    def read(self, *_a):
        return self._data

    def close(self):
        pass


def test_both_adapters_yield_the_same_shape_of_credential(monkeypatch):
    store = creds.CredentialStore()
    api = creds.credential_from_api_key(FAKE_KEY, store=store)

    opener, _ = _fake_exchange(monkeypatch, {"key": FAKE_KEY_2, "user_id": "7",
                                             "scope": "api"})
    pkce = creds.exchange_code("fake-code", "fake-verifier", store=store,
                               opener=opener, account="7")

    assert api.provider_id == pkce.provider_id == creds.PROVIDER_ID
    assert api.source == "api_key" and pkce.source == "pkce"
    assert isinstance(api.api_key, str) and isinstance(pkce.api_key, str)
    # Identical contract: downstream code reads only api_key and provider_id.
    assert set(vars(api)) == set(vars(pkce))


def test_pkce_exchange_posts_the_right_path_body_and_method(monkeypatch):
    opener, record = _fake_exchange(monkeypatch, {"key": FAKE_KEY, "user_id": "1",
                                                  "scope": "api"})
    session = creds.AuthorizationSession(flow="oob")
    creds.finish_connect(session, code="fake-code", store=creds.CredentialStore(),
                         opener=opener)

    assert record["url"] == "https://www.orcarouter.ai/api/v1/auth/keys"
    assert record["body"] == {"code": "fake-code",
                              "code_verifier": session.verifier,
                              "code_challenge_method": "S256"}
    # The verifier travels in the POST body, never in the URL.
    assert session.verifier not in record["url"]


def test_pkce_result_is_reused_across_restarts_and_never_refreshed(monkeypatch):
    store = creds.CredentialStore()
    opener, _ = _fake_exchange(monkeypatch, {"key": FAKE_KEY, "user_id": "42",
                                             "scope": "api"})
    first = creds.finish_connect(creds.AuthorizationSession(flow="oob"),
                                 code="c1", store=store, opener=opener)
    # A second read is a plain store read: no exchange, no new key minted.
    stored = store.get(account=first.account)
    assert stored["key"] == first.api_key
    assert not hasattr(creds, "refresh")
    assert not hasattr(store, "refresh")


def test_scope_is_read_from_the_response_not_assumed(monkeypatch):
    opener, _ = _fake_exchange(monkeypatch, {"key": FAKE_KEY, "user_id": "9",
                                             "scope": "api"})
    result = creds.finish_connect(
        creds.AuthorizationSession(flow="oob", scope="connector"),
        code="c", store=creds.CredentialStore(), opener=opener)
    assert result.scope == "api"          # granted, not requested
    assert result.scope_downgraded is True


# ---------------------------------------------------------------------------
# Failure semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,needs_reauth", [(400, False), (403, True), (429, False)])
def test_exchange_errors_are_classified(monkeypatch, status, needs_reauth):
    opener, _ = _fake_exchange(monkeypatch, {"error": "x", "error_description": "y"},
                               status=status)
    with pytest.raises(creds.CredentialError) as excinfo:
        creds.exchange_code("c", "v", store=creds.CredentialStore(), opener=opener)
    assert excinfo.value.needs_reauth is needs_reauth
    assert excinfo.value.status == status


def test_exchange_network_failure_does_not_leak_the_verifier(monkeypatch):
    def opener(request, timeout=None):
        raise urllib.error.URLError("no route to host")

    verifier = "super-secret-verifier-value"
    with pytest.raises(creds.CredentialError) as excinfo:
        creds.exchange_code("c", verifier, store=creds.CredentialStore(), opener=opener)
    assert verifier not in str(excinfo.value)


def test_rejected_exchange_never_writes_a_key(monkeypatch):
    store = creds.CredentialStore()
    opener, _ = _fake_exchange(monkeypatch, {"error": "access_denied"}, status=403)
    with pytest.raises(creds.CredentialError):
        creds.exchange_code("c", "v", store=store, opener=opener)
    assert store.get() is None


# ---------------------------------------------------------------------------
# Flow A — state, denial, timeout, single-use
# ---------------------------------------------------------------------------

def _drive_loopback(session, params):
    """Deliver a callback to the session's real loopback listener."""
    query = urllib.parse.urlencode(params)
    with urllib.request.urlopen(session.callback_url + "?" + query, timeout=10) as r:
        r.read()


def test_flow_a_exchanges_the_code_and_keeps_the_verifier_out_of_the_url(monkeypatch):
    opener, record = _fake_exchange(monkeypatch, {"key": FAKE_KEY, "user_id": "3",
                                                  "scope": "api"})
    session = creds.AuthorizationSession(flow="loopback")
    assert session.callback_url.startswith("http://127.0.0.1:")
    assert providers.valid_callback_url(session.callback_url)

    threading.Thread(target=_drive_loopback,
                     args=(session, {"code": "fake-code", "state": session.state}),
                     daemon=True).start()
    result = creds.finish_connect(session, store=creds.CredentialStore(), opener=opener)
    assert result.api_key == FAKE_KEY
    assert session.verifier not in session.url
    assert record["body"]["code_verifier"] == session.verifier


def test_flow_a_state_mismatch_is_refused_before_any_exchange(monkeypatch):
    called = {"exchange": False}

    def opener(request, timeout=None):
        called["exchange"] = True
        raise AssertionError("must not exchange after a state mismatch")

    session = creds.AuthorizationSession(flow="loopback")
    threading.Thread(target=_drive_loopback,
                     args=(session, {"code": "attacker-code", "state": "wrong-state"}),
                     daemon=True).start()
    with pytest.raises(creds.CredentialError) as excinfo:
        creds.finish_connect(session, store=creds.CredentialStore(), opener=opener)
    assert "state" in str(excinfo.value)
    assert called["exchange"] is False


def test_flow_a_denial_is_reported_and_exits():
    session = creds.AuthorizationSession(flow="loopback")
    threading.Thread(target=_drive_loopback,
                     args=(session, {"error": "access_denied", "state": session.state}),
                     daemon=True).start()
    with pytest.raises(creds.CredentialError) as excinfo:
        creds.finish_connect(session, store=creds.CredentialStore())
    assert "refused" in str(excinfo.value)


def test_flow_a_timeout_does_not_hang_forever():
    session = creds.AuthorizationSession(flow="loopback")
    with pytest.raises(creds.CredentialError) as excinfo:
        creds.finish_connect(session, store=creds.CredentialStore(), timeout=0.4)
    assert "timed out" in str(excinfo.value)
    session.close()


def test_flow_a_listener_is_released_after_a_denial():
    session = creds.AuthorizationSession(flow="loopback")
    port = urllib.parse.urlsplit(session.callback_url).port
    session.close()
    session.close()  # idempotent
    with pytest.raises(OSError):
        # The port must be free again: the login lock is genuinely released.
        urllib.request.urlopen("http://127.0.0.1:%d/cb" % port, timeout=2).read()


def test_code_is_single_use_the_second_exchange_is_refused(monkeypatch):
    store = creds.CredentialStore()
    opener, _ = _fake_exchange(monkeypatch, {"key": FAKE_KEY, "user_id": "5",
                                             "scope": "api"})
    creds.exchange_code("one-shot", "v", store=store, opener=opener)
    # The server refuses a reused code with 403; our client surfaces it as a
    # terminal, non-retryable credential error.
    opener2, _ = _fake_exchange(monkeypatch, {"error": "invalid_grant"}, status=403)
    with pytest.raises(creds.CredentialError) as excinfo:
        creds.exchange_code("one-shot", "v", store=store, opener=opener2)
    assert excinfo.value.needs_reauth is True
    assert excinfo.value.status == 403


# ---------------------------------------------------------------------------
# Terminal 401 and generation-safe reauth
# ---------------------------------------------------------------------------

def test_mark_needs_reauth_targets_only_the_rejected_generation():
    store = creds.CredentialStore()
    gen1 = store.put(FAKE_KEY, "pkce", scope="api")
    gen2 = store.put(FAKE_KEY_2, "pkce", scope="api")
    assert (gen1, gen2) == (1, 2)

    # A late 401 from a request made with generation 1 must not condemn the
    # credential that replaced it.
    assert store.mark_needs_reauth(generation=gen1) is False
    assert store.get()["needs_reauth"] is False

    assert store.mark_needs_reauth(generation=gen2) is True
    assert store.get()["needs_reauth"] is True
    # The old secret is not silently deleted; only flagged.
    assert store.get()["key"] == FAKE_KEY_2


def test_a_new_login_clears_needs_reauth_and_bumps_the_generation():
    store = creds.CredentialStore()
    gen1 = store.put(FAKE_KEY, "pkce")
    store.mark_needs_reauth(generation=gen1)
    gen2 = store.put(FAKE_KEY_2, "pkce")
    assert gen2 == gen1 + 1
    assert store.get()["needs_reauth"] is False


def test_prefix_check_is_a_format_check_only_and_sends_no_request(monkeypatch):
    """An sk-orca- prefix is not proof of validity.

    We must not spend a paid inference request to make a form read "valid", so
    acceptance is purely syntactic and the first real request establishes truth.
    """
    calls = {"n": 0}

    def opener(*_a, **_k):
        calls["n"] += 1
        raise AssertionError("storing a key must not make a network request")

    monkeypatch.setattr(urllib.request, "urlopen", opener)
    result = creds.credential_from_api_key(FAKE_KEY, store=creds.CredentialStore())
    assert result.api_key == FAKE_KEY
    assert result.scope == "api"        # requested default; not a validated claim
    assert calls["n"] == 0


def test_a_non_orcarouter_key_is_rejected_before_storage():
    store = creds.CredentialStore()
    with pytest.raises(creds.CredentialError) as excinfo:
        creds.credential_from_api_key("sk-or-v1-abcdef", store=store)
    assert "sk-orca-" in str(excinfo.value)
    assert store.get() is None


def test_corrupt_credential_file_degrades_to_empty_not_a_crash(tmp_path):
    path = creds.credentials_path()
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("{not json")
    store = creds.CredentialStore()
    assert store.get() is None
    gen = store.put(FAKE_KEY, "api_key")
    assert gen == 1
