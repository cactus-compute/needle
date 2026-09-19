"""End-to-end wiring tests for the OrcaRouter provider.

These drive the real synthesis code path (``generate_examples`` -> ``_openrouter``)
with a stubbed transport, so what is asserted is the URL, headers and body the
implementation actually produces — not a helper in isolation.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

import pytest

from needle.model import credentials as creds
from needle.model import finetune, providers

FAKE_KEY = "sk-orca-00000000000000000000000000000000"
PKCE_KEY = "sk-orca-22222222222222222222222222222222"

TOOLS = [{"name": "set_lights",
          "parameters": {"type": "object", "properties": {"room": {"type": "string"}}}}]
REPLY = json.dumps({"choices": [{"message": {"content": json.dumps(
    [{"query": "dim the lights", "answers": [{"name": "set_lights",
                                              "arguments": {"room": "study"}}]}])}}]})


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("NEEDLE_CACHE_DIR", str(tmp_path))
    for name in ("ORCA_BASE_URL", "ORCA_AUTH_BASE_URL", "ORCA_API_BASE_URL",
                 "ORCAROUTER_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_URL"):
        monkeypatch.delenv(name, raising=False)
    yield


def _capture_transport(body=REPLY, status=None, payload=None):
    """Stub urllib so the real request object is inspectable."""
    calls = []

    class _Response:
        def read(self, *_a):
            return body.encode()

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def urlopen(request, timeout=None):
        calls.append({"url": request.full_url, "timeout": timeout,
                      "headers": dict(request.headers),
                      "body": json.loads(request.data.decode())})
        if status is not None:
            raise urllib.error.HTTPError(request.full_url, status, "err", {},
                                         _BodyFile(payload or b"{}"))
        return _Response()

    return urlopen, calls


class _BodyFile:
    def __init__(self, data):
        self._data = data

    def read(self, *_a):
        return self._data

    def close(self):
        pass


def _authorize_and_exchange(monkeypatch, store, api_key=PKCE_KEY, scope="api"):
    """Run the PKCE adapter end to end through a fake auth server."""
    events = []

    class _Response:
        def __init__(self, payload):
            self._payload = json.dumps(payload).encode()

        def read(self, *_a):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def opener(request, timeout=None):
        events.append({"url": request.full_url,
                       "body": json.loads(request.data.decode())})
        return _Response({"key": api_key, "user_id": "u-1", "scope": scope})

    session = creds.AuthorizationSession(flow="oob")
    return creds.finish_connect(session, code="fake-code", store=store,
                                opener=opener), session, events


# ---------------------------------------------------------------------------
# Both adapters reach the same inference path
# ---------------------------------------------------------------------------

def test_both_adapters_reach_an_identical_inference_request(monkeypatch):
    """The downstream provider path cannot tell which adapter produced the key."""
    store = creds.CredentialStore()
    provider = providers.get_provider("orcarouter")

    # Adapter 1: a pasted API key.
    api_key_credential = creds.credential_from_api_key(FAKE_KEY, store=store)

    transport, calls = _capture_transport()
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    rows_api = finetune.generate_examples(TOOLS, n=1, provider_id="orcarouter",
                                          api_key=api_key_credential.api_key,
                                          store=store)
    api_call = dict(calls[0])

    # Adapter 2: PKCE. Same store, same downstream call.
    pkce_credential, session, _ = _authorize_and_exchange(monkeypatch, store)
    assert pkce_credential.source == "pkce"

    calls.clear()
    rows_pkce = finetune.generate_examples(TOOLS, n=1, provider_id="orcarouter",
                                           api_key=pkce_credential.api_key,
                                           store=store)
    pkce_call = dict(calls[0])

    assert rows_api == rows_pkce
    assert api_call["url"] == pkce_call["url"] == \
        "https://api.orcarouter.ai/v1/chat/completions"
    # Only the credential differs; everything else about the request is equal.
    assert api_call["headers"]["Authorization"] != pkce_call["headers"]["Authorization"]
    for key in set(api_call) - {"headers"}:
        assert api_call[key] == pkce_call[key], key
    assert provider.api_base == "https://api.orcarouter.ai/v1"


def test_generate_examples_uses_the_provider_default_model(monkeypatch):
    transport, calls = _capture_transport()
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    finetune.generate_examples(TOOLS, n=1, model=None, provider_id="orcarouter",
                               api_key=FAKE_KEY)
    assert calls[0]["body"]["model"] == "orcarouter/auto"


def test_openrouter_path_is_unchanged(monkeypatch):
    transport, calls = _capture_transport()
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    finetune.generate_examples(TOOLS, n=1, provider_id="openrouter", api_key="sk-or-fake")
    assert calls[0]["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert calls[0]["body"]["model"] == "deepseek/deepseek-flash-latest"


def test_synthesis_reads_the_environment_when_no_key_is_passed(monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", FAKE_KEY)
    transport, calls = _capture_transport()
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    finetune.generate_examples(TOOLS, n=1, provider_id="orcarouter")
    assert calls[0]["headers"]["Authorization"] == "Bearer " + FAKE_KEY


def test_an_unknown_provider_is_refused_before_any_request(monkeypatch):
    transport, calls = _capture_transport()
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    with pytest.raises(ValueError):
        finetune.generate_examples(TOOLS, n=1, provider_id="nope", api_key=FAKE_KEY)
    assert calls == []


# ---------------------------------------------------------------------------
# Origin separation
# ---------------------------------------------------------------------------

def _orcarouter_urls():
    provider = providers.get_provider("orcarouter")
    auth_base, api_base = providers.resolve_orcarouter_origins({})
    session = creds.AuthorizationSession(flow="oob", auth_base=auth_base)
    return {
        "authorize": session.url,
        "exchange": providers.exchange_url(auth_base),
        "chat": provider.chat_completions_url,
        "models": provider.models_url,
    }


def test_auth_requests_only_ever_go_to_the_auth_origin():
    urls = _orcarouter_urls()
    for name in ("authorize", "exchange"):
        host = urllib.parse.urlsplit(urls[name]).netloc
        assert host == "www.orcarouter.ai", (name, urls[name])
    assert urls["exchange"] == "https://www.orcarouter.ai/api/v1/auth/keys"


def test_inference_and_catalog_only_ever_go_to_the_api_origin():
    urls = _orcarouter_urls()
    for name in ("chat", "models"):
        parsed = urllib.parse.urlsplit(urls[name])
        assert parsed.netloc == "api.orcarouter.ai", (name, urls[name])
        assert parsed.path.startswith("/v1/")
    assert urls["models"] == "https://api.orcarouter.ai/v1/models"


def test_the_wrong_auth_path_is_never_constructed():
    """``api.orcarouter.ai/v1/auth/keys`` is a 404 — the classic mistake."""
    urls = _orcarouter_urls()
    assert "/v1/auth/keys" not in urls["chat"]
    assert "/v1/auth/keys" not in urls["models"]
    assert not urls["exchange"].startswith("https://api.orcarouter.ai")
    for url in urls.values():
        assert "api.orcarouter.ai/v1/auth" not in url


def test_a_self_hosted_base_keeps_auth_and_inference_on_one_origin():
    auth_base, api_base = providers.resolve_orcarouter_origins(
        {"ORCA_BASE_URL": "https://orca.internal"})
    assert providers.exchange_url(auth_base) == \
        "https://orca.internal/api/v1/auth/keys"
    provider = providers.get_provider("orcarouter")
    assert api_base == "https://orca.internal/v1"


# ---------------------------------------------------------------------------
# Terminal 401, no fake refresh
# ---------------------------------------------------------------------------

def test_a_401_is_terminal_and_attempts_exactly_one_request(monkeypatch):
    transport, calls = _capture_transport(status=401)
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    with pytest.raises(creds.CredentialError) as excinfo:
        finetune.generate_examples(TOOLS, n=1, provider_id="orcarouter",
                                   api_key=FAKE_KEY)
    assert excinfo.value.needs_reauth is True
    assert excinfo.value.status == 401
    # No retry, no second request: a revoked durable key is not refreshed.
    assert len(calls) == 1


def test_a_revoked_key_marks_reauthentication_instead_of_refreshing(monkeypatch):
    store = creds.CredentialStore()
    generation = store.put(FAKE_KEY, "pkce", scope="api")

    transport, _ = _capture_transport(status=401)
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    with pytest.raises(creds.CredentialError):
        finetune.generate_examples(TOOLS, n=1, provider_id="orcarouter",
                                   api_key=FAKE_KEY, store=store)

    # The rejected generation is marked by the request path itself, not by the
    # caller: a revoked durable key must leave the account unusable.
    entry = store.get()
    assert entry["needs_reauth"] is True
    assert entry["key"] == FAKE_KEY          # not silently deleted
    assert entry["generation"] == generation
    # There is no refresh grant to reach for.
    assert not hasattr(creds, "refresh_credential")
    assert not hasattr(finetune, "refresh")


def test_the_revoked_key_is_then_refused_without_another_request(monkeypatch):
    """The marked account stays unusable until a new login succeeds."""
    store = creds.CredentialStore()
    store.put(FAKE_KEY, "pkce", scope="api")

    transport, calls = _capture_transport(status=401)
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    with pytest.raises(creds.CredentialError):
        finetune.generate_examples(TOOLS, n=1, provider_id="orcarouter",
                                   api_key=FAKE_KEY, store=store)
    assert len(calls) == 1

    # A later generation-safe login clears the flag and is usable again.
    store.put(PKCE_KEY, "pkce", scope="api")
    assert store.get()["needs_reauth"] is False

    transport, calls = _capture_transport()
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    rows = finetune.generate_examples(TOOLS, n=1, provider_id="orcarouter",
                                      api_key=PKCE_KEY, store=store)
    assert rows and calls[0]["headers"]["Authorization"] == "Bearer " + PKCE_KEY


def test_a_late_401_cannot_mark_a_newer_credential_generation(monkeypatch):
    """The exact-account, exact-generation rule, driven through the real call."""
    store = creds.CredentialStore()
    old_generation = store.put(FAKE_KEY, "pkce", scope="api")
    stale = creds.CredentialResult(provider_id="orcarouter", api_key=FAKE_KEY,
                                   source="pkce", account="orcarouter",
                                   generation=old_generation)

    # A newer login replaces the stored credential before the stale request
    # fails. Its rejection must not touch what replaced it.
    store.put(PKCE_KEY, "pkce", scope="api")
    transport, _ = _capture_transport(status=401)
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    with pytest.raises(creds.CredentialError):
        finetune.generate_examples(TOOLS, n=1, provider_id="orcarouter",
                                   credential=stale, store=store)

    entry = store.get()
    assert entry["needs_reauth"] is False
    assert entry["key"] == PKCE_KEY


def test_an_environment_key_cannot_mark_a_stored_record(monkeypatch):
    """The store never issued that key, so it may not invalidate one it holds."""
    store = creds.CredentialStore()
    store.put(PKCE_KEY, "pkce", scope="api")

    monkeypatch.setenv("ORCAROUTER_API_KEY", FAKE_KEY)
    transport, _ = _capture_transport(status=401)
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    with pytest.raises(creds.CredentialError):
        finetune.generate_examples(TOOLS, n=1, provider_id="orcarouter", store=store)

    entry = store.get()
    assert entry["needs_reauth"] is False
    assert entry["key"] == PKCE_KEY


def test_a_429_is_retryable_not_a_reauth(monkeypatch):
    transport, calls = _capture_transport(status=429)
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    with pytest.raises(creds.CredentialError) as excinfo:
        finetune.generate_examples(TOOLS, n=1, provider_id="orcarouter",
                                   api_key=FAKE_KEY)
    assert excinfo.value.needs_reauth is False
    assert excinfo.value.status == 429
    assert len(calls) == 1


def test_a_network_failure_is_reported_without_leaking_the_key(monkeypatch):
    def urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(creds.CredentialError) as excinfo:
        finetune.generate_examples(TOOLS, n=1, provider_id="orcarouter",
                                   api_key=FAKE_KEY)
    assert FAKE_KEY not in str(excinfo.value)
    assert "Bearer" not in str(excinfo.value)


def test_no_secret_appears_in_any_error_message(monkeypatch, capsys):
    for status in (401, 429, 500):
        transport, _ = _capture_transport(status=status)
        monkeypatch.setattr(urllib.request, "urlopen", transport)
        with pytest.raises(creds.CredentialError) as excinfo:
            finetune.generate_examples(TOOLS, n=1, provider_id="orcarouter",
                                       api_key=FAKE_KEY)
        assert FAKE_KEY not in str(excinfo.value)
    captured = capsys.readouterr()
    assert FAKE_KEY not in captured.out
    assert FAKE_KEY not in captured.err


# ---------------------------------------------------------------------------
# Multimodal entry point
#
# An image attachment is the non-text modality this repository's synthesis entry
# point can carry. The catalog decides which models may receive it, and the
# request body changes shape only once that is settled.
# ---------------------------------------------------------------------------

VISION_MODEL = "deepseek/deepseek-v4-flash-vision-exp"
IMAGE_URL = "https://example.invalid/receipt.png"

CATALOG_BODY = json.dumps({"data": [
    {"id": "deepseek/deepseek-v4-flash", "supported_endpoint_types": ["openai"],
     "architecture": {"input_modalities": ["text"]}},
    {"id": VISION_MODEL, "supported_endpoint_types": ["openai", "anthropic"],
     "architecture": {"input_modalities": ["text", "image"]}},
]})


def _catalog_then_chat(chat_body=REPLY):
    """Transport that answers the catalog request and then the completion."""
    calls = []

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def read(self, *_a):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def urlopen(request, timeout=None):
        calls.append({"url": request.full_url,
                      "body": json.loads(request.data.decode())
                      if request.data else None})
        if request.full_url.endswith("/models"):
            return _Response(CATALOG_BODY.encode())
        return _Response(chat_body.encode())

    return urlopen, calls


def test_an_image_prompt_sends_the_multimodal_content_form(monkeypatch):
    transport, calls = _catalog_then_chat()
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    rows = finetune.generate_examples(TOOLS, n=1, model=VISION_MODEL,
                                      provider_id="orcarouter", api_key=FAKE_KEY,
                                      input_modality="image", image_url=IMAGE_URL)
    assert rows and rows[0]["query"] == "dim the lights"
    chat = [c for c in calls if not c["url"].endswith("/models")]
    assert len(chat) == 1
    user = chat[0]["body"]["messages"][1]
    assert isinstance(user["content"], list)
    assert user["content"][0]["type"] == "text"
    assert user["content"][1] == {"type": "image_url", "image_url": {"url": IMAGE_URL}}
    assert chat[0]["body"]["model"] == VISION_MODEL


def test_a_text_prompt_keeps_the_plain_string_form(monkeypatch):
    transport, calls = _catalog_then_chat()
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    finetune.generate_examples(TOOLS, n=1, model=VISION_MODEL,
                               provider_id="orcarouter", api_key=FAKE_KEY)
    chat = [c for c in calls if not c["url"].endswith("/models")]
    assert isinstance(chat[0]["body"]["messages"][1]["content"], str)


def test_a_text_only_model_is_refused_before_the_request_is_sent(monkeypatch):
    """The selector would not offer it; if a stale value reaches here it stops."""
    transport, calls = _catalog_then_chat()
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    with pytest.raises(creds.CredentialError) as excinfo:
        finetune.generate_examples(TOOLS, n=1, model="deepseek/deepseek-v4-flash",
                                   provider_id="orcarouter", api_key=FAKE_KEY,
                                   input_modality="image", image_url=IMAGE_URL)
    assert "does not declare image input" in str(excinfo.value)
    assert not [c for c in calls if not c["url"].endswith("/models")], \
        "an incompatible pair must never reach the completions endpoint"


def test_a_model_absent_from_the_catalog_is_refused(monkeypatch):
    transport, calls = _catalog_then_chat()
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    with pytest.raises(creds.CredentialError) as excinfo:
        finetune.generate_examples(TOOLS, n=1, model="vendor/not-in-catalog",
                                   provider_id="orcarouter", api_key=FAKE_KEY,
                                   input_modality="image", image_url=IMAGE_URL)
    assert "not in the OrcaRouter catalog" in str(excinfo.value)
    assert not [c for c in calls if not c["url"].endswith("/models")]


def test_the_modality_guard_reads_the_live_catalog_not_a_name(monkeypatch):
    """A vision-looking id that declares no image input is still refused."""
    body = json.dumps({"data": [
        {"id": "vendor/looks-like-vision", "supported_endpoint_types": ["openai"],
         "architecture": {"input_modalities": ["text"]}}]})

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def read(self, *_a):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    calls = []

    def urlopen(request, timeout=None):
        calls.append(request.full_url)
        return _Response(body.encode() if request.full_url.endswith("/models")
                         else REPLY.encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(creds.CredentialError) as excinfo:
        finetune.generate_examples(TOOLS, n=1, model="vendor/looks-like-vision",
                                   provider_id="orcarouter", api_key=FAKE_KEY,
                                   input_modality="image", image_url=IMAGE_URL)
    assert "does not declare image input" in str(excinfo.value)
    assert calls == ["https://api.orcarouter.ai/v1/models"], \
        "only the catalog should have been consulted"


def test_other_providers_keep_their_behaviour_for_an_image_prompt(monkeypatch):
    """OpenRouter has no catalog here, so it is not second-guessed."""
    transport, calls = _catalog_then_chat()
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    rows = finetune.generate_examples(TOOLS, n=1, model="vendor/whatever",
                                      provider_id="openrouter", api_key="sk-or-x",
                                      input_modality="image", image_url=IMAGE_URL)
    assert rows
    assert calls[0]["url"].startswith("https://openrouter.ai/"), \
        "no OrcaRouter catalog call should be made for another provider"
