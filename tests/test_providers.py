"""Provider registry, origin policy and model-catalog tests.

Fixtures deliberately include records for every capability the repository's
filtering rules talk about — text-only chat, image-input chat, embedding, image
generation, video and rerank — so a filter can be shown to exclude what it must
rather than merely to accept what it should.
"""

import json
import urllib.error
import urllib.request

import pytest

from needle.model import providers
from needle.model.providers import (ModelCatalog, Provider, discover_models,
                                    filter_models, get_provider, parse_catalog,
                                    resolve_orcarouter_origins, validate_origin)

CHAT = ["openai", "openai-response"]
VISION = ["openai", "anthropic"]

FIXTURES = [
    {"id": "deepseek/deepseek-v4-flash", "object": "model",
     "supported_endpoint_types": CHAT, "context_length": 1048576,
     "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}},
    {"id": "deepseek/deepseek-v4-flash-vision-exp", "object": "model",
     "supported_endpoint_types": VISION, "context_length": 1048576,
     "architecture": {"input_modalities": ["text", "image"],
                      "output_modalities": ["text"]}},
    {"id": "openai/gpt-5.5", "object": "model", "supported_endpoint_types": ["openai"],
     "context_length": 400000,
     "architecture": {"input_modalities": ["text", "image"],
                      "output_modalities": ["text"]},
     "reasoning_efforts": ["low", "medium", "high", "xhigh"]},
    {"id": "google/text-embedding-005", "object": "model",
     "supported_endpoint_types": ["embeddings"],
     "architecture": {"input_modalities": ["text"]}},
    {"id": "openai/gpt-image-1", "object": "model",
     "supported_endpoint_types": ["image-generation"]},
    {"id": "openai/sora-2", "object": "model",
     "supported_endpoint_types": ["openai-video"]},
    {"id": "jina/jina-reranker-v2", "object": "model",
     "supported_endpoint_types": ["jina-rerank"]},
    # Declares no architecture at all: must fail closed for multimodal.
    {"id": "vendor/undeclared-modal", "object": "model",
     "supported_endpoint_types": ["openai"]},
    # A bare name with no vendor namespace cannot be routed.
    {"id": "no-namespace", "object": "model", "supported_endpoint_types": ["openai"]},
    "not-a-record",
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_orcarouter_is_a_first_class_named_provider():
    provider = get_provider("orcarouter")
    assert provider.id == "orcarouter"
    assert provider.label == "OrcaRouter"
    assert provider.env_key == "ORCAROUTER_API_KEY"
    assert provider.chat_completions_url == "https://api.orcarouter.ai/v1/chat/completions"
    assert provider.models_url == "https://api.orcarouter.ai/v1/models"
    assert "orcarouter" in providers.providers()
    assert "openrouter" in providers.providers()


def test_openrouter_stays_the_default_and_is_untouched():
    assert get_provider(None).id == "openrouter"
    assert get_provider("openrouter").chat_completions_url == \
        "https://openrouter.ai/api/v1/chat/completions"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        get_provider("definitely-not-a-provider")


def test_legacy_openrouter_url_is_used_verbatim(monkeypatch):
    """``OPENROUTER_URL`` was always the completions URL; keep passing it through."""
    monkeypatch.setenv("OPENROUTER_URL", "https://gateway.example.test/v1/chat")
    provider = get_provider("openrouter")
    assert provider.chat_completions_url == "https://gateway.example.test/v1/chat"

    monkeypatch.setenv("OPENROUTER_URL",
                       "https://gateway.example.test/v1/chat/completions")
    provider = get_provider("openrouter")
    assert provider.chat_completions_url == \
        "https://gateway.example.test/v1/chat/completions"
    assert provider.api_base == "https://gateway.example.test/v1"

    monkeypatch.delenv("OPENROUTER_URL", raising=False)
    assert get_provider("openrouter").chat_completions_url == \
        "https://openrouter.ai/api/v1/chat/completions"


# ---------------------------------------------------------------------------
# Origins
# ---------------------------------------------------------------------------

def test_public_defaults_use_two_distinct_origins():
    auth, api = resolve_orcarouter_origins({})
    assert auth == "https://www.orcarouter.ai"
    assert api == "https://api.orcarouter.ai/v1"
    # The inference origin is never derived from the auth origin by host swap.
    assert "www." not in api and api != auth
    assert providers.exchange_url(auth) == "https://www.orcarouter.ai/api/v1/auth/keys"


def test_shared_self_hosted_base_supplies_both_origins():
    auth, api = resolve_orcarouter_origins({"ORCA_BASE_URL": "https://orca.internal"})
    assert auth == "https://orca.internal"
    assert api == "https://orca.internal/v1"


def test_explicit_overrides_win_over_the_shared_base():
    auth, api = resolve_orcarouter_origins({
        "ORCA_BASE_URL": "https://shared.internal",
        "ORCA_AUTH_BASE_URL": "https://auth.internal",
        "ORCA_API_BASE_URL": "https://api.internal/v1",
    })
    assert auth == "https://auth.internal"
    assert api == "https://api.internal/v1"


def test_http_is_only_allowed_for_loopback():
    assert validate_origin("http://127.0.0.1:8080", "x") == "http://127.0.0.1:8080"
    assert validate_origin("http://localhost:9000", "x") == "http://localhost:9000"
    with pytest.raises(ValueError):
        validate_origin("http://orca.example.com", "x")
    with pytest.raises(ValueError):
        validate_origin("ftp://orca.example.com", "x")


def test_a_loopback_self_hosted_base_is_accepted():
    auth, api = resolve_orcarouter_origins({"ORCA_BASE_URL": "http://127.0.0.1:8081"})
    assert auth == "http://127.0.0.1:8081"
    assert api == "http://127.0.0.1:8081/v1"


# ---------------------------------------------------------------------------
# Catalog parsing
# ---------------------------------------------------------------------------

def test_parse_catalog_keeps_good_records_and_drops_malformed_ones():
    models = parse_catalog({"data": FIXTURES})
    ids = [m["id"] for m in models]
    assert "deepseek/deepseek-v4-flash" in ids
    assert "no-namespace" not in ids      # no vendor namespace: unroutable
    assert "not-a-record" not in ids
    assert len(models) == 8


def test_parse_catalog_preserves_metadata_including_reasoning_efforts():
    models = {m["id"]: m for m in parse_catalog({"data": FIXTURES})}
    gpt = models["openai/gpt-5.5"]
    assert gpt["context_length"] == 400000
    assert gpt["architecture"]["input_modalities"] == ["text", "image"]
    assert gpt["reasoning_efforts"] == ["low", "medium", "high", "xhigh"]


def test_parse_catalog_rejects_non_list_payloads():
    assert parse_catalog({"data": "nope"}) == []
    assert parse_catalog(None) == []


def test_parse_catalog_is_bounded(monkeypatch):
    monkeypatch.setattr(providers, "CATALOG_MAX_ITEMS", 3)
    records = [{"id": "v/m%d" % i, "supported_endpoint_types": CHAT} for i in range(50)]
    assert len(parse_catalog({"data": records})) == 3


# ---------------------------------------------------------------------------
# Capability filtering
# ---------------------------------------------------------------------------

def _ids(models):
    return {m["id"] for m in models}


def test_chat_filter_excludes_non_text_specialist_models():
    ids = _ids(filter_models(FIXTURES, capability="chat"))
    assert "deepseek/deepseek-v4-flash" in ids
    assert "openai/gpt-5.5" in ids
    # image generation, video and rerank are not chat.
    assert "openai/gpt-image-1" not in ids
    assert "openai/sora-2" not in ids
    assert "jina/jina-reranker-v2" not in ids
    assert "google/text-embedding-005" not in ids


def test_each_capability_selects_exactly_its_own_models():
    assert _ids(filter_models(FIXTURES, capability="embedding")) == \
        {"google/text-embedding-005"}
    assert _ids(filter_models(FIXTURES, capability="image")) == {"openai/gpt-image-1"}
    assert _ids(filter_models(FIXTURES, capability="video")) == {"openai/sora-2"}
    assert _ids(filter_models(FIXTURES, capability="rerank")) == {"jina/jina-reranker-v2"}


def test_multimodal_filter_keeps_only_models_declaring_the_modality():
    ids = _ids(filter_models(FIXTURES, capability="chat", input_modalities=["image"]))
    assert ids == {"deepseek/deepseek-v4-flash-vision-exp", "openai/gpt-5.5"}
    # The text-only model is gone, and so is the record that declared nothing.
    assert "deepseek/deepseek-v4-flash" not in ids
    assert "vendor/undeclared-modal" not in ids


def test_multimodal_filter_fails_closed_on_undeclared_capability():
    ids = _ids(filter_models(FIXTURES, capability="chat", input_modalities=["text"]))
    # "text" is the default and must not narrow the chat list.
    assert "vendor/undeclared-modal" in ids
    # But an undeclared record never satisfies a non-text modality.
    for modality in ("image", "audio", "video"):
        selected = _ids(filter_models(FIXTURES, capability="chat",
                                      input_modalities=[modality]))
        assert "vendor/undeclared-modal" not in selected


def test_audio_filter_excludes_models_that_only_declare_text():
    ids = _ids(filter_models(FIXTURES, capability="chat", input_modalities=["audio"]))
    assert ids == set()


def test_unknown_capability_is_rejected():
    with pytest.raises(ValueError):
        filter_models(FIXTURES, capability="telepathy")


# ---------------------------------------------------------------------------
# Discovery, fallback and degraded state
# ---------------------------------------------------------------------------

def _fake_opener(payload=None, error=None):
    record = {}

    class _Response:
        def __init__(self, body):
            self._body = body

        def read(self, *_a):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def opener(request, timeout=None):
        record["url"] = request.full_url
        record["timeout"] = timeout
        record["headers"] = dict(request.headers)
        if error is not None:
            raise error
        return _Response(json.dumps(payload).encode())

    return opener, record


def test_live_discovery_is_authoritative_and_never_mixed_with_the_seed():
    provider = get_provider("orcarouter")
    opener, record = _fake_opener({"data": [
        {"id": "vendor/only-live-model", "supported_endpoint_types": CHAT,
         "architecture": {"input_modalities": ["text"]}}]})
    catalog = discover_models(provider, api_key="sk-orca-fake", opener=opener)

    assert catalog.source == "live"
    assert catalog.degraded is False
    assert catalog.ids() == ["vendor/only-live-model"]
    # The verified seed must not leak into a successful live result.
    for seeded in providers.SEED_MODEL_IDS:
        assert seeded not in catalog.ids()
    assert record["url"] == "https://api.orcarouter.ai/v1/models"
    assert record["headers"]["Authorization"] == "Bearer sk-orca-fake"


def test_discovery_falls_back_to_the_verified_seed_on_failure():
    provider = get_provider("orcarouter")
    opener, _ = _fake_opener(error=urllib.error.URLError("offline"))
    catalog = discover_models(provider, api_key="sk-orca-fake", opener=opener)

    assert catalog.degraded is True
    assert catalog.source == "seed"
    assert catalog.error
    assert set(providers.SEED_MODEL_IDS).issubset(set(catalog.ids()))
    # The seed carries real metadata, so a fallback is still filterable.
    by_id = {m["id"]: m for m in catalog.models}
    assert by_id["openai/gpt-5.5"]["reasoning_efforts"] == \
        ["low", "medium", "high", "xhigh"]
    assert by_id["openai/gpt-5.5"]["architecture"]["input_modalities"] == \
        ["text", "image"]
    assert len(filter_models(catalog.models, capability="chat")) == 5


def test_discovery_prefers_last_known_good_over_the_seed():
    provider = get_provider("orcarouter")
    lkg = [{"id": "vendor/cached", "supported_endpoint_types": CHAT,
            "architecture": {"input_modalities": ["text"]}}]
    opener, _ = _fake_opener(error=urllib.error.URLError("offline"))
    catalog = discover_models(provider, api_key="sk-orca-fake", opener=opener,
                              last_known_good=lkg)
    assert catalog.degraded is True
    assert catalog.source == "last-known-good"
    assert catalog.ids() == ["vendor/cached"]


def test_an_empty_live_catalog_is_treated_as_degraded_not_as_a_real_answer():
    provider = get_provider("orcarouter")
    opener, _ = _fake_opener({"data": []})
    catalog = discover_models(provider, api_key="sk-orca-fake", opener=opener)
    assert catalog.degraded is True
    assert set(providers.SEED_MODEL_IDS).issubset(set(catalog.ids()))


def test_an_oversized_catalog_response_is_refused(monkeypatch):
    provider = get_provider("orcarouter")
    monkeypatch.setattr(providers, "CATALOG_MAX_BYTES", 16)
    opener, _ = _fake_opener({"data": FIXTURES})
    catalog = discover_models(provider, api_key="sk-orca-fake", opener=opener)
    assert catalog.degraded is True


def test_catalog_request_is_bounded_by_a_timeout():
    provider = get_provider("orcarouter")
    opener, record = _fake_opener({"data": []})
    discover_models(provider, api_key="sk-orca-fake", opener=opener)
    assert record["timeout"] == providers.CATALOG_TIMEOUT_SECONDS
    assert record["timeout"] and record["timeout"] < 60


def test_a_provider_without_discovery_does_not_invent_models():
    catalog = discover_models(get_provider("openrouter"))
    assert catalog.models == []
