"""Live OrcaRouter checks.

These make real requests through the provider code this change adds — the same
``generate_examples`` / ``discover_models`` / ``filter_models`` path the CLI and
the playground use. They need ``ORCAROUTER_API_KEY`` and are skipped without it,
so the offline suite stays runnable and no key is ever committed.

Run:  ORCAROUTER_API_KEY=... python3 -m pytest tests/test_orcarouter_live.py -q
"""

import json
import os

import pytest

from needle.model import finetune, providers
from needle.model.credentials import CredentialError, credential_for

API_KEY = os.environ.get("ORCAROUTER_API_KEY")

requires_key = pytest.mark.skipif(
    not API_KEY, reason="ORCAROUTER_API_KEY is not set; live checks skipped")

#: A real live test may not assume a particular vendor is present in the
#: workspace, so the model is whichever chat model discovery offers first.
TOOLS = [{"name": "set_lights",
          "parameters": {"type": "object",
                         "properties": {"room": {"type": "string"},
                                        "state": {"type": "string",
                                                  "enum": ["on", "off"]}},
                         "required": ["room", "state"]}}]


@pytest.fixture(scope="module")
def credential():
    if not API_KEY:
        pytest.skip("ORCAROUTER_API_KEY is not set")
    return credential_for("orcarouter")


@requires_key
def test_live_catalog_is_reachable_and_filterable(credential):
    provider = providers.get_provider("orcarouter")
    catalog = providers.discover_models(provider, api_key=credential.api_key)

    assert catalog.degraded is False, "live discovery fell back: %s" % catalog.error
    assert catalog.source == "live"
    assert catalog.models, "the workspace returned no models"

    chat = providers.filter_models(catalog.models, capability="chat")
    assert chat, "no chat-capable model in the live catalog"
    # Vendor/model namespace is preserved verbatim.
    assert all("/" in m["id"] for m in catalog.models)

    # Every chat model must prove a chat route; specialist routes stay out.
    for model in chat:
        endpoints = set(model["supported_endpoint_types"])
        assert endpoints & providers.CHAT_ENDPOINT_TYPES
        assert "image-generation" not in endpoints or endpoints & {"openai"}


@requires_key
def test_live_inference_runs_through_the_implemented_provider_path(credential):
    """Real generation through the code path the CLI and playground use.

    The catalog is the workspace's, but a key may be scoped to a subset of it,
    so candidates are tried in catalog order until one is permitted. At least
    one must succeed; a 403 for a scoped-out model is a legitimately different
    outcome from a failure to route the request at all.
    """
    provider = providers.get_provider("orcarouter")
    catalog = providers.discover_models(provider, api_key=credential.api_key)
    models = providers.filter_models(catalog.models, capability="chat")
    if not models:
        pytest.skip("no chat-capable model available to this workspace")

    attempts, last_error = [], None
    for model in [m["id"] for m in models[:32]]:
        try:
            rows = finetune.generate_examples(TOOLS, n=2, model=model,
                                              provider_id="orcarouter",
                                              api_key=credential.api_key)
        except CredentialError as exc:
            last_error = exc
            assert model not in str(exc) or "denied access" in str(exc)
            attempts.append((model, getattr(exc, "status", None)))
            continue
        assert isinstance(rows, list) and rows, "live generation returned no rows"
        for row in rows:
            assert "query" in row and "answers" in row
            assert row["tools"] == TOOLS
        return
    pytest.fail("no model callable with this key; attempts=%r last=%s"
                % (attempts, last_error))


@requires_key
def test_a_403_for_a_scoped_out_model_is_not_a_credential_failure(credential):
    """Per-key model scoping is reported as a permissions problem, not a reauth.

    ``/v1/models`` describes the workspace, so a listed model can still be
    denied to one key. That must not mark the credential for reauthentication.
    """
    provider = providers.get_provider("orcarouter")
    catalog = providers.discover_models(provider, api_key=credential.api_key)
    models = providers.filter_models(catalog.models, capability="chat")
    if not models:
        pytest.skip("no chat-capable model available to this workspace")

    for model in [m["id"] for m in models[:32]]:
        try:
            finetune.generate_examples(TOOLS, n=1, model=model,
                                       provider_id="orcarouter",
                                       api_key=credential.api_key)
        except CredentialError as exc:
            if getattr(exc, "status", None) == 403:
                assert exc.needs_reauth is False
                assert "denied access" in str(exc)
                return          # exercised the path we wanted to test
            continue
        return                  # this key is unscoped; nothing to assert
    pytest.skip("no scoped-out model encountered")


@requires_key
def test_live_request_targets_the_inference_origin_not_the_auth_origin(credential):
    provider = providers.get_provider("orcarouter")
    auth_base, api_base = providers.resolve_orcarouter_origins()

    assert provider.api_base == api_base
    assert provider.chat_completions_url.startswith("https://api.orcarouter.ai/v1/")
    assert providers.exchange_url(auth_base).startswith("https://www.orcarouter.ai/")

    # Reachability of each origin, independently.
    catalog = providers.discover_models(provider, api_key=credential.api_key)
    assert catalog.degraded is False


@requires_key
def test_live_rejects_a_bogus_key_instead_of_succeeding(credential):
    """A made-up key must fail against the real relay, not appear to work."""
    provider = providers.get_provider("orcarouter")
    with pytest.raises(Exception) as excinfo:
        providers.fetch_catalog(provider, api_key="sk-orca-" + "0" * 32)
    assert "sk-orca-" + "0" * 32 not in str(excinfo.value)


def _tiny_png_data_uri():
    """A real 8x8 PNG as a data URI, so the check needs no image egress."""
    import base64
    import struct
    import zlib

    raw = b"".join(b"\x00" + bytes((200, 30, 30)) * 8 for _ in range(8))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + \
            struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


@requires_key
def test_live_image_prompt_reaches_a_vision_model(credential):
    """The multimodal entry point end to end, through the implemented path.

    The catalog picks the model; the guard rejects anything that does not
    declare an image input; the request itself carries the multimodal content
    form. If this workspace offers no image-input chat model the check is
    skipped rather than pointed at a model chosen by hand.
    """
    provider = providers.get_provider("orcarouter")
    catalog = providers.discover_models(provider, api_key=credential.api_key)
    vision = providers.filter_models(catalog.models, capability="chat",
                                     input_modalities=["image"])
    if not vision:
        pytest.skip("no image-input chat model in this workspace")

    image = _tiny_png_data_uri()
    last_error = None
    for model in [m["id"] for m in vision[:8]]:
        try:
            rows = finetune.generate_examples(
                TOOLS, n=1, model=model, provider_id="orcarouter",
                api_key=credential.api_key, input_modality="image",
                image_url=image)
        except CredentialError as exc:
            last_error = exc          # a scoped-out model is not a code failure
            continue
        assert rows, "a live multimodal generation returned no rows"
        assert rows[0]["tools"] == TOOLS
        return
    pytest.fail("no image-input model callable with this key: %s" % last_error)


@requires_key
def test_live_text_only_model_is_refused_an_image_prompt(credential):
    """The modality guard is a catalog fact, not a model-name guess."""
    provider = providers.get_provider("orcarouter")
    catalog = providers.discover_models(provider, api_key=credential.api_key)
    text_only = [m["id"] for m in
                 providers.filter_models(catalog.models, capability="chat")
                 if "image" not in ((m.get("architecture") or {}).get(
                     "input_modalities") or [])]
    if not text_only:
        pytest.skip("every chat model in this workspace declares an image input")

    with pytest.raises(CredentialError) as excinfo:
        finetune.generate_examples(TOOLS, n=1, model=text_only[0],
                                   provider_id="orcarouter",
                                   api_key=credential.api_key,
                                   input_modality="image",
                                   image_url=_tiny_png_data_uri())
    assert "does not declare image input" in str(excinfo.value)
