"""Provider registry, origin policy and model-catalog discovery.

This is the single place that knows where an upstream AI gateway lives. Every
call site that talks to a provider — data synthesis in ``finetune.py``, the CLI,
and the playground server — goes through :func:`chat_completions_url` and
:func:`discover_models` rather than holding a URL constant of its own.

OrcaRouter is a first-class provider here, not a custom base URL: it has its own
id, label, credential environment variable and seed catalog. Its authentication
origin and its inference origin are different hosts and are never derived from
one another (``www.orcarouter.ai`` vs ``api.orcarouter.ai``).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DEFAULT_AUTH_BASE = "https://www.orcarouter.ai"
DEFAULT_API_BASE = "https://api.orcarouter.ai/v1"

ORCAROUTER_API_KEY_ENV = "ORCAROUTER_API_KEY"

#: Endpoint types that can serve an OpenAI-wire chat completion.
CHAT_ENDPOINT_TYPES = frozenset({"openai", "anthropic", "gemini", "openai-response"})

#: Endpoint types that are explicitly *not* text chat, whatever else they claim.
NON_CHAT_ENDPOINT_TYPES = frozenset({
    "image-generation", "openai-video", "jina-rerank", "embeddings",
})

#: capability id -> the endpoint type that proves it.
CAPABILITY_ENDPOINT_TYPES = {
    "chat": CHAT_ENDPOINT_TYPES,
    "embedding": frozenset({"embeddings"}),
    "image": frozenset({"image-generation"}),
    "video": frozenset({"openai-video"}),
    "rerank": frozenset({"jina-rerank"}),
}

#: Bounds on a catalog response, so a hostile or broken endpoint cannot make the
#: client allocate without limit.
CATALOG_TIMEOUT_SECONDS = 20
CATALOG_MAX_BYTES = 2 * 1024 * 1024
CATALOG_MAX_ITEMS = 2000

#: Verified cold-start fallback. Used only when live discovery fails, and always
#: reported as degraded so the UI can say so. ``supported_endpoint_types`` and
#: modality metadata are carried so a fallback entry is filtered exactly like a
#: discovered one, and so reasoning-effort ladders survive an outage.
SEED_MODEL_IDS = (
    "openai/gpt-5.5",
    "anthropic/claude-opus-4.8",
    "google/gemini-3.5-flash",
    "deepseek/deepseek-v4-pro",
    "orcarouter/auto",
)


def _seed(model_id, context_length, modalities, reasoning_efforts=()):
    return {
        "id": model_id,
        "object": "model",
        "owned_by": model_id.split("/", 1)[0],
        "supported_endpoint_types": ["openai"],
        "context_length": context_length,
        "architecture": {"input_modalities": list(modalities),
                         "output_modalities": ["text"]},
        "reasoning_efforts": list(reasoning_efforts),
        "verified": True,
    }


SEED_CATALOG = (
    _seed("openai/gpt-5.5", 400000, ("text", "image"),
          ("low", "medium", "high", "xhigh")),
    _seed("anthropic/claude-opus-4.8", 200000, ("text", "image")),
    _seed("google/gemini-3.5-flash", 1000000, ("text", "image")),
    _seed("deepseek/deepseek-v4-pro", 1048576, ("text",)),
    _seed("orcarouter/auto", 1000000, ("text",)),
)


@dataclass(frozen=True)
class Provider:
    """One selectable provider."""

    id: str
    label: str
    api_base: str
    env_key: str
    default_model: str
    auth_base: Optional[str] = None
    aliases: Tuple[str, ...] = ()
    seed: Tuple[dict, ...] = ()
    #: When set, the completions route is used exactly as given rather than
    #: derived from ``api_base``. This preserves the historical semantics of
    #: ``OPENROUTER_URL``, which was always the completions URL itself.
    completions_url: Optional[str] = None

    @property
    def chat_completions_url(self) -> str:
        if self.completions_url:
            return self.completions_url
        return self.api_base.rstrip("/") + "/chat/completions"

    @property
    def models_url(self) -> str:
        return self.api_base.rstrip("/") + "/models"


def _orcarouter() -> Provider:
    auth_base, api_base = resolve_orcarouter_origins()
    return Provider(
        id="orcarouter",
        label="OrcaRouter",
        api_base=api_base,
        auth_base=auth_base,
        env_key=ORCAROUTER_API_KEY_ENV,
        default_model="orcarouter/auto",
        aliases=("orca",),
        seed=SEED_CATALOG,
    )


OPENROUTER_DEFAULT_API_BASE = "https://openrouter.ai/api/v1"


def _openrouter() -> Provider:
    """Resolve the OpenRouter entry, honouring the legacy ``OPENROUTER_URL``.

    That variable has always been the completions URL itself, so it is used
    verbatim; ``api_base`` is only derived for the model-list route, which
    OpenRouter does not use here.
    """
    configured = (os.environ.get("OPENROUTER_URL") or "").strip()
    if configured:
        completions_url = configured
        api_base = configured[: -len("/chat/completions")] \
            if configured.endswith("/chat/completions") else configured
    else:
        completions_url = None
        api_base = OPENROUTER_DEFAULT_API_BASE
    return Provider(
        id="openrouter",
        label="OpenRouter",
        api_base=api_base,
        env_key="OPENROUTER_API_KEY",
        default_model="deepseek/deepseek-flash-latest",
        completions_url=completions_url,
    )


def providers() -> Dict[str, Provider]:
    """The registry, rebuilt on access so environment overrides stay live."""
    return {"openrouter": _openrouter(), "orcarouter": _orcarouter()}


def get_provider(provider_id: Optional[str]) -> Provider:
    registry = providers()
    if provider_id is None:
        return registry["openrouter"]
    key = provider_id.strip().lower()
    for provider in registry.values():
        if key == provider.id or key in provider.aliases:
            return provider
    raise ValueError(
        "unknown provider %r (known: %s)"
        % (provider_id, ", ".join(sorted(registry)))
    )


def chat_completions_url(provider: Provider) -> str:
    return provider.chat_completions_url


# --------------------------------------------------------------------------
# Origin policy
# --------------------------------------------------------------------------

def _is_loopback_host(host: str) -> bool:
    host = (host or "").strip("[]").lower()
    return host in ("localhost", "127.0.0.1", "::1")


def validate_origin(origin: str, what: str) -> str:
    """Reject an origin that would send a credential somewhere unsafe.

    HTTPS is required for anything remote; plain HTTP is allowed only for
    loopback, which is what a self-hosted or test deployment uses.
    """
    origin = (origin or "").strip().rstrip("/")
    if not origin:
        raise ValueError("%s must not be empty" % what)
    parsed = urllib.parse.urlsplit(origin)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("%s must be an absolute URL, got %r" % (what, origin))
    if parsed.scheme == "https":
        return origin
    if parsed.scheme == "http" and _is_loopback_host(parsed.hostname or ""):
        return origin
    raise ValueError(
        "%s must use https (http is allowed only for loopback), got %r" % (what, origin)
    )


def resolve_orcarouter_origins(environ=None):
    """Resolve the auth and inference origins.

    ``ORCA_BASE_URL`` is a shared self-hosted origin: authentication lives at
    the origin itself and inference at ``<origin>/v1``. Explicit
    ``ORCA_AUTH_BASE_URL`` / ``ORCA_API_BASE_URL`` always win over it. The two
    public origins are never derived from each other by swapping a hostname.
    """
    env = os.environ if environ is None else environ
    shared = (env.get("ORCA_BASE_URL") or "").strip().rstrip("/")

    auth = (env.get("ORCA_AUTH_BASE_URL") or "").strip().rstrip("/")
    if not auth:
        auth = shared or DEFAULT_AUTH_BASE
    auth = validate_origin(auth, "ORCA_AUTH_BASE_URL")

    api = (env.get("ORCA_API_BASE_URL") or "").strip().rstrip("/")
    if not api:
        api = (shared + "/v1") if shared else DEFAULT_API_BASE
    api = validate_origin(api, "ORCA_API_BASE_URL")

    return auth, api


def authorize_url(auth_base, callback_url, code_challenge, state, app_name,
                  scope="api", login_hint=None, workspace_hint=None, prompt=None):
    """Build the consent-screen URL. The verifier never appears here."""
    url = urllib.parse.urlsplit(auth_base.rstrip("/") + "/auth")
    query = [
        ("callback_url", callback_url),
        ("code_challenge", code_challenge),
        ("code_challenge_method", "S256"),
        ("state", state),
        ("app_name", app_name),
        ("scope", scope),
    ]
    if login_hint:
        query.append(("login_hint", login_hint))
    if workspace_hint:
        query.append(("workspace_hint", workspace_hint))
    if prompt:
        query.append(("prompt", prompt))
    return urllib.parse.urlunsplit(
        (url.scheme, url.netloc, url.path, urllib.parse.urlencode(query), "")
    )


def exchange_url(auth_base) -> str:
    """The token endpoint. Note ``/api/v1/auth/keys``, *not* ``/v1/auth/keys``."""
    return auth_base.rstrip("/") + "/api/v1/auth/keys"


def valid_callback_url(callback_url: str) -> bool:
    """Mirror the server's callback rules so we fail before opening a browser."""
    try:
        parsed = urllib.parse.urlsplit(callback_url)
    except ValueError:
        return False
    if not parsed.scheme or not parsed.netloc:
        return False
    if parsed.username or parsed.password or parsed.fragment:
        return False
    if parsed.scheme == "https":
        return True
    return parsed.scheme == "http" and _is_loopback_host(parsed.hostname or "")


# --------------------------------------------------------------------------
# Model catalog
# --------------------------------------------------------------------------

@dataclass
class ModelCatalog:
    """A catalog plus where it came from, so callers can show degraded state."""

    models: List[dict] = field(default_factory=list)
    source: str = "seed"          # "live" | "seed" | "last-known-good"
    degraded: bool = False
    error: Optional[str] = None

    def ids(self):
        return [m.get("id") for m in self.models]


def _string_list(value):
    if not isinstance(value, (list, tuple)):
        return []
    return [v for v in value if isinstance(v, str) and v]


def normalize_model(record) -> Optional[dict]:
    """Accept a catalog record only if it has the shape we can actually route."""
    if not isinstance(record, dict):
        return None
    model_id = record.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        return None
    if "/" not in model_id:
        # OrcaRouter keeps the vendor/model namespace; a bare name is not
        # something we can route, so fail closed rather than guess a vendor.
        return None
    normalized = {
        "id": model_id,
        "object": record.get("object", "model"),
        "owned_by": record.get("owned_by"),
        "supported_endpoint_types": _string_list(record.get("supported_endpoint_types")),
    }
    for key in ("name", "description", "context_length", "max_completion_tokens"):
        if record.get(key) is not None:
            normalized[key] = record[key]
    architecture = record.get("architecture")
    if isinstance(architecture, dict):
        normalized["architecture"] = {
            "input_modalities": _string_list(architecture.get("input_modalities")),
            "output_modalities": _string_list(architecture.get("output_modalities")),
        }
    efforts = _string_list(record.get("reasoning_efforts"))
    if efforts:
        normalized["reasoning_efforts"] = efforts
    return normalized


def parse_catalog(payload) -> List[dict]:
    """Parse a ``/v1/models`` body, bounded and shape-checked."""
    if isinstance(payload, dict):
        data = payload.get("data", payload.get("models"))
    else:
        data = payload
    if not isinstance(data, list):
        return []
    models = []
    for record in data[:CATALOG_MAX_ITEMS]:
        normalized = normalize_model(record)
        if normalized is not None:
            models.append(normalized)
    return models


def fetch_catalog(provider, api_key=None, capability=None, opener=None) -> List[dict]:
    """One bounded live request to ``GET {api_base}/models``."""
    url = provider.models_url
    if capability:
        url += "?" + urllib.parse.urlencode({"capability": capability})
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(url, headers=headers)
    open_url = opener or urllib.request.urlopen
    with open_url(request, timeout=CATALOG_TIMEOUT_SECONDS) as response:
        body = response.read(CATALOG_MAX_BYTES + 1)
    if len(body) > CATALOG_MAX_BYTES:
        raise ValueError("catalog response exceeded %d bytes" % CATALOG_MAX_BYTES)
    return parse_catalog(json.loads(body.decode("utf-8")))


def discover_models(provider, api_key=None, capability=None, opener=None,
                    last_known_good=None) -> ModelCatalog:
    """Live discovery, falling back to a verified catalog when it fails.

    A live success is authoritative and the seed is never mixed into it.
    """
    if provider.id != "orcarouter":
        return ModelCatalog(models=[], source="live", degraded=False)
    try:
        models = fetch_catalog(provider, api_key=api_key, capability=capability,
                               opener=opener)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError,
            json.JSONDecodeError) as exc:
        return _fallback_catalog(provider, last_known_good, _describe_error(exc))
    if not models:
        return _fallback_catalog(provider, last_known_good,
                                 "catalog endpoint returned no usable models")
    return ModelCatalog(models=models, source="live", degraded=False)


def _fallback_catalog(provider, last_known_good, error):
    models = list(last_known_good) if last_known_good else list(provider.seed)
    source = "last-known-good" if last_known_good else "seed"
    return ModelCatalog(models=models, source=source, degraded=True, error=error)


def _describe_error(exc) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return "catalog request failed with HTTP %s" % exc.code
    return "catalog request failed: %s" % type(exc).__name__


def filter_models(models, capability="chat", input_modalities=None) -> List[dict]:
    """Filter a catalog for one entry point's capability.

    Rules are driven by catalog metadata only — never by the model name.
    A record that does not declare a capability is excluded rather than assumed
    compatible, so an unknown modality never reaches a selector.
    """
    if capability not in CAPABILITY_ENDPOINT_TYPES:
        raise ValueError("unknown capability %r" % capability)

    required_modalities = [m for m in (input_modalities or []) if m and m != "text"]
    accepted_endpoints = CAPABILITY_ENDPOINT_TYPES[capability]
    selected = []

    for model in models:
        if not isinstance(model, dict):
            continue
        endpoints = set(_string_list(model.get("supported_endpoint_types")))
        # A record must *prove* the route it is selected for. A model
        # advertising only image-generation has no chat route and is excluded
        # from the chat selector by this intersection alone.
        if not (endpoints & accepted_endpoints):
            continue

        if required_modalities:
            declared = _declared_input_modalities(model)
            if declared is None:
                continue  # fail closed: undeclared modality
            if any(modality not in declared for modality in required_modalities):
                continue
        selected.append(model)
    return selected


def _declared_input_modalities(model):
    architecture = model.get("architecture")
    if not isinstance(architecture, dict):
        return None
    modalities = architecture.get("input_modalities")
    if not isinstance(modalities, (list, tuple)) or not modalities:
        return None
    return set(_string_list(modalities))
