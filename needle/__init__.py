from __future__ import annotations

import ctypes
import decimal
import json
import os
import re
import sys
import warnings

from .agent.tools import Field, build_schema, pydantic_schema, tool, _is_pydantic_model
from ._telemetry import track as _track
from ._worker import FineTuneWorker

__version__ = "2.0.12"
__all__ = ["Needle", "ExtractionValidationError", "tool", "Field", "extract",
           "__version__"]


class ExtractionValidationError(ValueError):
    """The engine produced structured values that are not grounded in the input."""


_CACT_GENERATIONS = {
    0x05E12A83: 2,
    0x05E12A84: 3,
}


def _weight_generation(path):
    with open(path, "rb") as handle:
        tag_bytes = handle.read(4)
    if len(tag_bytes) != 4:
        raise RuntimeError(f"{path} is not a complete .cact archive")
    tag = int.from_bytes(tag_bytes, "little")
    try:
        return _CACT_GENERATIONS[tag]
    except KeyError as exc:
        raise RuntimeError(
            f"{path} has unknown .cact format tag 0x{tag:08x}; "
            "cannot choose a compatible Needle engine") from exc


def _library_path(generation=2):
    from .agent import fetch

    generation = int(generation)
    override = os.environ.get(f"NEEDLE{generation}_LIB_PATH")
    if generation == 2 and not override:
        # NEEDLE_LIB_PATH predates multi-generation dispatch and therefore
        # names the Needle 2 engine.  Never route a v3 archive through it.
        override = os.environ.get("NEEDLE_LIB_PATH")
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    lib_name = fetch._lib_name()
    stem, suffix = os.path.splitext(lib_name)
    local_names = [f"{stem}{generation}{suffix}"]
    if generation == 2:
        # Wheels published before the split shipped Needle 2 as libneedle.*.
        local_names.append(lib_name)
    for name in local_names:
        local = os.path.join(here, name)
        if os.path.exists(local):
            return local
    version = fetch.engine_version(generation)
    cache = os.path.join(os.path.expanduser("~"), ".cache", "cactus-needle",
                         f"v{generation}", version)
    cached = os.path.join(cache, fetch._lib_name())
    if os.path.exists(cached):
        return cached
    os.makedirs(cache, exist_ok=True)
    return fetch.fetch_library(version, cache, generation=generation)


_lib_handles = {}
_active = {}


class _NeedleAudio(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_uint64),
        ("sample_rate", ctypes.c_int),
        ("channels", ctypes.c_int),
        ("format", ctypes.c_int),
    ]


_AUDIO_FORMATS = {"wav": 1, "pcm16": 2, "float32": 3}


def _prepare_audio(audio, audio_format="wav", sample_rate=0, channels=1):
    if audio is None:
        return None
    if isinstance(audio, (str, os.PathLike)):
        with open(os.fspath(audio), "rb") as handle:
            data = handle.read()
    else:
        try:
            data = bytes(audio)
        except (TypeError, ValueError) as exc:
            raise TypeError("audio must be a path or bytes-like object") from exc
    if not data:
        raise ValueError("audio is empty")
    try:
        format_code = _AUDIO_FORMATS[str(audio_format).lower()]
    except KeyError as exc:
        raise ValueError("audio_format must be 'wav', 'pcm16', or 'float32'") from exc
    if format_code != _AUDIO_FORMATS["wav"] and (
            int(sample_rate) <= 0 or int(channels) <= 0):
        raise ValueError("PCM audio requires positive sample_rate and channels")
    return {"data": data, "format": format_code,
            "sample_rate": int(sample_rate), "channels": int(channels)}


def _native_audio(payload):
    if payload is None:
        return None, None
    data = payload["data"]
    storage = ctypes.create_string_buffer(data, len(data))
    descriptor = _NeedleAudio(
        ctypes.cast(storage, ctypes.c_void_p), len(data),
        payload["sample_rate"], payload["channels"], payload["format"])
    return ctypes.byref(descriptor), (storage, descriptor)


def _lib(generation=2):
    generation = int(generation)
    if generation not in _lib_handles:
        lib = ctypes.CDLL(_library_path(generation))
        lib.needle_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.needle_init.restype = ctypes.c_int
        if generation >= 3:
            lib.needle_complete.argtypes = [
                ctypes.c_char_p, ctypes.POINTER(_NeedleAudio), ctypes.c_int,
                ctypes.c_char_p, ctypes.c_int]
            lib.needle_embed.argtypes = [
                ctypes.c_char_p, ctypes.POINTER(_NeedleAudio),
                ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            lib.needle_embed.restype = ctypes.c_int
        else:
            lib.needle_complete.argtypes = [
                ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.needle_complete.restype = ctypes.c_int
        lib.needle_reset.argtypes = []
        lib.needle_reset.restype = None
        lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
        lib.needle_load.restype = ctypes.c_int
        _lib_handles[generation] = lib
    return _lib_handles[generation]


class Needle:
    def __init__(self, tools=None, system=None, weights=None, tool_index_path=None, buffer_size=65536):
        self._functions = {}
        self._weights = os.fspath(weights) if weights is not None else None
        self._generation = _weight_generation(self._weights) if self._weights else 2
        self._worker = None
        self._closed = False
        if weights:
            warnings.warn("finetuning does not update the confidence head, so scores are "
                          "uncalibrated for tuned weights; this agent reports confidence as None",
                          stacklevel=2)
        self._system_text = system or ""
        self._system = self._system_text.encode("utf-8")
        tools_json = tools if isinstance(tools, str) else json.dumps(self._resolve(tools))
        self._tools_json = tools_json.encode("utf-8")
        try:
            parsed_tools = json.loads(tools_json)
            self._n_tools = len(parsed_tools)
        except (json.JSONDecodeError, TypeError):
            parsed_tools, self._n_tools = [], None
        self._tool_schemas = [entry for entry in (parsed_tools or [])
                              if isinstance(entry, dict)]
        self._seen_years = set()
        self._tool_index_path = (os.fspath(tool_index_path).encode("utf-8")
                                 if tool_index_path else None)
        self._buffer = ctypes.create_string_buffer(buffer_size)
        if self._weights:
            self._worker = FineTuneWorker(
                _library_path(self._generation), self._weights,
                self._system.decode("utf-8"), self._tools_json.decode("utf-8"),
                os.fspath(tool_index_path) if tool_index_path else None,
                buffer_size, generation=self._generation)
        else:
            self._bind()

    def _bind(self):
        if self._closed:
            raise RuntimeError("Needle instance is closed")
        if self._worker is not None:
            return
        generation = self._generation
        lib = _lib(generation)
        if _active.get(generation) is self:
            return
        if lib.needle_init(self._system, self._tools_json, self._tool_index_path) < 0:
            _active.pop(generation, None)
            raise RuntimeError("needle_init failed")
        _active[generation] = self
        self._seen_years = set()

    def _resolve(self, tools):
        schemas = []
        for entry in tools or []:
            if _is_pydantic_model(entry):
                schema = pydantic_schema(entry)
                self._functions[schema["name"]] = entry
                schemas.append(schema)
            elif callable(entry):
                schema = getattr(entry, "_needle_tool", None) or build_schema(entry)
                self._functions[schema["name"]] = entry
                schemas.append(schema)
            elif isinstance(entry, dict):
                schemas.append(entry)
        return schemas

    def _track_props(self):
        return {"n_tools": self._n_tools, "tuned": bool(self._weights),
                "generation": self._generation}

    def complete(self, text: str = "", max_new_tokens: int = 256,
                 audio=None, audio_format="wav", sample_rate=0,
                 channels=1) -> dict:
        _track("complete", self._track_props())
        payload = _prepare_audio(audio, audio_format, sample_rate, channels)
        return self._complete(text, max_new_tokens, payload)

    def _complete(self, text: str, max_new_tokens: int = 256,
                  audio=None, ground: bool = True) -> dict:
        if audio is not None and self._generation < 3:
            raise ValueError("audio requires a Needle 3 model")
        self._bind()
        self._seen_years |= _source_years(text or "")
        if self._worker is not None:
            raw = self._worker.complete(text, max_new_tokens, audio)
        else:
            lib = _lib(self._generation)
            if self._generation >= 3:
                native_audio, keepalive = _native_audio(audio)
                rc = lib.needle_complete(
                    text.encode("utf-8"), native_audio, int(max_new_tokens),
                    self._buffer, len(self._buffer))
            else:
                rc = lib.needle_complete(
                    text.encode("utf-8"), int(max_new_tokens), self._buffer,
                    len(self._buffer))
            if rc < 0:
                detail = self._buffer.value.decode("utf-8", "replace")
                raise RuntimeError(detail or f"needle_complete failed (code {rc})")
            raw = self._buffer.value.decode("utf-8")
        try:
            response = json.loads(raw)
        except json.JSONDecodeError as err:
            raise RuntimeError(
                f"engine returned an unparseable envelope ({err}); this is an "
                f"engine bug - please report it with the prompt and schema") from err
        if self._weights:
            response["confidence"] = None
        if ground:
            _annotate_ungrounded(response, self._tool_schemas, self._seen_years,
                                 self._system_text)
        return response

    def embed(self, text: str = "", audio=None, audio_format="wav",
              sample_rate=0, channels=1) -> list[float]:
        if self._generation < 3:
            raise ValueError("embeddings require a Needle 3 model")
        payload = _prepare_audio(audio, audio_format, sample_rate, channels)
        self._bind()
        if self._worker is not None:
            return self._worker.embed(text, payload)
        lib = _lib(self._generation)
        native_audio, keepalive = _native_audio(payload)
        dim = lib.needle_embed(text.encode("utf-8"), native_audio, None, 0)
        if dim <= 0:
            raise RuntimeError(f"needle_embed failed (code {dim})")
        output = (ctypes.c_float * dim)()
        rc = lib.needle_embed(text.encode("utf-8"), native_audio, output, dim)
        if rc != dim:
            raise RuntimeError(f"needle_embed failed (code {rc})")
        return list(output)

    def run(self, query: str = "", max_steps: int = 8,
            max_new_tokens: int = 256, audio=None, audio_format="wav",
            sample_rate=0, channels=1, strict: bool = True) -> dict:
        _track("run", self._track_props())
        payload = _prepare_audio(audio, audio_format, sample_rate, channels)
        response = self._complete(query, max_new_tokens, payload)
        executed = []
        for _ in range(max_steps):
            calls = response.get("function_calls") or []
            if response.get("type") != "call" or not calls:
                break
            ungrounded = _ungrounded_paths(response)
            results = []
            for call in calls:
                name = str(call.get("name"))
                fabricated = sorted(ungrounded.get(name, ()))
                if strict and fabricated:
                    grounded = _grounded_number_paths(
                        call.get("arguments") or {}, query, self._system_text)
                    fabricated = [path for path in fabricated
                                  if path not in grounded]
                if strict and fabricated:
                    results.append({"error": "ungrounded " + ", ".join(fabricated)})
                    continue
                fn = self._functions.get(call.get("name"))
                if fn is None:
                    results.append({"error": "unknown tool: " + name})
                    continue
                try:
                    results.append(fn(**(call.get("arguments") or {})))
                except Exception as exc:
                    results.append({"error": str(exc)})
            executed.extend(results)
            response = self._complete(json.dumps(results, default=_jsonable),
                                      max_new_tokens, ground=False)
        response["results"] = executed
        return response

    def extract(self, text: str, schema: type | dict, max_new_tokens: int = 256,
                strict: bool = True) -> object:
        return extract(text, schema, max_new_tokens=max_new_tokens,
                       weights=self._weights, strict=strict)

    def reset(self):
        self._bind()
        if self._worker is not None:
            self._worker.reset()
        else:
            _lib(self._generation).needle_reset()
        self._seen_years = set()

    def close(self):
        if self._worker is not None:
            try:
                self._worker.close()
            finally:
                self._worker = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict") and _is_pydantic_model(type(value)):
        return value.dict()
    return str(value)


def _schema_parameters(schema):
    raw = pydantic_schema(schema) if _is_pydantic_model(schema) else schema
    return raw.get("parameters", raw) if isinstance(raw, dict) else {}


def _resolve_ref(node, root):
    seen = set()
    while isinstance(node, dict) and "$ref" in node:
        if node["$ref"] in seen:
            break
        seen.add(node["$ref"])
        target = root
        for part in node["$ref"].removeprefix("#/").split("/"):
            target = target.get(part.replace("~1", "/").replace("~0", "~"), {})
        if target is node:
            break
        node = target
    return node


def _source_years(text):
    months = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
              r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
              r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
    patterns = [
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{months}[\s,]+(\d{{1,4}})(?![0-9A-Za-z])",
        rf"\b{months}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s+(\d{{1,4}})(?![0-9A-Za-z])",
        rf"\b{months}[\s,]+(\d{{3,4}})(?![0-9A-Za-z])",
        r"\byear\s+(\d{1,4})(?![0-9A-Za-z])",
        r"(?<![0-9])(\d{1,4})(?=[-/]\d{1,2}[-/]\d{1,2}(?![0-9]))",
    ]
    lowered = text.lower()
    return {int(match.group(1)) for pattern in patterns
            for match in re.finditer(pattern, lowered)}


def _licensed_years(seen_years, system=None):
    years = set(seen_years)
    if years and system:
        years |= _source_years(system)
    return years


def _walk_grounding(schema, arguments, years):
    root = _schema_parameters(schema)
    checked, failures = set(), set()

    def walk(value, node, path):
        node = _resolve_ref(node, root)
        variants = node.get("anyOf") or node.get("oneOf") or []
        concrete = [v for v in variants if _resolve_ref(v, root).get("type") != "null"]
        if len(concrete) == 1:
            node = _resolve_ref(concrete[0], root)
        fmt = node.get("format")
        if fmt in ("date", "date-time") and isinstance(value, str):
            match = re.match(r"^(\d{4})-", value)
            if match and years:
                checked.add(path)
                if int(match.group(1)) not in years:
                    failures.add(path)
            return
        if isinstance(value, dict):
            properties = node.get("properties", {})
            for key, item in value.items():
                if key in properties:
                    walk(item, properties[key], f"{path}.{key}" if path else key)
        elif isinstance(value, list) and "items" in node:
            for index, item in enumerate(value):
                walk(item, node["items"], f"{path}[{index}]")

    walk(arguments, root, "")
    return checked, failures


def _temporal_grounding(text, schema, arguments, system=None):
    years = _licensed_years(_source_years(text), system)
    return _walk_grounding(schema, arguments, years)


_NUMBER_TOKEN = re.compile(
    r"(?<![\w.,])(?:[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)(?!\d)")


def _numeric_paths(arguments, path=""):
    """Yield ``(path, value)`` for every numeric leaf, arguments-relative.

    Paths match the form both call sites derive after stripping the engine's
    tool prefix, so a flagged ``Tool.path`` resolves to a helper path.
    """
    if isinstance(arguments, bool):
        return
    if isinstance(arguments, (int, float)):
        yield path, arguments
    elif isinstance(arguments, dict):
        for key, item in arguments.items():
            yield from _numeric_paths(item, f"{path}.{key}" if path else str(key))
    elif isinstance(arguments, list):
        for index, item in enumerate(arguments):
            yield from _numeric_paths(item, f"{path}[{index}]")


def _source_numbers(*sources):
    """Numeric values written in the source texts, separators normalized."""
    text = "\n".join(source for source in sources if source)
    return {decimal.Decimal(match.group(0).replace(",", ""))
            for match in _NUMBER_TOKEN.finditer(text)}


def _grounded_number_paths(arguments, *sources):
    numbers = _source_numbers(*sources)
    if not numbers:
        return set()
    grounded = set()
    for path, value in _numeric_paths(arguments):
        try:
            if decimal.Decimal(str(value)) in numbers:
                grounded.add(path)
        except (decimal.InvalidOperation, ValueError):
            continue
    return grounded


def _ungrounded_paths(response):
    validation = response.get("validation") or {}
    grouped = {}
    for name in validation.get("ungrounded") or []:
        tool, _, path = str(name).partition(".")
        grouped.setdefault(tool, set()).add(path or tool)
    return grouped


def _annotate_ungrounded(response, tool_schemas, seen_years, system=None):
    calls = response.get("function_calls") or []
    years = _licensed_years(seen_years, system)
    if not calls or not years:
        return
    schemas = {entry.get("name"): entry for entry in tool_schemas}
    found = []
    for call in calls:
        schema = schemas.get(call.get("name"))
        if schema is None:
            continue
        _, failures = _walk_grounding(schema, call.get("arguments") or {}, years)
        found.extend(f"{call['name']}.{path}" for path in sorted(failures))
    if not found:
        return
    validation = response.setdefault("validation", {}) or {}
    response["validation"] = validation
    existing = list(validation.get("ungrounded") or [])
    validation["ungrounded"] = existing + [
        name for name in found if name not in existing]


def _validate_extraction(text, schema, arguments, response, system=None):
    checked, failures = _temporal_grounding(text, schema, arguments, system)
    validation = response.get("validation") or {}
    flagged = validation.get("ungrounded") or []
    grounded = _grounded_number_paths(arguments, text, system) if flagged else set()
    for name in flagged:
        path = name.split(".", 1)[-1]
        if path in grounded:
            continue
        if path not in checked or path in failures:
            failures.add(path)
    if validation.get("negation"):
        failures.add("negated request")
    if failures:
        detail = ", ".join(sorted(failures))
        raise ExtractionValidationError(
            f"extraction returned values not grounded in the input: {detail}")


def extract(text: str, schema: type | dict, system: str | None = None,
            max_new_tokens: int = 256, weights: str | None = None,
            strict: bool = True) -> object:
    """One-shot structured extraction using the matching native engine.

    With ``strict=True`` (the default), temporal values that contradict a literal
    year in the input, plus engine-reported fabricated or negated values, raise
    :class:`ExtractionValidationError` instead of being returned silently.
    """
    selected = weights
    generation = _weight_generation(selected) if selected else 2
    _track("extract", {"n_tools": 1, "tuned": bool(selected),
                       "generation": generation})
    agent = Needle(tools=[schema], system=system, weights=selected)
    try:
        response = agent._complete(text, max_new_tokens)
    finally:
        agent.close()
    calls = response.get("function_calls") or []
    if not calls:
        return None
    arguments = calls[0].get("arguments") or {}
    if strict:
        _validate_extraction(text, schema, arguments, response, system)
    return schema(**arguments) if _is_pydantic_model(schema) else arguments
