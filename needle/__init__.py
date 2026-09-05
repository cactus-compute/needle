from __future__ import annotations

import ctypes
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


def _lib(generation=2):
    generation = int(generation)
    if generation not in _lib_handles:
        lib = ctypes.CDLL(_library_path(generation))
        lib.needle_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.needle_init.restype = ctypes.c_int
        lib.needle_complete.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
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
        self._system = (system or "").encode("utf-8")
        tools_json = tools if isinstance(tools, str) else json.dumps(self._resolve(tools))
        self._tools_json = tools_json.encode("utf-8")
        try:
            self._n_tools = len(json.loads(tools_json))
        except (json.JSONDecodeError, TypeError):
            self._n_tools = None
        self._tool_index_path = (os.fspath(tool_index_path).encode("utf-8")
                                 if tool_index_path else None)
        self._buffer = ctypes.create_string_buffer(buffer_size)
        if self._weights:
            self._worker = FineTuneWorker(
                _library_path(self._generation), self._weights,
                self._system.decode("utf-8"), self._tools_json.decode("utf-8"),
                os.fspath(tool_index_path) if tool_index_path else None,
                buffer_size)
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

    def complete(self, text: str, max_new_tokens: int = 256) -> dict:
        _track("complete", self._track_props())
        return self._complete(text, max_new_tokens)

    def _complete(self, text: str, max_new_tokens: int = 256) -> dict:
        self._bind()
        if self._worker is not None:
            raw = self._worker.complete(text, max_new_tokens)
        else:
            lib = _lib(self._generation)
            rc = lib.needle_complete(
                text.encode("utf-8"), int(max_new_tokens), self._buffer,
                len(self._buffer))
            if rc < 0:
                raise RuntimeError(f"needle_complete failed (code {rc})")
            raw = self._buffer.value.decode("utf-8")
        try:
            response = json.loads(raw)
        except json.JSONDecodeError as err:
            raise RuntimeError(
                f"engine returned an unparseable envelope ({err}); this is an "
                f"engine bug - please report it with the prompt and schema") from err
        if self._weights:
            response["confidence"] = None
        return response

    def run(self, query: str, max_steps: int = 8, max_new_tokens: int = 256) -> dict:
        _track("run", self._track_props())
        response = self._complete(query, max_new_tokens)
        executed = []
        for _ in range(max_steps):
            calls = response.get("function_calls") or []
            if response.get("type") != "call" or not calls:
                break
            results = []
            for call in calls:
                fn = self._functions.get(call.get("name"))
                if fn is None:
                    results.append({"error": "unknown tool: " + str(call.get("name"))})
                    continue
                try:
                    results.append(fn(**(call.get("arguments") or {})))
                except Exception as exc:
                    results.append({"error": str(exc)})
            executed.extend(results)
            response = self._complete(json.dumps(results, default=_jsonable), max_new_tokens)
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


def _temporal_grounding(text, schema, arguments):
    root = _schema_parameters(schema)
    years = _source_years(text)
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


def _validate_extraction(text, schema, arguments, response):
    checked, failures = _temporal_grounding(text, schema, arguments)
    validation = response.get("validation") or {}
    for name in validation.get("ungrounded") or []:
        path = name.split(".", 1)[-1]
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
        _validate_extraction(text, schema, arguments, response)
    return schema(**arguments) if _is_pydantic_model(schema) else arguments
