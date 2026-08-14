import ctypes
import json
import os
import sys
import warnings

from .agent.tools import Field, build_schema, pydantic_schema, tool, _is_pydantic_model

__version__ = "2.0.0"
__all__ = ["Needle", "tool", "Field", "extract", "__version__"]


def _library_path():
    from .agent import fetch

    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(here, fetch._lib_name())
    if os.path.exists(local):
        return local
    cache = os.path.join(os.path.expanduser("~"), ".cache", "cactus-needle", fetch.ENGINE_VERSION)
    cached = os.path.join(cache, fetch._lib_name())
    if os.path.exists(cached):
        return cached
    os.makedirs(cache, exist_ok=True)
    return fetch.fetch_library(fetch.ENGINE_VERSION, cache)


_lib_handle = None
_active = None
_active_weights = None


def _lib():
    global _lib_handle
    if _lib_handle is None:
        lib = ctypes.CDLL(_library_path())
        lib.needle_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.needle_init.restype = ctypes.c_int
        lib.needle_complete.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.needle_complete.restype = ctypes.c_int
        lib.needle_reset.argtypes = []
        lib.needle_reset.restype = None
        lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
        lib.needle_load.restype = ctypes.c_int
        _lib_handle = lib
    return _lib_handle


class Needle:
    def __init__(self, tools=None, system=None, weights=None, tool_index_path=None, buffer_size=65536):
        self._functions = {}
        self._weights = weights
        self._weight_blob = None
        if weights:
            warnings.warn("finetuning does not update the confidence head, so scores are "
                          "uncalibrated for tuned weights; this agent reports confidence as None",
                          stacklevel=2)
        self._system = (system or "").encode("utf-8")
        tools_json = tools if isinstance(tools, str) else json.dumps(self._resolve(tools))
        self._tools_json = tools_json.encode("utf-8")
        self._tool_index_path = tool_index_path.encode("utf-8") if tool_index_path else None
        self._buffer = ctypes.create_string_buffer(buffer_size)
        self._bind()

    def _bind(self):
        global _active, _active_weights
        if _active is self:
            return
        if self._weights and _active_weights != self._weights:
            if self._weight_blob is None:
                with open(self._weights, "rb") as handle:
                    # needle_load does not copy the input buffer and retains/dereferences
                    # it during inference, so the bytes object must be kept alive on self.
                    self._weight_blob = handle.read()
            if _lib().needle_load(self._weight_blob, len(self._weight_blob)) != 0:
                raise RuntimeError(
                    f"failed to load weights from {self._weights} - the .cact "
                    f"format is tied to the engine version, so an archive "
                    f"exported by an older cactus-needle will not load; re-run "
                    f"`needle build` on your checkpoint with this package version")
            _active_weights = self._weights
        if _lib().needle_init(self._system, self._tools_json, self._tool_index_path) < 0:
            _active = None
            raise RuntimeError("needle_init failed")
        _active = self

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

    def complete(self, text, max_new_tokens=256):
        self._bind()
        _lib().needle_complete(text.encode("utf-8"), int(max_new_tokens),
                               self._buffer, len(self._buffer))
        response = json.loads(self._buffer.value.decode("utf-8"))
        if self._weights:
            response["confidence"] = None
        return response

    def run(self, query, max_steps=8, max_new_tokens=256):
        response = self.complete(query, max_new_tokens)
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
            response = self.complete(json.dumps(results, default=_jsonable), max_new_tokens)
        response["results"] = executed
        return response

    def extract(self, text, schema, max_new_tokens=256):
        return extract(text, schema, max_new_tokens=max_new_tokens)

    def reset(self):
        self._bind()
        _lib().needle_reset()


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict") and _is_pydantic_model(type(value)):
        return value.dict()
    return str(value)


def extract(text, schema, system=None, max_new_tokens=256):
    """One-shot structured extraction: declare `schema` as the only tool and return
    the parsed object (a Pydantic instance if `schema` is a model, else a dict).
    Re-initializes the shared engine with this single schema."""
    agent = Needle(tools=[schema], system=system)
    response = agent.complete(text, max_new_tokens)
    calls = response.get("function_calls") or []
    if not calls:
        return None
    arguments = calls[0].get("arguments") or {}
    return schema(**arguments) if _is_pydantic_model(schema) else arguments
