import enum
import inspect
import re
import types
import typing
from typing import Callable

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean",
               list: "array", dict: "object"}

# PEP 604 unions (X | Y) have origin types.UnionType, which only exists on
# Python 3.10+; on 3.9 the getattr falls back to typing.Union harmlessly.
_UNION_ORIGINS = (typing.Union, getattr(types, "UnionType", typing.Union))

_MISSING = object()


class Field:
    def __init__(self, default=_MISSING, *, description=None, enum=None, const=_MISSING,
                 ge=None, le=None, gt=None, lt=None, multiple_of=None,
                 min_length=None, max_length=None, pattern=None, format=None,
                 min_items=None, max_items=None, unique_items=None):
        self.default = default
        self.description = description
        self.enum = enum
        self.const = const
        self.ge, self.le, self.gt, self.lt = ge, le, gt, lt
        self.multiple_of = multiple_of
        self.min_length, self.max_length = min_length, max_length
        self.pattern, self.format = pattern, format
        self.min_items, self.max_items, self.unique_items = min_items, max_items, unique_items

    def has_default(self):
        return self.default is not _MISSING

    def apply(self, schema):
        pairs = (("description", self.description), ("enum", self.enum),
                 ("minimum", self.ge), ("maximum", self.le),
                 ("exclusiveMinimum", self.gt), ("exclusiveMaximum", self.lt),
                 ("multipleOf", self.multiple_of), ("minLength", self.min_length),
                 ("maxLength", self.max_length), ("pattern", self.pattern),
                 ("format", self.format), ("minItems", self.min_items),
                 ("maxItems", self.max_items), ("uniqueItems", self.unique_items))
        for key, value in pairs:
            if value is not None:
                schema[key] = list(value) if key == "enum" else value
        if self.const is not _MISSING:
            schema["const"] = self.const
        return schema


def _is_optional(annotation):
    return (typing.get_origin(annotation) in _UNION_ORIGINS
            and type(None) in typing.get_args(annotation))


def _json_type(annotation):
    if annotation is inspect.Parameter.empty or annotation is None:
        return {"type": "string"}
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _json_type(typing.get_args(annotation)[0])
    if origin is None:
        if annotation in _JSON_TYPES:
            return {"type": _JSON_TYPES[annotation]}
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            return {"type": "string", "enum": [e.value for e in annotation]}
        if _is_pydantic_model(annotation):
            return pydantic_schema(annotation)["parameters"]
        return {"type": "string"}
    if origin is typing.Literal:
        return {"type": "string", "enum": list(typing.get_args(annotation))}
    if origin in (list, typing.List):
        args = typing.get_args(annotation)
        return {"type": "array", "items": _json_type(args[0]) if args else {"type": "string"}}
    if origin in (dict, typing.Dict):
        return {"type": "object"}
    if origin in _UNION_ORIGINS:
        rest = [a for a in typing.get_args(annotation) if a is not type(None)]
        if rest:
            return _json_type(rest[0])
    return {"type": "string"}


def _field_of(annotation, default):
    if isinstance(default, Field):
        return default
    if typing.get_origin(annotation) is typing.Annotated:
        for meta in typing.get_args(annotation)[1:]:
            if isinstance(meta, Field):
                return meta
    return None


def _parse_doc(doc):
    if not doc:
        return "", {}
    lines = doc.strip("\n").splitlines()
    heads = ("args:", "arguments:", "parameters:", "params:")
    desc, args, i = [], {}, 0
    while i < len(lines) and lines[i].strip().lower() not in heads:
        desc.append(lines[i].strip())
        i += 1
    for line in lines[i + 1:]:
        m = re.match(r"\s+(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)", line)
        if m:
            args[m.group(1)] = m.group(2).strip()
    return " ".join(w for w in desc if w).strip(), args


def build_schema(fn: Callable) -> dict:
    signature = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}
    description, arg_docs = _parse_doc(fn.__doc__)
    properties, required = {}, []
    for name, param in signature.parameters.items():
        if name in ("self", "cls") or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = hints.get(name, param.annotation)
        schema = _json_type(annotation)
        if name in arg_docs and "description" not in schema:
            schema["description"] = arg_docs[name]
        field = _field_of(annotation, param.default)
        if field:
            field.apply(schema)
        properties[name] = schema
        has_default = param.default is not param.empty and not isinstance(param.default, Field)
        if field and field.has_default():
            has_default = True
        if not has_default and not _is_optional(annotation):
            required.append(name)
    parameters = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    out = {"name": fn.__name__}
    if description:
        out["description"] = description
    out["parameters"] = parameters
    return out


def _is_pydantic_model(obj):
    return isinstance(obj, type) and any(
        base.__module__.startswith("pydantic") and base.__name__ == "BaseModel"
        for base in obj.__mro__)


def pydantic_schema(model: type) -> dict:
    raw = model.model_json_schema() if hasattr(model, "model_json_schema") else model.schema()
    parameters = {"type": "object", "properties": raw.get("properties", {})}
    for key in ("required", "$defs", "definitions"):
        if key in raw:
            parameters[key] = raw[key]
    out = {"name": raw.get("title", model.__name__), "parameters": parameters}
    doc = (model.__doc__ or "").strip()
    if doc:
        out["description"] = doc
    return out


def tool(fn: Callable) -> Callable:
    fn._needle_tool = build_schema(fn)
    return fn
