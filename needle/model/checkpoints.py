import json
import os
import pickle

import numpy as np

SAFETENSORS_SUFFIX = ".safetensors"


def is_safetensors(path):
    return os.fspath(path).endswith(SAFETENSORS_SUFFIX)


def _contiguous(value):
    array = np.asarray(value)
    return np.ascontiguousarray(array) if array.ndim else array


def flatten(tree, prefix=""):
    flat = {}
    for key, value in tree.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten(value, name))
        else:
            flat[name] = _contiguous(value)
    return flat


def unflatten(flat):
    tree = {}
    for name, value in flat.items():
        node = tree
        parts = name.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return tree


def _safetensors_numpy():
    try:
        from safetensors import numpy as safetensors_numpy
    except ImportError as err:
        raise ImportError("reading or writing .safetensors needs the safetensors package: "
                          "pip install safetensors") from err
    return safetensors_numpy


def _json_or_default(text, default):
    if text in (None, ""):
        return default
    return json.loads(text)


def _int_or_none(text):
    if text in (None, ""):
        return None
    return int(text)


def read_checkpoint(path):
    if not is_safetensors(path):
        raise ValueError(f"checkpoint must be a .safetensors file for security (got {path})")
    library = _safetensors_numpy()
    from safetensors import safe_open

    with safe_open(os.fspath(path), framework="np") as handle:
        metadata = handle.metadata() or {}
    return {
        "format_version": _int_or_none(metadata.get("format_version")),
        "params": unflatten(library.load_file(os.fspath(path))),
        "config": _json_or_default(metadata.get("config"), {}),
        "step": _int_or_none(metadata.get("step")),
        "run": _json_or_default(metadata.get("run"), {}),
    }


def write_checkpoint(path, checkpoint):
    if not is_safetensors(path):
        raise ValueError(f"checkpoint must be a .safetensors file for security (got {path})")
    library = _safetensors_numpy()
    metadata = {
        "format_version": str(checkpoint.get("format_version", "")),
        "config": json.dumps(checkpoint.get("config", {}), default=str),
        "step": "" if checkpoint.get("step") is None else str(checkpoint["step"]),
        "run": json.dumps(checkpoint.get("run") or {}, default=str),
    }
    library.save_file(flatten(checkpoint["params"]), os.fspath(path), metadata=metadata)


_ADAPTER_FIELDS = ("scale", "base", "rank", "seed")


def write_adapter(path, adapter):
    if not is_safetensors(path):
        raise ValueError(f"adapter must be a .safetensors file for security (got {path})")
    library = _safetensors_numpy()
    tensors = {}
    for name, value in adapter["lora"].items():
        tensors[f"lora/{name}/A"] = _contiguous(value["A"])
        tensors[f"lora/{name}/B"] = _contiguous(value["B"])
    metadata = {key: json.dumps(adapter.get(key), default=str) for key in _ADAPTER_FIELDS}
    library.save_file(tensors, os.fspath(path), metadata=metadata)


def read_adapter(path):
    if not is_safetensors(path):
        raise ValueError(f"adapter must be a .safetensors file for security (got {path})")
    library = _safetensors_numpy()
    from safetensors import safe_open

    with safe_open(os.fspath(path), framework="np") as handle:
        metadata = handle.metadata() or {}
    lora = {}
    for name, value in library.load_file(os.fspath(path)).items():
        _, key, matrix = name.rsplit("/", 2) if name.count("/") == 2 else (None, *name[5:].rsplit("/", 1))
        lora.setdefault(key, {})[matrix] = value
    adapter = {"lora": lora}
    for key in _ADAPTER_FIELDS:
        adapter[key] = _json_or_default(metadata.get(key), None)
    return adapter
