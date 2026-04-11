import os
import pickle
import threading
import jax
import jax.numpy as jnp
import numpy as np

_HF_CHECKPOINT_REPO = "Cactus-Compute/checkpoints"


def _download_checkpoint(name_or_path):
    """Return a local path to a checkpoint, downloading from HF if needed.

    If `name_or_path` already exists on disk, it's returned unchanged.
    Otherwise treats the basename as a file in `Cactus-Compute/checkpoints`.
    """
    if os.path.exists(name_or_path):
        return name_or_path
    from huggingface_hub import hf_hub_download
    local_dir = os.path.dirname(name_or_path) or "checkpoints"
    os.makedirs(local_dir, exist_ok=True)
    print(f"  Downloading {os.path.basename(name_or_path)} from {_HF_CHECKPOINT_REPO}...", flush=True)
    local_path = hf_hub_download(
        repo_id=_HF_CHECKPOINT_REPO,
        filename=os.path.basename(name_or_path),
        repo_type="model",
        local_dir=local_dir,
    )
    print(f"  Downloaded to {local_path}", flush=True)
    return local_path


def _flatten_params(tree, prefix=()):
    """Flatten a nested dict of params into {tuple-path: leaf}."""
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(_flatten_params(v, prefix + (k,)))
    else:
        out[prefix] = tree
    return out


def _unflatten_params(flat):
    """Inverse of _flatten_params: rebuild a nested dict."""
    root = {}
    for path, leaf in flat.items():
        node = root
        for k in path[:-1]:
            node = node.setdefault(k, {})
        node[path[-1]] = leaf
    return root


def partial_load_params(init_params, loaded_params):
    """Merge `loaded_params` into `init_params` by matching path + shape.

    For every leaf in `init_params`:
      - if the same path exists in `loaded_params` and shapes agree,
        replace it with the loaded value (cast to the init leaf's dtype);
      - otherwise keep the init leaf (random init).

    Extra leaves in `loaded_params` (not present in `init_params`) are ignored.

    Returns (merged_params, stats_dict).
    """
    init_flat = _flatten_params(init_params)
    loaded_flat = _flatten_params(loaded_params)

    merged = {}
    n_loaded = 0
    n_random = 0
    shape_mismatches = []
    missing_in_ckpt = []
    for path, init_leaf in init_flat.items():
        loaded_leaf = loaded_flat.get(path)
        if loaded_leaf is None:
            merged[path] = init_leaf
            n_random += 1
            missing_in_ckpt.append(path)
            continue
        if tuple(loaded_leaf.shape) != tuple(init_leaf.shape):
            merged[path] = init_leaf
            n_random += 1
            shape_mismatches.append((path, tuple(loaded_leaf.shape), tuple(init_leaf.shape)))
            continue
        merged[path] = jnp.asarray(loaded_leaf, dtype=init_leaf.dtype)
        n_loaded += 1

    extra_in_ckpt = [p for p in loaded_flat.keys() if p not in init_flat]

    stats = {
        "loaded": n_loaded,
        "random_init": n_random,
        "shape_mismatches": shape_mismatches,
        "missing_in_ckpt": missing_in_ckpt,
        "extra_in_ckpt": extra_in_ckpt,
    }
    return _unflatten_params(merged), stats


def load_pretrained_params(path_or_name):
    """Download (if needed) + pickle.load a checkpoint, return its params dict."""
    local = _download_checkpoint(path_or_name)
    with open(local, "rb") as f:
        data = pickle.load(f)
    return data["params"], data.get("config", None), local


def _replicate(tree):
    """Replicate a pytree across all local devices (multi-host safe)."""
    devices = jax.local_devices()
    return jax.tree.map(
        lambda x: jax.device_put_replicated(x, devices), tree)


def _unreplicate(tree):
    """Get a single copy from a replicated pytree (multi-host safe).

    Uses addressable_shards instead of x[0] indexing, which fails
    in multi-host pmap because the array spans devices on other hosts.
    Each shard has shape (1, ...) — we index [0] to strip the leading dim.
    """
    def _get_first(x):
        if hasattr(x, 'addressable_shards') and x.addressable_shards:
            shard = x.addressable_shards[0].data
            # Shard has shape (1, ...) from the pmap partition; strip it
            return jax.device_get(shard[0]) if shard.ndim > 0 else jax.device_get(shard)
        return x
    return jax.tree.map(_get_first, tree)


def shard_batch(batch, num_devices):
    """Reshape a batch array so leading dim is (num_devices, per_device_batch, ...)."""
    return batch.reshape(num_devices, -1, *batch.shape[1:])


def _upload_checkpoint(ckpt_path):
    """Upload a checkpoint file to HuggingFace Hub in a background thread."""
    import threading

    def _upload():
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(_HF_CHECKPOINT_REPO, repo_type="model", private=True, exist_ok=True)
            filename = os.path.basename(ckpt_path)
            print(f"[hf] Uploading {filename} to {_HF_CHECKPOINT_REPO} ...")
            api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo=filename,
                repo_id=_HF_CHECKPOINT_REPO,
                repo_type="model",
            )
            print(f"[hf] Checkpoint uploaded: {_HF_CHECKPOINT_REPO}/{filename}")
        except Exception as e:
            print(f"[hf] Warning: checkpoint upload failed: {e}")

    threading.Thread(target=_upload, daemon=True).start()
