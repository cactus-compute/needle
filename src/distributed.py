import os
import threading
import jax
import numpy as np

_HF_CHECKPOINT_REPO = "Cactus-Compute/checkpoints"


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
