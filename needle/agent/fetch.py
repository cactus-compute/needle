import os
import platform
import sys
import zipfile

ENGINE_REPOS = {
    2: "Cactus-Compute/needle2",
    3: "Cactus-Compute/needle3",
}
ENGINE_VERSIONS = {
    2: "2.0.4",
    3: "3.0.1",
}

BASE_WEIGHTS = {
    2: "needle2.cact",
    3: "needle3.cact",
}
CHECKPOINT_PREFIX = "checkpoints"

# Backwards-compatible aliases for callers that explicitly fetch Needle 2.
HF_REPO = ENGINE_REPOS[2]

PLATFORMS = ("macos-arm64", "linux-x86_64", "linux-arm64", "linux-armv7",
             "linux-riscv64", "linux-mipsel", "windows-x86_64", "windows-arm64",
             "android-arm64", "android-armv7", "android-riscv64",
             "ios-arm64", "ios-sim-arm64", "tvos-arm64", "watchos-arm64", "wasm",
             "wasm-component")


def _lib_name_for(tag):
    if tag.startswith("macosx"):
        return "libneedle.dylib"
    if tag.startswith("win"):
        return "libneedle.dll"
    return "libneedle.so"


def _lib_name():
    if sys.platform == "darwin":
        return "libneedle.dylib"
    if sys.platform == "win32":
        return "libneedle.dll"
    return "libneedle.so"


def _is_musl():
    if sys.platform != "linux":
        return False
    libc, _ = platform.libc_ver()
    if libc:
        return False
    try:
        with open("/proc/self/maps", "rb") as maps:
            return b"musl" in maps.read()
    except OSError:
        return True


def _platform_tag():
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
        return "macosx_11_0_" + arch
    if sys.platform == "win32":
        return "win_arm64" if machine in ("arm64", "aarch64") else "win_amd64"
    arch = "aarch64" if machine in ("aarch64", "arm64") else "x86_64"
    family = "musllinux_1_2_" if _is_musl() else "manylinux2014_"
    return family + arch


def engine_repo(generation=2):
    try:
        return ENGINE_REPOS[int(generation)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unsupported Needle generation: {generation}") from exc


def engine_version(generation=2):
    try:
        return ENGINE_VERSIONS[int(generation)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unsupported Needle generation: {generation}") from exc


def _register_download(generation=2):
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(repo_id=engine_repo(generation), filename="config.json", repo_type="model",
                        force_download=True)
    except Exception:
        pass


def download_platform(name, out_dir, generation=2, dest=None):
    import shutil
    import stat
    from huggingface_hub import hf_hub_download, list_repo_files

    repo = engine_repo(generation)
    _register_download(generation)
    files = [f for f in list_repo_files(repo) if f.startswith(name + "/")]
    if name not in PLATFORMS or not files:
        raise FileNotFoundError(f"{name} is not a published platform folder in {repo}")
    dest = dest or os.path.join(out_dir, name)
    os.makedirs(dest, exist_ok=True)
    out = []
    for f in files:
        cached = hf_hub_download(repo_id=repo, filename=f, repo_type="model")
        target = os.path.join(dest, os.path.basename(f))
        shutil.copyfile(cached, target)
        if os.path.basename(target) in ("needle", "needle.exe"):
            os.chmod(target, os.stat(target).st_mode | stat.S_IXUSR
                     | stat.S_IXGRP | stat.S_IXOTH)
        out.append(target)
    return out


def base_weights(generation=2):
    try:
        return BASE_WEIGHTS[int(generation)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unsupported Needle generation: {generation}") from exc


def cache_dir(generation=2):
    return os.path.join(os.path.expanduser("~"), ".cache", "cactus-needle",
                        f"v{int(generation)}", engine_version(generation))


def fetch_weights(generation=2, dest_dir=None, force=False):
    """Download the base .cact archive of a generation (cached next to its engine).

    ``force`` fetches it again even when the cache holds a copy, so every build
    is a download of the published model.
    """
    import shutil
    from huggingface_hub import hf_hub_download

    name = base_weights(generation)
    dest_dir = dest_dir or cache_dir(generation)
    out = os.path.join(dest_dir, name)
    if os.path.exists(out) and not force:
        return out
    _register_download(generation)
    cached = hf_hub_download(repo_id=engine_repo(generation), filename=name, repo_type="model",
                             force_download=force)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copyfile(cached, out)
    return out


def fetch_checkpoint(name, dest_dir, generation=3, force=False):
    """Download a training checkpoint (needle3.safetensors, ...) for finetune/build."""
    import shutil
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    repo = engine_repo(generation)
    _register_download(generation)
    base = os.path.basename(name)
    cached = None
    for candidate in (f"{CHECKPOINT_PREFIX}/{base}", base):
        try:
            cached = hf_hub_download(repo_id=repo, filename=candidate, repo_type="model",
                                     force_download=force)
            break
        except EntryNotFoundError:
            continue
    if cached is None:
        raise FileNotFoundError(f"{base} is not published in {repo}")
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, base)
    shutil.copyfile(cached, out)
    return out


def fetch_library(version=None, dest_dir=None, tag=None, generation=2):
    from huggingface_hub import hf_hub_download

    if dest_dir is None:
        raise TypeError("dest_dir is required")
    version = version or engine_version(generation)
    tag = tag or _platform_tag()
    wheel = "cactus_needle-{}-py3-none-{}.whl".format(version, tag)
    repo = engine_repo(generation)
    _register_download(generation)
    path = hf_hub_download(repo_id=repo, filename="python/" + wheel, repo_type="model")
    lib = _lib_name_for(tag)
    stem, suffix = os.path.splitext(lib)
    member = f"{stem}{generation}{suffix}" if int(generation) >= 3 else lib
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        data = archive.read("needle/" + member)
    out = os.path.join(dest_dir, lib)
    with open(out, "wb") as handle:
        handle.write(data)
    return out
