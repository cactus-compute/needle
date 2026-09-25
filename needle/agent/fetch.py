import os
import platform
import sysconfig
import sys
import warnings
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
    """True on a musl system (Alpine and friends).

    platform.libc_ver() scans the interpreter binary and reports glibc on some
    musl builds, so the loader and the filesystem are asked first.
    """
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/self/maps", "rb") as maps:
            blob = maps.read()
        if b"musl" in blob:
            return True
        if b"/libc.so.6" in blob or b"/ld-linux" in blob:
            return False
    except OSError:
        pass
    import glob
    if glob.glob("/lib/ld-musl-*.so.1"):
        return True
    if glob.glob("/lib*/libc.so.6") or glob.glob("/lib/*-linux-gnu/libc.so.6"):
        return False
    if "musl" in (sysconfig.get_config_var("HOST_GNU_TYPE") or ""):
        return True
    libc, _ = platform.libc_ver()
    return not libc


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


def other_libc_tag():
    """The tag for the other Linux libc family, or None off Linux."""
    tag = _platform_tag()
    if tag.startswith("manylinux2014_"):
        return tag.replace("manylinux2014_", "musllinux_1_2_")
    if tag.startswith("musllinux_1_2_"):
        return tag.replace("musllinux_1_2_", "manylinux2014_")
    return None


def _wheel_name(version, tag):
    return f"cactus_needle-{version}-py3-none-{tag}.whl"


def _version_key(version):
    """Numeric sort key, so 3.0.10 ranks above 3.0.9 rather than below it."""
    return tuple(int(part) if part.isdigit() else -1 for part in version.split("."))


def published_engine_versions(repo, tag):
    """Engine versions whose ``tag`` wheel is actually in ``repo``, newest first.

    ``ENGINE_VERSIONS`` names the engine the Python side is written against, and
    the wheels for it are uploaded to the Hub separately.  A bump that reaches
    ``main`` before its own upload therefore names an engine nobody can fetch,
    and this is what ``fetch_library`` falls back to when that happens.
    """
    from huggingface_hub import list_repo_files

    prefix, suffix = "python/cactus_needle-", f"-py3-none-{tag}.whl"
    found = set()
    for name in list_repo_files(repo):
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        version = name[len(prefix):-len(suffix)]
        if version and "/" not in version:
            found.add(version)
    return sorted(found, key=_version_key, reverse=True)


def _download_wheel(repo, version, tag):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename="python/" + _wheel_name(version, tag),
                           repo_type="model")


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
    if dest_dir is None:
        raise TypeError("dest_dir is required")
    from huggingface_hub.errors import EntryNotFoundError

    version = version or engine_version(generation)
    tag = tag or _platform_tag()
    repo = engine_repo(generation)
    _register_download(generation)
    fell_back = False
    try:
        path = _download_wheel(repo, version, tag)
    except EntryNotFoundError:
        # The pinned engine is not on the Hub for this platform.  Take the newest
        # one that is, so a version bump that outruns its own upload costs a
        # warning instead of a 404 that fails every fresh install.
        published = published_engine_versions(repo, tag)
        if not published or published[0] == version:
            raise
        fallback = published[0]
        warnings.warn(
            f"the {tag} engine {version} is not published in {repo}; using "
            f"{fallback} instead (published: {', '.join(published)})", stacklevel=3)
        fell_back = True
        version, path = fallback, _download_wheel(repo, fallback, tag)
    lib = _lib_name_for(tag)
    stem, suffix = os.path.splitext(lib)
    member = f"{stem}{generation}{suffix}" if int(generation) >= 3 else lib
    # A fallback engine is written under the version it actually is, so a cache
    # directory named after the missing one cannot keep serving it once the real
    # engine has been uploaded.
    out_dir = os.path.join(dest_dir, version) if fell_back else dest_dir
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        data = archive.read("needle/" + member)
    out = os.path.join(out_dir, lib)
    with open(out, "wb") as handle:
        handle.write(data)
    return out
