import sys

import pytest

def test_lib_path_env_override(tmp_path, monkeypatch):
    import needle

    fake = tmp_path / "libneedle.dylib"
    fake.write_bytes(b"x")
    monkeypatch.setenv("NEEDLE_LIB_PATH", str(fake))
    assert needle._library_path() == str(fake)


def test_legacy_lib_override_cannot_capture_v3(tmp_path, monkeypatch):
    import needle
    from needle.agent import fetch

    v2 = tmp_path / "libneedle-v2.dylib"
    v3 = tmp_path / "libneedle-v3.dylib"
    v2.write_bytes(b"v2")
    v3.write_bytes(b"v3")
    monkeypatch.setenv("NEEDLE_LIB_PATH", str(v2))
    monkeypatch.setenv("NEEDLE3_LIB_PATH", str(v3))
    assert needle._library_path(2) == str(v2)
    assert needle._library_path(3) == str(v3)

    monkeypatch.delenv("NEEDLE3_LIB_PATH")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(fetch, "fetch_library", lambda *args, **kwargs: str(v3))
    assert needle._library_path(3) == str(v3)


def test_download_target_kinds():
    import pytest
    from needle.cli import _download_target

    assert _download_target("macos-arm64") == ("platform", "macos-arm64")
    assert _download_target("needle3") == ("base", 3)
    assert _download_target("needle2.cact") == ("base", 2)
    assert _download_target("needle3.safetensors") == ("checkpoint", "needle3.safetensors")
    assert _download_target("acme/tuned/model.cact") == ("hub", "acme/tuned/model.cact")
    with pytest.raises(SystemExit, match="unknown download"):
        _download_target("needle4")


def test_fetch_weights_copies_the_base_archive_and_reuses_it(tmp_path, monkeypatch):
    from needle.agent import fetch

    archive = tmp_path / "needle3.cact"
    archive.write_bytes(b"weights")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs["filename"])
        return str(archive)

    monkeypatch.setattr(fetch, "_register_download", lambda generation: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    dest = tmp_path / "cache"
    out = fetch.fetch_weights(3, str(dest))
    assert out == str(dest / "needle3.cact") and open(out, "rb").read() == b"weights"
    assert calls == ["needle3.cact"]
    assert fetch.fetch_weights(3, str(dest)) == out and calls == ["needle3.cact"]


def test_fetch_checkpoint_prefers_the_checkpoints_folder(tmp_path, monkeypatch):
    from huggingface_hub.errors import EntryNotFoundError
    from needle.agent import fetch

    source = tmp_path / "needle3.safetensors"
    source.write_bytes(b"ckpt")

    def fake_download(**kwargs):
        if kwargs["filename"] != "checkpoints/needle3.safetensors":
            raise EntryNotFoundError("missing")
        return str(source)

    monkeypatch.setattr(fetch, "_register_download", lambda generation: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    out = fetch.fetch_checkpoint("needle3.safetensors", str(tmp_path / "dl" / "checkpoints"))
    assert out.endswith("checkpoints/needle3.safetensors") and open(out, "rb").read() == b"ckpt"


def test_weights_spec_parsing():
    from needle.cli import _weights_spec

    assert _weights_spec("acme/tuned/model.cact") == ("acme/tuned", "model.cact")
    assert _weights_spec("acme/tuned") == ("acme/tuned", None)
    assert _weights_spec("acme/tuned/sub/dir/m.cact") == ("acme/tuned", "sub/dir/m.cact")


def test_lib_name_for_tags():
    from needle.agent.fetch import _lib_name_for

    assert _lib_name_for("macosx_11_0_arm64") == "libneedle.dylib"
    assert _lib_name_for("win_amd64") == "libneedle.dll"
    assert _lib_name_for("manylinux2014_aarch64") == "libneedle.so"
    assert _lib_name_for("musllinux_1_2_x86_64") == "libneedle.so"


def test_component_platform_is_downloadable():
    from needle.agent.fetch import PLATFORMS

    assert "wasm-component" in PLATFORMS


def test_fetch_library_creates_destination(tmp_path, monkeypatch):
    import zipfile
    from needle.agent import fetch

    wheel = tmp_path / "engine.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("needle/libneedle.so", b"engine")
    monkeypatch.setattr(fetch, "_register_download", lambda generation: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda **kwargs: str(wheel))
    out = fetch.fetch_library("2.0.4", tmp_path / "new", tag="manylinux2014_x86_64")

    assert (tmp_path / "new" / "libneedle.so").read_bytes() == b"engine"
    assert out == str(tmp_path / "new" / "libneedle.so")


def test_published_engine_versions_is_newest_first_for_the_tag(monkeypatch):
    from needle.agent import fetch

    files = [
        "python/cactus_needle-3.0.9-py3-none-manylinux2014_x86_64.whl",
        "python/cactus_needle-3.0.10-py3-none-manylinux2014_x86_64.whl",
        "python/cactus_needle-3.0.10-py3-none-macosx_11_0_arm64.whl",
        "needle3.cact",
        "linux-x86_64/needle",
    ]
    monkeypatch.setattr("huggingface_hub.list_repo_files", lambda repo: files)
    assert fetch.published_engine_versions("Cactus-Compute/needle3", "manylinux2014_x86_64") \
        == ["3.0.10", "3.0.9"]


def _engine_wheel(tmp_path):
    import zipfile

    wheel = tmp_path / "engine.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("needle/libneedle3.so", b"engine")
    return str(wheel)


def test_fetch_library_falls_back_when_the_pinned_engine_is_unpublished(tmp_path, monkeypatch):
    from huggingface_hub.errors import EntryNotFoundError
    from needle.agent import fetch

    wheel = _engine_wheel(tmp_path)
    published = {"python/cactus_needle-3.0.10-py3-none-manylinux2014_x86_64.whl",
                 "python/cactus_needle-3.0.9-py3-none-manylinux2014_x86_64.whl"}
    asked = []

    def fake_download(**kwargs):
        asked.append(kwargs["filename"])
        if kwargs["filename"] not in published:
            raise EntryNotFoundError("404")
        return wheel

    monkeypatch.setattr(fetch, "_register_download", lambda generation: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr("huggingface_hub.list_repo_files", lambda repo: sorted(published))

    dest = tmp_path / "cache" / "v3" / "3.0.11"
    with pytest.warns(UserWarning, match="engine 3.0.11 is not published"):
        out = fetch.fetch_library("3.0.11", dest, tag="manylinux2014_x86_64", generation=3)

    # The pin is tried first, the newest published engine second.
    assert asked == ["python/cactus_needle-3.0.11-py3-none-manylinux2014_x86_64.whl",
                     "python/cactus_needle-3.0.10-py3-none-manylinux2014_x86_64.whl"]
    # A fallback is cached under the version it really is, so the directory named
    # after the missing one never keeps serving it once 3.0.11 is uploaded.
    assert out == str(dest / "3.0.10" / "libneedle.so")
    assert (dest / "3.0.10" / "libneedle.so").read_bytes() == b"engine"
    assert not (dest / "libneedle.so").exists()


def test_fetch_library_keeps_the_cache_layout_when_the_pin_is_published(tmp_path, monkeypatch):
    from needle.agent import fetch

    wheel = _engine_wheel(tmp_path)
    monkeypatch.setattr(fetch, "_register_download", lambda generation: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: wheel)
    monkeypatch.setattr("huggingface_hub.list_repo_files",
                        lambda repo: pytest.fail("no listing needed when the pin resolves"))

    dest = tmp_path / "cache" / "v3" / "3.0.11"
    out = fetch.fetch_library("3.0.11", dest, tag="manylinux2014_x86_64", generation=3)
    assert out == str(dest / "libneedle.so")


def test_fetch_library_reraises_when_no_engine_serves_the_platform(tmp_path, monkeypatch):
    from huggingface_hub.errors import EntryNotFoundError
    from needle.agent import fetch

    def fake_download(**kwargs):
        raise EntryNotFoundError("404")

    monkeypatch.setattr(fetch, "_register_download", lambda generation: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr("huggingface_hub.list_repo_files", lambda repo: [])

    with pytest.raises(EntryNotFoundError):
        fetch.fetch_library("3.0.1", tmp_path / "cache", tag="macosx_11_0_arm64", generation=3)


def test_pinned_engine_versions_are_published():
    """Every pinned engine needs a wheel for this platform on the Hub.

    ``ENGINE_VERSIONS`` and the Hub uploads land separately, so a bump can name
    an engine that was never uploaded.  That 404s every fresh install, which is
    what #142 and #146 report.  Set ``NEEDLE_SKIP_HUB_CHECK=1`` to opt out.
    """
    import os

    from needle.agent import fetch

    if os.environ.get("NEEDLE_SKIP_HUB_CHECK"):
        pytest.skip("NEEDLE_SKIP_HUB_CHECK is set")
    tag = fetch._platform_tag()
    for generation, version in fetch.ENGINE_VERSIONS.items():
        repo = fetch.engine_repo(generation)
        try:
            published = fetch.published_engine_versions(repo, tag)
        except Exception as exc:  # offline, rate limited, repo renamed
            pytest.skip(f"{repo} is not reachable: {exc}")
        assert version in published, (
            f"ENGINE_VERSIONS[{generation}] pins {version} but {repo} publishes "
            f"{published or 'no engine'} for {tag}")


def test_engine_gate_finds_the_cache_the_runtime_loads_from(tmp_path, monkeypatch):
    import needle
    from needle.agent import fetch
    from conftest import _engine_available

    package = tmp_path / "pkg"
    package.mkdir()
    monkeypatch.setattr(needle, "__file__", str(package / "__init__.py"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("NEEDLE_LIB_PATH", raising=False)
    monkeypatch.delenv("NEEDLE3_LIB_PATH", raising=False)

    assert not _engine_available()

    cache = tmp_path / ".cache" / "cactus-needle" / "v3" / fetch.engine_version(3)
    cache.mkdir(parents=True)
    (cache / fetch._lib_name()).write_bytes(b"")

    assert _engine_available()


def test_engine_gate_honours_the_library_override(tmp_path, monkeypatch):
    import needle
    from needle.agent import fetch
    from conftest import _engine_available

    package = tmp_path / "pkg"
    package.mkdir()
    monkeypatch.setattr(needle, "__file__", str(package / "__init__.py"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    engine = tmp_path / fetch._lib_name()
    engine.write_bytes(b"")
    monkeypatch.setenv("NEEDLE3_LIB_PATH", str(engine))
    assert _engine_available()

    monkeypatch.setenv("NEEDLE3_LIB_PATH", str(tmp_path / "gone"))
    assert not _engine_available()


def test_musl_detected_when_libc_ver_claims_glibc(monkeypatch, tmp_path):
    import platform
    from needle.agent import fetch

    monkeypatch.setattr(platform, "libc_ver", lambda *a, **k: ("glibc", "2.9"))
    monkeypatch.setattr(sys, "platform", "linux")
    maps = tmp_path / "maps"
    maps.write_bytes(b"7f0000000000-7f0000001000 r-xp /lib/ld-musl-x86_64.so.1\n")
    real_open = open
    monkeypatch.setattr("builtins.open",
                        lambda p, *a, **k: real_open(maps, *a, **k) if p == "/proc/self/maps"
                        else real_open(p, *a, **k))
    assert fetch._is_musl() is True


def test_glibc_detected_from_maps(monkeypatch, tmp_path):
    import platform
    from needle.agent import fetch

    monkeypatch.setattr(platform, "libc_ver", lambda *a, **k: ("", ""))
    monkeypatch.setattr(sys, "platform", "linux")
    maps = tmp_path / "maps"
    maps.write_bytes(b"7f0000000000-7f0000001000 r-xp /usr/lib/x86_64-linux-gnu/libc.so.6\n")
    real_open = open
    monkeypatch.setattr("builtins.open",
                        lambda p, *a, **k: real_open(maps, *a, **k) if p == "/proc/self/maps"
                        else real_open(p, *a, **k))
    assert fetch._is_musl() is False


def test_other_libc_tag_swaps_families(monkeypatch):
    from needle.agent import fetch

    monkeypatch.setattr(fetch, "_platform_tag", lambda: "manylinux2014_aarch64")
    assert fetch.other_libc_tag() == "musllinux_1_2_aarch64"
    monkeypatch.setattr(fetch, "_platform_tag", lambda: "musllinux_1_2_x86_64")
    assert fetch.other_libc_tag() == "manylinux2014_x86_64"
    monkeypatch.setattr(fetch, "_platform_tag", lambda: "macosx_11_0_arm64")
    assert fetch.other_libc_tag() is None


def test_engine_load_falls_back_to_the_other_libc(monkeypatch, tmp_path):
    import ctypes
    import needle
    from needle.agent import fetch

    good = tmp_path / "musl" / "libneedle.so"
    good.parent.mkdir()
    good.write_bytes(b"x")
    monkeypatch.setattr(needle, "_library_path", lambda generation=2: str(tmp_path / "libneedle.so"))
    monkeypatch.setattr(fetch, "other_libc_tag", lambda: "musllinux_1_2_x86_64")
    monkeypatch.setattr(fetch, "fetch_library",
                        lambda version=None, dest_dir=None, tag=None, generation=2: str(good))
    loaded = {}

    def fake_cdll(path):
        if path.endswith("musl/libneedle.so"):
            loaded["path"] = path
            return "handle"
        raise OSError("Error relocating libneedle.so: strtoll_l: symbol not found")

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    with pytest.warns(UserWarning, match="using the musllinux"):
        assert needle._load_cdll(3) == "handle"
    assert loaded["path"].endswith("musl/libneedle.so")
