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
    monkeypatch.setattr(fetch, "fetch_library", lambda *args, **kwargs: str(v3))
    assert needle._library_path(3) == str(v3)


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
    monkeypatch.delenv("NEEDLE2_LIB_PATH", raising=False)

    assert not _engine_available()

    cache = tmp_path / ".cache" / "cactus-needle" / "v2" / fetch.engine_version(2)
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
    monkeypatch.setenv("NEEDLE_LIB_PATH", str(engine))
    assert _engine_available()

    monkeypatch.setenv("NEEDLE_LIB_PATH", str(tmp_path / "gone"))
    assert not _engine_available()
