import array
import math
import os
import struct
import sys
import wave

import pytest

from conftest import _engine_available


def _whistle_weights():
    from needle.agent import fetch

    return os.environ.get("NEEDLE_WHISTLE_WEIGHTS") or os.path.join(fetch.cache_dir(fetch.WHISTLE), fetch.base_weights(fetch.WHISTLE))


requires_whistle = pytest.mark.skipif(not (_engine_available("whistle") and os.path.exists(_whistle_weights())),
                                      reason="Whistle engine or whistle.cact not installed (auto-fetched from HF on first real use)")


def test_engine_and_weights_resolve_like_needle(tmp_path, monkeypatch):
    import zipfile
    import needle
    from needle import whistle
    from needle.agent import fetch

    monkeypatch.setattr(os.path, "expanduser", lambda path: str(tmp_path))
    monkeypatch.setattr(needle, "__file__", str(tmp_path / "package" / "__init__.py"))
    monkeypatch.setattr(fetch, "_register_download", lambda generation: None)
    monkeypatch.delenv("NEEDLE_WHISTLE_WEIGHTS", raising=False)
    monkeypatch.setenv("NEEDLE_WHISTLE_LIB_PATH", "/opt/libwhistle.so")
    assert needle._library_path(fetch.WHISTLE) == "/opt/libwhistle.so"
    monkeypatch.delenv("NEEDLE_WHISTLE_LIB_PATH")

    lib = fetch.lib_name(fetch.WHISTLE)
    wheel, weights, fetched = tmp_path / "engine.whl", tmp_path / "published.cact", []
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("needle/" + lib, b"engine")
    weights.write_bytes(b"weights")
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda **kwargs: fetched.append((kwargs["repo_id"], kwargs["filename"])) or str(weights if kwargs["filename"].endswith(".cact") else wheel))
    cache = tmp_path / ".cache" / "cactus-needle" / "whistle" / fetch.ENGINE_VERSIONS[fetch.WHISTLE]
    assert lib.startswith("libwhistle.") and needle._library_path(fetch.WHISTLE) == str(cache / lib) and (cache / lib).read_bytes() == b"engine"
    assert whistle._weights_path() == str(cache / "whistle.cact") and (cache / "whistle.cact").read_bytes() == b"weights"
    assert fetched == [("Cactus-Compute/whistle", f"python/cactus_whistle-{fetch.ENGINE_VERSIONS[fetch.WHISTLE]}-py3-none-{fetch._platform_tag()}.whl"),
                       ("Cactus-Compute/whistle", "whistle.cact")]
    assert needle._library_path(fetch.WHISTLE) and whistle._weights_path() and len(fetched) == 2
    monkeypatch.setenv("NEEDLE_WHISTLE_WEIGHTS", "/opt/whistle.cact")
    assert whistle._weights_path() == "/opt/whistle.cact"


def test_cli_fetches_the_whistle_engine_and_weights(tmp_path, monkeypatch, capsys):
    import sys
    import needle._telemetry
    import needle.cli
    from needle.agent import fetch

    calls = []
    monkeypatch.setattr(needle._telemetry, "track", lambda *a, **k: None)
    monkeypatch.setattr(fetch, "fetch_library", lambda version, dest, tag=None, generation=2: calls.append((version, tag, generation)) or os.path.join(dest, "libwhistle.so"))
    monkeypatch.setattr(fetch, "fetch_weights", lambda generation, dest: calls.append((generation, dest)) or __file__)
    monkeypatch.setattr(sys, "argv", ["needle", "whistle", "fetch", "--out", str(tmp_path), "--platform-tag", "manylinux2014_aarch64"])
    needle.cli.main()
    monkeypatch.setattr(sys, "argv", ["needle", "whistle", "download", "whistle", "--out", str(tmp_path)])
    needle.cli.main()
    assert calls == [(fetch.ENGINE_VERSIONS[fetch.WHISTLE], "manylinux2014_aarch64", fetch.WHISTLE), (fetch.WHISTLE, str(tmp_path))]
    out = capsys.readouterr().out
    assert "NEEDLE_WHISTLE_LIB_PATH" in out and "needle.Whistle(weights=" in out


def _tone(seconds, rate=16000):
    return [0.3 * math.sin(2 * math.pi * 220 * i / rate) for i in range(int(seconds * rate))]


def _write_wav(path, samples, rate, channels=1, width=2):
    peak = (1 << (8 * width - 1)) - 1
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        for value in samples:
            level = int(value * peak)
            frame = bytes([level + 128]) if width == 1 else level.to_bytes(width, "little", signed=True)
            out.writeframes(frame * channels)


@pytest.mark.parametrize("rate,channels,width", [(16000, 1, 2), (8000, 1, 2), (44100, 2, 2), (16000, 1, 1), (22050, 2, 3), (48000, 1, 4)])
def test_wav_files_become_16k_mono_floats(tmp_path, rate, channels, width):
    from needle.whistle import _read_wav

    if rate != 16000:
        pytest.importorskip("soxr")
    path = tmp_path / "tone.wav"
    _write_wav(path, _tone(0.5, rate), rate, channels, width)
    samples = _read_wav(str(path))
    assert abs(len(samples) - 8000) <= 1
    assert 0.28 < max(samples) < 0.31 and -0.31 < min(samples) < -0.28
    crossings = sum(1 for a, b in zip(samples, samples[1:]) if a < 0 <= b)
    assert 108 <= crossings <= 111


def test_samples_accept_lists_arrays_and_float32_bytes():
    from needle.whistle import _samples

    values = [0.0, 0.5, -0.25]
    inputs = [values, array.array("f", values), struct.pack("<3f", *values), (v for v in values)]
    numpy = pytest.importorskip("numpy")
    inputs.append(numpy.array(values, numpy.float64))
    for audio in inputs:
        buffer, count = _samples(audio)
        assert count == 3 and list(buffer) == values
    assert _samples([])[1] == 0


@requires_whistle
def test_silence_is_an_empty_transcript():
    from needle import Whistle

    result = Whistle().transcribe([0.0] * 16000, word_timestamps=True)
    assert result == {"text": "", "language": "", "words": [], "ttft_ms": 0.0, "decode_tps": 0.0}


@requires_whistle
def test_transcribe_returns_text_language_and_timed_words(tmp_path):
    from needle import Whistle
    from needle.whistle import LANGUAGES

    whistle = Whistle()
    sound = [v * (0.2 + 0.8 * abs(math.sin(math.pi * 3 * i / 16000))) for i, v in enumerate(_tone(3))]
    plain = whistle.transcribe(sound)
    assert set(plain) == {"text", "language", "ttft_ms", "decode_tps"} and plain["language"] in LANGUAGES + ("",)
    assert (plain["ttft_ms"] > 0) == bool(plain["language"]) and plain["decode_tps"] >= 0
    path = tmp_path / "sound.wav"
    _write_wav(path, sound, 16000)
    timed = whistle.transcribe(path, language="en", keywords=["Siobhan", "Krzysztof"], word_timestamps=True)
    assert timed["language"] == "en" and isinstance(timed["text"], str)
    starts = [word["start"] for word in timed["words"]]
    assert starts == sorted(starts)
    assert all(0 <= word["start"] <= word["end"] <= 3.01 and 0 <= word["probability"] <= 1 for word in timed["words"])


@requires_whistle
def test_embed_and_the_30_second_limit():
    from needle import Whistle

    whistle = Whistle()
    embedding = whistle.embed(_tone(1))
    assert len(embedding) > 0 and all(isinstance(v, float) for v in embedding[:8])
    with pytest.raises(RuntimeError, match="30 s"):
        whistle.transcribe([0.0] * (30 * 16000 + 1))


@requires_whistle
def test_missing_weights_fail_clearly(tmp_path):
    from needle import Whistle

    broken = tmp_path / "broken.cact"
    broken.write_bytes(b"not a model at all")
    with pytest.raises(RuntimeError):
        Whistle(weights=broken)
    whistle = Whistle()
    assert os.path.getsize(whistle.weights) > 0 and whistle.transcribe([0.0] * 16000)["text"] == ""


class _Whistle:
    weights = __file__

    def __init__(self, result):
        self.result = result
        self.calls = []

    def transcribe(self, audio, **options):
        self.calls.append((audio, options))
        return self.result


class _Microphone:
    PortAudioError = OSError

    def __init__(self, rate, seconds):
        self.rate, self.seconds, self.opened = rate, seconds, None

    def query_devices(self, kind):
        return {"default_samplerate": float(self.rate)}

    def InputStream(self, **options):
        import numpy

        microphone = self

        class Stream:
            def __enter__(self):
                microphone.opened = options
                options["callback"](numpy.full((microphone.rate * microphone.seconds, 1), 0.25, numpy.float32), None, None, None)

            def __exit__(self, *_):
                return False

        return Stream()


def test_record_resamples_the_microphone_to_16_khz_and_keeps_30_s(monkeypatch, capsys):
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("soxr")
    from needle.whistle.playground import record

    microphone = _Microphone(48000, 1)
    monkeypatch.setitem(sys.modules, "sounddevice", microphone)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    audio = record()
    assert microphone.opened["samplerate"] == 48000 and microphone.opened["channels"] == 1
    assert audio.dtype == numpy.float32 and len(audio) == 16000 and abs(float(audio[8000]) - 0.25) < 0.01
    assert "recording, Enter to stop" in capsys.readouterr().out
    monkeypatch.setitem(sys.modules, "sounddevice", _Microphone(16000, 31))
    assert len(record()) == 30 * 16000
    assert "keeping the first 30 s" in capsys.readouterr().out


def test_record_says_what_is_missing(monkeypatch):
    from needle.whistle.playground import record

    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(RuntimeError, match=r"cactus-needle\[whistle\]"):
        record()


def test_rate_counts_steps_after_the_first_mark():
    from needle.whistle.compare import rate

    assert rate([1.0, 1.5, 2.0]) == 2.0
    assert rate([1.0]) == 0.0 and rate([]) == 0.0 and rate([2.0, 2.0]) == 0.0


def test_compare_prints_one_line_per_model_with_dashes_for_missing_timing(capsys):
    from needle.whistle.compare import compare

    models = [("whistle", 17e6, lambda audio: ("hello there", 0.013, 1264.0)),
              ("moonshine tiny v2", 45e6, lambda audio: ("", None, None))]
    compare(models, [0.0] * 16000)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("  whistle             17 MB  ttft  13 ms  decode 1264 tok/s  total")
    assert lines[0].endswith("ms   hello there")
    assert lines[1].startswith("  moonshine tiny v2   45 MB  ttft      -  decode          -  total")
    assert lines[1].endswith("ms   (no speech)")


def test_playground_and_compare_share_one_status_line():
    from needle.whistle.playground import status

    assert status("whistle", 17e6, 0.0137, 1264.4, 0.048) == "  whistle             17 MB  ttft  14 ms  decode 1264 tok/s  total   48 ms"
    assert status("moonshine tiny v2", 145e6, None, None, 0.21) == "  moonshine tiny v2  145 MB  ttft      -  decode          -  total  210 ms"


def test_status_reads_a_slow_run_in_seconds():
    from needle.whistle.playground import status

    assert status("whistle", 17e6, 0.0137, 1264.4, 12.5).endswith("total  12.5 s")
    assert status("whistle", 17e6, 0.0137, 1264.4, 9.9).endswith("total 9900 ms")


def test_audio_path_takes_a_path_as_a_terminal_hands_it_over():
    from needle.whistle.playground import audio_path

    assert audio_path(" clip.wav ") == "clip.wav"
    assert audio_path('"my clips/a b.wav"') == "my clips/a b.wav"
    assert audio_path("'my clips/a b.wav'") == "my clips/a b.wav"
    assert audio_path("my\\ clips/a\\ b.wav") == "my clips/a b.wav"


def test_playground_prints_text_words_and_timing(capsys):
    from needle.whistle.playground import transcribe

    whistle = _Whistle({"text": "hello there", "language": "en", "ttft_ms": 12.6, "decode_tps": 900.4,
                        "words": [{"word": "hello", "start": 0.1, "end": 0.4, "probability": 0.98}]})
    transcribe(whistle, "clip.wav", {"language": "en", "keywords": ["Siobhan"], "timestamps": True})
    out = capsys.readouterr().out.splitlines()
    assert whistle.calls == [("clip.wav", {"language": "en", "keywords": ["Siobhan"], "word_timestamps": True})]
    assert out[0] == "hello there"
    assert out[1].split() == ["0.10", "-", "0.40", "hello", "0.98"]
    assert out[2].startswith("  whistle           ") and "ttft  13 ms  decode  900 tok/s  total" in out[2] and out[2].endswith("ms   en")


def test_playground_reports_engine_errors_instead_of_raising(capsys):
    from needle.whistle.playground import transcribe

    class Broken:
        def transcribe(self, audio, **options):
            raise RuntimeError("audio limit is 30 s")

    transcribe(Broken(), "long.wav", {"language": None, "keywords": [], "timestamps": False})
    assert capsys.readouterr().out == "  audio limit is 30 s\n"


def test_playground_keeps_the_language_when_the_code_is_not_one_of_ours(monkeypatch, capsys):
    from needle.whistle import playground

    lines = iter(["/language de", "/language EN", "/language", "/quit"])
    monkeypatch.setattr(playground, "prompt", lambda: next(lines))
    monkeypatch.setattr("needle.whistle.Whistle", lambda weights=None: _Whistle({}))
    playground.main(type("Args", (), {"audio": None, "language": None, "keywords": "",
                                      "word_timestamps": False, "weights": None})())
    out = capsys.readouterr().out
    assert "language de" in out and "language is one of en de fr es it nl pl" in out and "language detected" in out


def test_compare_runs_a_file_the_same_way_the_playground_does(monkeypatch, capsys):
    pytest.importorskip("numpy")
    from needle.whistle import compare as whistle_compare

    seen, read = [], []
    monkeypatch.setattr(whistle_compare, "compare", lambda models, audio: seen.append(len(audio)))
    monkeypatch.setattr("needle.whistle._read_wav", lambda path: read.append(path) or [0.0] * 16000)
    lines = iter(['/file "my clips/a b.wav"', "/quit"])
    monkeypatch.setattr(whistle_compare, "prompt", lambda: next(lines))
    monkeypatch.setattr(whistle_compare, "load_whistle", lambda weights: (17e6, lambda audio: ("", None, None)))
    monkeypatch.setattr(whistle_compare, "load_whisper", lambda size: (76e6, lambda audio: ("", None, None)))
    monkeypatch.setattr(whistle_compare, "load_moonshine", lambda: (45e6, lambda audio: ("", None, None)))
    whistle_compare.main(type("Args", (), {"audio": None, "weights": None})())
    assert read == ["my clips/a b.wav"] and seen == [16000]


def test_compare_needs_its_extra(monkeypatch):
    from needle.whistle import compare as whistle_compare

    monkeypatch.setitem(sys.modules, "whisper", None)
    with pytest.raises(SystemExit, match=r"cactus-needle\[whistle,whistle-compare\]"):
        whistle_compare.main(type("Args", (), {"audio": None, "weights": None})())


def test_cli_routes_the_whistle_commands(monkeypatch):
    import needle._telemetry
    import needle.cli
    import needle.whistle.playground
    import needle.whistle.compare

    seen = []
    monkeypatch.setattr(needle._telemetry, "track", lambda *a, **k: None)
    monkeypatch.setattr(needle.whistle.playground, "main", lambda args: seen.append(("playground", args)))
    monkeypatch.setattr(needle.whistle.compare, "main", lambda args: seen.append(("compare", args)))
    monkeypatch.setattr(sys, "argv", ["needle", "whistle", "playground", "clip.wav", "--language", "de", "--keywords", "Siobhan, Krzysztof", "--word-timestamps"])
    needle.cli.main()
    monkeypatch.setattr(sys, "argv", ["needle", "whistle", "compare", "--weights", "w.cact"])
    needle.cli.main()
    assert [s[0] for s in seen] == ["playground", "compare"]
    assert (seen[0][1].audio, seen[0][1].language, seen[0][1].keywords, seen[0][1].word_timestamps) == ("clip.wav", "de", "Siobhan, Krzysztof", True)
    assert (seen[1][1].audio, seen[1][1].weights) == (None, "w.cact")
    monkeypatch.setattr(sys, "argv", ["needle", "whistle"])
    with pytest.raises(SystemExit, match="needle whistle playground \\| compare \\| fetch \\| download"):
        needle.cli.main()


def test_needle3_whistle_resolves_inside_the_needle3_repo(tmp_path, monkeypatch):
    """The combined engine shares needle3's repo, so every path it fetches is prefixed."""
    import zipfile
    import needle
    from needle.agent import fetch

    audio = fetch.NEEDLE3_WHISTLE
    assert fetch.engine_repo(audio) == fetch.engine_repo(3)
    assert fetch.engine_version(audio) == "3.0.0"
    assert fetch.lib_name(audio) == fetch.lib_name(3).replace("needle", audio)
    assert fetch.engine_wheel("3.0.0", "win_arm64", audio).startswith(audio + "/python/")
    assert fetch.engine_wheel("3.0.3", "win_arm64", 3).startswith("python/")

    monkeypatch.setattr(os.path, "expanduser", lambda path: str(tmp_path))
    monkeypatch.setattr(needle, "__file__", str(tmp_path / "package" / "__init__.py"))
    monkeypatch.setattr(fetch, "_register_download", lambda generation: None)
    monkeypatch.setenv(fetch.lib_path_env(audio), "/opt/libneedle3_whistle.so")
    assert needle._library_path(audio) == "/opt/libneedle3_whistle.so"
    monkeypatch.delenv(fetch.lib_path_env(audio))

    wheel, fetched = tmp_path / "engine.whl", []
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("needle/" + fetch.lib_name(audio), b"engine")
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda **kwargs: fetched.append(kwargs["filename"]) or str(wheel))
    resolved = needle._library_path(audio)
    assert resolved == str(tmp_path / ".cache" / "cactus-needle" / audio / "3.0.0" / fetch.lib_name(audio))
    assert fetched == [fetch.engine_wheel("3.0.0", fetch._platform_tag(), audio)]
