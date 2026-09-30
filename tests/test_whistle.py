import array
import math
import os
import struct
import wave

import pytest


def _whistle_available():
    from needle import whistle

    engine = os.environ.get("NEEDLE_WHISTLE_LIB_PATH") or os.path.join(whistle._cache_dir(), whistle._lib_name())
    weights = os.environ.get("NEEDLE_WHISTLE_WEIGHTS") or os.path.join(whistle._cache_dir(), whistle.WHISTLE_WEIGHTS)
    return os.path.exists(engine) and os.path.exists(weights)


requires_whistle = pytest.mark.skipif(not _whistle_available(), reason="no Whistle engine or whistle.cact on this machine (fetched on first real use)")


def test_engine_and_weights_resolve_like_needle(tmp_path, monkeypatch):
    import zipfile
    from needle import whistle

    monkeypatch.setattr(os.path, "expanduser", lambda path: str(tmp_path))
    monkeypatch.setattr(whistle, "__file__", str(tmp_path / "package" / "whistle.py"))
    monkeypatch.delenv("NEEDLE_WHISTLE_WEIGHTS", raising=False)
    monkeypatch.setenv("NEEDLE_WHISTLE_LIB_PATH", "/opt/libwhistle.so")
    assert whistle._library_path() == "/opt/libwhistle.so"
    monkeypatch.delenv("NEEDLE_WHISTLE_LIB_PATH")

    wheel, weights, fetched = tmp_path / "engine.whl", tmp_path / "published.cact", []
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("needle/" + whistle._lib_name(), b"engine")
    weights.write_bytes(b"weights")
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda **kwargs: fetched.append((kwargs["repo_id"], kwargs["filename"])) or str(weights if kwargs["filename"].endswith(".cact") else wheel))
    cache = tmp_path / ".cache" / "cactus-needle" / "whistle" / whistle.ENGINE_VERSION
    assert whistle._library_path() == str(cache / whistle._lib_name()) and (cache / whistle._lib_name()).read_bytes() == b"engine"
    assert whistle._weights_path() == str(cache / "whistle.cact") and (cache / "whistle.cact").read_bytes() == b"weights"
    assert [repo for repo, _ in fetched] == ["Cactus-Compute/whistle"] * 2
    assert fetched[0][1].startswith(f"python/cactus_whistle-{whistle.ENGINE_VERSION}-py3-none-") and fetched[1][1] == "whistle.cact"
    assert whistle._library_path() and whistle._weights_path() and len(fetched) == 2


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
