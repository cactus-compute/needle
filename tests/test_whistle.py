import array
import math
import os
import struct
import wave

import pytest

from conftest import _engine_available


def _whistle_available():
    try:
        from needle import whistle

        weights = os.environ.get("NEEDLE_WHISTLE_WEIGHTS") or os.path.join(
            os.path.expanduser("~"), ".cache", "cactus-needle", "whistle", whistle.WHISTLE_WEIGHTS)
        return _engine_available(3) and os.path.exists(weights) and hasattr(whistle._lib(), "whistle_transcribe")
    except Exception:
        return False


requires_whistle = pytest.mark.skipif(not _whistle_available(), reason="no Needle engine with Whistle, or no whistle.cact, on this machine")


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
    assert result == {"text": "", "words": [], "language": ""}


@requires_whistle
def test_transcribe_returns_text_language_and_timed_words(tmp_path):
    from needle import Whistle
    from needle.whistle import LANGUAGES

    whistle = Whistle()
    sound = [v * (0.2 + 0.8 * abs(math.sin(math.pi * 3 * i / 16000))) for i, v in enumerate(_tone(3))]
    plain = whistle.transcribe(sound)
    assert set(plain) == {"text", "language"} and plain["language"] in LANGUAGES + ("",)
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
    assert Whistle().transcribe([0.0] * 16000)["text"] == ""
