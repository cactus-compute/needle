import math
import os
import sys
import wave

import pytest


def _write_wav(path, samples, rate, channels=1):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        for value in samples:
            out.writeframes(int(value * 32767).to_bytes(2, "little", signed=True) * channels)


class _Whistle:
    weights = __file__

    def __init__(self, result):
        self.result = result
        self.calls = []

    def transcribe(self, audio, **options):
        self.calls.append((audio, options))
        return self.result


def test_read_mixes_channels_to_mono_floats(tmp_path):
    numpy = pytest.importorskip("numpy")
    from needle.playground.compare import read

    tone = [0.5 * math.sin(2 * math.pi * 440 * i / 16000) for i in range(1600)]
    path = tmp_path / "stereo.wav"
    _write_wav(path, tone, 16000, channels=2)
    audio = read(str(path))
    assert audio.dtype == numpy.float32 and len(audio) == 1600
    assert abs(float(audio.max()) - 0.5) < 0.01 and abs(float(audio.min()) + 0.5) < 0.01


def test_read_resamples_to_16_khz(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("soxr")
    from needle.playground.compare import read

    path = tmp_path / "wide.wav"
    _write_wav(path, [0.0] * 4800, 48000)
    assert len(read(str(path))) == 1600


def test_read_rejects_other_sample_widths(tmp_path):
    pytest.importorskip("numpy")
    from needle.playground.compare import read

    path = tmp_path / "bytes.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(1)
        out.setframerate(16000)
        out.writeframes(bytes([128] * 160))
    with pytest.raises(ValueError, match="16-bit"):
        read(str(path))


def test_rate_counts_steps_after_the_first_mark():
    from needle.playground.compare import rate

    assert rate([1.0, 1.5, 2.0]) == 2.0
    assert rate([1.0]) == 0.0 and rate([]) == 0.0 and rate([2.0, 2.0]) == 0.0


def test_compare_prints_one_line_per_model_with_dashes_for_missing_timing(capsys):
    from needle.playground.compare import compare

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
    from needle.playground.whistle import status

    assert status("whistle", 17e6, 0.0137, 1264.4, 0.048) == "  whistle             17 MB  ttft  14 ms  decode 1264 tok/s  total   48 ms"
    assert status("moonshine tiny v2", 145e6, None, None, 0.21) == "  moonshine tiny v2  145 MB  ttft      -  decode          -  total  210 ms"


def test_playground_prints_text_words_and_timing(capsys):
    from needle.playground.whistle import transcribe

    whistle = _Whistle({"text": "hello there", "language": "en", "ttft_ms": 12.6, "decode_tps": 900.4,
                        "words": [{"word": "hello", "start": 0.1, "end": 0.4, "probability": 0.98}]})
    transcribe(whistle, "clip.wav", {"lang": "en", "keywords": ["Siobhan"], "timestamps": True})
    out = capsys.readouterr().out.splitlines()
    assert whistle.calls == [("clip.wav", {"language": "en", "keywords": ["Siobhan"], "word_timestamps": True})]
    assert out[0] == "hello there"
    assert out[1].split() == ["0.10", "-", "0.40", "hello", "0.98"]
    assert out[2].startswith("  whistle           ") and "ttft  13 ms  decode  900 tok/s  total" in out[2] and out[2].endswith("ms   en")


def test_playground_reports_engine_errors_instead_of_raising(capsys):
    from needle.playground.whistle import transcribe

    class Broken:
        def transcribe(self, audio, **options):
            raise RuntimeError("audio limit is 30 s")

    transcribe(Broken(), "long.wav", {"lang": None, "keywords": [], "timestamps": False})
    assert capsys.readouterr().out == "  audio limit is 30 s\n"


def test_compare_needs_its_extra(monkeypatch):
    from needle.playground import compare

    monkeypatch.setitem(sys.modules, "whisper", None)
    with pytest.raises(SystemExit, match=r"cactus-needle\[compare\]"):
        compare.main(type("Args", (), {"audio": None, "weights": None})())


def test_cli_routes_whistle_and_compare(monkeypatch):
    import needle._telemetry
    import needle.cli
    import needle.playground.compare
    import needle.playground.whistle

    seen = []
    monkeypatch.setattr(needle._telemetry, "track", lambda *a, **k: None)
    monkeypatch.setattr(needle.playground.whistle, "main", lambda args: seen.append(("whistle", args)))
    monkeypatch.setattr(needle.playground.compare, "main", lambda args: seen.append(("compare", args)))
    monkeypatch.setattr(sys, "argv", ["needle", "whistle", "clip.wav", "--lang", "de", "--keywords", "Siobhan, Krzysztof", "--word-timestamps"])
    needle.cli.main()
    monkeypatch.setattr(sys, "argv", ["needle", "compare", "--weights", "w.cact"])
    needle.cli.main()
    assert [s[0] for s in seen] == ["whistle", "compare"]
    assert (seen[0][1].audio, seen[0][1].lang, seen[0][1].keywords, seen[0][1].word_timestamps) == ("clip.wav", "de", "Siobhan, Krzysztof", True)
    assert (seen[1][1].audio, seen[1][1].weights) == (None, "w.cact")
