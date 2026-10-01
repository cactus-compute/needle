import sys

import pytest


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
