import json
import struct
import warnings

import pytest


class _Engine:
    def __init__(self):
        self.calls = []
        self.audio = None

    def needle_init(self, system, tools, index):
        return 0

    def needle_load(self, blob, size):
        return 0

    def needle_reset(self):
        pass

    def needle_set_audio(self, language, keywords, word_timestamps, tool_schema_keywords):
        self.audio = (language, keywords, word_timestamps, tool_schema_keywords)

    def needle_complete(self, text, pcm, samples, max_new_tokens, buffer, capacity):
        self.calls.append((text, samples))
        if text is None:
            envelope = {"type": "call", "function_calls": [{"name": "lights", "arguments": {"room": "kitchen"}}], "confidence": 0.9,
                        "audio_text": f"heard {samples} samples in June 2031", "audio_language": "en", "audio_ttft_ms": 12.0, "audio_decode_tps": 800.0}
        else:
            envelope = {"type": "text", "function_calls": [], "text": "done", "confidence": 0.9}
        buffer.value = json.dumps(envelope).encode("utf-8")
        return 0


class _Speech:
    bound = 0

    def __init__(self, weights=None):
        self.weights = weights

    def _bind(self):
        _Speech.bound += 1


@pytest.fixture
def engine(monkeypatch, tmp_path):
    import needle

    fake = _Engine()
    base = tmp_path / "needle3.cact"
    base.write_bytes((0x05E12A84).to_bytes(4, "little") + b"base weights")
    monkeypatch.setattr(needle, "_lib", lambda generation=3: fake)
    monkeypatch.setattr(needle, "_library_path", lambda generation=3: "/tmp/libneedle3")
    monkeypatch.setattr(needle, "_base_weights_path", lambda generation: str(base))
    monkeypatch.setattr(needle, "_active", {})
    monkeypatch.setattr(needle, "_loaded_base", {})
    monkeypatch.setattr(needle._whistle, "_shared_model", lambda weights: _Speech(weights))
    monkeypatch.setattr(needle._whistle, "_loaded", None)
    monkeypatch.setattr("needle._telemetry.track", lambda *a, **k: None)
    return fake


def test_an_audio_turn_passes_samples_and_reports_the_transcript(engine):
    import needle

    def lights(room: str):
        return {"room": room, "on": True}

    agent = needle.Needle(tools=[lights])
    response = agent.complete(audio=[0.0] * 16000)
    assert engine.calls == [(None, 16000)] and _Speech.bound >= 1
    assert response["audio_text"] == "heard 16000 samples in June 2031" and response["function_calls"][0]["name"] == "lights"
    assert engine.audio == (None, None, 0, 1)
    assert 2031 in agent._seen_years
    response = agent.run(audio=struct.pack("<3f", 0.1, 0.2, 0.3))
    assert engine.calls[1] == (None, 3) and response["results"] == [{"room": "kitchen", "on": True}]
    assert engine.calls[2][0] is not None, "the tool result goes back as text"


def test_audio_turns_take_the_transcription_options_of_transcribe(engine):
    import needle

    agent = needle.Needle(tools="[]")
    agent.complete(audio=[0.0] * 10, language="de", keywords=["Siobhan", "Krzysztof"], word_timestamps=True)
    assert engine.audio == (b"de", b"Siobhan\nKrzysztof", 1, 1)
    agent.run(audio=[0.0] * 10, keywords="one\ntwo", tool_schema_keywords=False)
    assert engine.audio == (None, b"one\ntwo", 0, 0)


def test_text_and_audio_together_are_refused(engine):
    import needle

    agent = needle.Needle(tools="[]")
    with pytest.raises(ValueError, match="not both"):
        agent.complete("hello", audio=[0.0] * 10)
    with pytest.raises(ValueError, match="not both"):
        agent.run("hello", audio=[0.0] * 10)
    assert engine.calls == []


def test_audio_turns_use_the_whistle_the_process_loaded(engine, monkeypatch):
    import needle

    chosen = []
    monkeypatch.setattr(needle._whistle, "_shared_model", lambda weights: chosen.append(weights) or _Speech(weights))
    needle.Needle(tools="[]").complete(audio=[0.0] * 10)
    monkeypatch.setattr(needle._whistle, "_loaded", "tuned-whistle.cact")
    needle.Needle(tools="[]").complete(audio=[0.0] * 10)
    assert chosen == [None, "tuned-whistle.cact"]
