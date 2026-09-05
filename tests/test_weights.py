import json
import warnings

import pytest

ENVELOPE = json.dumps({"type": "call", "confidence": 0.9,
                       "function_calls": [{"name": "City",
                                           "arguments": {"city": "Paris"}}]}).encode("utf-8")


class _Stub:
    def __init__(self, calls):
        self.calls = calls

    def needle_init(self, system, tools, index):
        self.calls.append("init")
        return 0

    def needle_load(self, blob, size):
        self.calls.append("load")
        return 0

    def needle_complete(self, text, max_new_tokens, buffer, size):
        self.calls.append("complete")
        buffer.value = ENVELOPE
        return 0

    def needle_reset(self):
        self.calls.append("reset")


class _WorkerStub:
    next_pid = 1000

    def __init__(self, calls, library, weights, system, tools, tool_index,
                 buffer_size):
        self.calls = calls
        self.pid = type(self).next_pid
        type(self).next_pid += 1
        self.calls.append(("worker_start", self.pid, library, weights))

    def complete(self, text, max_new_tokens):
        self.calls.append(("worker_complete", self.pid, text))
        return ENVELOPE.decode("utf-8")

    def reset(self):
        self.calls.append(("worker_reset", self.pid))

    def close(self):
        self.calls.append(("worker_close", self.pid))


@pytest.fixture
def engine(monkeypatch):
    import needle

    calls = []
    stubs = {2: _Stub(calls), 3: _Stub(calls)}
    monkeypatch.setattr(needle, "_lib", lambda generation=2: stubs[generation])
    monkeypatch.setattr(needle, "_library_path",
                        lambda generation=2: f"/tmp/libneedle{generation}")
    monkeypatch.setattr(
        needle, "FineTuneWorker",
        lambda library, weights, system, tools, tool_index, buffer_size:
            _WorkerStub(calls, library, weights, system, tools, tool_index,
                        buffer_size))
    monkeypatch.setattr(needle, "_active", {})
    return calls


@pytest.fixture
def tuned(tmp_path):
    path = tmp_path / "tuned.cact"
    path.write_bytes((0x05E12A83).to_bytes(4, "little") + b"tuned weights")
    return str(path)


@pytest.fixture
def tuned_v3(tmp_path):
    path = tmp_path / "tuned-v3.cact"
    path.write_bytes((0x05E12A84).to_bytes(4, "little") + b"tuned weights")
    return str(path)


def _tuned_agent(path):
    import needle

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return needle.Needle(tools="[]", weights=path)


def test_base_agent_and_tuned_worker_coexist(engine, tuned):
    import needle

    tuned_agent = _tuned_agent(tuned)
    base = needle.Needle(tools="[]")
    tuned_agent.complete("tuned")
    base.complete("base")
    assert any(call[0] == "worker_complete" for call in engine if isinstance(call, tuple))
    assert "complete" in engine


def test_tuned_worker_does_not_rebind_existing_base_agent(engine, tuned):
    import needle

    base = needle.Needle(tools="[]")
    base.complete("hello")
    tuned_agent = _tuned_agent(tuned)
    tuned_agent.complete("hello")
    base.complete("again")
    assert engine.count("load") == 0
    assert engine.count("complete") == 2


def test_tuned_agent_loads_one_worker_and_reuses_it(engine, tuned):
    agent = _tuned_agent(tuned)
    agent.complete("one")
    agent.complete("two")
    starts = [call for call in engine
              if isinstance(call, tuple) and call[0] == "worker_start"]
    assert len(starts) == 1
    assert len([call for call in engine
                if isinstance(call, tuple) and call[0] == "worker_complete"]) == 2


def test_extract_uses_an_explicit_tuned_worker(engine, tuned):
    import needle

    schema = {"name": "City", "parameters": {"type": "object",
                                             "properties": {"city": {"type": "string"}}}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert needle.extract("the weather in Paris", schema, weights=tuned) == {
            "city": "Paris"}
    assert any(call[0] == "worker_start" and call[3] == tuned
               for call in engine if isinstance(call, tuple))


def test_v2_and_v3_weights_route_to_independent_engines(engine, tuned, tuned_v3,
                                                        monkeypatch):
    import needle

    routed = []
    monkeypatch.setattr(needle, "_library_path",
                        lambda generation=2: routed.append(generation) or
                        f"/tmp/libneedle{generation}")

    v2 = _tuned_agent(tuned)
    v3 = _tuned_agent(tuned_v3)
    assert v2._generation == 2
    assert v3._generation == 3
    assert routed == [2, 3]


def test_unknown_archive_tag_is_rejected_before_loading(engine, tmp_path):
    path = tmp_path / "unknown.cact"
    path.write_bytes(b"NOPE")
    with pytest.raises(RuntimeError, match="unknown .cact format tag"):
        _tuned_agent(str(path))


def test_two_finetunes_use_independent_workers(engine, tuned, tmp_path):
    other = tmp_path / "other.cact"
    other.write_bytes((0x05E12A83).to_bytes(4, "little") + b"other weights")

    first = _tuned_agent(tuned)
    second = _tuned_agent(str(other))
    assert first._worker.pid != second._worker.pid
    first.complete("one")
    second.complete("two")
    assert ("worker_complete", first._worker.pid, "one") in engine
    assert ("worker_complete", second._worker.pid, "two") in engine
    first.close()
    second.close()


def test_extraction_rejects_fabricated_temporal_year():
    import datetime
    import needle
    import pydantic

    class Invoice(pydantic.BaseModel):
        due_date: datetime.datetime

    response = {"validation": {"ungrounded": ["Invoice.due_date"]}}
    correct = {"due_date": "2034-09-05T00:00:00Z"}
    needle._validate_extraction("due on 5th September 2034", Invoice,
                                correct, response)

    fabricated = {"due_date": "2024-09-05T00:00:00Z"}
    with pytest.raises(needle.ExtractionValidationError, match="due_date"):
        needle._validate_extraction("due on 5th September 2034", Invoice,
                                    fabricated, response)

    with pytest.raises(needle.ExtractionValidationError, match="due_date"):
        needle._validate_extraction(
            "due on 5th September 42", Invoice,
            {"due_date": "2026-09-05T00:00:00Z"}, response)

    assert needle._source_years("due September 5, 2034") == {2034}
    assert needle._source_years("due September 5") == set()
    assert needle._source_years("due 5th September 42") == {42}
    assert needle._source_years("Invoice 42 is due tomorrow at 5") == set()


def test_agent_extract_carries_its_own_system_facts(engine, monkeypatch):
    import needle

    seen = {}

    def spy(text, schema, system=None, max_new_tokens=256, weights=None, strict=True):
        seen.update(text=text, system=system, weights=weights, strict=strict)
        return None

    monkeypatch.setattr(needle, "extract", spy)
    facts = "date: 2026-07-21 Tue 14:30; locale: en-US"
    agent = needle.Needle(tools="[]", system=facts)

    assert agent.extract("dinner tomorrow at 7", {"type": "object"}) is None
    assert seen["system"] == facts
    assert seen["text"] == "dinner tomorrow at 7"


def test_agent_extract_without_system_facts_sends_none(engine, monkeypatch):
    import needle

    seen = {}
    monkeypatch.setattr(
        needle, "extract",
        lambda text, schema, system=None, **kwargs: seen.update(system=system))
    needle.Needle(tools="[]").extract("anything", {"type": "object"})

    assert seen["system"] is None


def test_agent_extract_still_carries_weights_and_strict(engine, tuned, monkeypatch):
    import needle

    seen = {}
    monkeypatch.setattr(
        needle, "extract",
        lambda text, schema, system=None, max_new_tokens=256, weights=None, strict=True:
            seen.update(system=system, weights=weights, strict=strict))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent = needle.Needle(tools="[]", weights=tuned, system="device: phone")
    agent.extract("anything", {"type": "object"}, strict=False)

    assert seen == {"system": "device: phone", "weights": tuned, "strict": False}
