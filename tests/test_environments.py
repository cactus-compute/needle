import json
import re
import types

import pytest

from conftest import requires_engine
from needle import environments, tool
from needle.environments import _harness

CATEGORIES = {"positive", "missing", "irrelevant", "negation", "invalid", "parallel"}


@pytest.fixture(params=sorted(environments.ENVIRONMENTS))
def env(request):
    return environments.ENVIRONMENTS[request.param]


def _schemas(env):
    return {fn._needle_tool["name"]: fn._needle_tool for fn in env.TOOLS}


def test_registry_matches_modules():
    for name, module in environments.ENVIRONMENTS.items():
        assert module.__name__ == f"needle.environments.{name}"
        assert module.SYSTEM and module.TOOLS and module.TEST_CASES


def test_tool_surface(env):
    schemas = _schemas(env)
    assert 0 < len(schemas) <= 5
    for schema in schemas.values():
        assert schema["description"]
        for prop in schema["parameters"]["properties"].values():
            assert "type" in prop


def test_cases_cover_all_categories(env):
    assert {case["category"] for case in env.TEST_CASES} == CATEGORIES


def test_cases_match_declared_tools(env):
    schemas = _schemas(env)
    for case in env.TEST_CASES:
        for call in case["calls"]:
            parameters = schemas[call["name"]]["parameters"]
            arguments = call["arguments"]
            assert set(arguments) <= set(parameters["properties"])
            assert set(parameters.get("required", [])) <= set(arguments)
            for key, value in arguments.items():
                prop = parameters["properties"][key]
                if "enum" in prop:
                    assert value in prop["enum"]
                if "minimum" in prop:
                    assert value >= prop["minimum"]
                if "maximum" in prop:
                    assert value <= prop["maximum"]
                if "pattern" in prop:
                    assert re.match(prop["pattern"], value)


def test_tools_execute(env):
    schemas = {fn._needle_tool["name"]: fn for fn in env.TOOLS}
    for case in env.TEST_CASES:
        for call in case["calls"]:
            result = schemas[call["name"]](**call["arguments"])
            assert result["ok"] is True


@requires_engine
def test_smart_home_smoke():
    from needle.environments import smart_home

    response = smart_home.agent.complete("turn on the kitchen lights")
    assert response.get("type") in ("call", "text", "refuse")


_MATCHING = {"name": "set_level", "arguments": {"level": 5}}
_OTHER = {"name": "set_level", "arguments": {"level": 9}}
_PASS = "pass"
_LOW = "low"
_FAIL = "fail"


@tool
def set_level(level: int):
    """Set the level.

    Args:
        level: The level to set.
    """
    return {"ok": True, "level": level}


class _Stub:
    """Engine stand-in whose reply is chosen from the query, so a suite's
    outcome is decided by its own cases rather than by a model."""

    def needle_init(self, system, tools, index):
        return 0

    def needle_load(self, blob, size):
        return 0

    def needle_complete(self, text, *args):
        # The output buffer is the second-to-last argument in every
        # needle_complete signature, for both engine generations.
        buffer = args[-2]
        prompt = text.decode("utf-8").strip()
        matched = not prompt.startswith(_FAIL)
        confidence = 0.2 if prompt.startswith(_LOW) else 0.9
        buffer.value = json.dumps({
            "type": "call",
            "confidence": confidence,
            "function_calls": [_MATCHING if matched else _OTHER],
        }).encode("utf-8")
        return 0

    def needle_reset(self):
        pass


@pytest.fixture
def stub(monkeypatch):
    import needle

    engine = _Stub()
    monkeypatch.setenv("NEEDLE_TELEMETRY", "0")
    # agent_for sets this with setdefault, so claiming it here keeps the change
    # from leaking into the rest of the process.
    monkeypatch.setenv("NEEDLE_STRICT_VALIDATE", "1")
    monkeypatch.setattr(needle, "_lib", lambda generation=2: engine)
    monkeypatch.setattr(needle, "_active", {})
    monkeypatch.setattr(_harness, "_agents", {})


def _synthetic(name, passes, failures, critical=0):
    """A suite whose queries drive the stub: a _PASS query is answered with the
    call its case expects, a _FAIL query with a different one."""
    cases = [{"query": f"{_PASS} {i}", "calls": [_MATCHING], "category": "positive"}
             for i in range(passes)]
    for i in range(failures):
        case = {"query": f"{_FAIL} {i}", "calls": [_MATCHING], "category": "synthetic"}
        if i < critical:
            case["critical"] = True
        cases.append(case)
    return types.SimpleNamespace(__name__=name, TOOLS=[set_level], SYSTEM="",
                                 TEST_CASES=cases)


def test_suite_passes_at_29_of_32(stub):
    assert _harness.run_tests(_synthetic("a", 29, 3), verbose=False) is True


def test_suite_fails_at_28_of_32(stub):
    assert _harness.run_tests(_synthetic("b", 28, 4), verbose=False) is False


def test_one_critical_failure_fails_a_suite_that_clears_the_rate(stub):
    assert _harness.run_tests(_synthetic("c", 31, 1, critical=1), verbose=False) is False


def test_min_confidence_treats_a_low_confidence_call_as_a_refusal(stub):
    module = types.SimpleNamespace(
        __name__="d", TOOLS=[set_level], SYSTEM="",
        TEST_CASES=[{"query": _LOW, "calls": [], "category": "missing"}])

    assert _harness.run_tests(module, 0.0, verbose=False) is False
    assert _harness.run_tests(module, 0.4, verbose=False) is True


def test_min_confidence_keeps_a_call_at_the_threshold(stub):
    module = types.SimpleNamespace(
        __name__="e", TOOLS=[set_level], SYSTEM="",
        TEST_CASES=[{"query": _PASS, "calls": [_MATCHING], "category": "positive"}])

    # "at or above the threshold" is a strict comparison, so a call whose
    # confidence equals the gate is acted on rather than treated as a refusal.
    assert _harness.run_tests(module, 0.9, verbose=False) is True


def test_aggregate_reports_failure_when_any_environment_fails(monkeypatch):
    class _Module:
        def __init__(self, ok):
            self._ok = ok

        def run_tests(self, min_confidence=0.0, verbose=True):
            return self._ok

    failing = environments._NAMES[-1]
    monkeypatch.setattr(environments, "_load", lambda name: _Module(name != failing))
    assert environments.run_tests(verbose=False) is False

    monkeypatch.setattr(environments, "_load", lambda name: _Module(True))
    assert environments.run_tests(verbose=False) is True
