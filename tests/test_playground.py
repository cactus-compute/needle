import pytest


class _Agent:
    def __init__(self, tools, weights):
        self.tools = tools
        self.weights = weights
        self.calls = []

    def complete(self, query, max_new_tokens=256):
        self.calls.append(("complete", query))
        return {"type": "text", "text": query}

    def reset(self):
        self.calls.append("reset")

    def close(self):
        self.calls.append("close")


@pytest.fixture
def agents(monkeypatch):
    """Stand in for Needle, and fail the test if the global engine is touched."""
    import needle

    made = []

    def build(tools=None, weights=None, **_kwargs):
        agent = _Agent(tools, weights)
        made.append(agent)
        return agent

    def refuse(*_args, **_kwargs):
        raise AssertionError("the playground reached the process-global engine")

    monkeypatch.setattr(needle, "Needle", build)
    monkeypatch.setattr(needle, "_lib", refuse)
    return made


def test_a_second_query_on_the_same_tools_rewinds_that_agent(agents):
    from needle.playground.server import Engine

    engine = Engine()
    engine.complete("[]", "one")
    engine.complete("[]", "two")

    assert len(agents) == 1
    assert agents[0].calls == [("complete", "one"), "reset", ("complete", "two")]


def test_new_tools_build_a_fresh_agent_without_rewinding(agents):
    from needle.playground.server import Engine

    engine = Engine()
    engine.complete("[]", "one")
    engine.complete('[{"name": "t"}]', "two")

    assert len(agents) == 2
    assert "reset" not in agents[0].calls
    assert agents[1].calls == [("complete", "two")]


def test_reset_closes_the_agent_it_drops(agents):
    from needle.playground.server import Engine

    engine = Engine()
    engine.complete("[]", "one")
    engine.reset()

    assert agents[0].calls[-1] == "close"
    assert engine.agent is None
    assert engine.tools_json is None


def test_a_tuned_agent_is_rewound_through_itself(agents):
    from needle.playground.server import Engine

    engine = Engine(weights="tuned.cact")
    engine.complete("[]", "one")
    engine.complete("[]", "two")

    assert agents[0].weights == "tuned.cact"
    assert "reset" in agents[0].calls
