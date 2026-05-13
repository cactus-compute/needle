"""Unit tests for grammar-constrained decoding.

Covers:
- ToolConstraints param trie: flat, JSON Schema, and description-bearing formats
- Trie operations
- JsonStateMachine state transitions
- apply_constraints logit masking
- _check_token_valid token validation
- ConstrainedDecoder confidence tracking
"""

import numpy as np
import pytest

from needle.model.constrained import (
    JsonState,
    JsonStateMachine,
    Trie,
    ToolConstraints,
    ConstrainedDecoder,
    TokenIndex,
    _check_token_valid,
    apply_constraints,
    build_token_strings,
)


# ── Trie ──────────────────────────────────────────────────────────────────────

class TestTrie:
    def test_insert_and_get_node(self):
        t = Trie()
        t.insert("location")
        assert t.get_node("loc") is not None

    def test_terminal_flag(self):
        t = Trie()
        t.insert("location")
        assert t.get_node("location").is_terminal

    def test_prefix_not_terminal(self):
        t = Trie()
        t.insert("location")
        assert not t.get_node("loc").is_terminal

    def test_missing_prefix_returns_none(self):
        t = Trie()
        t.insert("location")
        assert t.get_node("xyz") is None

    def test_empty_prefix_returns_root(self):
        t = Trie()
        t.insert("ab")
        node = t.get_node("")
        assert node is not None
        assert "a" in node.children

    def test_words_property(self):
        t = Trie()
        for w in ["alpha", "beta", "gamma"]:
            t.insert(w)
        assert sorted(t.words) == ["alpha", "beta", "gamma"]


# ── ToolConstraints ────────────────────────────────────────────────────────────

class TestToolConstraints:
    def test_flat_string_params(self):
        """Regression: flat {\"key\": \"string\"} must populate param trie."""
        tools = '[{"name":"get_weather","parameters":{"location":"string"}}]'
        tc = ToolConstraints(tools)
        trie = tc.get_param_trie("get_weather")
        assert trie is not None
        assert trie.get_node("location").is_terminal

    def test_flat_multiple_params(self):
        tools = '[{"name":"set_alarm","parameters":{"time":"string","label":"string"}}]'
        tc = ToolConstraints(tools)
        trie = tc.get_param_trie("set_alarm")
        assert trie.get_node("time").is_terminal
        assert trie.get_node("label").is_terminal

    def test_json_schema_properties_format(self):
        tools = '[{"name":"search","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}]'
        tc = ToolConstraints(tools)
        trie = tc.get_param_trie("search")
        assert trie is not None
        assert trie.get_node("query").is_terminal

    def test_description_field_ignored_in_param_trie(self):
        """description at tool level must not leak into param trie."""
        tools = '[{"name":"get_weather","description":"Get current weather","parameters":{"location":"string"}}]'
        tc = ToolConstraints(tools)
        trie = tc.get_param_trie("get_weather")
        assert trie.get_node("location").is_terminal
        assert trie.get_node("description") is None

    def test_description_does_not_break_name_trie(self):
        tools = '[{"name":"get_weather","description":"Get current weather","parameters":{"location":"string"}}]'
        tc = ToolConstraints(tools)
        assert tc.name_trie.get_node("get_weather").is_terminal

    def test_name_trie_populated(self):
        tools = '[{"name":"get_weather","parameters":{"location":"string"}},{"name":"set_alarm","parameters":{"time":"string"}}]'
        tc = ToolConstraints(tools)
        assert tc.name_trie.get_node("get_weather").is_terminal
        assert tc.name_trie.get_node("set_alarm").is_terminal

    def test_unknown_tool_returns_none(self):
        tools = '[{"name":"get_weather","parameters":{"location":"string"}}]'
        tc = ToolConstraints(tools)
        assert tc.get_param_trie("nonexistent") is None

    def test_invalid_json_does_not_raise(self):
        tc = ToolConstraints("not json at all")
        assert tc.name_trie.get_node("x") is None

    def test_empty_tools_list(self):
        tc = ToolConstraints("[]")
        assert tc.name_trie.words == []

    def test_multi_tool_playground_format(self):
        """Matches exact format used in playground and test.py."""
        tools = '''[
            {"name": "get_weather",   "description": "Get weather",    "parameters": {"location": "string"}},
            {"name": "set_alarm",     "description": "Set an alarm",   "parameters": {"time": "string", "label": "string"}},
            {"name": "search_web",    "description": "Search the web", "parameters": {"query": "string"}},
            {"name": "send_message",  "description": "Send a message", "parameters": {"to": "string", "body": "string"}}
        ]'''
        tc = ToolConstraints(tools)
        assert tc.get_param_trie("get_weather").get_node("location").is_terminal
        assert tc.get_param_trie("set_alarm").get_node("time").is_terminal
        assert tc.get_param_trie("set_alarm").get_node("label").is_terminal
        assert tc.get_param_trie("search_web").get_node("query").is_terminal
        assert tc.get_param_trie("send_message").get_node("to").is_terminal
        assert tc.get_param_trie("send_message").get_node("body").is_terminal


# ── _check_token_valid ────────────────────────────────────────────────────────

class TestCheckTokenValid:
    def setup_method(self):
        self.trie = Trie()
        self.trie.insert("location")
        self.trie.insert("lang")

    def test_valid_prefix_token(self):
        node = self.trie.get_node("")
        assert _check_token_valid("loc", node)

    def test_valid_full_word_plus_quote(self):
        node = self.trie.get_node("")
        assert _check_token_valid('location"', node)

    def test_invalid_token_not_in_trie(self):
        node = self.trie.get_node("")
        assert not _check_token_valid("xyz", node)

    def test_quote_at_nonterminal_invalid(self):
        node = self.trie.get_node("loc")
        assert not _check_token_valid('"', node)

    def test_quote_at_terminal_valid(self):
        node = self.trie.get_node("location")
        assert _check_token_valid('"', node)


# ── apply_constraints ────────────────────────────────────────────────────────

class TestApplyConstraints:
    def _make_token_strings(self):
        return ["", "[", "{", '"', "l", "o", "c", "a", "t", "i", "n", "x", 'location"']

    def _make_token_index(self, token_strings):
        return TokenIndex(token_strings)

    def test_valid_tokens_unmasked(self):
        trie = Trie()
        trie.insert("location")
        token_strings = self._make_token_strings()
        token_index = self._make_token_index(token_strings)
        logits = np.zeros(len(token_strings))
        result = apply_constraints(logits, JsonState.IN_ARG_KEY, trie.root, token_strings, token_index)
        assert result[4] == 0.0       # "l" valid
        assert result[11] == -np.inf  # "x" invalid

    def test_fallback_on_empty_trie(self):
        trie = Trie()
        token_strings = self._make_token_strings()
        token_index = self._make_token_index(token_strings)
        logits = np.ones(len(token_strings))
        result = apply_constraints(logits, JsonState.IN_ARG_KEY, trie.root, token_strings, token_index)
        np.testing.assert_array_equal(result, logits)


# ── JsonStateMachine ──────────────────────────────────────────────────────────

class TestJsonStateMachine:
    def test_initial_state_is_free(self):
        assert JsonStateMachine().state == JsonState.FREE

    def test_enters_in_name_after_name_prefix(self):
        m = JsonStateMachine()
        m.feed('[{"name":"')
        assert m.state == JsonState.IN_NAME

    def test_captures_tool_name(self):
        m = JsonStateMachine()
        m.feed('[{"name":"get_weather"')
        assert m.current_function == "get_weather"
        assert m.state == JsonState.FREE

    def test_enters_in_arg_key_after_arguments(self):
        m = JsonStateMachine()
        m.feed('[{"name":"get_weather","arguments":{"')
        assert m.state == JsonState.IN_ARG_KEY

    def test_captures_arg_key(self):
        m = JsonStateMachine()
        m.feed('[{"name":"get_weather","arguments":{"location"')
        assert m.state == JsonState.FREE

    def test_second_arg_key_triggers_in_arg_key(self):
        m = JsonStateMachine()
        m.feed('[{"name":"set_alarm","arguments":{"time":"7am","')
        assert m.state == JsonState.IN_ARG_KEY

    def test_constrained_buf_resets_on_close_quote(self):
        m = JsonStateMachine()
        m.feed('[{"name":"get_weather","arguments":{"location"')
        assert m.constrained_buf == ""


# ── ConstrainedDecoder confidence ────────────────────────────────────────────

class TestConstrainedDecoderConfidence:
    def _make_decoder(self, tools_json: str, vocab: list[str]) -> ConstrainedDecoder:
        token_index = TokenIndex(vocab)
        tc = ToolConstraints(tools_json)
        return ConstrainedDecoder([tc], vocab, token_index)

    def test_confidence_starts_at_zero(self):
        vocab = ["", "g", "s", '"', "et_weather", 'et_weather"']
        decoder = self._make_decoder('[{"name":"get_weather","parameters":{"location":"string"}}]', vocab)
        assert decoder.get_confidence(0) == 0.0

    def test_confidence_set_after_in_name_step(self):
        tools = '[{"name":"get_weather","parameters":{"location":"string"}}]'
        # vocab includes 'g' (start of get_weather) and 's' (start of search)
        vocab = ["", "g", "s", "e", "t", "_", "w", "a", "r", "h", "o", '"',
                 "et_weather", 'et_weather"', "earch_web", 'earch_web"']
        decoder = self._make_decoder(tools, vocab)

        # Drive machine to IN_NAME state
        decoder.machines[0].feed('[{"name":"')
        assert decoder.machines[0].state == JsonState.IN_NAME
        assert decoder.machines[0].constrained_buf == ""

        # Apply constraints — confidence should be captured
        logits = np.zeros(len(vocab))
        logits[1] = 2.0  # 'g' — high probability
        logits[2] = 0.5  # 's' — lower
        decoder.constrain_logits(logits, 0)

        conf = decoder.get_confidence(0)
        assert 0.0 < conf <= 1.0

    def test_confidence_is_higher_when_model_more_certain(self):
        tools = '[{"name":"get_weather","parameters":{"location":"string"}},{"name":"search_web","parameters":{"query":"string"}}]'
        vocab = ["", "g", "s", '"']
        decoder_certain = self._make_decoder(tools, vocab)
        decoder_uncertain = self._make_decoder(tools, vocab)

        for decoder in (decoder_certain, decoder_uncertain):
            decoder.machines[0].feed('[{"name":"')

        # Certain: model heavily favors 'g'
        logits_certain = np.array([0.0, 10.0, 0.1, 0.0])
        decoder_certain.constrain_logits(logits_certain, 0)

        # Uncertain: roughly equal between 'g' and 's'
        logits_uncertain = np.array([0.0, 1.0, 1.0, 0.0])
        decoder_uncertain.constrain_logits(logits_uncertain, 0)

        assert decoder_certain.get_confidence(0) > decoder_uncertain.get_confidence(0)

    def test_confidence_in_valid_range(self):
        tools = '[{"name":"get_weather","parameters":{"location":"string"}}]'
        vocab = ["", "g", "s", '"']
        decoder = self._make_decoder(tools, vocab)
        decoder.machines[0].feed('[{"name":"')
        logits = np.random.randn(len(vocab))
        decoder.constrain_logits(logits, 0)
        conf = decoder.get_confidence(0)
        assert 0.0 <= conf <= 1.0
