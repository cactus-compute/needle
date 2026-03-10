import re

import numpy as np

from .data import (
    EOS_ID,
    JSON_COLON,
    JSON_COMMA,
    JSON_FALSE,
    JSON_KEY_ARGUMENTS,
    JSON_KEY_NAME,
    JSON_LBRACE,
    JSON_LBRACK,
    JSON_NULL,
    JSON_QUOTE,
    JSON_RBRACE,
    JSON_RBRACK,
    JSON_TRUE,
    _normalize_tool_schema_spec,
)


_NUMBER_PREFIX_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+-]?\d*)?$|^-?(?:0|[1-9]\d*)?$|^-?(?:0|[1-9]\d*)\.$|^-?(?:0|[1-9]\d*)(?:\.\d+)?[eE]?$|^-?(?:0|[1-9]\d*)(?:\.\d+)?[eE][+-]?$")
_NUMBER_COMPLETE_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def _is_number_prefix(text):
    return bool(text) and _NUMBER_PREFIX_RE.match(text) is not None


def _is_complete_number(text):
    return bool(text) and _NUMBER_COMPLETE_RE.match(text) is not None


def _schema_type_name(value):
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"str", "string", "text"}:
            return "string"
        if lowered in {"bool", "boolean"}:
            return "boolean"
        if lowered in {"int", "integer", "long"}:
            return "integer"
        if lowered in {"float", "double", "number", "numeric"}:
            return "number"
        if lowered in {"list", "array"}:
            return "array"
        if lowered in {"dict", "map", "object"}:
            return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "any"


def _build_trie(sequences):
    root = {}
    for index, sequence in sequences:
        node = root
        for token_id in sequence:
            node = node.setdefault(int(token_id), {})
        node["_end"] = index
    return root


class JsonValueParser:
    def __init__(self, tokenizer, type_hint="any"):
        self.tokenizer = tokenizer
        self.type_hint = type_hint
        self.regular_ids = tokenizer.regular_token_ids
        self.quote_id = tokenizer.json_token_ids[JSON_QUOTE]
        self.lbrace_id = tokenizer.json_token_ids[JSON_LBRACE]
        self.rbrace_id = tokenizer.json_token_ids[JSON_RBRACE]
        self.lbrack_id = tokenizer.json_token_ids[JSON_LBRACK]
        self.rbrack_id = tokenizer.json_token_ids[JSON_RBRACK]
        self.colon_id = tokenizer.json_token_ids[JSON_COLON]
        self.comma_id = tokenizer.json_token_ids[JSON_COMMA]
        self.true_id = tokenizer.json_token_ids[JSON_TRUE]
        self.false_id = tokenizer.json_token_ids[JSON_FALSE]
        self.null_id = tokenizer.json_token_ids[JSON_NULL]
        self.mode = "expect_value"
        self.stack = []
        self.number_text = ""
        self.complete = False
        self._regular_surfaces = {
            int(token_id): tokenizer.token_surface(int(token_id))
            for token_id in self.regular_ids
        }
        self._number_prefix_ids = np.array(
            [token_id for token_id, text in self._regular_surfaces.items() if _is_number_prefix(text)],
            dtype=np.int32,
        )
        self._number_cache = {}

    def _start_value_ids(self, type_hint=None):
        hint = type_hint or self.type_hint
        if hint == "string":
            return np.array([self.quote_id], dtype=np.int32)
        if hint == "boolean":
            return np.array([self.true_id, self.false_id], dtype=np.int32)
        if hint in {"integer", "number"}:
            return self._number_prefix_ids
        if hint == "array":
            return np.array([self.lbrack_id], dtype=np.int32)
        if hint == "object":
            return np.array([self.lbrace_id], dtype=np.int32)
        return np.concatenate(
            [
                np.array([self.quote_id, self.lbrace_id, self.lbrack_id, self.true_id, self.false_id, self.null_id], dtype=np.int32),
                self._number_prefix_ids,
            ]
        )

    def _number_allowed_ids(self, training=False):
        if training:
            return None
        cached = self._number_cache.get(self.number_text)
        if cached is not None:
            return cached
        allowed = []
        for token_id, surface in self._regular_surfaces.items():
            if _is_number_prefix(self.number_text + surface):
                allowed.append(token_id)
        if self._is_value_complete():
            allowed.extend(self._value_end_ids().tolist())
        cached = np.array(sorted(set(allowed)), dtype=np.int32)
        self._number_cache[self.number_text] = cached
        return cached

    def _value_end_ids(self):
        if not self.stack:
            return np.array([EOS_ID], dtype=np.int32)
        frame = self.stack[-1]
        if frame["kind"] == "array":
            return np.array([self.comma_id, self.rbrack_id], dtype=np.int32)
        return np.array([self.comma_id, self.rbrace_id], dtype=np.int32)

    def _value_finished(self):
        if not self.stack:
            self.complete = True
            self.mode = "done"
            return
        frame = self.stack[-1]
        if frame["kind"] == "array":
            frame["mode"] = "after_value"
            self.mode = "array_after_value"
        else:
            frame["mode"] = "after_value"
            self.mode = "object_after_value"

    def allowed_next_ids(self, training=False):
        if self.mode == "done":
            return np.array([EOS_ID], dtype=np.int32)
        if self.mode == "expect_value":
            ids = self._start_value_ids()
            if training and self.type_hint == "any":
                return np.array([self.quote_id, self.lbrace_id, self.lbrack_id, self.true_id, self.false_id, self.null_id], dtype=np.int32)
            return ids
        if self.mode == "string":
            if training:
                return None
            return np.concatenate([self.regular_ids, np.array([self.quote_id], dtype=np.int32)])
        if self.mode == "number":
            return self._number_allowed_ids(training=training)
        if self.mode == "array_value_or_end":
            return np.concatenate([np.array([self.rbrack_id], dtype=np.int32), self._start_value_ids("any")])
        if self.mode == "array_after_value":
            return np.array([self.comma_id, self.rbrack_id], dtype=np.int32)
        if self.mode == "object_key_or_end":
            return np.array([self.quote_id, self.rbrace_id], dtype=np.int32)
        if self.mode == "object_key_string":
            if training:
                return None
            return np.concatenate([self.regular_ids, np.array([self.quote_id], dtype=np.int32)])
        if self.mode == "object_after_key":
            return np.array([self.colon_id], dtype=np.int32)
        if self.mode == "object_after_value":
            return np.array([self.comma_id, self.rbrace_id], dtype=np.int32)
        return None

    def step(self, token_id):
        token_id = int(token_id)
        if self.mode == "expect_value":
            if token_id == self.quote_id:
                self.mode = "string"
                return
            if token_id == self.lbrack_id:
                self.stack.append({"kind": "array", "mode": "value_or_end"})
                self.mode = "array_value_or_end"
                return
            if token_id == self.lbrace_id:
                self.stack.append({"kind": "object", "mode": "key_or_end"})
                self.mode = "object_key_or_end"
                return
            if token_id in {self.true_id, self.false_id, self.null_id}:
                self._value_finished()
                return
            self.mode = "number"
            self.number_text = self._regular_surfaces.get(token_id, "")
            return

        if self.mode == "string":
            if token_id == self.quote_id:
                self._value_finished()
            return

        if self.mode == "number":
            if token_id in set(self._value_end_ids().tolist()):
                if token_id == EOS_ID:
                    self.complete = True
                    self.mode = "done"
                    return
                frame = self.stack[-1]
                if frame["kind"] == "array":
                    if token_id == self.comma_id:
                        frame["mode"] = "value"
                        self.mode = "expect_value"
                        self.type_hint = "any"
                    else:
                        self.stack.pop()
                        self._value_finished()
                else:
                    if token_id == self.comma_id:
                        frame["mode"] = "key_or_end"
                        self.mode = "object_key_or_end"
                    else:
                        self.stack.pop()
                        self._value_finished()
                self.number_text = ""
                return
            self.number_text += self._regular_surfaces.get(token_id, "")
            return

        if self.mode == "array_value_or_end":
            if token_id == self.rbrack_id:
                self.stack.pop()
                self._value_finished()
                return
            self.mode = "expect_value"
            self.type_hint = "any"
            self.step(token_id)
            return

        if self.mode == "array_after_value":
            frame = self.stack[-1]
            if token_id == self.comma_id:
                frame["mode"] = "value"
                self.mode = "expect_value"
                self.type_hint = "any"
                return
            self.stack.pop()
            self._value_finished()
            return

        if self.mode == "object_key_or_end":
            if token_id == self.rbrace_id:
                self.stack.pop()
                self._value_finished()
                return
            self.mode = "object_key_string"
            return

        if self.mode == "object_key_string":
            if token_id == self.quote_id:
                self.mode = "object_after_key"
            return

        if self.mode == "object_after_key":
            self.mode = "expect_value"
            self.type_hint = "any"
            return

        if self.mode == "object_after_value":
            frame = self.stack[-1]
            if token_id == self.comma_id:
                frame["mode"] = "key_or_end"
                self.mode = "object_key_or_end"
                return
            self.stack.pop()
            self._value_finished()

    def _is_value_complete(self):
        return _is_complete_number(self.number_text)


class ToolCallCFG:
    def __init__(self, tokenizer, tools_text):
        self.tokenizer = tokenizer
        self.tools_text = tools_text
        self.tools = _normalize_tool_schema_spec(tools_text)
        self.tool_names = [tool["name"] for tool in self.tools]
        self.param_names = {
            tool["name"]: [name for name, _ in tool["parameters"]]
            for tool in self.tools
        }
        self.param_types = {
            tool["name"]: {name: _schema_type_name(param_type) for name, param_type in tool["parameters"]}
            for tool in self.tools
        }
        self.ids = tokenizer.json_token_ids
        self.quote_id = self.ids[JSON_QUOTE]
        self.tool_name_trie = _build_trie(
            (i, tokenizer.encode_json_string_content(name))
            for i, name in enumerate(self.tool_names)
        )
        self.state = "start_array"
        self.current_tool = None
        self.current_tool_index = None
        self.current_param_start = 0
        self.current_param_name = None
        self.current_param_type = "any"
        self._name_trie_node = None
        self._selected_name_index = None
        self.value_parser = None
        self.invalid = False

    def _remaining_param_trie(self):
        if self.current_tool is None:
            return {}
        params = self.param_names.get(self.current_tool, [])
        sequences = []
        for i in range(self.current_param_start, len(params)):
            sequences.append((i, self.tokenizer.encode_json_string_content(params[i])))
        return _build_trie(sequences)

    def _set_name_state(self, trie, next_state):
        self.state = next_state
        self._name_trie_node = trie
        self._selected_name_index = None

    def allowed_next_ids(self, training=False):
        if self.invalid:
            return None
        if self.value_parser is not None:
            return self.value_parser.allowed_next_ids(training=training)

        if self.state == "done":
            return np.array([EOS_ID], dtype=np.int32)
        if self.state == "start_array":
            return np.array([self.ids[JSON_LBRACK]], dtype=np.int32)
        if self.state == "after_array_start":
            return np.array([self.ids[JSON_RBRACK], self.ids[JSON_LBRACE]], dtype=np.int32)
        if self.state == "after_call":
            return np.array([self.ids[JSON_COMMA], self.ids[JSON_RBRACK]], dtype=np.int32)
        if self.state == "expect_key_name":
            return np.array([self.ids[JSON_KEY_NAME]], dtype=np.int32)
        if self.state == "expect_name_colon":
            return np.array([self.ids[JSON_COLON]], dtype=np.int32)
        if self.state == "expect_tool_name_quote":
            return np.array([self.quote_id], dtype=np.int32)
        if self.state == "tool_name_content":
            allowed = [token_id for token_id in self._name_trie_node.keys() if token_id != "_end"]
            if "_end" in self._name_trie_node:
                allowed.append(self.quote_id)
            return np.array(sorted(set(allowed)), dtype=np.int32)
        if self.state == "expect_tool_name_comma":
            return np.array([self.ids[JSON_COMMA]], dtype=np.int32)
        if self.state == "expect_key_arguments":
            return np.array([self.ids[JSON_KEY_ARGUMENTS]], dtype=np.int32)
        if self.state == "expect_arguments_colon":
            return np.array([self.ids[JSON_COLON]], dtype=np.int32)
        if self.state == "expect_arguments_object":
            return np.array([self.ids[JSON_LBRACE]], dtype=np.int32)
        if self.state == "arg_name_or_end":
            params = self.param_names.get(self.current_tool, [])
            if self.current_param_start >= len(params):
                return np.array([self.ids[JSON_RBRACE]], dtype=np.int32)
            return np.array([self.ids[JSON_RBRACE], self.quote_id], dtype=np.int32)
        if self.state == "arg_name_content":
            allowed = [token_id for token_id in self._name_trie_node.keys() if token_id != "_end"]
            if "_end" in self._name_trie_node:
                allowed.append(self.quote_id)
            return np.array(sorted(set(allowed)), dtype=np.int32)
        if self.state == "expect_arg_colon":
            return np.array([self.ids[JSON_COLON]], dtype=np.int32)
        if self.state == "after_arg_value":
            has_more = self.current_param_start < len(self.param_names.get(self.current_tool, []))
            if has_more:
                return np.array([self.ids[JSON_COMMA], self.ids[JSON_RBRACE]], dtype=np.int32)
            return np.array([self.ids[JSON_RBRACE]], dtype=np.int32)
        if self.state == "expect_call_end":
            return np.array([self.ids[JSON_RBRACE]], dtype=np.int32)
        return None

    def step(self, token_id):
        token_id = int(token_id)
        if self.invalid:
            return
        if self.value_parser is not None:
            self.value_parser.step(token_id)
            if self.value_parser.complete:
                self.value_parser = None
                self.state = "after_arg_value"
            return

        if self.state == "start_array":
            self.state = "after_array_start"
            return
        if self.state == "after_array_start":
            if token_id == self.ids[JSON_RBRACK]:
                self.state = "done"
            else:
                self.state = "expect_key_name"
            return
        if self.state == "after_call":
            if token_id == self.ids[JSON_COMMA]:
                self.state = "expect_key_name"
            else:
                self.state = "done"
            return
        if self.state == "expect_key_name":
            self.state = "expect_name_colon"
            return
        if self.state == "expect_name_colon":
            self.state = "expect_tool_name_quote"
            return
        if self.state == "expect_tool_name_quote":
            self._set_name_state(self.tool_name_trie, "tool_name_content")
            return
        if self.state == "tool_name_content":
            if token_id == self.quote_id:
                if "_end" not in self._name_trie_node:
                    self.invalid = True
                    return
                self.current_tool_index = self._name_trie_node["_end"]
                self.current_tool = self.tool_names[self.current_tool_index]
                self.current_param_start = 0
                self.state = "expect_tool_name_comma"
            else:
                if token_id not in self._name_trie_node:
                    self.invalid = True
                    return
                self._name_trie_node = self._name_trie_node[token_id]
            return
        if self.state == "expect_tool_name_comma":
            self.state = "expect_key_arguments"
            return
        if self.state == "expect_key_arguments":
            self.state = "expect_arguments_colon"
            return
        if self.state == "expect_arguments_colon":
            self.state = "expect_arguments_object"
            return
        if self.state == "expect_arguments_object":
            self.state = "arg_name_or_end"
            return
        if self.state == "arg_name_or_end":
            if token_id == self.ids[JSON_RBRACE]:
                self.state = "expect_call_end"
            else:
                self._set_name_state(self._remaining_param_trie(), "arg_name_content")
            return
        if self.state == "arg_name_content":
            if token_id == self.quote_id:
                if "_end" not in self._name_trie_node:
                    self.invalid = True
                    return
                param_idx = self._name_trie_node["_end"]
                self.current_param_name = self.param_names[self.current_tool][param_idx]
                self.current_param_type = self.param_types[self.current_tool].get(self.current_param_name, "any")
                self.current_param_start = param_idx + 1
                self.state = "expect_arg_colon"
            else:
                if token_id not in self._name_trie_node:
                    self.invalid = True
                    return
                self._name_trie_node = self._name_trie_node[token_id]
            return
        if self.state == "expect_arg_colon":
            self.value_parser = JsonValueParser(self.tokenizer, type_hint=self.current_param_type)
            return
        if self.state == "after_arg_value":
            if token_id == self.ids[JSON_COMMA]:
                self.state = "arg_name_or_end"
            else:
                self.state = "expect_call_end"
            return
        if self.state == "expect_call_end":
            self.state = "after_call"


def _extract_tools_tokens(tgt_in, loss_mask):
    supervised = np.flatnonzero(loss_mask > 0)
    if len(supervised) == 0:
        return []
    tool_stop = int(supervised[0]) + 1
    return [int(token_id) for token_id in tgt_in[2:tool_stop] if int(token_id) != 0]


def build_cfg_training_constraints(tokenizer, tgt_in_batch, tgt_out_batch, loss_mask_batch):
    batch_size, seq_len = tgt_out_batch.shape
    batch_allowed = [[None] * seq_len for _ in range(batch_size)]
    max_allowed = 0

    for i in range(batch_size):
        tools_tokens = _extract_tools_tokens(tgt_in_batch[i], loss_mask_batch[i])
        tools_text = tokenizer.decode_structured(tools_tokens)
        cfg = ToolCallCFG(tokenizer, tools_text)
        active_positions = np.flatnonzero(loss_mask_batch[i] > 0)
        for pos in active_positions:
            if cfg.invalid:
                break
            allowed = cfg.allowed_next_ids(training=True)
            if allowed is not None and len(allowed) > 0:
                allowed = np.asarray(allowed, dtype=np.int32)
                batch_allowed[i][int(pos)] = allowed
                max_allowed = max(max_allowed, len(allowed))
            cfg.step(int(tgt_out_batch[i, pos]))

    if max_allowed == 0:
        return (
            np.full((batch_size, seq_len, 1), -1, dtype=np.int32),
            np.zeros((batch_size, seq_len), dtype=np.int32),
        )

    allowed_ids = np.full((batch_size, seq_len, max_allowed), -1, dtype=np.int32)
    allowed_counts = np.zeros((batch_size, seq_len), dtype=np.int32)
    for i in range(batch_size):
        for pos, allowed in enumerate(batch_allowed[i]):
            if allowed is None:
                continue
            count = len(allowed)
            allowed_ids[i, pos, :count] = allowed
            allowed_counts[i, pos] = count
    return allowed_ids, allowed_counts
