"""Grammar-constrained decoding for tool-call generation.

Constrains the decoder to only produce valid tool names and argument keys
by tracking position in the output JSON and masking invalid tokens via a
character-level trie built from the tool definitions.
"""

import json
import logging
import warnings
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)

class TrieNode:
    __slots__ = ("children", "is_terminal")

    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.is_terminal: bool = False


class Trie:
    """Character-level prefix tree for matching valid names/keys."""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_terminal = True

    def get_node(self, prefix: str) -> TrieNode | None:
        """Walk the trie for *prefix* and return the node, or None."""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    @property
    def words(self) -> list[str]:
        """Return all words stored in the trie."""
        result = []
        def _dfs(node, path):
            if node.is_terminal:
                result.append("".join(path))
            for ch, child in sorted(node.children.items()):
                _dfs(child, path + [ch])
        _dfs(self.root, [])
        return result

class ToolConstraints:
    """Holds one name trie and one param trie per function, built from tool JSON."""

    def __init__(self, tools_json: str):
        self.name_trie = Trie()
        self.param_tries: dict[str, Trie] = {} 

        try:
            tools = json.loads(tools_json)
        except (json.JSONDecodeError, TypeError):
            tools = []

        if not isinstance(tools, list):
            tools = []

        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name", "")
            if not name:
                continue
            self.name_trie.insert(name)

            params = tool.get("parameters", {})
            if isinstance(params, dict):
                param_trie = Trie()
                for key in params:
                    param_trie.insert(key)
                self.param_tries[name] = param_trie

    def get_param_trie(self, function_name: str) -> Trie | None:
        return self.param_tries.get(function_name)


class JsonState(Enum):
    FREE = auto()
    IN_NAME = auto()
    IN_ARG_KEY = auto()


class JsonStateMachine:
    """Tracks position in output JSON to know when to constrain decoding.

    Transitions:
      - ``"name":"`` suffix → IN_NAME
      - ``"arguments":{`` suffix → sets in_arguments; then ``{"`` or ``,"``
        at nesting depth 0 → IN_ARG_KEY
      - Closing ``"`` while in constrained state → FREE
    """

    def __init__(self):
        self.state = JsonState.FREE
        self.buffer = ""          
        self.constrained_buf = "" 
        self.current_function = "" 
        self.in_arguments = False
        self.arguments_depth = 0  # nesting depth of the arguments {
        self.nesting_depth = 0
        self.in_string_value = False
        self.prev_char_escape = False

    def feed(self, text: str):
        """Feed generated text character-by-character to drive transitions."""
        for ch in text:
            self._feed_char(ch)

    def _feed_char(self, ch: str):
        if self.state in (JsonState.IN_NAME, JsonState.IN_ARG_KEY):
            if ch == '"':
                if self.state == JsonState.IN_NAME:
                    self.current_function = self.constrained_buf
                self.constrained_buf = ""
                self.state = JsonState.FREE
            else:
                self.constrained_buf += ch
            self.buffer += ch
            return

        self.buffer += ch

        if self.in_string_value:
            if self.prev_char_escape:
                self.prev_char_escape = False
                return
            if ch == '\\':
                self.prev_char_escape = True
                return
            if ch == '"':
                self.in_string_value = False
            return

        if ch in '{[':
            self.nesting_depth += 1
        elif ch in '}]':
            self.nesting_depth = max(0, self.nesting_depth - 1)
            if ch == '}' and self.in_arguments and self.nesting_depth < self.arguments_depth:
                self.in_arguments = False

        if self.buffer.endswith('"name":"'):
            self.state = JsonState.IN_NAME
            self.constrained_buf = ""
            return

        if self.buffer.endswith('"arguments":{'):
            self.in_arguments = True
            self.arguments_depth = self.nesting_depth  # depth of the arguments { itself
            return

        if self.in_arguments and self._at_arg_key_start():
            self.state = JsonState.IN_ARG_KEY
            self.constrained_buf = ""
            return

        if ch == '"' and self._is_value_string_start():
            self.in_string_value = True

    def _at_arg_key_start(self) -> bool:
        """Check if buffer ends with ``{"`` or ``,"`` while in arguments."""
        if len(self.buffer) < 2:
            return False
        return self.buffer[-2:] in ('{"', ',"')

    def _is_value_string_start(self) -> bool:
        """Check if the current ``"`` opens a JSON string value (preceded by ``:``).

        Keys open after ``{`` or ``,``; values open after ``:``.
        """
        for j in range(len(self.buffer) - 2, -1, -1):
            c = self.buffer[j]
            if c in ' \t\n\r':
                continue
            return c == ':'
        return False


def build_token_strings(tokenizer) -> list[str]:
    """Map each vocab ID to its decoded text.

    Handles SentencePiece control tokens (IDs 0..N that decode to empty)
    and byte-fallback pieces.
    """
    vocab_size = tokenizer.vocab_size
    sp = tokenizer.sp
    strings = []
    for i in range(vocab_size):
        if sp.IsControl(i):
            strings.append("")
        elif sp.IsUnknown(i):
            strings.append("")
        elif sp.IsByte(i):
            # Byte-fallback piece like <0x41>
            piece = sp.IdToPiece(i)
            try:
                byte_val = int(piece[1:-1], 16) if piece.startswith("<0x") else ord(piece)
                strings.append(chr(byte_val))
            except (ValueError, IndexError):
                strings.append("")
        else:
            # Normal BPE piece — decode via SentencePiece
            text = sp.Decode([i])
            strings.append(text)
    return strings


class TokenIndex:
    """Maps first character → list of token IDs for fast candidate filtering."""

    def __init__(self, token_strings: list[str]):
        self._index: dict[str, list[int]] = {}
        for tid, s in enumerate(token_strings):
            if not s:
                continue
            first = s[0]
            if first not in self._index:
                self._index[first] = []
            self._index[first].append(tid)

    def candidates_for(self, first_char: str) -> list[int]:
        return self._index.get(first_char, [])

    @property
    def all_nonempty(self) -> list[int]:
        """All token IDs with non-empty strings."""
        result = []
        for ids in self._index.values():
            result.extend(ids)
        return result


def _check_token_valid(token_text: str, trie_node: TrieNode) -> bool:
    """Check if *token_text* is a valid continuation from *trie_node*.

    The token text may contain a closing ``"`` which signals end of the
    constrained span — at that point the trie node must be terminal.
    """
    node = trie_node
    for i, ch in enumerate(token_text):
        if ch == '"':
            return node.is_terminal
        if ch not in node.children:
            return False
        node = node.children[ch]
    return True  


def apply_constraints(
    logits: np.ndarray,
    state: JsonState,
    trie_node: TrieNode,
    token_strings: list[str],
    token_index: TokenIndex,
) -> np.ndarray:
    """Mask logits so only valid tokens survive.

    Args:
        logits: shape (vocab_size,) float array
        state: current constrained state (IN_NAME or IN_ARG_KEY)
        trie_node: current position in the relevant trie
        token_strings: mapping from token ID to text
        token_index: first-char index for fast filtering

    Returns:
        Modified logits with invalid tokens set to -inf.
    """
    vocab_size = logits.shape[0]
    mask = np.full(vocab_size, False)

    valid_first_chars = set(trie_node.children.keys())
    if trie_node.is_terminal:
        valid_first_chars.add('"')

    for first_char in valid_first_chars:
        for tid in token_index.candidates_for(first_char):
            if not mask[tid]:
                text = token_strings[tid]
                if _check_token_valid(text, trie_node):
                    mask[tid] = True

    if not mask.any():
        logger.warning("Constrained decoding: no valid tokens found, falling back to unconstrained")
        return logits

    masked_logits = logits.copy()
    masked_logits[~mask] = -np.inf
    return masked_logits

class ConstrainedDecoder:
    """Batch-aware constrained decoder.

    Holds per-example ``JsonStateMachine`` instances and shared token
    metadata. Call ``constrain_logits`` between logit computation and
    argmax, then ``update`` after selecting the token.
    """

    def __init__(self, tool_constraints_list: list[ToolConstraints],
                 token_strings: list[str], token_index: TokenIndex):
        self.batch_size = len(tool_constraints_list)
        self.tool_constraints = tool_constraints_list
        self.machines = [JsonStateMachine() for _ in range(self.batch_size)]
        self.token_strings = token_strings
        self.token_index = token_index

    def is_active(self, batch_idx: int) -> bool:
        """Return True if this batch element is currently in a constrained state."""
        return self.machines[batch_idx].state != JsonState.FREE

    def constrain_logits(self, logits: np.ndarray, batch_idx: int) -> np.ndarray:
        """Apply grammar constraints to logits for a single batch element.

        Args:
            logits: shape (vocab_size,) numpy array
            batch_idx: index into the batch

        Returns:
            Possibly-masked logits (same shape).
        """
        machine = self.machines[batch_idx]
        tc = self.tool_constraints[batch_idx]

        if machine.state == JsonState.FREE:
            return logits

        if machine.state == JsonState.IN_NAME:
            trie = tc.name_trie
        elif machine.state == JsonState.IN_ARG_KEY:
            trie = tc.get_param_trie(machine.current_function)
            if trie is None:
                return logits 
        else:
            return logits

        node = trie.get_node(machine.constrained_buf)
        if node is None:
            logger.warning("Constrained decoding: off-trie at %r, falling back", machine.constrained_buf)
            return logits

        return apply_constraints(logits, machine.state, node, self.token_strings, self.token_index)

    def update(self, batch_idx: int, token_id: int):
        """Advance the state machine for *batch_idx* with the selected token."""
        text = self.token_strings[token_id]
        self.machines[batch_idx].feed(text)


_token_cache: dict[int, tuple[list[str], TokenIndex]] = {}


def _get_token_data(tokenizer) -> tuple[list[str], TokenIndex]:
    """Return (token_strings, token_index), cached per tokenizer instance."""
    key = id(tokenizer)
    if key not in _token_cache:
        ts = build_token_strings(tokenizer)
        _token_cache[key] = (ts, TokenIndex(ts))
    return _token_cache[key]


def build_constrained_decoder(
    tools_json_list: list[str],
    tokenizer,
) -> ConstrainedDecoder:
    """Convenience factory: build a ConstrainedDecoder for a batch of examples.

    Args:
        tools_json_list: list of tool JSON strings (one per batch element)
        tokenizer: NeedleTokenizer with .sp SentencePiece model
    """
    token_strings, token_index = _get_token_data(tokenizer)
    tc_list = [ToolConstraints(tj) for tj in tools_json_list]
    return ConstrainedDecoder(tc_list, token_strings, token_index)
