"""The tokenizer module must stay importable without the ``train`` extra.

``sentencepiece`` is declared under the ``train`` extra, but the marker and id
constants in :mod:`needle.model.tokenizer` are pulled in by inference, export,
rendering and dataset-synthesis paths. Importing the module must therefore not
require the native library; only constructing a tokenizer may.
"""

import builtins
import importlib
import sys

import pytest


@pytest.fixture
def without_sentencepiece(monkeypatch):
    """Make ``import sentencepiece`` fail even when it is installed."""
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "sentencepiece" or name.startswith("sentencepiece."):
            raise ImportError("No module named 'sentencepiece'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.delitem(sys.modules, "sentencepiece", raising=False)
    monkeypatch.delitem(sys.modules, "needle.model.tokenizer", raising=False)
    yield
    sys.modules.pop("needle.model.tokenizer", None)


def test_module_imports_without_sentencepiece(without_sentencepiece):
    tokenizer = importlib.import_module("needle.model.tokenizer")

    assert tokenizer.spm is None
    # The constants the rest of the package imports are still usable.
    assert tokenizer.PAD_ID == 0
    assert tokenizer.IM_START == "<|im_start|>"
    assert tokenizer.TOOL_CALL_START in tokenizer.CHAT_MARKERS


def test_loading_a_tokenizer_reports_the_missing_extra(without_sentencepiece):
    tokenizer = importlib.import_module("needle.model.tokenizer")

    with pytest.raises(RuntimeError, match=r"cactus-needle\[train\]"):
        tokenizer.SANTokenizer("nonexistent.model")
