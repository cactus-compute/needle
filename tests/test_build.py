import os
import subprocess
import sys
import types

import pytest

from conftest import requires_engine

pytestmark = pytest.mark.slow


def _build_args(checkpoint, out, bits="4", lora=None):
    return types.SimpleNamespace(checkpoint=checkpoint, lora=lora, out=out,
                                 upload=False, bits=bits)


def test_build_exports_loadable_cact(tiny_checkpoint, tmp_path):
    from needle.model.finetune import build_main
    from needle.model.export import read_export

    out = str(tmp_path / "tiny.cact")
    build_main(_build_args(tiny_checkpoint, out, bits="4"))

    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
    header, tensors = read_export(out)
    assert header["num_tensors"] > 0
    assert len(tensors) == header["num_tensors"]
    assert any(isinstance(t, (bytes, bytearray)) for t in tensors)


def test_build_at_two_bits(tiny_checkpoint, tmp_path):
    from needle.model.finetune import build_main
    from needle.model.export import read_export

    out = str(tmp_path / "tiny_w2.cact")
    build_main(_build_args(tiny_checkpoint, out, bits="2"))
    header, _ = read_export(out)
    assert header["num_tensors"] > 0


def test_export_round_trips_a_projection(tiny_checkpoint, tmp_path):
    import pickle
    import numpy as np
    from needle.model.export import write_export, read_export
    from needle.model.architecture import TransformerConfig, effective_kv_window
    from needle.model.tokenizer import get_tokenizer

    with open(tiny_checkpoint, "rb") as handle:
        ckpt = pickle.load(handle)
    params, config = ckpt["params"], TransformerConfig(**ckpt["config"])

    out = str(tmp_path / "rt.cact")
    write_export(params, config, out, bits=4,
                 tokenizer=get_tokenizer(config.vocab_size),
                 kv_window=effective_kv_window(config))
    header, tensors = read_export(out)

    original = np.asarray(params["stack"]["layers"]["block"]["self_attn"]["q_proj"]["kernel"][0]).T
    dequant = tensors[2]
    assert dequant.shape == original.shape
    assert np.corrcoef(dequant.ravel(), original.ravel())[0, 1] > 0.9


@requires_engine
@pytest.mark.parametrize("bits", ["2", "4"])
def test_native_engine_loads_and_runs_built_cact(engine_checkpoint, tmp_path, bits):
    from needle.model.finetune import build_main

    out = str(tmp_path / f"engine_w{bits}.cact")
    build_main(_build_args(engine_checkpoint, out, bits=bits))

    code = (
        "import needle\n"
        "@needle.tool\n"
        "def dummy_tool(query: str):\n"
        "    return 'ok'\n"
        f"agent = needle.Needle(weights={out!r}, tools=[dummy_tool])\n"
        "response = agent.complete('test query')\n"
        "assert isinstance(response, dict) and 'type' in response\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"Native engine subprocess failed (exit code {proc.returncode}):\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"


@requires_engine
def test_malformed_cact_raises_clean_exception(tmp_path):
    bad_path = str(tmp_path / "bad.cact")
    with open(bad_path, "wb") as f:
        f.write(b"NOT_A_VALID_CACT_FILE_HEADER_BYTES_1234567890")

    code = (
        "import needle\n"
        "@needle.tool\n"
        "def dummy_tool(query: str):\n"
        "    return 'ok'\n"
        "try:\n"
        f"    needle.Needle(weights={bad_path!r}, tools=[dummy_tool])\n"
        "    exit(1)\n"
        "except RuntimeError as e:\n"
        "    if 'failed to load weights' in str(e):\n"
        "        exit(0)\n"
        "    exit(2)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"Malformed load subprocess failed (exit code {proc.returncode}):\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
