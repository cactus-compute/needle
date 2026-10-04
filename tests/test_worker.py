import io
import json
import shutil
import subprocess
import sys
import warnings

import pytest

from needle._worker import FineTuneWorker, _read_message, _write_message


def test_worker_protocol_round_trip():
    stream = io.BytesIO()
    message = {"operation": "complete", "text": "héllo\nworld",
               "max_new_tokens": 17}
    _write_message(stream, message)
    stream.seek(0)
    assert _read_message(stream) == message


@pytest.fixture
def stub_engine(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None or sys.platform == "win32":
        pytest.skip("a C compiler is required for the worker integration test")
    source = tmp_path / "stub.c"
    source.write_text(r'''
#include <stdint.h>
#include <stdio.h>

static int model_id;
static int resets;

int needle_load(const unsigned char* data, unsigned long long size) {
    if (!data || size < 6 || data[4] == 0) return -1;
    model_id = data[4];
    return 0;
}

int needle_init(const char* system, const char* tools, const char* index) {
    (void)system;
    (void)tools;
    (void)index;
    return 7;
}

static char language[8];

int needle_embed(const char* text, const float* pcm, int samples, float* output, int capacity) {
    (void)text;
    (void)pcm;
    (void)samples;
    (void)output;
    (void)capacity;
    return 0;
}

void needle_set_audio(const char* lang, const char* keywords, int word_timestamps, int tool_schema_keywords) {
    (void)keywords;
    (void)word_timestamps;
    (void)tool_schema_keywords;
    snprintf(language, sizeof(language), "%s", lang ? lang : "");
}

int needle_complete(const char* input, const float* pcm, int samples,
                    int max_new_tokens, char* output, int capacity) {
    (void)max_new_tokens;
    if (!input)
        snprintf(output, (size_t)capacity,
                 "{\"type\":\"text\",\"model\":%d,\"resets\":%d,\"samples\":%d,"
                 "\"last\":%.2f,\"language\":\"%s\"}", model_id, resets, samples, samples ? pcm[samples - 1] : 0.0f, language);
    else
        snprintf(output, (size_t)capacity,
                 "{\"type\":\"text\",\"model\":%d,\"resets\":%d,"
                 "\"input\":\"%s\"}", model_id, resets, input);
    return 1;
}

void needle_reset(void) {
    resets++;
}
''')
    if sys.platform == "darwin":
        library = tmp_path / "libstub.dylib"
        command = [compiler, "-dynamiclib", str(source), "-o", str(library)]
    else:
        library = tmp_path / "libstub.so"
        command = [compiler, "-shared", "-fPIC", str(source), "-o", str(library)]
    subprocess.run(command, check=True)
    return library


def _weights(path, model_id):
    path.write_bytes((0x05E12A83).to_bytes(4, "little") +
                     bytes([model_id, 1]))
    return path


def test_finetunes_run_in_independent_processes(stub_engine, tmp_path):
    first_path = _weights(tmp_path / "first.cact", 11)
    second_path = _weights(tmp_path / "second.cact", 29)
    first = FineTuneWorker(stub_engine, first_path, "", "[]", None, 4096)
    second = FineTuneWorker(stub_engine, second_path, "", "[]", None, 4096)
    try:
        assert first.pid != second.pid
        assert json.loads(first.complete("first", 8))["model"] == 11
        assert json.loads(second.complete("second", 8))["model"] == 29
        first.reset()
        assert json.loads(first.complete("again", 8))["resets"] == 1
        assert json.loads(second.complete("again", 8))["resets"] == 0
    finally:
        first.close()
        second.close()
    assert first._process.poll() is not None
    assert second._process.poll() is not None


def test_tuned_agents_use_independent_workers(stub_engine, tmp_path,
                                               monkeypatch):
    import needle

    monkeypatch.setattr(needle, "_library_path",
                        lambda generation=2: str(stub_engine))
    first_path = _weights(tmp_path / "agent-first.cact", 7)
    second_path = _weights(tmp_path / "agent-second.cact", 13)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = needle.Needle(tools="[]", weights=first_path)
        second = needle.Needle(tools="[]", weights=second_path)
    try:
        assert first._worker.pid != second._worker.pid
        assert first.complete("first")["model"] == 7
        assert second.complete("second")["model"] == 13
        first.reset()
        assert first.complete("again")["resets"] == 1
        assert second.complete("again")["resets"] == 0
    finally:
        first.close()
        second.close()


def test_worker_carries_audio_turns(stub_engine, tmp_path):
    import struct

    weights = _weights(tmp_path / "tuned.cact", 5)
    speech = _weights(tmp_path / "whistle.cact", 9)
    worker = FineTuneWorker(stub_engine, weights, "", "[]", None, 4096, generation=3)
    try:
        heard = json.loads(worker.complete(None, 8, audio=struct.pack("<3f", 0.1, 0.2, 0.75), speech=str(speech), options=("de", None, 0, 1)))
        assert heard["samples"] == 3 and heard["last"] == 0.75 and heard["language"] == "de" and heard["model"] == 9
        assert json.loads(worker.complete("text", 8))["input"] == "text"
    finally:
        worker.close()


def test_worker_propagates_native_load_failure(stub_engine, tmp_path):
    bad = _weights(tmp_path / "bad.cact", 0)
    with pytest.raises(RuntimeError, match="needle_load failed"):
        FineTuneWorker(stub_engine, bad, "", "[]", None, 4096)
