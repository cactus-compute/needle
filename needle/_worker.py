from __future__ import annotations

import ctypes
import json
import os
import queue
import struct
import subprocess
import sys
import threading


_HEADER = struct.Struct("!Q")
_EOF = object()


def _read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == size:
                return None
            raise EOFError("truncated worker message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_message(stream):
    header = _read_exact(stream, _HEADER.size)
    if header is None:
        return None
    size = _HEADER.unpack(header)[0]
    payload = _read_exact(stream, size)
    if payload is None:
        raise EOFError("truncated worker message")
    return json.loads(payload.decode("utf-8"))


def _write_message(stream, message):
    payload = json.dumps(message, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _load_library(path, generation=2):
    lib = ctypes.CDLL(path)
    lib.needle_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                ctypes.c_char_p]
    lib.needle_init.restype = ctypes.c_int
    lib.needle_complete.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                    ctypes.c_char_p, ctypes.c_int]
    lib.needle_complete.restype = ctypes.c_int
    if int(generation) >= 3:
        lib.needle_embed.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.needle_embed.restype = ctypes.c_int
    lib.needle_reset.argtypes = []
    lib.needle_reset.restype = None
    lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
    lib.needle_load.restype = ctypes.c_int
    return lib


def _child():
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    source = sys.stdin.buffer
    try:
        config = _read_message(source)
        if config is None:
            return 1
        generation = int(config.get("generation", 2))
        lib = _load_library(config["library"], generation)
        with open(config["weights"], "rb") as handle:
            weights = handle.read()
        if lib.needle_load(weights, len(weights)) < 0:
            raise RuntimeError(f"needle_load failed for {config['weights']}")
        system = config["system"].encode("utf-8")
        tools = config["tools"].encode("utf-8")
        index = config.get("tool_index")
        index = index.encode("utf-8") if index else None
        prefix = lib.needle_init(system, tools, index)
        if prefix < 0:
            raise RuntimeError(f"needle_init failed (code {prefix})")
        output = ctypes.create_string_buffer(int(config["buffer_size"]))
        _write_message(protocol, {"status": "ready", "prefix_tokens": prefix})
        while True:
            request = _read_message(source)
            if request is None:
                break
            operation = request.get("operation")
            if operation == "complete":
                code = lib.needle_complete(
                    request["text"].encode("utf-8"),
                    int(request["max_new_tokens"]), output, len(output))
                if code < 0:
                    _write_message(protocol, {
                        "status": "error",
                        "message": f"needle_complete failed (code {code})",
                    })
                else:
                    _write_message(protocol, {
                        "status": "ok",
                        "response": output.value.decode("utf-8"),
                    })
            elif operation == "embed":
                if generation < 3:
                    _write_message(protocol, {
                        "status": "error",
                        "message": "embeddings require a Needle 3 model",
                    })
                    continue
                text = request["text"].encode("utf-8")
                dim = lib.needle_embed(text, None, 0)
                if dim <= 0:
                    _write_message(protocol, {
                        "status": "error",
                        "message": f"needle_embed failed (code {dim})",
                    })
                    continue
                embedding = (ctypes.c_float * dim)()
                code = lib.needle_embed(text, embedding, dim)
                if code != dim:
                    _write_message(protocol, {
                        "status": "error",
                        "message": f"needle_embed failed (code {code})",
                    })
                else:
                    _write_message(protocol, {
                        "status": "ok",
                        "embedding": list(embedding),
                    })
            elif operation == "reset":
                lib.needle_reset()
                _write_message(protocol, {"status": "ok"})
            elif operation == "close":
                _write_message(protocol, {"status": "ok"})
                break
            else:
                _write_message(protocol, {
                    "status": "error",
                    "message": f"unknown worker operation: {operation}",
                })
        return 0
    except BaseException as exc:
        try:
            _write_message(protocol, {
                "status": "fatal",
                "message": f"{type(exc).__name__}: {exc}",
            })
        except BaseException:
            pass
        return 1
    finally:
        protocol.close()


class FineTuneWorker:
    def __init__(self, library, weights, system, tools, tool_index,
                 buffer_size, startup_timeout=300, generation=2):
        self._lock = threading.Lock()
        self._messages = queue.Queue()
        self._closed = False
        self._process = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--child"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._reader = threading.Thread(target=self._read_messages,
                                        daemon=True)
        self._reader.start()
        try:
            self._send({
                "library": os.fspath(library),
                "weights": os.path.abspath(os.fspath(weights)),
                "system": system,
                "tools": tools,
                "tool_index": tool_index,
                "buffer_size": int(buffer_size),
                "generation": int(generation),
            })
            ready = self._receive(startup_timeout)
            if ready.get("status") != "ready":
                raise RuntimeError(ready.get("message", "worker failed to start"))
        except BaseException:
            self.close()
            raise

    @property
    def pid(self):
        return self._process.pid

    def _read_messages(self):
        try:
            while True:
                message = _read_message(self._process.stdout)
                if message is None:
                    break
                self._messages.put(message)
        except BaseException as exc:
            self._messages.put(exc)
        finally:
            self._messages.put(_EOF)

    def _send(self, message):
        if self._process.stdin is None:
            raise RuntimeError("Needle worker input is closed")
        try:
            _write_message(self._process.stdin, message)
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("Needle worker exited unexpectedly") from exc

    def _receive(self, timeout=None):
        try:
            message = self._messages.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("Needle worker did not initialize in time") from exc
        if message is _EOF:
            code = self._process.poll()
            raise RuntimeError(f"Needle worker exited unexpectedly (code {code})")
        if isinstance(message, BaseException):
            raise RuntimeError("Needle worker protocol failed") from message
        return message

    def _request(self, message):
        with self._lock:
            if self._closed:
                raise RuntimeError("Needle worker is closed")
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"Needle worker exited unexpectedly (code {self._process.returncode})")
            self._send(message)
            response = self._receive()
            if response.get("status") != "ok":
                raise RuntimeError(response.get("message", "Needle worker failed"))
            return response

    def complete(self, text, max_new_tokens):
        return self._request({
            "operation": "complete",
            "text": text,
            "max_new_tokens": int(max_new_tokens),
        })["response"]

    def embed(self, text):
        return self._request({
            "operation": "embed",
            "text": text,
        })["embedding"]

    def reset(self):
        self._request({"operation": "reset"})

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._process.poll() is None:
                try:
                    self._send({"operation": "close"})
                    self._receive(2)
                except BaseException:
                    pass
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except OSError:
                    pass
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()


if __name__ == "__main__":
    raise SystemExit(_child() if sys.argv[1:] == ["--child"] else 2)
