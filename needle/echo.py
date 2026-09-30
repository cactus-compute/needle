from __future__ import annotations

import array
import ctypes
import json
import os
import sys
import wave

ECHO_REPO = "Cactus-Compute/echo"
ECHO_WEIGHTS = "echo.cact"
SAMPLE_RATE = 16000
LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")

_handle = None
_loaded = None


def _weights_path():
    override = os.environ.get("NEEDLE_ECHO_WEIGHTS")
    if override:
        return override
    cache = os.path.join(os.path.expanduser("~"), ".cache", "cactus-needle", "echo")
    local = os.path.join(cache, ECHO_WEIGHTS)
    if os.path.exists(local):
        return local
    import shutil
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=ECHO_REPO, filename=ECHO_WEIGHTS, repo_type="model")
    os.makedirs(cache, exist_ok=True)
    shutil.copyfile(cached, local)
    return local


def _lib():
    global _handle
    if _handle is None:
        from . import _load_cdll

        lib = _load_cdll(3)
        if not hasattr(lib, "echo_transcribe"):
            raise RuntimeError("this Needle engine was built without Echo; point NEEDLE3_LIB_PATH at an engine that has it")
        samples = ctypes.POINTER(ctypes.c_float)
        lib.echo_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
        lib.echo_load.restype = ctypes.c_int
        lib.echo_transcribe.argtypes = [samples, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
                                        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
        lib.echo_transcribe.restype = ctypes.c_int
        lib.echo_embed.argtypes = [samples, ctypes.c_int, samples, ctypes.c_int]
        lib.echo_embed.restype = ctypes.c_int
        lib.echo_last_error.argtypes = []
        lib.echo_last_error.restype = ctypes.c_char_p
        _handle = lib
    return _handle


def _read_wav(path):
    with wave.open(path, "rb") as source:
        channels, width, rate = source.getnchannels(), source.getsampwidth(), source.getframerate()
        raw = source.readframes(source.getnframes())
    if width == 3:
        values = [int.from_bytes(raw[i:i + 3], "little", signed=True) for i in range(0, len(raw), 3)]
    else:
        values = array.array({1: "B", 2: "h", 4: "i"}[width])
        values.frombytes(raw)
        if sys.byteorder == "big":
            values.byteswap()
    offset, scale = (128, 128.0) if width == 1 else (0, float(1 << (8 * width - 1)))
    scale *= channels
    mono = [(sum(frame) - offset * channels) / scale for frame in zip(*(values[c::channels] for c in range(channels)))]
    if rate == SAMPLE_RATE or not mono:
        return mono
    step, last = rate / SAMPLE_RATE, len(mono) - 1
    out = []
    for i in range(int(len(mono) / step)):
        at = int(i * step)
        out.append(mono[at] + (mono[min(at + 1, last)] - mono[at]) * (i * step - at))
    return out


def _samples(audio):
    if isinstance(audio, (str, os.PathLike)):
        audio = _read_wav(os.fspath(audio))
    data = array.array("f")
    if isinstance(audio, (bytes, bytearray)):
        data.frombytes(audio)
    elif isinstance(audio, array.array) and audio.typecode == "f":
        data = audio
    else:
        data.extend(audio)
    return (ctypes.c_float * len(data)).from_buffer(data) if len(data) else (ctypes.c_float * 1)(), len(data)


class Echo:
    """Speech-to-text in English, German, French, Spanish, Italian, Dutch and Polish.

    Audio is a WAV file path or 16 kHz mono float samples in [-1, 1], at most 30 s.
    One model is loaded per process; it is not thread-safe.
    """

    def __init__(self, weights=None, buffer_size=1 << 18):
        self._weights = os.fspath(weights) if weights is not None else _weights_path()
        self._buffer = ctypes.create_string_buffer(buffer_size)
        self._bind()

    def _bind(self):
        global _loaded
        lib = _lib()
        if _loaded == self._weights:
            return lib
        with open(self._weights, "rb") as handle:
            data = handle.read()
        if lib.echo_load(data, len(data)) < 0:
            _loaded = None
            raise RuntimeError(lib.echo_last_error().decode("utf-8", "replace"))
        _loaded = self._weights
        return lib

    def transcribe(self, audio, language=None, phrases=None, word_timestamps=False) -> dict:
        """Returns {"text", "language"}, plus "words" with start, end and probability when word_timestamps is set.

        language is one of LANGUAGES, or None to detect it. phrases are words or phrases to favour.
        """
        lib = self._bind()
        samples, count = _samples(audio)
        if phrases is not None and not isinstance(phrases, str):
            phrases = "\n".join(phrases)
        detected = ctypes.create_string_buffer(4)
        code = lib.echo_transcribe(samples, count, language.encode("utf-8") if language else None,
                                   phrases.encode("utf-8") if phrases else None, int(bool(word_timestamps)),
                                   self._buffer, len(self._buffer), detected)
        if code < 0:
            raise RuntimeError(lib.echo_last_error().decode("utf-8", "replace"))
        text = self._buffer.value.decode("utf-8", "replace")
        result = json.loads(text) if word_timestamps else {"text": text}
        result["language"] = detected.value.decode("utf-8")
        return result

    def embed(self, audio) -> list[float]:
        """The encoder output, one row of floats per 80 ms frame, flattened."""
        lib = self._bind()
        samples, count = _samples(audio)
        size = lib.echo_embed(samples, count, None, 0)
        if size < 0:
            raise RuntimeError(lib.echo_last_error().decode("utf-8", "replace"))
        output = (ctypes.c_float * size)()
        if lib.echo_embed(samples, count, output, size) != size:
            raise RuntimeError(lib.echo_last_error().decode("utf-8", "replace"))
        return list(output)
