from __future__ import annotations

import array
import ctypes
import json
import os
import sys
import wave

SAMPLE_RATE = 16000
LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")

_handle = None
_loaded = None


def _weights_path():
    from .. import _base_weights_path
    from ..agent import fetch

    return os.environ.get("NEEDLE_WHISTLE_WEIGHTS") or _base_weights_path(fetch.WHISTLE)


def _lib():
    global _handle
    if _handle is None:
        from .. import _load_cdll
        from ..agent import fetch

        lib = _load_cdll(fetch.WHISTLE)
        samples = ctypes.POINTER(ctypes.c_float)
        lib.whistle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
        lib.whistle_load.restype = ctypes.c_int
        lib.whistle_transcribe.argtypes = [samples, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
                                        ctypes.c_char_p, ctypes.c_int]
        lib.whistle_transcribe.restype = ctypes.c_int
        lib.whistle_embed.argtypes = [samples, ctypes.c_int, samples, ctypes.c_int]
        lib.whistle_embed.restype = ctypes.c_int
        lib.whistle_last_error.argtypes = []
        lib.whistle_last_error.restype = ctypes.c_char_p
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
    try:
        import numpy
        import soxr
    except ImportError:
        raise RuntimeError(f"resampling {rate} Hz audio to {SAMPLE_RATE} Hz needs soxr: pip install cactus-needle[whistle]") from None
    return soxr.resample(numpy.asarray(mono, numpy.float32), rate, SAMPLE_RATE, quality="HQ")


def _samples(audio):
    if isinstance(audio, (str, os.PathLike)):
        audio = _read_wav(os.fspath(audio))
    data = array.array("f")
    if isinstance(audio, (bytes, bytearray)):
        data.frombytes(audio)
    elif isinstance(audio, array.array) and audio.typecode == "f":
        data = audio
    elif hasattr(audio, "tobytes") and getattr(audio, "dtype", None) is not None:
        data.frombytes(audio.astype("float32", copy=False).tobytes())
    else:
        data.extend(audio)
    return (ctypes.c_float * len(data)).from_buffer(data) if len(data) else (ctypes.c_float * 1)(), len(data)


class Whistle:
    """Speech-to-text in English, German, French, Spanish, Italian, Dutch and Polish.

    Audio is a WAV file path or 16 kHz mono float samples in [-1, 1], at most 30 s.
    One model is loaded per process; it is not thread-safe.
    """

    def __init__(self, weights=None, buffer_size=1 << 18):
        self.weights = os.fspath(weights) if weights is not None else _weights_path()
        self._buffer = ctypes.create_string_buffer(buffer_size)
        self._bind()

    def _bind(self):
        global _loaded
        lib = _lib()
        if _loaded == self.weights:
            return lib
        with open(self.weights, "rb") as handle:
            data = handle.read()
        if lib.whistle_load(data, len(data)) < 0:
            _loaded = None
            raise RuntimeError(lib.whistle_last_error().decode("utf-8", "replace"))
        _loaded = self.weights
        return lib

    def transcribe(self, audio, language=None, keywords=None, word_timestamps=False) -> dict:
        """Returns {"text", "language", "ttft_ms", "decode_tps"}, plus "words" with start, end and probability when word_timestamps is set.

        language is one of LANGUAGES, or None to detect it. keywords are words or phrases for keyword biasing.
        """
        lib = self._bind()
        samples, count = _samples(audio)
        if keywords is not None and not isinstance(keywords, str):
            keywords = "\n".join(keywords)
        code = lib.whistle_transcribe(samples, count, language.encode("utf-8") if language else None,
                                   keywords.encode("utf-8") if keywords else None, int(bool(word_timestamps)),
                                   self._buffer, len(self._buffer))
        if code < 0:
            raise RuntimeError(lib.whistle_last_error().decode("utf-8", "replace"))
        return json.loads(self._buffer.value.decode("utf-8", "replace"))

    def embed(self, audio) -> list[float]:
        """The encoder output, one row of floats per 80 ms frame, flattened."""
        lib = self._bind()
        samples, count = _samples(audio)
        size = lib.whistle_embed(samples, count, None, 0)
        if size < 0:
            raise RuntimeError(lib.whistle_last_error().decode("utf-8", "replace"))
        output = (ctypes.c_float * size)()
        if lib.whistle_embed(samples, count, output, size) != size:
            raise RuntimeError(lib.whistle_last_error().decode("utf-8", "replace"))
        return list(output)
