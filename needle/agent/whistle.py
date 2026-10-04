from __future__ import annotations

import array
import ctypes
import json
import os
import queue
import shutil
import sys
import threading
import time
import wave

SAMPLE_RATE = 16000
LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")

_handle = None
_loaded = None


def _weights_path():
    from .. import _base_weights_path
    from . import fetch

    return os.environ.get("NEEDLE_WHISTLE_WEIGHTS") or _base_weights_path(fetch.WHISTLE)


def _lib():
    global _handle
    if _handle is None:
        from .. import _load_cdll

        lib = _load_cdll(3)
        samples = ctypes.POINTER(ctypes.c_float)
        lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
        lib.needle_load.restype = ctypes.c_int
        lib.needle_transcribe.argtypes = [samples, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
                                          ctypes.c_char_p, ctypes.c_int]
        lib.needle_transcribe.restype = ctypes.c_int
        lib.needle_stream_transcribe_process.argtypes = [samples, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                                                         ctypes.c_int]
        lib.needle_stream_transcribe_process.restype = ctypes.c_int
        lib.needle_stream_transcribe_stop.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.needle_stream_transcribe_stop.restype = ctypes.c_int
        lib.needle_embed.argtypes = [ctypes.c_char_p, samples, ctypes.c_int, samples, ctypes.c_int]
        lib.needle_embed.restype = ctypes.c_int
        lib.needle_last_error.argtypes = []
        lib.needle_last_error.restype = ctypes.c_char_p
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
        raise RuntimeError(f"resampling {rate} Hz audio to {SAMPLE_RATE} Hz needs soxr: pip install cactus-needle[mic]") from None
    return soxr.resample(numpy.asarray(mono, numpy.float32), rate, SAMPLE_RATE, quality="HQ")


def _options(language, keywords):
    if keywords is not None and not isinstance(keywords, str):
        keywords = "\n".join(keywords)
    return language.encode("utf-8") if language else None, keywords.encode("utf-8") if keywords else None


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
        if lib.needle_load(data, len(data)) < 0:
            _loaded = None
            raise RuntimeError(lib.needle_last_error().decode("utf-8", "replace"))
        _loaded = self.weights
        return lib

    def _result(self, code):
        if code < 0:
            raise RuntimeError(_lib().needle_last_error().decode("utf-8", "replace"))
        return json.loads(self._buffer.value.decode("utf-8", "replace"))

    def transcribe(self, audio, language=None, keywords=None, word_timestamps=False) -> dict:
        """Returns {"text", "language", "ttft_ms", "decode_tps"}, plus "words" with start, end and probability when word_timestamps is set.

        language is one of LANGUAGES, or None to detect it. keywords are words or phrases for keyword biasing.
        """
        lib = self._bind()
        samples, count = _samples(audio)
        language, keywords = _options(language, keywords)
        return self._result(lib.needle_transcribe(samples, count, language, keywords, int(bool(word_timestamps)),
                                                  self._buffer, len(self._buffer)))

    def stream(self, chunks, language=None, keywords=None):
        """Live transcription: one dict per chunk, then one for the tail when the chunks end.

        chunks are 16 kHz mono float sample buffers, about a second each, with no limit on the total. Each dict has the
        "text" and "words" committed by that chunk, with times from the start of the stream, the unconfirmed "pending"
        tail, the "language", the seconds "received" and the "pass_ms" it took. Join the texts with a space.
        """
        lib = self._bind()
        language, keywords = _options(language, keywords)
        try:
            for chunk in chunks:
                samples, count = _samples(chunk)
                yield self._result(lib.needle_stream_transcribe_process(samples, count, language, keywords, self._buffer,
                                                                        len(self._buffer)))
        except BaseException:
            lib.needle_stream_transcribe_stop(self._buffer, len(self._buffer))
            raise
        yield self._result(lib.needle_stream_transcribe_stop(self._buffer, len(self._buffer)))

    def embed(self, audio) -> list[float]:
        """The encoder output, one row of floats per 80 ms frame, flattened."""
        lib = self._bind()
        samples, count = _samples(audio)
        size = lib.needle_embed(None, samples, count, None, 0)
        if size < 0:
            raise RuntimeError(lib.needle_last_error().decode("utf-8", "replace"))
        output = (ctypes.c_float * size)()
        if lib.needle_embed(None, samples, count, output, size) != size:
            raise RuntimeError(lib.needle_last_error().decode("utf-8", "replace"))
        return list(output)


_shared = {}


def _shared_model(weights):
    key = os.fspath(weights) if weights is not None else None
    if key not in _shared:
        _shared[key] = Whistle(weights=weights)
    return _shared[key]


def transcribe(audio, language=None, keywords=None, word_timestamps=False, weights=None) -> dict:
    """Speech to text, on the model this process already has loaded.

    audio is a WAV file path or 16 kHz mono float samples in [-1, 1], at most 30 s.
    Returns {"text", "language", "ttft_ms", "decode_tps"}, plus "words" with start,
    end and probability when word_timestamps is set. language is one of LANGUAGES,
    or None to detect it; keywords are words or phrases for keyword biasing.
    """
    from .._telemetry import track

    track("transcribe", {"tuned": bool(weights), "timestamps": bool(word_timestamps)})
    return _shared_model(weights).transcribe(audio, language=language, keywords=keywords,
                                             word_timestamps=word_timestamps)


def stream(chunks, language=None, keywords=None, weights=None):
    """Live transcription on the model this process already has loaded: Whistle.stream with no limit on length.

    chunks are 16 kHz mono float sample buffers, about a second each. Yields one dict per chunk with the "text"
    and "words" committed by it, the unconfirmed "pending" tail, the "language", the seconds "received" and
    "pass_ms", then one for the tail when the chunks end.
    """
    from .._telemetry import track

    track("stream", {"tuned": bool(weights)})
    return _shared_model(weights).stream(chunks, language=language, keywords=keywords)


PLAYGROUND_HELP = """  Enter          speak, Enter again to stop
  /language de   force a language (en de fr es it nl pl), bare to detect it
  /keywords      Siobhan, Krzysztof
  /timestamps    toggle word times
  /stream        toggle live transcription, the words printing as they are committed
  /file clip.wav transcribe a file
  /quit          leave"""
COMPARE_HELP = """  Enter          speak, Enter again to stop
  /file clip.wav run a file through every model
  /quit          leave"""
COMPARE_INSTALL = 'pip install "cactus-needle[mic,compare]"'
CLEAR_ABOVE = "\x1b[1A\x1b[2K"
DIM = "\x1b[90m"
PLAIN = "\x1b[0m"
LIMIT_SECONDS = 30


def audio_path(rest):
    """The path in a /file line, as a terminal hands it over: quoted, or with escaped spaces."""
    path = rest.strip()
    if len(path) > 1 and path[0] == path[-1] and path[0] in "'\"":
        return path[1:-1]
    return path.replace("\\ ", " ")


def status(name, size, ttft, tokens, total):
    ttft = f"{ttft * 1000:.0f} ms" if ttft is not None else "-"
    tokens = f"{tokens:.0f} tok/s" if tokens is not None else "-"
    spent = f"{total:.1f} s" if total >= 10 else f"{total * 1000:.0f} ms"
    return f"  {name:<18}{size / 1e6:>4.0f} MB  ttft {ttft:>6}  decode {tokens:>10}  total {spent:>7}"


def record():
    try:
        import numpy
        import sounddevice
        import soxr
    except ImportError as error:
        raise RuntimeError(f'{error.name} is not installed: pip install "cactus-needle[mic]"') from None
    chunks = []
    try:
        rate = int(sounddevice.query_devices(kind="input")["default_samplerate"])
        with sounddevice.InputStream(samplerate=rate, channels=1, dtype="float32", callback=lambda data, *_: chunks.append(data.copy())):
            print(CLEAR_ABOVE + "● recording, Enter to stop", end="", flush=True)
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                print()
    except sounddevice.PortAudioError as error:
        raise RuntimeError(f"no microphone: {error}") from None
    print(CLEAR_ABOVE, end="", flush=True)
    heard = numpy.concatenate(chunks)[:, 0] if chunks else numpy.zeros(0, numpy.float32)
    audio = heard[:LIMIT_SECONDS * rate]
    if len(audio) < len(heard):
        print(f"  keeping the first {LIMIT_SECONDS} s")
    return audio if rate == SAMPLE_RATE else soxr.resample(audio, rate, SAMPLE_RATE, quality="HQ")


def listen(whistle, state):
    try:
        import numpy
        import sounddevice
        import soxr
    except ImportError as error:
        raise RuntimeError(f'{error.name} is not installed: pip install "cactus-needle[mic]"') from None
    heard, steps, failed = queue.Queue(), [], []

    def chunks():
        block, held = [], 0
        while True:
            data = heard.get()
            if data is not None:
                block.append(data)
                held += len(data)
            if block and (held >= SAMPLE_RATE or data is None):
                yield numpy.concatenate(block)
                block, held = [], 0
            if data is None:
                return

    def show():
        committed, drawn = "", ()
        try:
            for step in whistle.stream(chunks(), language=state["language"], keywords=state["keywords"]):
                steps.append(step)
                committed += (" " if committed and step["text"] else "") + step["text"]
                columns = max(shutil.get_terminal_size().columns, 1)
                up = sum((len(line) - 1) // columns for line in drawn if line) + 1 if drawn else 0
                drawn = (committed, step["pending"])
                print((f"\r\x1b[{up}A\x1b[J" if up else "\r\x1b[J") + committed + "\n" + DIM + step["pending"] + PLAIN, end="", flush=True)
        except (RuntimeError, OSError) as error:
            failed.append(error)

    worker = threading.Thread(target=show, daemon=True)
    try:
        rate = int(sounddevice.query_devices(kind="input")["default_samplerate"])
        resampler = soxr.ResampleStream(rate, SAMPLE_RATE, 1, dtype="float32", quality="HQ") if rate != SAMPLE_RATE else None
        capture = lambda data, *_: heard.put(resampler.resample_chunk(data[:, 0]) if resampler else data[:, 0].copy())
        with sounddevice.InputStream(samplerate=rate, channels=1, dtype="float32", callback=capture):
            print(CLEAR_ABOVE + "● recording, Enter to stop")
            worker.start()
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                print()
            print("\x1b[1A", end="", flush=True)
        if resampler:
            heard.put(resampler.resample_chunk(numpy.zeros(0, numpy.float32), last=True))
    except sounddevice.PortAudioError as error:
        raise RuntimeError(f"no microphone: {error}") from None
    finally:
        heard.put(None)
        if worker.is_alive():
            worker.join()
    if failed:
        raise RuntimeError(failed[0])
    words = [word for step in steps for word in step["words"]]
    if not words:
        print("(no speech)")
    if state["timestamps"]:
        for word in words:
            print(f"  {word['start']:6.2f} - {word['end']:5.2f}  {word['word']:<20} {word['probability']:.2f}")
    passes = [step["pass_ms"] for step in steps if step["pass_ms"]]
    print(f"  {'whistle':<18}{os.path.getsize(whistle.weights) / 1e6:>4.0f} MB  audio {steps[-1]['received']:5.1f} s"
          f"  pass {sum(passes) / max(len(passes), 1):4.0f} ms   {steps[-1]['language'] or 'no speech'}")


def prompt():
    try:
        return input("› ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "/quit"


def show_transcript(whistle, audio, state):
    started = time.perf_counter()
    try:
        result = whistle.transcribe(audio, language=state["language"], keywords=state["keywords"], word_timestamps=state["timestamps"])
    except (RuntimeError, OSError, EOFError) as error:
        print(f"  {error}")
        return
    total = time.perf_counter() - started
    print(result["text"] or "(no speech)")
    for word in result.get("words", []):
        print(f"  {word['start']:6.2f} - {word['end']:5.2f}  {word['word']:<20} {word['probability']:.2f}")
    print(status("whistle", os.path.getsize(whistle.weights), result["ttft_ms"] / 1000, result["decode_tps"], total),
          " ", result["language"] or "no speech")


def playground(args):
    state = {"language": args.language, "keywords": [k.strip() for k in args.keywords.split(",") if k.strip()], "timestamps": args.word_timestamps,
             "stream": False}
    print("whistle playground: downloading and initializing the model...", flush=True)
    whistle = Whistle(weights=args.weights)
    if args.audio:
        return show_transcript(whistle, args.audio, state)
    print(PLAYGROUND_HELP)
    while True:
        line = prompt()
        command, _, rest = line.partition(" ")
        if command == "/quit":
            return
        if command == "/language":
            language = rest.strip()
            if language and language not in LANGUAGES:
                print("  language is one of", " ".join(LANGUAGES))
                continue
            state["language"] = language or None
            print("  language", state["language"] or "detected")
        elif command in ("/keywords", "/keyword"):
            state["keywords"] = [k.strip() for k in rest.split(",") if k.strip()]
            print("  keywords", ", ".join(state["keywords"]) or "none")
        elif command == "/timestamps":
            state["timestamps"] = not state["timestamps"]
            print("  word timestamps", "on" if state["timestamps"] else "off")
        elif command == "/stream":
            state["stream"] = not state["stream"]
            print("  live transcription", "on" if state["stream"] else "off")
        elif command == "/file":
            show_transcript(whistle, audio_path(rest), state)
        elif line == "":
            try:
                listen(whistle, state) if state["stream"] else show_transcript(whistle, record(), state)
            except RuntimeError as error:
                print(f"  {error}")
        else:
            print(PLAYGROUND_HELP)


def rate(marks):
    return (len(marks) - 1) / (marks[-1] - marks[0]) if len(marks) > 1 and marks[-1] > marks[0] else 0.0


def load_whistle(weights):
    whistle = Whistle(weights=weights)

    def transcribe(audio):
        result = whistle.transcribe(audio, language="en")
        return result["text"], result["ttft_ms"] / 1000, result["decode_tps"]

    return os.path.getsize(whistle.weights), transcribe


def load_whisper(size):
    import whisper

    model = whisper.load_model(size, device="cpu")
    cache = os.path.join(os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")), "whisper")
    marks = []
    model.decoder.register_forward_hook(lambda *_: marks.append(time.perf_counter()))

    def transcribe(audio):
        marks.clear()
        started = time.perf_counter()
        text = model.transcribe(audio, language="en", fp16=False)["text"].strip()
        return text, marks[0] - started, rate(marks)

    return os.path.getsize(os.path.join(cache, f"{size}.pt")), transcribe


def load_moonshine():
    import moonshine_voice

    path, arch = moonshine_voice.get_model_for_language("en", moonshine_voice.ModelArch.TINY_STREAMING)
    model = moonshine_voice.Transcriber(model_path=path, model_arch=arch)

    def transcribe(audio):
        lines = model.transcribe_without_streaming(audio.tolist(), SAMPLE_RATE).lines
        return " ".join(line.text.strip() for line in lines).strip(), None, None

    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)), transcribe


def run_models(models, audio):
    for name, size, transcribe in models:
        started = time.perf_counter()
        text, ttft, tokens = transcribe(audio)
        print(status(name, size, ttft, tokens, time.perf_counter() - started), " ", text or "(no speech)")


def compare(args):
    try:
        import moonshine_voice  # noqa: F401
        import numpy
        import whisper  # noqa: F401
    except ImportError as error:
        raise SystemExit(f"{error.name} is not installed: {COMPARE_INSTALL}")
    print("whistle compare: downloading and initializing the models...", flush=True)
    models = [
        ("whistle", *load_whistle(args.weights)),
        ("whisper tiny", *load_whisper("tiny")),
        ("whisper base", *load_whisper("base")),
        ("moonshine tiny v2", *load_moonshine()),
    ]
    beat = numpy.arange(SAMPLE_RATE) * (numpy.pi / SAMPLE_RATE)
    warmup = (numpy.sin(440 * beat) * numpy.abs(numpy.sin(3 * beat))).astype(numpy.float32)
    for _, _, transcribe in models:
        transcribe(warmup)
    if args.audio:
        return run_models(models, numpy.asarray(_read_wav(args.audio), numpy.float32))
    print(COMPARE_HELP)
    while True:
        line = prompt()
        command, _, rest = line.partition(" ")
        if command == "/quit":
            return
        if command == "/file":
            try:
                run_models(models, numpy.asarray(_read_wav(audio_path(rest)), numpy.float32))
            except OSError as error:
                print(f"  {error}")
        elif line == "":
            try:
                run_models(models, numpy.asarray(record(), numpy.float32))
            except RuntimeError as error:
                print(f"  {error}")
        else:
            print(COMPARE_HELP)
