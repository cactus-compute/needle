import os
import tempfile
import time
import wave

from .whistle import prompt, record, status

RATE = 16000
HELP = """  Enter          speak, Enter again to stop
  /quit          leave"""
INSTALL = 'pip install "cactus-needle[compare]"'


def read(path):
    import numpy

    with wave.open(path, "rb") as file:
        channels, width, rate = file.getnchannels(), file.getsampwidth(), file.getframerate()
        data = file.readframes(file.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit WAV is supported")
    audio = numpy.frombuffer(data, numpy.int16).reshape(-1, channels).mean(axis=1) / 32768.0
    if rate != RATE:
        import soxr

        audio = soxr.resample(audio.astype(numpy.float32), rate, RATE, quality="HQ")
    return audio.astype(numpy.float32)


def rate(marks):
    return (len(marks) - 1) / (marks[-1] - marks[0]) if len(marks) > 1 and marks[-1] > marks[0] else 0.0


def load_whistle(weights):
    from ..whistle import Whistle

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
        lines = model.transcribe_without_streaming(audio.tolist(), RATE).lines
        return " ".join(line.text.strip() for line in lines).strip(), None, None

    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)), transcribe


def compare(models, audio):
    for name, size, transcribe in models:
        started = time.perf_counter()
        text, ttft, tokens = transcribe(audio)
        print(status(name, size, ttft, tokens, time.perf_counter() - started), " ", text or "(no speech)")


def main(args):
    try:
        import moonshine_voice  # noqa: F401
        import numpy
        import whisper  # noqa: F401
    except ImportError as error:
        raise SystemExit(f"{error.name} is not installed: {INSTALL}")
    models = [
        ("whistle", *load_whistle(args.weights)),
        ("whisper tiny", *load_whisper("tiny")),
        ("whisper base", *load_whisper("base")),
        ("moonshine tiny v2", *load_moonshine()),
    ]
    warmup = numpy.sin(numpy.arange(RATE) * (2 * numpy.pi * 220 / RATE)) * numpy.abs(numpy.sin(numpy.arange(RATE) * (3 * numpy.pi / RATE)))
    for _, _, transcribe in models:
        transcribe(warmup.astype(numpy.float32))
    if args.audio:
        return compare(models, read(args.audio))
    recording = os.path.join(tempfile.mkdtemp(prefix="needle-compare-"), "recording.wav")
    print(HELP)
    while True:
        line = prompt()
        if line == "/quit":
            return
        if line:
            print(HELP)
            continue
        error = record(recording)
        print(error) if error else compare(models, read(recording))
