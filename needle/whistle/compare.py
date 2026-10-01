import os
import time

from .playground import audio_path, prompt, record, status

HELP = """  Enter          speak, Enter again to stop
  /file clip.wav run a file through every model
  /quit          leave"""
INSTALL = 'pip install "cactus-needle[whistle,whistle-compare]"'


def rate(marks):
    return (len(marks) - 1) / (marks[-1] - marks[0]) if len(marks) > 1 and marks[-1] > marks[0] else 0.0


def load_whistle(weights):
    from . import Whistle

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

    from . import SAMPLE_RATE

    path, arch = moonshine_voice.get_model_for_language("en", moonshine_voice.ModelArch.TINY_STREAMING)
    model = moonshine_voice.Transcriber(model_path=path, model_arch=arch)

    def transcribe(audio):
        lines = model.transcribe_without_streaming(audio.tolist(), SAMPLE_RATE).lines
        return " ".join(line.text.strip() for line in lines).strip(), None, None

    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)), transcribe


def compare(models, audio):
    for name, size, transcribe in models:
        started = time.perf_counter()
        text, ttft, tokens = transcribe(audio)
        print(status(name, size, ttft, tokens, time.perf_counter() - started), " ", text or "(no speech)")


def main(args):
    from . import SAMPLE_RATE, _read_wav

    try:
        import moonshine_voice  # noqa: F401
        import numpy
        import whisper  # noqa: F401
    except ImportError as error:
        raise SystemExit(f"{error.name} is not installed: {INSTALL}")
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
        return compare(models, numpy.asarray(_read_wav(args.audio), numpy.float32))
    print(HELP)
    while True:
        line = prompt()
        command, _, rest = line.partition(" ")
        if command == "/quit":
            return
        if command == "/file":
            try:
                compare(models, numpy.asarray(_read_wav(audio_path(rest)), numpy.float32))
            except OSError as error:
                print(f"  {error}")
        elif line == "":
            try:
                compare(models, numpy.asarray(record(), numpy.float32))
            except RuntimeError as error:
                print(f"  {error}")
        else:
            print(HELP)
