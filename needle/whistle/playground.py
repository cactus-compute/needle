import os
import time

HELP = """  Enter          speak, Enter again to stop
  /language de   force a language (en de fr es it nl pl), bare to detect it
  /keywords      Siobhan, Krzysztof
  /timestamps    toggle word times
  /file clip.wav transcribe a file
  /quit          leave"""
CLEAR_ABOVE = "\x1b[1A\x1b[2K"
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
    from . import SAMPLE_RATE

    try:
        import numpy
        import sounddevice
        import soxr
    except ImportError as error:
        raise RuntimeError(f'{error.name} is not installed: pip install "cactus-needle[whistle]"') from None
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


def prompt():
    try:
        return input("› ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "/quit"


def transcribe(whistle, audio, state):
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


def main(args):
    from . import LANGUAGES, Whistle

    state = {"language": args.language, "keywords": [k.strip() for k in args.keywords.split(",") if k.strip()], "timestamps": args.word_timestamps}
    print("whistle playground: downloading and initializing the model...", flush=True)
    whistle = Whistle(weights=args.weights)
    if args.audio:
        return transcribe(whistle, args.audio, state)
    print(HELP)
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
        elif command == "/file":
            transcribe(whistle, audio_path(rest), state)
        elif line == "":
            try:
                transcribe(whistle, record(), state)
            except RuntimeError as error:
                print(f"  {error}")
        else:
            print(HELP)
