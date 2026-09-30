import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HELP = """  Enter          speak, Enter again to stop
  /lang de       force a language (en de fr es it nl pl), bare to detect it
  /keywords      Siobhan, Krzysztof
  /timestamps    toggle word times
  /file clip.wav transcribe a file
  /quit          leave"""
CLEAR_LINE = "\r\x1b[2K"
CLEAR_ABOVE = "\x1b[1A\x1b[2K"


def status(name, size, ttft, tokens, total):
    ttft = f"{ttft * 1000:.0f} ms" if ttft is not None else "-"
    tokens = f"{tokens:.0f} tok/s" if tokens is not None else "-"
    return f"  {name:<18}{size / 1e6:.0f} MB  ttft {ttft}  decode {tokens}  total {total * 1000:.0f} ms"


def record(path):
    if shutil.which("rec"):
        command = ["rec", "-q", "-b", "16", path, "channels", "1", "rate", "16000"]
    elif shutil.which("ffmpeg"):
        command = ["ffmpeg", "-loglevel", "error", "-y", "-f", "avfoundation", "-i", ":0", "-ac", "1", "-ar", "16000", path]
    else:
        return "recording needs sox (rec) or ffmpeg"
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    print(CLEAR_ABOVE + "● recording, Enter to stop", end="", flush=True)
    pressed = bool(select.select([sys.stdin], [], [], 30)[0])
    if pressed:
        sys.stdin.readline()
    process.send_signal(signal.SIGINT)
    process.wait()
    print(CLEAR_ABOVE if pressed else CLEAR_LINE, end="", flush=True)
    return None


def prompt():
    try:
        return input("› ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "/quit"


def transcribe(whistle, audio, state):
    started = time.perf_counter()
    try:
        result = whistle.transcribe(audio, language=state["lang"], keywords=state["keywords"], word_timestamps=state["timestamps"])
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
    from ..whistle import LANGUAGES, Whistle

    state = {"lang": args.lang, "keywords": [k.strip() for k in args.keywords.split(",") if k.strip()], "timestamps": args.word_timestamps}
    whistle = Whistle(weights=args.weights)
    if args.audio:
        return transcribe(whistle, args.audio, state)
    recording = os.path.join(tempfile.mkdtemp(prefix="needle-whistle-"), "recording.wav")
    print(HELP)
    while True:
        line = prompt()
        command, _, rest = line.partition(" ")
        if command == "/quit":
            return
        if command == "/lang":
            state["lang"] = rest.strip() if rest.strip() in LANGUAGES else None
            print("  language", state["lang"] or "detected")
        elif command in ("/keywords", "/keyword"):
            state["keywords"] = [k.strip() for k in rest.split(",") if k.strip()]
            print("  keywords", ", ".join(state["keywords"]) or "none")
        elif command == "/timestamps":
            state["timestamps"] = not state["timestamps"]
            print("  word timestamps", "on" if state["timestamps"] else "off")
        elif command == "/file":
            transcribe(whistle, rest.strip(), state)
        elif line == "":
            error = record(recording)
            print(error) if error else transcribe(whistle, recording, state)
        else:
            print(HELP)
