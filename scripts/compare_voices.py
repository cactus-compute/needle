#!/usr/bin/env python3
"""Generate sample TTS audio with Wavenet vs Neural2 for quality comparison."""

import io
import wave
from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()

QUERIES = [
    "What's the weather like in San Francisco today?",
    "Book me a flight from London to Tokyo next Friday",
    "Send an email to the team about the meeting at 3pm",
]

VOICES = [
    ("wavenet_male_us", "en-US-Wavenet-D", "en-US"),
    ("chirp3_us", "en-US-Chirp3-HD-Achernar", "en-US"),
    ("journey_male_us", "en-US-Journey-D", "en-US"),
    ("wavenet_female_gb", "en-GB-Wavenet-A", "en-GB"),
    ("chirp3_gb", "en-GB-Chirp3-HD-Leda", "en-GB"),
    ("journey_female_us", "en-US-Journey-F", "en-US"),
    ("wavenet_male_au", "en-AU-Wavenet-B", "en-AU"),
    ("chirp3_au", "en-AU-Chirp3-HD-Charon", "en-AU"),
    ("journey_female_us2", "en-US-Journey-O", "en-US"),
]


def pcm_to_wav(pcm_bytes, sr=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


import os
os.makedirs("voice_samples", exist_ok=True)

for qi, query in enumerate(QUERIES):
    for label, voice_name, lang in VOICES:
        print(f"  {label} / query {qi+1}...", end="", flush=True)
        response = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=query),
            voice=texttospeech.VoiceSelectionParams(
                language_code=lang, name=voice_name,
            ),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
            ),
        )
        wav = pcm_to_wav(response.audio_content)
        path = f"voice_samples/q{qi+1}_{label}.wav"
        with open(path, "wb") as f:
            f.write(wav)
        print(f" saved {path}")

print(f"\nDone! {len(QUERIES) * len(VOICES)} files in voice_samples/")
