"""E2E repro: play TTS speech through speakers, record via recorder in a given
source mode, save WAV, and transcribe with faster-whisper directly.

Usage: python scripts/repro_system_transcribe.py [mic|system|both]
"""
from __future__ import annotations

import sys
import threading
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from voiceflow.core.audio_enhanced import EnhancedAudioRecorder
from voiceflow.core.config import Config

PHRASE = "The quick brown fox jumps over the lazy dog near the river bank"
OUT = Path(__file__).resolve().parent / "_repro_capture.wav"


def speak():
    import subprocess
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Speak('{PHRASE}')"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False)


def main() -> int:
    source = sys.argv[1] if len(sys.argv) > 1 else "both"
    cfg = Config()
    cfg.audio_input_source = source
    rec = EnhancedAudioRecorder(cfg)

    t = threading.Thread(target=speak, daemon=True)
    rec.start()
    time.sleep(0.3)
    t.start()
    t.join(timeout=15)
    time.sleep(0.5)
    audio = rec.stop()
    rec.stop_continuous()
    rec._stop_system_capture()

    rms = float(np.sqrt((audio.astype(np.float64) ** 2).mean())) if audio.size else 0.0
    print(f"source={source} samples={len(audio)} dur={len(audio)/16000:.2f}s rms={rms:.4f} "
          f"max={float(np.abs(audio).max()) if audio.size else 0:.4f}")

    with wave.open(str(OUT), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    print(f"saved {OUT}")

    from faster_whisper import WhisperModel
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(audio, language="en")
    text = " ".join(s.text.strip() for s in segments).strip()
    print(f"TRANSCRIPT: {text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
