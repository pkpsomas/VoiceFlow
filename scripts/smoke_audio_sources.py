"""Smoke test for audio source selection (mic / system loopback / both).

Plays a 440 Hz tone through the default output while recording, so the
loopback track has deterministic content. Run from repo root:

    venv\\Scripts\\python.exe scripts\\smoke_audio_sources.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import sounddevice as sd

from voiceflow.core.audio_enhanced import EnhancedAudioRecorder
from voiceflow.core.config import Config


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x.astype(np.float64) ** 2).mean())) if x.size else 0.0


def tone(duration_s: float, sr: int = 48000, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def run_case(source: str, play_tone: bool, record_s: float = 2.0) -> np.ndarray:
    print(f"\n=== source={source} (tone={'on' if play_tone else 'off'}) ===")
    cfg = Config()
    cfg.audio_input_source = source
    rec = EnhancedAudioRecorder(cfg)
    if play_tone:
        sd.play(tone(record_s + 1.0), samplerate=48000)
    time.sleep(0.5)  # let playback/loopback spin up
    rec.start()
    time.sleep(record_s)
    audio = rec.stop()
    sd.stop()
    rec.stop_continuous()
    rec._stop_system_capture()
    print(f"RESULT source={source}: samples={len(audio)} "
          f"({len(audio)/cfg.sample_rate:.2f}s) rms={rms(audio):.4f}")
    return audio


def main() -> int:
    ok = True

    sys_audio = run_case("system", play_tone=True)
    if rms(sys_audio) < 0.01:
        print("FAIL: system loopback captured no tone")
        ok = False

    both_audio = run_case("both", play_tone=True)
    if rms(both_audio) < 0.01:
        print("FAIL: 'both' mix captured no tone")
        ok = False

    mic_audio = run_case("mic", play_tone=False)
    if mic_audio.size == 0:
        print("FAIL: mic recording returned no samples")
        ok = False

    print("\nSMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
