from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def system_audio_supported() -> bool:
    """Check whether WASAPI loopback capture is available (soundcard installed)."""
    try:
        import soundcard  # noqa: F401

        return True
    except Exception:
        return False


class SystemAudioCapture:
    """Continuous WASAPI loopback capture of system output audio.

    Records whatever is playing on the default output device (speakers/headset)
    and delivers mono float32 chunks at the requested sample rate via `on_chunk`.
    Runs in a daemon thread; survives device errors by re-resolving the default
    speaker and reopening the loopback recorder.
    """

    def __init__(
        self,
        sample_rate: int,
        blocksize: int,
        on_chunk: Callable[[np.ndarray], None],
    ):
        self.sample_rate = int(sample_rate)
        self.blocksize = max(64, int(blocksize))
        self._on_chunk = on_chunk
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if self.is_running():
            return True
        if not system_audio_supported():
            logger.error("[SystemAudio] soundcard library not available; cannot capture system audio")
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="SystemAudioCapture", daemon=True
        )
        self._thread.start()
        print("[SystemAudio] Loopback capture thread started")
        return True

    def stop(self):
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            # record() blocks while the system is silent; daemon thread exits on
            # its own once audio flows or the process ends, so don't wait long.
            thread.join(timeout=2.0)
        print("[SystemAudio] Loopback capture stopped")

    def _run(self):
        import warnings

        import soundcard as sc

        # Silence-gap discontinuities are routine for loopback capture and
        # would otherwise flood stderr with one warning per chunk.
        warnings.filterwarnings(
            "ignore", category=sc.SoundcardRuntimeWarning, message=".*discontinuity.*"
        )

        # Re-resolve the default speaker periodically so capture follows the
        # user when they switch output devices mid-session.
        device_check_samples = self.sample_rate * 3

        while not self._stop.is_set():
            try:
                speaker = sc.default_speaker()
                loopback = sc.get_microphone(speaker.id, include_loopback=True)
                print(f"[SystemAudio] Capturing loopback of: {speaker.name}")
                samples_since_check = 0
                with loopback.recorder(
                    samplerate=self.sample_rate, channels=1, blocksize=self.blocksize
                ) as rec:
                    while not self._stop.is_set():
                        data = rec.record(numframes=self.blocksize)
                        if self._stop.is_set():
                            return
                        chunk = np.asarray(data, dtype=np.float32).reshape(-1)
                        if chunk.size:
                            self._on_chunk(chunk)
                        samples_since_check += chunk.size
                        if samples_since_check >= device_check_samples:
                            samples_since_check = 0
                            if sc.default_speaker().id != speaker.id:
                                print("[SystemAudio] Default output changed; reopening loopback")
                                break
            except Exception as e:
                if self._stop.is_set():
                    return
                logger.warning(f"[SystemAudio] Capture error, retrying in 1s: {e}")
                self._stop.wait(1.0)
