from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def system_audio_supported() -> bool:
    """Check whether WASAPI loopback capture is available (PyAudioWPatch)."""
    try:
        import pyaudiowpatch  # noqa: F401

        return True
    except Exception:
        return False


def _resample_to_target(mono: np.ndarray, native_rate: int, target_rate: int) -> np.ndarray:
    """Linear-resample a mono float32 signal; adequate for speech ASR."""
    if native_rate == target_rate or mono.size == 0:
        return mono
    n_out = int(round(len(mono) * float(target_rate) / float(native_rate)))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, mono).astype(np.float32)


class SystemAudioCapture:
    """Continuous WASAPI loopback capture of system output audio.

    Uses PyAudioWPatch (a PortAudio fork with WASAPI loopback). PortAudio
    negotiates the endpoint format natively, so this avoids the Python-side
    format assertion that the `soundcard` package hit on headsets in
    communications/call mode. Delivers mono float32 chunks resampled to the
    requested sample rate via `on_chunk`.
    """

    # Give up after this many consecutive open failures instead of
    # crash-looping forever (e.g. an endpoint that never accepts loopback).
    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(
        self,
        sample_rate: int,
        blocksize: int,
        on_chunk: Callable[[np.ndarray], None],
        on_failed: Optional[Callable[[str], None]] = None,
    ):
        self.sample_rate = int(sample_rate)
        self.blocksize = max(64, int(blocksize))
        self._on_chunk = on_chunk
        self._on_failed = on_failed
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.failed = False

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if self.is_running():
            return True
        if not system_audio_supported():
            logger.error("[SystemAudio] PyAudioWPatch not available; cannot capture system audio")
            return False
        self._stop.clear()
        self.failed = False
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
            thread.join(timeout=2.0)
        print("[SystemAudio] Loopback capture stopped")

    def _resolve_loopback(self, p):
        """Return the loopback device info dict for the default output, or None."""
        import pyaudiowpatch as pa

        wasapi = p.get_host_api_info_by_type(pa.paWASAPI)
        default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if default_out.get("isLoopbackDevice"):
            return default_out
        # Loopback devices are duplicated at the end of the list, matched by name.
        for cand in p.get_loopback_device_info_generator():
            if default_out["name"] in cand["name"]:
                return cand
        return None

    def _run(self):
        import pyaudiowpatch as pa

        consecutive_failures = 0

        while not self._stop.is_set():
            p = None
            stream = None
            try:
                p = pa.PyAudio()
                device = self._resolve_loopback(p)
                if device is None:
                    raise RuntimeError("no WASAPI loopback device for the default output")

                native_rate = int(device["defaultSampleRate"])
                channels = max(1, int(device["maxInputChannels"]))
                device_index = int(device["index"])
                default_out_index = int(
                    p.get_host_api_info_by_type(pa.paWASAPI)["defaultOutputDevice"]
                )
                print(f"[SystemAudio] Capturing loopback of: {device['name']}")

                def _callback(in_data, frame_count, time_info, status):  # noqa: ARG001
                    if self._stop.is_set():
                        return (None, pa.paComplete)
                    try:
                        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                        if channels > 1:
                            samples = samples.reshape(-1, channels).mean(axis=1)
                        mono = _resample_to_target(samples, native_rate, self.sample_rate)
                        if mono.size:
                            self._on_chunk(mono)
                    except Exception as cb_err:
                        logger.debug(f"[SystemAudio] callback error: {cb_err}")
                    return (None, pa.paContinue)

                stream = p.open(
                    format=pa.paInt16,
                    channels=channels,
                    rate=native_rate,
                    frames_per_buffer=self.blocksize,
                    input=True,
                    input_device_index=device_index,
                    stream_callback=_callback,
                )
                stream.start_stream()
                consecutive_failures = 0  # a successful open resets the counter

                # Keep the stream alive; reopen if the default output device
                # changes (user switched headset/speakers) or the stream drops.
                while not self._stop.is_set():
                    if not stream.is_active():
                        print("[SystemAudio] Stream inactive; reopening loopback")
                        break
                    try:
                        cur = int(
                            p.get_host_api_info_by_type(pa.paWASAPI)["defaultOutputDevice"]
                        )
                        if cur != default_out_index:
                            print("[SystemAudio] Default output changed; reopening loopback")
                            break
                    except Exception:
                        pass
                    self._stop.wait(0.5)
            except Exception as e:
                if self._stop.is_set():
                    break
                consecutive_failures += 1
                reason = str(e) or type(e).__name__
                if consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    self.failed = True
                    logger.error(
                        f"[SystemAudio] Giving up after {consecutive_failures} failures: {reason}"
                    )
                    print(f"[SystemAudio] Loopback capture unavailable: {reason}")
                    if self._on_failed is not None:
                        try:
                            self._on_failed(reason)
                        except Exception:
                            pass
                    self._safe_close(stream, p)
                    return
                logger.warning(
                    f"[SystemAudio] Capture error ({consecutive_failures}/"
                    f"{self.MAX_CONSECUTIVE_FAILURES}), retrying in 1s: {reason}"
                )
                self._stop.wait(1.0)
            finally:
                self._safe_close(stream, p)

    @staticmethod
    def _safe_close(stream, p):
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
        except Exception:
            pass
        try:
            if p is not None:
                p.terminate()
        except Exception:
            pass
