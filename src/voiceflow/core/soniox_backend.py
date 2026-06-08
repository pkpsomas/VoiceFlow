"""Soniox streaming ASR backend for VoiceFlow.

Implements the same transcribe(audio_data) interface as ModernWhisperASR so it
can be dropped in as a replacement.  Audio is streamed over a Soniox WebSocket
during transcription (not batched to a local model).

Protocol summary (mirrors untype-s SonioxTranscriber.swift):
  1. Connect to wss://stt.soniox.com/transcribe-stream
  2. Send JSON config frame (text)
  3. Send raw PCM-S16LE binary frames (audio data)
  4. Send {"type":"finalize"} (text)
  5. Drain responses for ~250 ms, then close
  6. Return the accumulated final-token text
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SonioxError(RuntimeError):
    pass


class SonioxASRBackend:
    """Drop-in replacement for ModernWhisperASR that uses Soniox cloud STT.

    Required config fields (read from a VoiceFlow Config object):
        soniox_api_key      – Soniox API key (required, no default)
        soniox_model        – model name, default "stt-rt-preview"
        soniox_endpoint     – WebSocket URL, default wss://stt.soniox.com/transcribe-stream
        soniox_languages    – list of language codes, default ["en"]
        soniox_enable_endpoint_detection – bool, default True
        soniox_context      – optional transcription context string
        sample_rate         – int, default 16000 (VoiceFlow standard)
        soniox_finalize_drain_ms – ms to wait after finalize, default 250
        verbose             – bool, extra logging
    """

    CHUNK_SAMPLES = 1600  # 100 ms at 16 kHz

    def __init__(self, cfg: Any) -> None:
        api_key: str = getattr(cfg, "soniox_api_key", "").strip()
        if not api_key:
            raise SonioxError(
                "soniox_api_key is required when transcriber='soniox'. "
                "Set it via SONIOX_API_KEY env-var or the config file."
            )

        self._api_key = api_key
        self._model: str = getattr(cfg, "soniox_model", "stt-rt-preview")
        self._endpoint: str = getattr(
            cfg, "soniox_endpoint", "wss://stt.soniox.com/transcribe-stream"
        )

        raw_langs = getattr(cfg, "soniox_languages", ["en"])
        self._languages: List[str] = list(raw_langs) if raw_langs else ["en"]

        self._enable_endpoint_detection: bool = bool(
            getattr(cfg, "soniox_enable_endpoint_detection", True)
        )
        self._context: Optional[str] = getattr(cfg, "soniox_context", None) or None
        self._sample_rate: int = int(getattr(cfg, "sample_rate", 16000))
        self._drain_ms: int = int(getattr(cfg, "soniox_finalize_drain_ms", 250))
        self._verbose: bool = bool(getattr(cfg, "verbose", False))

        # Statistics (legacy compat)
        self.session_transcription_count = 0
        self.total_audio_duration = 0.0
        self.total_processing_time = 0.0
        self._session_start = time.time()

        # Preload state (no local model to load)
        self._loaded = True
        self.skip_warmup = True  # signal ModelPreloader to skip synthetic warmup

    # ------------------------------------------------------------------
    # ModelPreloader / ASREngine compatibility surface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """No-op: Soniox is cloud-based, no local model to load."""

    def is_loaded(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def reset_session(self) -> None:
        self.session_transcription_count = 0
        self._session_start = time.time()
        self.total_audio_duration = 0.0
        self.total_processing_time = 0.0

    def get_statistics(self) -> Dict[str, Any]:
        return self.get_clean_statistics()

    def get_clean_statistics(self) -> Dict[str, Any]:
        elapsed = time.time() - self._session_start
        avg_speed = (
            self.total_audio_duration / self.total_processing_time
            if self.total_processing_time > 0
            else 0.0
        )
        return {
            "session_transcription_count": self.session_transcription_count,
            "transcription_count": self.session_transcription_count,
            "session_duration_seconds": elapsed,
            "total_audio_duration": self.total_audio_duration,
            "total_processing_time": self.total_processing_time,
            "average_speed_factor": avg_speed,
            "model_loaded": True,
            "backend": "soniox",
        }

    # ------------------------------------------------------------------
    # Main transcription interface
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        initial_prompt: Optional[str] = None,
        beam_size_override: Optional[int] = None,
        vad_filter_override: Optional[bool] = None,
    ) -> str:
        """Send audio to Soniox and return the final transcript text.

        Args:
            audio: float32 numpy array, 16 kHz mono (VoiceFlow standard).
                   Values are in [-1, 1]; converted to PCM-S16LE internally.
            initial_prompt: ignored (Soniox uses context instead).
            beam_size_override: ignored.
            vad_filter_override: ignored.

        Returns:
            Transcribed text string (stripped), empty string on silence/error.
        """
        if audio is None or audio.size == 0:
            return ""

        audio_duration = len(audio) / self._sample_rate
        start_time = time.time()

        try:
            text = self._run_in_new_loop(audio)
        except Exception as exc:
            logger.error("soniox_transcribe_error: %s", exc)
            text = ""

        elapsed = time.time() - start_time
        self.session_transcription_count += 1
        self.total_audio_duration += audio_duration
        self.total_processing_time += elapsed

        if self._verbose:
            logger.info(
                "soniox_transcribed chars=%d duration=%.2fs elapsed=%.2fs",
                len(text),
                audio_duration,
                elapsed,
            )

        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_in_new_loop(self, audio: np.ndarray) -> str:
        """Run the async Soniox session synchronously in a fresh event loop.

        VoiceFlow calls transcribe() from a ThreadPoolExecutor worker thread
        that has no running event loop, so we create one per call.
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._transcribe_async(audio))
        finally:
            loop.close()

    async def _transcribe_async(self, audio: np.ndarray) -> str:
        try:
            import websockets  # type: ignore[import]
        except ImportError as exc:
            raise SonioxError(
                "The 'websockets' package is required for the Soniox backend. "
                "Install it with: pip install 'websockets>=12.0'"
            ) from exc

        finals: List[str] = []
        committed = ""

        async with websockets.connect(self._endpoint) as ws:
            # 1. Send config frame
            await ws.send(self._build_config_frame())

            # 2. Stream audio as PCM-S16LE binary chunks
            pcm = _float32_to_pcm16le(audio)
            chunk_bytes = self.CHUNK_SAMPLES * 2  # 2 bytes per int16 sample
            for offset in range(0, len(pcm), chunk_bytes):
                await ws.send(pcm[offset : offset + chunk_bytes])

            # 3. Send finalize
            await ws.send(json.dumps({"type": "finalize"}))

            # 4. Drain responses until finished or timeout.
            # Allow at least 12 s after finalize — Soniox can take 5-8 s on
            # longer utterances before sending the finished frame.
            audio_duration_s = len(audio) / max(1, self._sample_rate)
            drain_window = max(12.0, audio_duration_s * 2.0) + (self._drain_ms / 1000.0)
            drain_deadline = asyncio.get_event_loop().time() + drain_window
            try:
                while True:
                    remaining = drain_deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        logger.warning("soniox_drain_timeout committed_chars=%d", len(committed))
                        break
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5.0))
                    done, chunk_committed = _handle_frame(raw, self._verbose)
                    if chunk_committed:
                        committed = _merge_finals(committed, chunk_committed)
                        if self._verbose:
                            logger.debug("soniox_frame_committed %r", chunk_committed)
                    if done:
                        break
            except asyncio.TimeoutError:
                logger.warning("soniox_recv_timeout after finalize committed_chars=%d", len(committed))
            except Exception as exc:
                # ConnectionClosedOK means server closed after sending all data — treat as done
                exc_name = type(exc).__name__
                if "ConnectionClosed" in exc_name:
                    logger.debug("soniox_connection_closed_by_server committed_chars=%d", len(committed))
                else:
                    logger.warning("soniox_drain_error: %s", exc)

        return committed.strip()

    def _build_config_frame(self) -> str:
        obj: Dict[str, Any] = {
            "api_key": self._api_key,
            "model": self._model,
            "audio_format": "pcm_s16le",
            "sample_rate": self._sample_rate,
            "num_channels": 1,
            "enable_endpoint_detection": self._enable_endpoint_detection,
        }

        if len(self._languages) == 1 and self._languages[0] == "auto":
            obj["enable_language_identification"] = True
        else:
            obj["language_hints"] = self._languages

        context = (self._context or "").strip()
        if context:
            obj["context"] = {"text": context}

        return json.dumps(obj, sort_keys=True)


# ---------------------------------------------------------------------------
# Module-level helpers (no self dependency)
# ---------------------------------------------------------------------------

def _float32_to_pcm16le(audio: np.ndarray) -> bytes:
    """Convert float32 [-1,1] mono audio to raw PCM-S16LE bytes."""
    clamped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clamped * 32767).astype(np.int16)
    return pcm16.tobytes()


def _handle_frame(raw: str | bytes, verbose: bool) -> tuple[bool, str]:
    """Parse one Soniox WebSocket frame.

    Returns (done, committed_finals_text).
    """
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return False, ""

    raw = raw.strip()
    if not raw:
        return False, ""

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return False, ""

    # Server error
    error_type = (
        obj.get("error_type")
        or obj.get("errorType")
        or (obj.get("type") if obj.get("type") == "error" else None)
    )
    if error_type:
        msg = obj.get("message") or obj.get("error") or f"Soniox error: {error_type}"
        raise SonioxError(msg)

    committed = ""

    tokens = obj.get("tokens")
    if tokens and isinstance(tokens, list):
        committed = _extract_finals(tokens)
        if verbose and committed:
            logger.debug("soniox_finals: %r", committed)

    msg_type = obj.get("type", "")
    finished = obj.get("finished", False)

    done = (
        finished is True
        or msg_type in ("finished", "finalized", "endpoint")
    )

    return done, committed


def _extract_finals(tokens: list) -> str:
    """Return concatenated text of all final tokens, skipping sentinels."""
    parts: List[str] = []
    for tok in tokens:
        if not isinstance(tok, dict):
            continue
        text = tok.get("text", "")
        if text in ("<end>", "<fin>", ""):
            continue
        if tok.get("is_final"):
            parts.append(text)
    return "".join(parts)


def _merge_finals(committed: str, incoming: str) -> str:
    """Merge two consecutive final-token strings, deduplicating overlap.

    Mirrors SonioxTranscriber.swift mergeFinalText().
    """
    if not committed:
        return incoming
    if not incoming:
        return committed
    if incoming.startswith(committed):
        return incoming
    if committed.endswith(incoming):
        return committed

    # Find longest suffix of committed that is a prefix of incoming
    max_len = min(len(committed), len(incoming))
    overlap = 0
    for length in range(max_len, 0, -1):
        if committed[-length:] == incoming[:length]:
            overlap = length
            break

    return committed + incoming[overlap:]
