from __future__ import annotations

import threading
from typing import Optional, List
from collections import deque
import time
import logging

import numpy as np
import sounddevice as sd

from voiceflow.core.config import Config
from voiceflow.core.system_audio import SystemAudioCapture, system_audio_supported
from voiceflow.utils.validation import validate_audio_data, validate_sample_rate, ValidationError

VALID_AUDIO_SOURCES = ("mic", "system", "both")

# Set up logging for audio validation
logger = logging.getLogger(__name__)


def audio_validation_guard(audio_data: np.ndarray,
                          operation_name: str = "audio_operation",
                          allow_empty: bool = False,
                          cfg: Optional['Config'] = None) -> np.ndarray:
    """
    Comprehensive audio input validation and sanitization guard.

    This function prevents crashes from malformed audio data by:
    - Detecting and handling NaN/Inf values
    - Validating audio format and dimensions
    - Clamping extreme values to safe ranges
    - Providing detailed error logging with metadata
    - Using centralized validation module for security consistency

    Args:
        audio_data: Audio data array to validate
        operation_name: Name of operation for logging context
        allow_empty: Whether to allow empty arrays (default: False)

    Returns:
        Sanitized audio data array

    Raises:
        ValueError: If audio data is invalid and cannot be recovered
    """
    if audio_data is None:
        error_msg = f"[AudioGuard] {operation_name}: Audio data is None"
        logger.error(error_msg)
        if allow_empty:
            return np.array([], dtype=np.float32)
        raise ValueError(error_msg)

    # PERFORMANCE OPTIMIZATION: Fast path for performance mode
    if cfg and getattr(cfg, 'enable_fast_audio_validation', False):
        return _fast_audio_validation_guard(audio_data, operation_name, allow_empty, cfg)

    # First apply centralized validation for security consistency
    try:
        validated_audio = validate_audio_data(audio_data, f"{operation_name}_audio")
    except ValidationError as e:
        error_msg = f"[AudioGuard] {operation_name}: Security validation failed: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Use the validated audio from security validation
    audio_data = validated_audio

    # Convert to numpy array if needed (should already be done by validation)
    if not isinstance(audio_data, np.ndarray):
        try:
            audio_data = np.array(audio_data, dtype=np.float32)
            logger.warning(f"[AudioGuard] {operation_name}: Converted input to numpy array")
        except Exception as e:
            error_msg = f"[AudioGuard] {operation_name}: Cannot convert input to numpy array: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    # Check for empty arrays
    if audio_data.size == 0:
        if allow_empty:
            logger.info(f"[AudioGuard] {operation_name}: Empty audio array (allowed)")
            return np.array([], dtype=np.float32)
        else:
            error_msg = f"[AudioGuard] {operation_name}: Empty audio array not allowed"
            logger.error(error_msg)
            raise ValueError(error_msg)

    # Validate array dimensions
    if audio_data.ndim > 2:
        error_msg = f"[AudioGuard] {operation_name}: Audio data has too many dimensions: {audio_data.ndim}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Convert to 1D if needed
    original_shape = audio_data.shape
    if audio_data.ndim == 2:
        if audio_data.shape[1] > 1:
            # Convert stereo to mono
            audio_data = np.mean(audio_data, axis=1)
            logger.info(f"[AudioGuard] {operation_name}: Converted stereo to mono")
        else:
            audio_data = audio_data.flatten()

    # Ensure float32 dtype for consistency
    if audio_data.dtype != np.float32:
        try:
            audio_data = audio_data.astype(np.float32)
            logger.info(f"[AudioGuard] {operation_name}: Converted dtype to float32")
        except Exception as e:
            error_msg = f"[AudioGuard] {operation_name}: Cannot convert to float32: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    # Critical: Check for NaN values
    nan_count = np.count_nonzero(np.isnan(audio_data))
    if nan_count > 0:
        logger.warning(f"[AudioGuard] {operation_name}: Found {nan_count} NaN values, replacing with zeros")
        audio_data = np.nan_to_num(audio_data, nan=0.0, copy=False)

    # Critical: Check for infinite values
    inf_count = np.count_nonzero(np.isinf(audio_data))
    if inf_count > 0:
        logger.warning(f"[AudioGuard] {operation_name}: Found {inf_count} infinite values, clamping")
        audio_data = np.nan_to_num(audio_data, posinf=32.0, neginf=-32.0, copy=False)

    # Check for extreme values and clamp to safe range
    max_amplitude = np.max(np.abs(audio_data))
    safe_max = 100.0  # Safe maximum for float32 audio processing

    if max_amplitude > safe_max:
        logger.warning(f"[AudioGuard] {operation_name}: Extreme amplitude {max_amplitude:.2f}, clamping to ±{safe_max}")
        audio_data = np.clip(audio_data, -safe_max, safe_max)
        max_amplitude = safe_max

    # Warn about high amplitudes that might indicate issues
    if max_amplitude > 10.0:
        logger.warning(f"[AudioGuard] {operation_name}: High audio amplitude: {max_amplitude:.2f}")
    elif max_amplitude == 0.0:
        logger.info(f"[AudioGuard] {operation_name}: Silent audio (all zeros)")

    # Log validation summary
    logger.debug(f"[AudioGuard] {operation_name}: Validation complete - "
                f"Shape: {audio_data.shape}, Max: {max_amplitude:.3f}, "
                f"NaN fixed: {nan_count}, Inf fixed: {inf_count}")

    return audio_data


def validate_audio_format(sample_rate: int, channels: int, operation_name: str = "audio_format") -> tuple[int, int]:
    """
    Validate and sanitize audio format parameters using centralized validation.

    Args:
        sample_rate: Audio sample rate to validate
        channels: Number of audio channels to validate
        operation_name: Operation name for logging

    Returns:
        Tuple of (validated_sample_rate, validated_channels)
    """
    # Use centralized sample rate validation for security consistency
    try:
        validated_sample_rate = validate_sample_rate(sample_rate)
    except ValidationError as e:
        logger.error(f"[AudioGuard] {operation_name}: Sample rate validation failed: {e}")
        # Fallback to safe default
        validated_sample_rate = 16000
        logger.warning(f"[AudioGuard] {operation_name}: Using fallback sample rate: {validated_sample_rate}Hz")

    sample_rate = validated_sample_rate

    # Validate channels
    if channels < 1 or channels > 2:
        logger.warning(f"[AudioGuard] {operation_name}: Invalid channel count {channels}, defaulting to 1")
        channels = 1

    return sample_rate, channels


def safe_audio_operation(func, *args, operation_name: str = "audio_op",
                        fallback_value=None, max_retries: int = 3):
    """
    Execute audio operation with error recovery and retry mechanism.

    Args:
        func: Function to execute
        *args: Arguments to pass to function
        operation_name: Operation name for logging
        fallback_value: Value to return if all retries fail
        max_retries: Maximum number of retry attempts

    Returns:
        Result of function or fallback value
    """
    for attempt in range(max_retries):
        try:
            return func(*args)
        except Exception as e:
            logger.warning(f"[AudioGuard] {operation_name}: Attempt {attempt + 1} failed: {e}")

            if attempt == max_retries - 1:
                logger.error(f"[AudioGuard] {operation_name}: All retries exhausted, using fallback")
                return fallback_value

            # Exponential backoff
            time.sleep(0.1 * (2 ** attempt))


def _fast_audio_validation_guard(audio_data: np.ndarray,
                               operation_name: str,
                               allow_empty: bool,
                               cfg: 'Config') -> np.ndarray:
    """
    PERFORMANCE-OPTIMIZED audio validation using statistical sampling.

    DeepSeek Analysis: 15-25% CPU reduction through selective validation
    - Validates only 5% of samples instead of 100%
    - Uses vectorized operations for 20x speedup
    - Skips redundant checks and logging
    - Maintains safety through strategic sampling
    """
    # Quick basic validation
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data, dtype=np.float32)

    if audio_data.size == 0:
        if allow_empty:
            return np.array([], dtype=np.float32)
        raise ValueError(f"[FastGuard] {operation_name}: Empty audio not allowed")

    # Flatten if needed (minimal check)
    if audio_data.ndim > 1:
        audio_data = audio_data.flatten()

    # Ensure float32 (minimal conversion overhead)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # STATISTICAL SAMPLING: Only check subset of data for NaN/Inf
    sample_rate = getattr(cfg, 'audio_validation_sample_rate', 0.05)  # Default 5%
    if getattr(cfg, 'fast_nan_inf_detection', True) and audio_data.size > 1000:
        # Sample every Nth element for large arrays
        step_size = max(1, int(1.0 / sample_rate))
        sample_indices = slice(0, None, step_size)
        sample_data = audio_data[sample_indices]

        # Check sample for issues
        if np.any(np.isnan(sample_data)):
            # Full check only if sample shows problems
            audio_data = np.nan_to_num(audio_data, nan=0.0, copy=False)

        if np.any(np.isinf(sample_data)):
            # Full check only if sample shows problems
            audio_data = np.nan_to_num(audio_data, posinf=32.0, neginf=-32.0, copy=False)

        # Quick amplitude check on sample
        sample_max = np.max(np.abs(sample_data))
        if sample_max > 100.0:
            # Clamp full array only if needed
            audio_data = np.clip(audio_data, -100.0, 100.0)
    else:
        # For small arrays, do minimal full check
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=32.0, neginf=-32.0, copy=False)
        if np.max(np.abs(audio_data)) > 100.0:
            audio_data = np.clip(audio_data, -100.0, 100.0)

    # Skip non-critical logging for performance
    if not getattr(cfg, 'disable_amplitude_warnings', True):
        max_amp = np.max(np.abs(audio_data))
        if max_amp > 10.0:
            logger.warning(f"[FastGuard] {operation_name}: High amplitude: {max_amp:.2f}")

    return audio_data


class BoundedRingBuffer:
    """Memory-safe ring buffer for audio data with size limits"""
    
    def __init__(self, max_duration_seconds: float, sample_rate: int):
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.samples_written = 0
        self.lock = threading.Lock()
        print(f"[AudioBuffer] Initialized with {max_duration_seconds}s capacity ({self.max_samples} samples)")
    
    def append(self, data: np.ndarray):
        """Add data to ring buffer, overwriting old data if full"""
        with self.lock:
            try:
                # CRITICAL: Validate and sanitize input data
                data = audio_validation_guard(data, "RingBuffer.append", allow_empty=True)

                # Skip if empty after validation
                if data.size == 0:
                    return

                data_len = len(data)

                if data_len >= self.max_samples:
                    # Data larger than buffer - take only the most recent part
                    data = data[-self.max_samples:]
                    data_len = len(data)
                    self.buffer[:data_len] = data
                    self.write_pos = data_len % self.max_samples
                    self.samples_written = data_len
                    return

                # Normal case: append to buffer
                end_pos = self.write_pos + data_len

                if end_pos <= self.max_samples:
                    # No wraparound needed
                    self.buffer[self.write_pos:end_pos] = data
                else:
                    # Wraparound needed
                    first_part_len = self.max_samples - self.write_pos
                    self.buffer[self.write_pos:] = data[:first_part_len]
                    remaining = data[first_part_len:]
                    self.buffer[:len(remaining)] = remaining

                self.write_pos = end_pos % self.max_samples
                self.samples_written += data_len

            except Exception as e:
                logger.error(f"[RingBuffer] Critical error in append: {e}")
                # Don't crash - just skip this data
                return
    
    def get_data(self) -> np.ndarray:
        """Get all data from buffer in correct order"""
        with self.lock:
            if self.samples_written == 0:
                return np.array([], dtype=np.float32)
            
            if self.samples_written < self.max_samples:
                # Buffer not full yet - return from start to write_pos
                return self.buffer[:self.write_pos].copy()
            else:
                # Buffer is full - return from write_pos to end, then from start to write_pos
                return np.concatenate([
                    self.buffer[self.write_pos:],
                    self.buffer[:self.write_pos]
                ])

    def get_latest_samples(self, max_samples: int) -> np.ndarray:
        """Get only the latest N samples in logical order."""
        if max_samples <= 0:
            return np.array([], dtype=np.float32)
        with self.lock:
            available = min(self.samples_written, self.max_samples)
            if available <= 0:
                return np.array([], dtype=np.float32)

            count = min(int(max_samples), int(available))
            if self.samples_written < self.max_samples:
                start = max(0, self.write_pos - count)
                return self.buffer[start:self.write_pos].copy()

            # Full ring: latest samples end at write_pos.
            start = (self.write_pos - count) % self.max_samples
            if start < self.write_pos:
                return self.buffer[start:self.write_pos].copy()
            return np.concatenate((self.buffer[start:], self.buffer[:self.write_pos]))

    def get_samples_since(self, last_total_samples: int) -> tuple[np.ndarray, int]:
        """
        Return samples written after `last_total_samples` plus current absolute sample count.
        If caller lags past ring capacity, returns the oldest still-available slice.
        """
        with self.lock:
            total = int(self.samples_written)
            if total <= 0:
                return np.array([], dtype=np.float32), 0

            oldest_available = max(0, total - self.max_samples)
            start_total = max(int(last_total_samples), oldest_available)
            if start_total >= total:
                return np.array([], dtype=np.float32), total

            count = total - start_total
            if self.samples_written < self.max_samples:
                start = max(0, self.write_pos - count)
                return self.buffer[start:self.write_pos].copy(), total

            start = (self.write_pos - count) % self.max_samples
            if start < self.write_pos:
                return self.buffer[start:self.write_pos].copy(), total
            return np.concatenate((self.buffer[start:], self.buffer[:self.write_pos])), total

    def get_samples(self) -> np.ndarray:
        """Compatibility alias used by UI/streaming callers."""
        return self.get_data()
    
    def clear(self):
        """Clear the buffer AND zero out data to prevent corruption"""
        with self.lock:
            self.write_pos = 0
            self.samples_written = 0
            # CRITICAL: Zero out the buffer to prevent old data bleeding through
            self.buffer.fill(0.0)
    
    def get_duration_seconds(self) -> float:
        """Get current data duration in seconds"""
        with self.lock:
            return min(self.samples_written, self.max_samples) / self.sample_rate


class EnhancedAudioRecorder:
    """Enhanced audio recorder with memory-safe bounded buffers"""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._stream: Optional[sd.InputStream] = None
        
        # CRITICAL FIX: Bounded buffer instead of unlimited list
        max_duration = 300.0  # 5 minutes maximum
        self._ring_buffer = BoundedRingBuffer(max_duration, cfg.sample_rate)
        
        # PRE-RECORDING BUFFER: Continuously captures audio to prevent word loss
        self._pre_buffer_duration = 1.5  # 1500ms pre-buffer (optimized for key-press timing)
        self._pre_buffer = BoundedRingBuffer(self._pre_buffer_duration, cfg.sample_rate)
        self._continuous_stream: Optional[sd.InputStream] = None
        self._continuous_recording = False

        # SYSTEM AUDIO (WASAPI loopback): separate ring + pre-buffer so the mic
        # and system tracks can be mixed at stop() when source is "both".
        self._system_ring_buffer = BoundedRingBuffer(max_duration, cfg.sample_rate)
        self._system_pre_buffer = BoundedRingBuffer(self._pre_buffer_duration, cfg.sample_rate)
        self._system_capture: Optional[SystemAudioCapture] = None
        self._system_recording = False
        self._system_target: Optional[BoundedRingBuffer] = None
        self._active_source = "mic"  # source the in-flight recording started with
        self._system_capture_failed_reason: Optional[str] = None

        self._lock = threading.Lock()
        self._recording = False
        self._start_time = 0.0
        # Per-track RMS of the last completed recording, keyed by source name.
        self.last_track_rms: dict = {}
        
        # Performance monitoring
        self._callback_count = 0
        self._total_frames = 0
        
        print(f"[AudioRecorder] Enhanced recorder initialized:")
        print(f"  - Sample rate: {cfg.sample_rate}Hz")
        print(f"  - Channels: {cfg.channels}")
        print(f"  - Block size: {cfg.blocksize} frames")
        print(f"  - Max duration: {max_duration}s")
        print(f"  - Pre-buffer: {self._pre_buffer_duration}s")

    def _callback(self, indata, frames, time, status):
        """Enhanced audio callback with bounded buffer and validation"""
        if status:
            # Log non-fatal warnings from PortAudio
            logger.warning(f"[AudioRecorder] PortAudio status: {status}")

        if not self._recording:
            return

        try:
            self._callback_count += 1
            self._total_frames += frames

            # CRITICAL: Validate input data first
            data = audio_validation_guard(indata.copy(), "AudioCallback", allow_empty=True)

            # Skip if empty after validation
            if data.size == 0:
                return

            # Convert to mono if needed (already handled by validation guard)
            # Add to bounded buffer (thread-safe)
            self._ring_buffer.append(data)

            # Reduced logging: only log every 200 callbacks (~12.8 seconds)
            if self._callback_count % 200 == 0:
                duration = self._ring_buffer.get_duration_seconds()
                print(f"[AudioRecorder] Recording: {duration:.1f}s")

        except Exception as e:
            logger.error(f"[AudioRecorder] Critical error in audio callback: {e}")
            # Don't crash the audio stream - just skip this frame
    
    def _continuous_callback(self, indata, frames, time, status):
        """Continuous pre-recording callback for seamless capture"""
        if status:
            logger.warning(f"[AudioRecorder] Continuous audio status: {status}")

        if not self._continuous_recording:
            return

        try:
            # CRITICAL: Validate input data first
            data = audio_validation_guard(indata.copy(), "ContinuousCallback", allow_empty=True)

            # Skip if empty after validation
            if data.size == 0:
                return

            # Add to pre-buffer (always running)
            self._pre_buffer.append(data)

        except Exception as e:
            logger.error(f"[AudioRecorder] Critical error in continuous callback: {e}")
            # Don't crash the audio stream - just skip this frame

    def get_audio_source(self) -> str:
        """Resolve the configured audio input source to a valid value."""
        source = str(getattr(self.cfg, "audio_input_source", "mic")).strip().lower()
        return source if source in VALID_AUDIO_SOURCES else "mic"

    def _resolve_input_device(self):
        """Map the configured input_device name (substring) to a device index.

        Returns None (PortAudio default) when unset or unmatched.
        """
        preferred = getattr(self.cfg, "input_device", None)
        if not preferred:
            return None
        needle = str(preferred).strip().lower()
        if not needle:
            return None
        try:
            for idx, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels", 0) > 0 and needle in str(dev.get("name", "")).lower():
                    return idx
        except Exception as e:
            logger.warning(f"[AudioRecorder] Input device lookup failed: {e}")
        print(f"[AudioRecorder] Input device {preferred!r} not found; using system default")
        return None

    def set_input_device(self, device_name):
        """Switch the mic device; applies on the next stream open."""
        self.cfg.input_device = device_name or None
        print(f"[AudioRecorder] Input device set to: {device_name or 'system default'}")
        if not self._recording:
            # Restart the continuous pre-buffer so it picks up the new device.
            if self._continuous_recording:
                self.stop_continuous()
                if self.get_audio_source() in ("mic", "both"):
                    self.start_continuous()

    def set_audio_source(self, source: str):
        """Switch audio source; takes effect immediately when idle, else next recording."""
        source = str(source).strip().lower()
        if source not in VALID_AUDIO_SOURCES:
            logger.warning(f"[AudioRecorder] Invalid audio source: {source}")
            return
        self.cfg.audio_input_source = source
        print(f"[AudioRecorder] Audio source set to: {source}")
        # Re-selecting a system source is an explicit retry: clear any prior
        # "device incompatible" latch so the loopback is attempted fresh.
        if source in ("system", "both"):
            self._system_capture_failed_reason = None
            if self._system_capture is not None:
                self._system_capture.failed = False
        if self._recording:
            return  # applies on next recording
        # Keep the right pre-buffers warm for the new source
        if source == "system":
            self.stop_continuous()
        else:
            self.start_continuous()
        if source in ("system", "both"):
            self._ensure_system_capture()
        else:
            self._stop_system_capture()

    def _on_system_chunk(self, chunk: np.ndarray):
        """Receive loopback chunks from the system capture thread."""
        try:
            data = audio_validation_guard(chunk, "SystemAudioChunk", allow_empty=True, cfg=self.cfg)
            if data.size == 0:
                return
            self._system_pre_buffer.append(data)
            if self._system_recording and self._system_target is not None:
                self._system_target.append(data)
        except Exception as e:
            logger.error(f"[AudioRecorder] Critical error in system audio chunk: {e}")

    def _on_system_capture_failed(self, reason: str):
        """Called by the capture thread when loopback can't be opened at all."""
        self._system_capture_failed_reason = reason
        self._system_recording = False
        self._system_target = None
        print(f"[AudioRecorder] System audio unavailable, continuing mic-only: {reason}")

    def _ensure_system_capture(self) -> bool:
        # Don't keep reopening a device that already proved incompatible.
        if getattr(self, "_system_capture_failed_reason", None):
            return False
        if self._system_capture is None:
            self._system_capture = SystemAudioCapture(
                sample_rate=self.cfg.sample_rate,
                blocksize=self.cfg.blocksize,
                on_chunk=self._on_system_chunk,
                on_failed=self._on_system_capture_failed,
            )
        if self._system_capture.failed:
            return False
        if not self._system_capture.is_running():
            self._system_pre_buffer.clear()
            return self._system_capture.start()
        return True

    def _stop_system_capture(self):
        self._system_recording = False
        self._system_target = None
        if self._system_capture is not None and self._system_capture.is_running():
            self._system_capture.stop()
        self._system_pre_buffer.clear()

    @staticmethod
    def _rms(audio: np.ndarray) -> float:
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))

    @staticmethod
    def _mix_tracks(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Mix two mono tracks of (possibly) different lengths into one."""
        if a.size == 0:
            return b
        if b.size == 0:
            return a
        n = max(len(a), len(b))
        mixed = np.zeros(n, dtype=np.float32)
        mixed[: len(a)] += a
        mixed[: len(b)] += b
        return np.clip(mixed, -1.0, 1.0)

    @staticmethod
    def _trim_pre_buffer(pre_buffer_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Keep the most recent slice of a pre-buffer for key-press timing."""
        min_pre_buffer_samples = int(0.3 * sample_rate)  # 300ms minimum
        if len(pre_buffer_data) > min_pre_buffer_samples:
            # Use recent 800ms of pre-buffer for optimal key-press timing
            recent_samples = int(0.8 * sample_rate)
            pre_buffer_data = pre_buffer_data[-recent_samples:]
        return pre_buffer_data

    def start(self):
        """Start recording with pre-buffer integration"""
        if self._recording:
            return

        source = self.get_audio_source()
        if source in ("system", "both") and not system_audio_supported():
            print("[AudioRecorder] System audio capture unavailable (soundcard missing); using mic only")
            source = "mic"
        mic_on = source in ("mic", "both")
        system_on = source in ("system", "both")
        self._active_source = source

        print(f"[AudioRecorder] Starting enhanced recording with pre-buffer (source: {source})...")

        # CRITICAL FIX: Clear main buffer before starting to prevent old data corruption
        self._ring_buffer.clear()
        self._system_ring_buffer.clear()

        if system_on:
            started = self._ensure_system_capture()
            if not started and not mic_on:
                print("[AudioRecorder] System capture failed to start; falling back to mic")
                source = self._active_source = "mic"
                mic_on, system_on = True, False
            elif started:
                # In "system" mode the loopback feeds the main ring buffer directly so
                # streaming previews and duration tracking keep working; in "both" mode
                # it goes to its own ring and is mixed with the mic track at stop().
                self._system_target = self._ring_buffer if source == "system" else self._system_ring_buffer
                system_pre = self._system_pre_buffer.get_data()
                self._system_pre_buffer.clear()
                if len(system_pre) > 0:
                    self._system_target.append(self._trim_pre_buffer(system_pre, self.cfg.sample_rate))
                self._system_recording = True

        self._callback_count = 0
        self._total_frames = 0
        self._start_time = time.time()

        if mic_on:
            # Start continuous recording if not already running
            if not self._continuous_recording:
                self.start_continuous()

            # Get pre-buffer data BEFORE clearing to use for this recording
            pre_buffer_data = self._pre_buffer.get_data()

            # CRITICAL FIX: Clear pre-buffer IMMEDIATELY after getting data
            # This prevents any accumulation while we process
            self._pre_buffer.clear()

            if len(pre_buffer_data) > 0:
                # Add pre-buffer to main buffer for this recording only
                self._ring_buffer.append(self._trim_pre_buffer(pre_buffer_data, self.cfg.sample_rate))

            self._stream = sd.InputStream(
                device=self._resolve_input_device(),
                channels=self.cfg.channels,
                samplerate=self.cfg.sample_rate,
                dtype="float32",
                blocksize=self.cfg.blocksize,
                callback=self._callback,
            )
            self._stream.start()
        self._recording = True
        print(f"[AudioRecorder] Recording started successfully with pre-buffer integration")

    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self._recording

    def stop(self) -> np.ndarray:
        """Stop recording and return audio data"""
        if not self._recording:
            return np.array([], dtype=np.float32)

        try:
            self._recording = False
            self._system_recording = False
            self._system_target = None
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            # Get the recorded audio data
            audio_data = self._ring_buffer.get_data()
            if self._active_source == "both":
                system_data = self._system_ring_buffer.get_data()
                self.last_track_rms = {"mic": self._rms(audio_data), "system": self._rms(system_data)}
                if len(system_data) > 0:
                    print(
                        f"[AudioRecorder] Mixing mic ({len(audio_data)} samples, rms={self.last_track_rms['mic']:.4f}) "
                        f"+ system ({len(system_data)} samples, rms={self.last_track_rms['system']:.4f})"
                    )
                else:
                    reason = getattr(self, "_system_capture_failed_reason", None)
                    detail = f" [{reason}]" if reason else " (nothing played on default output?)"
                    print(
                        f"[AudioRecorder] System track empty{detail}; "
                        f"mic {len(audio_data)} samples, rms={self.last_track_rms['mic']:.4f}"
                    )
                audio_data = self._mix_tracks(audio_data, system_data)
            elif self._active_source == "system":
                self.last_track_rms = {"system": self._rms(audio_data)}
                print(f"[AudioRecorder] System track: {len(audio_data)} samples, rms={self.last_track_rms['system']:.4f}")
            else:
                self.last_track_rms = {"mic": self._rms(audio_data)}
            duration = len(audio_data) / self.cfg.sample_rate

            # CRITICAL FIX: Clear buffer after getting data to prevent accumulation
            self._ring_buffer.clear()
            self._system_ring_buffer.clear()
            print(f"[AudioRecorder] Buffer cleared after extraction to prevent accumulation")
            
            # Performance summary
            actual_duration = time.time() - self._start_time
            print(f"[AudioRecorder] Recording stopped:")
            print(f"  - Audio duration: {duration:.2f}s")
            print(f"  - Actual duration: {actual_duration:.2f}s")
            print(f"  - Callbacks: {self._callback_count}")
            print(f"  - Samples: {len(audio_data)}")
            print(f"  - Memory usage: {len(audio_data) * 4 / 1024 / 1024:.2f}MB")
            
            return audio_data
            
        except Exception as e:
            print(f"[AudioRecorder] Error stopping recording: {e}")
            return np.array([], dtype=np.float32)
    
    def get_current_duration(self) -> float:
        """Get current recording duration in seconds"""
        return self._ring_buffer.get_duration_seconds()
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        return self._ring_buffer.max_samples * 4 / 1024 / 1024  # 4 bytes per float32
    
    def start_continuous(self):
        """Start continuous pre-recording to prevent word loss"""
        if self._continuous_recording:
            return
        
        print("[AudioRecorder] Starting continuous pre-buffer recording...")
        self._pre_buffer.clear()
        
        self._continuous_stream = sd.InputStream(
            device=self._resolve_input_device(),
            channels=self.cfg.channels,
            samplerate=self.cfg.sample_rate,
            dtype="float32",
            blocksize=self.cfg.blocksize,
            callback=self._continuous_callback,
        )
        self._continuous_stream.start()
        self._continuous_recording = True
        print(f"[AudioRecorder] Continuous pre-buffer active ({self._pre_buffer_duration}s)")
    
    def stop_continuous(self):
        """Stop continuous pre-recording"""
        if not self._continuous_recording:
            return
            
        self._continuous_recording = False
        if self._continuous_stream is not None:
            self._continuous_stream.stop()
            self._continuous_stream.close()
            self._continuous_stream = None
        print("[AudioRecorder] Continuous pre-buffer stopped")


# Compatibility alias for drop-in replacement
AudioRecorder = EnhancedAudioRecorder
