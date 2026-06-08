# Issues - Pending Items

## Pending Items

_(none)_

---

## Completed Items

### Soniox backend: poor transcription quality on Windows (RESOLVED 2026-06-09)

**Symptom:** Push-to-talk transcription via the Soniox cloud backend was erratic and
low quality — sentences came back as short fragments, wrong words, or empty — while
the Soniox web playground on the *same laptop and microphone* was excellent. Earlier,
the previous clipboard contents were pasted instead of the transcript.

**Root causes found (in order of discovery) and fixes:**

1. **Clipboard paste did nothing / pasted old clipboard.**
   - Cause: `pyperclip` was not installed in the runtime environment; `inject.py`
     silently falls back to a no-op clipboard stub, so every Ctrl+V pasted stale
     clipboard contents.
   - Fix: installed `pyperclip` (already declared in `pyproject.toml`).

2. **Audio shredded by PortAudio input overflow (primary quality killer).**
   - Cause chain:
     - Two simultaneous `sd.InputStream`s (continuous pre-buffer + main recording)
       competed for the mic device.
     - The capture callback ran the heavy `audio_validation_guard` (NaN/Inf scan)
       on every 512-frame block inside the real-time PortAudio thread.
     - Small block size + low latency gave no slack for GIL stalls from VoiceFlow's
       many background threads (overlay, tray, monitoring).
     - Result: tens of thousands of `input overflow` warnings; mic audio dropped and
       spliced, so a 5 s sentence arrived as ~1 s of disjointed audio.
   - Fixes (in `core/audio_enhanced.py`):
     - Disabled the redundant continuous pre-buffer stream; only the main recording
       stream captures the mic (single stream, no contention).
     - Removed the heavy validation from the real-time callback — it now does the
       minimum (`ring_buffer.append(indata.copy())`); validation happens downstream.
     - Enlarged the buffer: `blocksize=1600` (100 ms) and explicit `latency=0.5`
       to absorb callback stalls.
     - Result: input overflow → **0**; dropout splices eliminated.

3. **Quiet mic level + distorting normalization.**
   - Cause: the laptop mic is quiet (peak ~0.2). VoiceFlow's Whisper-oriented
     `AudioPreprocessor` RMS-normalized every clip to exactly 0.1 and **hard-clipped
     peaks to 1.0**, distorting the signal Soniox needs.
   - Fix (in `ui/cli_enhanced.py`): for the Soniox backend, bypass the Whisper
     preprocessing and instead apply a clean **peak normalization** — scale so the
     loudest sample reaches ~0.95 with no clipping (gain capped at 8×).

4. **Wrong Soniox endpoint/model (fixed earlier in the port).**
   - Correct endpoint `wss://stt-rt.soniox.com/transcribe-websocket`, model
     `stt-rt-v4`, `language_hints: ["el","en"]`, `audio_format pcm_s16le`,
     `sample_rate 16000`, mono. Verified on the wire.

**Reverted dead-end:** Selecting an explicit WASAPI input device index proved
unstable in-process (WDM-KS / USB enumeration errors, `Invalid sample rate -9997`),
so capture stays on the system default device. The overflow was instead solved by
the minimal-callback + large-buffer changes above.

**Verification:** Greek test sentence "Ο λαγός πήγε να κοιμηθεί γιατί είναι αργά το
βράδυ." transcribed correctly 3/3 times. Dumped-audio analysis: overflow=0, peak=0.95,
no clipping, voiced fraction restored, dropout splices gone.

### Dependency vetting log
- 2026-06-09: `pyperclip` (>=1.8.0, already in pyproject) installed into runtime env. No advisories.
- `websockets>=12.0` added for the Soniox WebSocket backend.
