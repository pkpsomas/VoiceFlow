# Dual‑channel transcription (mic + incoming audio) on Windows — research & upgrade options

> **Status: Options A + B implemented.** `soundcard` was replaced with **PyAudioWPatch**
> for WASAPI loopback (fixes the call‑mode crash at its root), and `both` mode now keeps
> the mic ("You") and incoming ("Them") tracks **separate**, transcribing each and injecting
> a labeled `You:` / `Them:` transcript. Option C (VB‑CABLE fallback) remains optional/manual.


Goal: transcribe **both** sides of a conversation at once — your Plantronics headset
**microphone** (you) and the **incoming/other‑party audio** (what you hear) — ideally
labeled by who spoke.

This document covers (1) why the current approach breaks, (2) the Windows audio‑capture
options that actually work, (3) existing GitHub projects that deliver this, and (4) a
concrete upgrade plan for VoiceFlow.

---

## 1. Why the current `both` mode fails

VoiceFlow captures incoming audio with the **`soundcard`** library's WASAPI loopback.
On this machine it throws a bare `AssertionError` (`assert wFormatTag == 0xFFFE`) the
moment the headset enters **communications/call mode** — exactly when you're on a call.
`soundcard` parses the Windows `WAVEFORMATEXTENSIBLE` header in Python and asserts the
endpoint is in extensible format; a headset in call mode exposes a plain PCM shared mix
format, so the assert fails and no incoming audio is ever captured. (Fixed in commit
`e091917` only to the extent that it no longer crash‑loops — it now degrades to mic‑only.)

The real fix is a **different capture method** for the render (output) endpoint.

---

## 2. Windows capture options for the incoming channel (the crux)

| Method | Reliability on headsets in call mode | User install? | Keeps streams separate? | Notes |
|---|---|---|---|---|
| **`soundcard`** (current) | ✗ breaks on PCM mix format | none | n/a (we mix) | Python‑side format assert is the failure |
| **PyAudioWPatch** (PortAudio fork, WASAPI loopback) | ✓ much better — native PortAudio negotiates the format, no Python‑side assert | none (pip wheel) | yes | Best pure‑software upgrade; wheels cover Py 3.9–3.15 incl. your 3.13 |
| **Virtual audio cable** (VB‑CABLE / VoiceMeeter) | ✓✓ bulletproof — fixed format, normal capture endpoint | yes (driver) | yes | What serious call‑recording setups use; needs routing config |
| **Stereo Mix** | ✗ usually disabled / absent on modern laptops | n/a | no | Legacy, not dependable |

**Key architectural point:** don't **mix** mic + incoming into one track (what VoiceFlow
does today). Capture them as **two separate, labeled streams** ("You" / "Speaker") and
transcribe each. That's how the mature projects do it, and it's what gives you a usable
two‑party transcript instead of an overlapping blur.

**PyAudioWPatch caveat:** because it's a native PortAudio backend rather than Python code
parsing the format header, it avoids the exact `soundcard` assert — but no library fully
escapes the reality that a headset *in call mode* may still refuse loopback. The
virtual‑cable route is the only one that is format‑independent. Recommended: ship
PyAudioWPatch as the default and **document VB‑CABLE as the fallback for calls**.

---

## 3. GitHub projects that already deliver this (Windows)

| Project | What it does | Capture | Streams | Backends | License / activity |
|---|---|---|---|---|---|
| **vivekuppal/transcribe** | Real‑time mic + speaker transcription, optional GPT reply suggestions. Closest off‑the‑shelf match. | Windows speaker **loopback** | **Separate & labeled** `You` / `Speaker`; choose mic / speaker / **both** | Local Whisper, OpenAI Whisper, **OpenAI Realtime**, Deepgram | MIT · 269★ · active (fork of ecoute) |
| **SevaSk/ecoute** | The original lighter tool: live mic + speaker transcript + GPT replies | Speaker loopback (WASAPI) | Both, default mic+speaker | Local Whisper tiny, OpenAI Whisper API | MIT · popular but less maintained |
| **Zackriya‑Solutions/meeting‑minutes (Meetily)** | Self‑hosted meeting assistant, live notes + local summaries | Select mic **and/or** system audio | Meeting‑oriented | Whisper.cpp local, Ollama summaries | Rust app, Win/macOS, active |
| **OpenWhispr** (LTMCEO / Knuckles92 forks) | Dictation + "Meeting Mode" capturing mic + system audio | WASAPI system capture | Meeting mode | Local Whisper / NVIDIA Parakeet | Check which fork; offline |
| **PyAudioWPatch** (s0d3s) | *Library*, not an app — the WASAPI loopback capture layer the others should use | — | — | — | Apache‑2.0 · maintained (0.2.12.9, Oct 2026) |

**Takeaways**
- **vivekuppal/transcribe** is the single best reference/alternative: MIT, Windows, both
  channels **labeled**, and it already supports an **OpenAI Realtime** streaming backend
  (same category as your Soniox integration). If you'd rather not build this, it's the
  drop‑in off‑the‑shelf option.
- Everyone doing this on Windows converges on **WASAPI loopback via a PortAudio‑based
  layer** (PyAudioWPatch) or a **virtual cable** — nobody relies on `soundcard` for calls.

---

## 4. Upgrade options for VoiceFlow

Three tiers, cheapest first. They stack — you can do A, then B, then C.

### Option A — Swap the loopback library (fixes the crash, smallest change)
- Replace `soundcard` with **PyAudioWPatch** in `src/voiceflow/core/system_audio.py`.
- Same "capture whatever plays on the default output" model; no user install; pip wheel
  exists for Python 3.13.
- Keeps the current mix‑at‑stop behavior, so the rest of the pipeline is untouched.
- **Effort:** rewrite one file (~the `_run` capture loop) + swap the dependency.
- **Limitation:** still one mixed track (overlapping voices); still at the mercy of the
  headset's call‑mode format, though far more robust than `soundcard`.

### Option B — Two labeled streams (the real feature you asked for)
- Capture mic and loopback **separately** (don't mix). You already have separate ring
  buffers (`_ring_buffer`, `_system_ring_buffer` in `audio_enhanced.py`) — stop mixing
  them at `stop()` and instead transcribe each.
- Transcribe both and inject/label as `You:` / `Them:` (or write to the history panel as
  two columns). This mirrors vivekuppal/transcribe's proven design.
- With **Soniox** (already integrated in `SonioxBackend`): send the two tracks as a
  **2‑channel** request or two requests; Soniox returns per‑token language *and speaker*,
  so you get proper attribution natively — a strong fit for your mixed Greek+English case.
- **Effort:** moderate — pipeline change in `audio_enhanced.py` + `cli_enhanced.py`
  transcription dispatch + a small UI change to show two speakers.

### Option C — Virtual‑cable fallback for calls (max reliability)
- Document/support **VB‑CABLE** (free) or **VoiceMeeter**: route the call app's output to
  the virtual device, then capture it as an ordinary input (fixed format — immune to the
  headset call‑mode problem).
- Add a tray option "Incoming audio device" so the user can point capture at the virtual
  cable instead of loopback.
- **Effort:** low code (reuse the existing `input_device` selection pattern), but requires
  the user to install a driver — hence a fallback, not the default.

### Recommended path
**A + B together:** move to **PyAudioWPatch** *and* switch to **two labeled streams**,
with **Soniox** doing per‑speaker attribution. Keep **C (VB‑CABLE)** documented as the
"calls that still won't cooperate" escape hatch. This gives you a real two‑party transcript
("You" vs "Them"), fixes the crash at its root, and plays to the Soniox backend you already
have working.

---

## Sources
- [vivekuppal/transcribe](https://github.com/vivekuppal/transcribe) — mic+speaker, labeled streams, mic/speaker/both
- [SevaSk/ecoute](https://github.com/SevaSk/ecoute) — original live mic+speaker tool
- [PyAudioWPatch (s0d3s)](https://github.com/s0d3s/PyAudioWPatch) · [PyPI](https://pypi.org/project/PyAudioWPatch/) — WASAPI loopback PortAudio fork
- [Meetily (Zackriya‑Solutions/meeting‑minutes)](https://github.com/Zackriya-Solutions/meeting-minutes) — self‑hosted meeting assistant
- [OpenWhispr](https://github.com/LTMCEO/openwhispr) — dictation + meeting mode
- [Microsoft: application/loopback audio capture](https://learn.microsoft.com/en-us/samples/microsoft/windows-classic-samples/applicationloopbackaudio-sample/) — background on WASAPI loopback
