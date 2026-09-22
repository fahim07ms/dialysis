# Dialysis — DTMF Signal Analysis Studio

An interactive desktop app for generating, visualizing, and decoding DTMF
(touch-tone) signals. Built with PyQt6 + pyqtgraph, styled after a Pixel-style
dark/teal quick-settings panel.

## Run it

```bash
pip install -r requirements.txt
python main.py
```

Linux users: `sounddevice` needs PortAudio at the OS level —
`sudo apt install libportaudio2` if it complains on startup.

## Project structure

```
main.py                     entry point: loads fonts, applies theme, launches window
core/
  dsp_interface.py            ALL signal-processing math lives here. No PyQt imports.
  utils.py                    small helpers (bit-reversal, zero-padding) used by the custom FFT
  audio_io.py                 device I/O: WAV read/write, speaker playback, MicStream (mic capture)
ui/
  theme.py                    colors + QSS stylesheet (dark/teal Pixel-style theme)
  keypad.py                   4x4 keypad widget
  left_panel.py                keypad, text entry, Play/Export/Import, live playback timer
  decoder_panel.py            algorithm picker, Decode button, live mic decoding — lives in the sidebar
  main_window.py               assembles left + right panel into the split window
  right_panel.py               tabbed analysis area
  tabs/
    waveform_tab.py            time-domain waveform + magnitude spectrum, live-updating during playback
    spectrogram_tab.py         time/frequency/intensity image, background-threaded
assets/fonts/                 bundled Nunito variable font
```

See **DSP_REPORT.md** for a full write-up of every function in
`core/dsp_interface.py` — what it does, the math behind it, and why it's
built the way it is.

## The contract

- `core/` never imports PyQt. Pure numpy/Python in, numpy/Python out — fully
  testable from a terminal with no GUI running.
- `ui/` never does math. It calls `core/dsp_interface.py` and displays
  whatever comes back, catching `NotImplementedError` gracefully wherever a
  phase isn't built yet instead of crashing.

## What's done

**Phase 0 — App skeleton.** Two-pane splitter layout, dark/teal QSS theme,
bundled Nunito font.

**Phase 1 — Tone synthesis.** `generate_dtmf_tone()`. Keypad taps and typed
sequences play immediately.

**Phase 2 — WAV export.** `sequence_to_wav()`. Export button writes a real
`.wav` file.

**Phase 3 — Spectrum analysis.** A hand-rolled iterative radix-2 Cooley-Tukey
FFT (`fft()`, with `fft_recursive()` as a reference/alternate implementation)
backs `compute_spectrum()`. Verified against `numpy.fft` to ~1e-7 accuracy.
The Waveform tab shows both the time-domain signal and the magnitude
spectrum, **live-updating during playback** (bounded/throttled — see
"Performance notes" below) and settling on the full, complete analysis the
instant playback finishes.

**Phase 4 — Segmentation.** `segment_tone_regions()` uses RMS energy over
10ms frames with a relative threshold, merging consecutive loud frames into
clean `(start, end)` tone regions. The Waveform tab overlays each detected
region as a shaded band.

**Phase 5 — Goertzel decoder.** `goertzel_single_freq()` + `goertzel_decode()`.
Verified against all 16 keypad digits and against a real 44.1kHz recording
(`freesound_community-dtmf_dial-105992.wav`) — decodes it perfectly.

**Phase 6 — FFT decoder.** `fft_decode()` — same job as Goertzel, via
spectrum peak-picking instead. Both algorithms agree on every test signal
tried so far, including the real recording. Selectable from the same
dropdown in the Decoder panel.

**Phase 7 — Spectrogram.** `compute_spectrogram()` via `scipy.signal.spectrogram`.
Rendered as a themed (teal, not rainbow) image, computed on a background
thread so a long recording can't freeze the window.

**UI layout.** The Decoder (algorithm picker, Decode button, result display)
was moved from its own tab into the left sidebar, right next to whatever
produced the signal — no tab-switching needed to decode something you just
played or imported.

**Live microphone decoding.** `core/audio_io.py`'s `MicStream` captures
continuously; `decoder_panel.py` orchestrates a rolling buffer, a
"safely-closed" check (won't trust a tone as finished without a trailing
silence margin), and a **confidence gate** (`_is_valid_dtmf_region`) that
rejects noise before it can masquerade as a digit — built entirely from
already-exposed `dsp_interface.py` primitives, no new core math needed.
No mic hardware exists in the dev sandbox this was built in, so it's been
validated with simulated real-time streaming and real recorded audio;
**real-hardware testing surfaced two live bugs, both since fixed**: an
`Invalid input sample format` error (fixed by requesting `float32` from the
device instead of `float64`, which is what real drivers actually support),
and background noise getting misread as digits (fixed by the confidence
gate above — measured empirically: pure noise never exceeds ~0.58
"dominance" across many trials, real tones never drop below ~0.99).

## Performance notes (read this if something feels slow)

Two real freezes were found and fixed during development, both from the
same root cause: the hand-written FFT is `O(N log N)` but in pure Python
loops, so calling it on a *large* array is genuinely slow (measured: ~4
seconds on the full 10-second real recording).

- **Live view** (during playback/mic listening): only ever runs
  `compute_spectrum()` on a small bounded trailing window, throttled to
  every other timer tick — not the whole growing signal.
- **Full analysis** (once playback/import finishes): runs on a background
  `QThread` so a long recording can't freeze the window, however long the
  computation takes.

If you add new analysis that calls `compute_spectrum()`, `segment_tone_regions()`,
or anything else that scales with signal length, keep this pattern in mind —
bound the input size for anything that runs on a timer, and background-thread
anything that runs once on a potentially-large complete signal.

## What's next

**Phase 8 — Pole-Zero visualizer.** `goertzel_poles()` — not implemented.
Needs the pole locations of the Goertzel resonator on the Z-plane for each
DTMF frequency (a 2nd-order recursive filter → a conjugate pole pair on the
unit circle per frequency). UI (Z-plane plot, unit circle, resonance
markers) not built yet either.

**Phase 9 — Impulse Response & Convolution.** `impulse_response()` and
`convolve_signals()` — not implemented. UI (h[n] plot + animated
convolution step-through) not built yet.

**Phase 10 — Noise Injector & Variable Sampling Rate.** `add_awgn_noise()`
and `resample_signal()` — not implemented. UI (AWGN slider, sample-rate
slider, live aliasing demo) not built yet.

**Phase 11 — Benchmark Analyzer.** `benchmark_decoders()` — not implemented.
UI (side-by-side FFT vs Goertzel timing/memory/accuracy) not built yet. The
Decoder panel already demonstrates the timing gap informally (Goertzel
~12ms vs FFT ~55ms on the same 6-digit signal) — this phase formalizes it.

## Known tuning knobs (things that may need adjusting on real hardware)

- `segment_tone_regions()`'s threshold (`0.2 * max(rms)`) is purely
  *relative* to the loudest frame in whatever signal it's given. It works
  well on clean synthetic/file audio. On a genuinely noisy live mic, an
  **absolute floor** (`threshold = max(0.2 * max(rms), MIN_ABSOLUTE_RMS)`)
  may help — not yet added, since the right floor value depends on your
  specific microphone's gain and hasn't been calibrated against real
  hardware.
- `MIC_SAFETY_MARGIN_SECONDS` (currently 0.15s, in `decoder_panel.py`) is
  how long a silence gap must be before a tone is trusted as "finished."
  If real dialing has shorter gaps than this, digits could get missed —
  worth tuning against how you actually dial.
- `MIC_MIN_DOMINANCE` (currently 0.75) is the confidence-gate threshold.
  Measured with a wide safety margin (noise tops out ~0.58, real tones stay
  above ~0.99), but real-world room acoustics may shift these numbers.
