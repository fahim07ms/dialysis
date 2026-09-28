# Dialysis — DTMF Signal Analysis Studio

Dialysis is an interactive desktop application for the synthesis, visualization,
and decoding of Dual-Tone Multi-Frequency (DTMF) signals. It implements a complete
transmit-and-receive signal processing pipeline: tone generation, WAV export,
recording analysis, real-time microphone decoding, and a set of visualizations
that explain the signal-processing theory behind each stage.

The application is written in Python using PyQt6 and pyqtgraph. All core
algorithms (FFT, Goertzel, convolution, and the decoding engine) are implemented
in the project itself rather than delegated to library routines.

---

## 1. Installation and Usage

```bash
pip install -r requirements.txt
python main.py
```

On Linux, `sounddevice` requires the PortAudio library at the operating-system
level:

```bash
sudo apt install libportaudio2
```

## 2. Features

### Signal generation and file handling
- A 4 × 4 virtual keypad and keyboard entry generate the composite DTMF signal
  `x[n] = sin(2πf₁n/fₛ) + sin(2πf₂n/fₛ)` across the low group (697–941 Hz) and
  high group (1209–1633 Hz), with immediate audio playback.
- Multi-digit sequences can be exported to uncompressed WAV files.
- External WAV recordings can be imported, played back, and decoded. The name of
  the loaded file is displayed in the interface.

### Analysis views
- **Waveform and magnitude spectrum.** The time-domain signal and its spectrum
  update continuously during playback and settle on the complete analysis when
  playback ends. The eight DTMF frequencies are marked on the spectrum, and
  detected tone regions are shaded on the waveform.
- **Spectrogram.** A time–frequency intensity plot with a 60 dB dynamic range,
  smooth interpolation, and optional DTMF frequency marker lines. It also updates
  from the live microphone stream.
- **Pole-zero visualizer.** An interactive Z-plane plot of the Goertzel
  resonators, with a frequency selector and the magnitude response of the
  selected resonator, illustrating its selectivity.
- **Impulse response and convolution.** Displays the impulse response `h[n]` of a
  resonator tuned to a chosen DTMF frequency and its convolution
  `y[n] = x[n] * h[n]` with the current signal.
- **Noise and sampling-rate controls.** Sliders for signal-to-noise ratio
  (additive white Gaussian noise) and sampling rate demonstrate how decoding
  degrades with noise and how it fails through aliasing when the rate falls below
  the Nyquist requirement (`fₛ > 3266 Hz`). The processed signal may be exported
  as a WAV file.
- **Benchmark.** A side-by-side comparison of the FFT and Goertzel decoders in
  execution time, peak memory, and agreement of results.

### Decoding
- **Goertzel decoder** and **FFT decoder**, selectable from the Decoder tab and
  producing the same digit sequence on the same input.
- **Live microphone decoding.** Digits are decoded from an open microphone in real
  time, with live per-frequency power bars, the noise floor and gate level, the
  reason for any rejected detection, an adjustable detection threshold, and an
  option to save the captured recording for later inspection.

---

## 3. Project Structure

```
main.py                        Application entry point (fonts, theme, window)
core/
  dsp_interface.py             All signal-processing algorithms; no PyQt imports
  utils.py                     Bit-reversal and zero-padding helpers for the FFT
  signal_lti.py                Reference discrete-signal and LTI-system classes
  dsp_interface_selftest.py    Cross-check of the convolution against signal_lti.py
  audio_io.py                  Device I/O: playback, WAV read/write, microphone capture
ui/
  theme.py                     Colour palette and stylesheet
  keypad.py                    Keypad widget
  left_panel.py                Keypad, text entry, playback, import and export
  decoder_panel.py             Decoder tab and live microphone interface
  right_panel.py               Tabbed analysis area
  main_window.py               Window layout and signal wiring
  tabs/
    waveform_tab.py            Waveform and magnitude spectrum
    spectrogram_tab.py         Spectrogram
    pole_zero_tab.py           Z-plane visualizer
    convolution_tab.py         Impulse response and convolution
    noise_sampling_tab.py      Noise and sampling-rate controls
    benchmark_tab.py           FFT and Goertzel benchmark
assets/fonts/                  Bundled Nunito font
DSP_REPORT.md                  Detailed description of the signal-processing module
```

### Architectural principle

The signal-processing layer and the user interface are strictly separated.

- `core/` does not import PyQt. Its functions accept and return plain NumPy
  arrays, numbers, strings, and dictionaries, so they can be tested from a
  terminal without the interface.
- `ui/` performs no signal-processing computation. It calls `core/` and displays
  the results.

---

## 4. Signal-Processing Methods

### Tone synthesis
Each key is generated as the sum of one row frequency and one column frequency.
Sequences are built by concatenating tones separated by silent gaps, so that
consecutive digits remain distinguishable.

### Fast Fourier Transform
The FFT is an iterative radix-2 Cooley–Tukey implementation with bit-reversal
reordering, written from scratch. Its output was verified against `numpy.fft`
to a precision of approximately 10⁻⁷. It provides the magnitude spectrum used in
the visualizations.

### Goertzel algorithm
The Goertzel recurrence `s[n] = x[n] + 2cos(ω)s[n−1] − s[n−2]` is a second-order
resonator with a conjugate pole pair on the unit circle at `e^{±jω}`. Evaluating it
over a block yields the signal power at a single target frequency, so only the
eight DTMF frequencies need to be measured. The classical per-sample form is
retained as a reference implementation, and a mathematically equivalent
vectorized form is used inside the decoding engine.

### Decoding engine
Both the file decoders and the live decoder use one shared frame-based engine
(`DtmfTracker`, with `decode_file_events` for whole recordings):

1. The signal is divided into overlapping frames of 20 ms with a 5 ms hop.
2. The power at each of the eight DTMF frequencies is measured per frame.
3. A frame is treated as tone-like if its strongest row–column pair exceeds an
   adaptive noise floor by a configurable margin (10 dB by default) and if that
   pair holds a sufficient share of the total power across the eight bins
   (dominance of at least 0.60). The noise floor may fall immediately but rises
   only slowly, so tones do not raise it.
4. A digit is latched after six consecutive agreeing frames and released after
   four disagreeing frames, so a repeated digit is reported twice while a brief
   dip inside a tone does not split it.
5. At the moment of latching, the digit is chosen by the selected algorithm
   (Goertzel bin selection, or FFT peak picking with a Hann window, zero-padding,
   and parabolic interpolation), and the detection is validated against
   ITU-T Q.24-style criteria: dial-tone rejection, limits on the level difference
   ("twist") between the two tones, frequency-offset tolerance, and second-harmonic
   rejection. Rejected detections are reported with their reason.

Because both algorithms share the same detection and validation stages, the
benchmark compares the two spectral measurements on identical detections.

### Convolution
Convolution is implemented from first principles using the superposition
principle: the output is the sum of shifted and scaled copies of one signal,
one for each sample of the other. The implementation is vectorized and its
output is verified against the reference classes in `core/signal_lti.py` and
against `numpy.convolve` by `core/dsp_interface_selftest.py`. The reference
classes are correct but scale approximately quadratically with signal length
and are therefore used for verification rather than for runtime processing.

### Noise and resampling
Noise is added at a specified signal-to-noise ratio by scaling zero-mean
Gaussian noise relative to the measured signal power. Resampling is performed with
`scipy.signal.resample`.

### Benchmarking
Each decoder is preceded by a warm-up run so that one-time setup costs are not
charged to either. Execution time is the minimum of three runs, and peak memory
is measured with `tracemalloc`.

---

## 5. Live Microphone Capture

- The microphone is opened at the input device's native sampling rate rather than
  a forced rate; the decoding engine adapts to whatever rate the device provides.
- A one-pole DC blocker (cut-off approximately 80 Hz) removes microphone offset
  without affecting the DTMF band.
- A bounded queue with drop-oldest behaviour prevents unbounded memory growth,
  and any dropped audio is counted and reported so the user is informed.
- Samples are requested from the device as 32-bit floating point, which is widely
  supported, and converted to 64-bit for processing.

---

## 6. Performance Considerations

The hand-written FFT is `O(N log N)` but is executed in Python, so applying it to
a long signal is comparatively slow (approximately four seconds for a ten-second
recording at 44.1 kHz). The interface therefore observes the following rules:

- **Live views** compute the spectrum only over a small, bounded trailing window
  and only on a subset of timer ticks, so the cost is independent of how long
  playback or listening continues.
- **Whole-signal analyses** (full spectrum, segmentation, spectrogram,
  convolution, noise processing, and benchmarking) run on background threads,
  and each result is tagged with a request identifier so that a stale result
  cannot overwrite a newer one.
- **Plots** of long signals use automatic downsampling for display only.

Any new analysis whose cost grows with signal length should follow the same
approach.

---

## 7. Validation

- All 16 keypad digits decode correctly with both algorithms.
- A real 10-second, 44.1 kHz recording of an actual phone being dialled decodes to
  the expected sequence with both algorithms.
- The Goertzel poles lie on the unit circle to numerical precision, and the
  impulse response agrees with the closed form `sin((n+1)ω)/sin(ω)`.
- Decoding remains correct down to the Nyquist limit of the highest DTMF
  frequency and fails below it, as predicted by the sampling theorem.
- The benchmark consistently shows the Goertzel decoder to be faster than the FFT
  decoder on the same input.

---

## 8. Configurable Parameters

The principal tunable parameters are defined as constants in
`core/dsp_interface.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `DTMF_FRAME_SECONDS` / `DTMF_HOP_SECONDS` | 20 ms / 5 ms | Analysis frame length and step |
| `DTMF_LATCH_FRAMES` / `DTMF_RELEASE_FRAMES` | 6 / 4 | Frames required to start and end a tone |
| `DTMF_GATE_DB_DEFAULT` | 10 dB | Margin of the tone pair above the noise floor |
| `DTMF_MIN_DOMINANCE_DEFAULT` | 0.60 | Minimum share of power in the winning pair |
| `DTMF_MAX_FORWARD_TWIST_DB` / `DTMF_MAX_REVERSE_TWIST_DB` | 15 dB / 12 dB | Permitted level difference between the two tones |
| `DTMF_MIN_HARMONIC_DROP_DB` | 6 dB | Required drop of the second harmonic below its fundamental |
| `DTMF_FREQ_TOLERANCE` | ±2.5 % | Permitted frequency offset from nominal |

The detection margin can also be adjusted at run time with the Detection
Threshold slider in the Decoder tab. Microphone gain and room acoustics vary, so
these values may need adjustment for a particular device and environment.

---

## 9. Possible Future Work

- Calibration of the detection thresholds automatically against the measured noise
  characteristics of the input device.
- Decoding of additional signalling formats that use the same analysis engine.
- An automated regression test suite covering the decoding engine with recorded
  real-world audio.