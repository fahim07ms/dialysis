"""
dsp_interface.py
=================

THIS IS YOUR FILE. This is the ONLY file where signal-processing math belongs.

RULES OF THE CONTRACT:
  1. Every function below takes plain numpy arrays / numbers in, and returns
     plain numpy arrays / numbers / strings out.
  2. NOTHING in this file may import PyQt, pyqtgraph, or touch any UI object.
     That keeps this module testable on its own, from a plain Python
     terminal, with no GUI running at all.
  3. Don't rename the functions or change their argument order — the UI
     code in ui/*.py calls these by exact name and signature. If you need
     to change a signature, tell me and I'll update the caller.
  4. Each function currently raises NotImplementedError with a TODO comment
     explaining exactly what to build. Replace the "raise" line with your
     real implementation.

You can develop and test everything in here completely separately from the
GUI. Just run this file directly:

    python core/dsp_interface.py

and the __main__ block at the bottom will call each function with dummy
values so you can check your work without ever opening the app window.
"""

from __future__ import annotations
import numpy as np
import math
from .utils import bit_reverse, bit_reversal, pad_with_zeros

# ---------------------------------------------------------------------------
# DTMF frequency table (this part is already done for you — it's just data,
# not an algorithm. Feel free to use it in your functions.)
# ---------------------------------------------------------------------------

LOW_FREQS = [697, 770, 852, 941]          # rows
HIGH_FREQS = [1209, 1336, 1477, 1633]     # columns

KEYPAD_LAYOUT = [
    ["1", "2", "3", "A"],
    ["4", "5", "6", "B"],
    ["7", "8", "9", "C"],
    ["*", "0", "#", "D"],
]

DIGIT_TO_FREQS = {}
for row_i, row in enumerate(KEYPAD_LAYOUT):
    for col_i, digit in enumerate(row):
        DIGIT_TO_FREQS[digit] = (LOW_FREQS[row_i], HIGH_FREQS[col_i])


# ---------------------------------------------------------------------------
# PHASE 1 — Tone synthesis
# ---------------------------------------------------------------------------

def generate_dtmf_tone(digit: str, fs: int = 8000, duration: float = 0.25) -> np.ndarray:
    """
    Build one DTMF tone for a single key.

    Inputs:
        digit    : one character, e.g. "5", "*", "A"
                   -> use DIGIT_TO_FREQS[digit] to get (f_low, f_high)
        fs       : sample rate in Hz (e.g. 8000)
        duration : tone length in seconds (e.g. 0.25)

    Output:
        1-D numpy float array, values roughly in [-1, 1], length = fs*duration
        x[n] = sin(2*pi*f_low*n/fs) + sin(2*pi*f_high*n/fs)

    Tip: n = np.arange(int(fs * duration))
    """
    f_low, f_high = DIGIT_TO_FREQS[digit]
    n = np.arange(int(fs * duration))
    x = np.sin(2 * np.pi * f_low * n / fs) + np.sin(2 * np.pi * f_high * n / fs)
    return x


# ---------------------------------------------------------------------------
# PHASE 2 — Sequence -> WAV export
# ---------------------------------------------------------------------------

def sequence_to_wav(digits: str, fs: int = 8000, tone_duration: float = 0.25,
                     gap_duration: float = 0.05) -> np.ndarray:
    """
    Turn a multi-digit string like "512*90" into one continuous waveform:
    tone, silence, tone, silence, ... (this is what real phones do so a
    listener can tell where one digit ends and the next begins).

    Inputs:
        digits        : e.g. "512*90"
        fs            : sample rate
        tone_duration : seconds per tone (reuse generate_dtmf_tone)
        gap_duration  : seconds of silence between tones

    Output:
        1-D numpy float array = concatenation of [tone, silence, tone, silence, ...]

    The UI's export button will take whatever you return here and write it
    straight to a .wav file — you don't need to touch any file I/O.
    """
    wav = np.array([])
    for digit in digits:
        wav = np.concatenate((wav, generate_dtmf_tone(digit, fs, tone_duration),
                               np.zeros(int(fs * gap_duration))))

    return wav


# ---------------------------------------------------------------------------
# PHASE 3 — Spectrum analysis
# ---------------------------------------------------------------------------
def fft_recursive(x: np.ndarray) -> np.ndarray:
    N = len(x)

    if N <= 1:
        return x

    x = pad_with_zeros(x)

    N_padded = len(x)

    # Divide into evens and odds
    e = x[::2]
    o = x[1::2]

    # Calculate G_k & H_k
    g_k = fft_recursive(e)
    h_k = fft_recursive(o)

    wnk = np.exp((-2j * np.pi * np.arange(N_padded // 2)) / N_padded)

    X = np.concatenate((g_k + wnk * h_k, g_k - wnk * h_k))
    return X


def fft(x: np.ndarray) -> np.ndarray:
    N = len(x)

    if N <= 1:
        return x

    x = pad_with_zeros(x).astype(np.complex64)

    N_padded = len(x)

    # Sort the array on Bit Reverse Order
    x = bit_reversal(x)

    # No. of stages
    n_stages = int(math.log2(N_padded))
    for s in range(1, n_stages + 1):
        M = 2**s
        WM = np.exp(-2j * np.pi / M)
        for l in range(0, N_padded - M + 1, M):
            # Twiddle factor
            W = 1
            for k in range(M // 2):
                g = x[l + k]
                h = W * x[l + k + M//2]
                x[l + k] = g + h
                x[l + k + M//2] = g - h
                W = W * WM

    return x

def compute_spectrum(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the magnitude spectrum of a signal.

    Inputs:
        x  : 1-D numpy array, the audio signal
        fs : sample rate

    Output:
        (freqs, magnitudes) — two 1-D numpy arrays of the SAME length.
        freqs       : frequency bins in Hz, only the positive half (0 .. fs/2)
        magnitudes  : magnitude (not power) at each bin
    """
    X = fft(x)
    N = len(X)
    freqs = np.arange(N) * fs / N
    magnitudes = np.abs(X) / len(x)
    half = N // 2
    return freqs[:half], magnitudes[:half]


# ---------------------------------------------------------------------------
# PHASE 4 — Segmentation of an uploaded/recorded signal
# ---------------------------------------------------------------------------

def segment_tone_regions(x: np.ndarray, fs: int) -> list[tuple[int, int]]:
    """
    TODO(you) — Phase 4

    Find where the "beeps" are in a longer recording that has silence
    between digits (e.g. a file the user uploaded).

    Inputs:
        x  : 1-D numpy array, full recording (tones + silence gaps)
        fs : sample rate

    Output:
        list of (start_sample, end_sample) tuples, one per detected tone,
        in chronological order. Example: [(0, 2000), (2400, 4400), ...]

    Tip: a simple energy threshold on short windows (e.g. RMS per 10ms
    frame) is a good starting point.
    """
    raise NotImplementedError("Phase 4: implement segment_tone_regions()")


# ---------------------------------------------------------------------------
# PHASE 5 — Goertzel decoder
# ---------------------------------------------------------------------------

def goertzel_decode(x: np.ndarray, fs: int) -> str:
    """
    TODO(you) — Phase 5

    Decode a full signal (tones + gaps) into a digit string using the
    Goertzel algorithm. This is the core of the whole app.

    Inputs:
        x  : 1-D numpy array, full recording
        fs : sample rate

    Output:
        decoded digit string, e.g. "512*90"

    Suggested approach:
        1. Use segment_tone_regions(x, fs) to find each tone's (start, end).
        2. For each region, run the Goertzel algorithm at all 8 DTMF
           frequencies (LOW_FREQS + HIGH_FREQS) and find the strongest
           low-group and strongest high-group frequency.
        3. Map that (f_low, f_high) pair back to a digit using
           DIGIT_TO_FREQS (just reverse the lookup).
    """
    raise NotImplementedError("Phase 5: implement goertzel_decode()")


def goertzel_single_freq(x: np.ndarray, fs: int, target_freq: float) -> float:
    """
    TODO(you) — Phase 5 (helper)

    Run the Goertzel algorithm for ONE target frequency and return its
    power/magnitude. goertzel_decode() will likely call this 8 times per
    tone region (once per DTMF frequency). This helper is also reused
    directly by Phase 8 (pole-zero view) to show what the algorithm
    "sees" at a single frequency, so keep it self-contained.

    Inputs:
        x           : 1-D numpy array, one tone region
        fs          : sample rate
        target_freq : frequency to test, in Hz

    Output:
        a single float: signal power at target_freq
    """
    raise NotImplementedError("Phase 5: implement goertzel_single_freq()")


# ---------------------------------------------------------------------------
# PHASE 6 — FFT decoder
# ---------------------------------------------------------------------------

def fft_decode(x: np.ndarray, fs: int) -> str:
    """
    TODO(you) — Phase 6

    Same job as goertzel_decode(), but using FFT peak-picking instead of
    the Goertzel algorithm. Lets the UI show the two decoders side by side.

    Inputs / Output: identical shape to goertzel_decode()
    """
    raise NotImplementedError("Phase 6: implement fft_decode()")


# ---------------------------------------------------------------------------
# PHASE 7 — Spectrogram
# ---------------------------------------------------------------------------

def compute_spectrogram(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TODO(you) — Phase 7

    Inputs:
        x  : 1-D numpy array
        fs : sample rate

    Output:
        (t, f, Sxx)
        t   : 1-D array of time bin centers (seconds)
        f   : 1-D array of frequency bin centers (Hz)
        Sxx : 2-D array, shape (len(f), len(t)), magnitude or dB values

    Tip: scipy.signal.spectrogram(x, fs) returns almost exactly this shape.
    """
    raise NotImplementedError("Phase 7: implement compute_spectrogram()")


# ---------------------------------------------------------------------------
# PHASE 8 — Pole-zero design for the Goertzel resonator
# ---------------------------------------------------------------------------

def goertzel_poles(target_freqs: list[float], fs: int) -> list[complex]:
    """
    TODO(you) — Phase 8

    For each target frequency, compute the pole location(s) of the
    Goertzel resonator on the Z-plane (it's a 2nd-order recursive filter,
    so each frequency gives a conjugate pole pair on the unit circle).

    Inputs:
        target_freqs : list of frequencies in Hz, e.g. LOW_FREQS + HIGH_FREQS
        fs            : sample rate

    Output:
        list of complex numbers (the pole locations). The UI will plot
        each one as a dot on the unit circle.
    """
    raise NotImplementedError("Phase 8: implement goertzel_poles()")


# ---------------------------------------------------------------------------
# PHASE 9 — Impulse response & convolution
# ---------------------------------------------------------------------------

def impulse_response(target_freq: float, fs: int, n_samples: int = 200) -> np.ndarray:
    """
    TODO(you) — Phase 9

    Inputs:
        target_freq : the resonator's target frequency in Hz
        fs           : sample rate
        n_samples    : length of h[n] to return

    Output:
        1-D numpy array h[n] of length n_samples
    """
    raise NotImplementedError("Phase 9: implement impulse_response()")


def convolve_signals(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    TODO(you) — Phase 9

    Output: y[n] = x[n] * h[n] (discrete convolution). np.convolve(x, h)
    is allowed here — the point of this feature is to VISUALIZE
    convolution, not to hand-roll the FFT-convolution theorem.
    """
    raise NotImplementedError("Phase 9: implement convolve_signals()")


# ---------------------------------------------------------------------------
# PHASE 10 — Noise + resampling
# ---------------------------------------------------------------------------

def add_awgn_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    """
    TODO(you) — Phase 10

    Add white Gaussian noise to x at the given signal-to-noise ratio.

    Inputs:
        x      : 1-D numpy array, clean signal
        snr_db : desired signal-to-noise ratio in dB (e.g. 10, 0, -5)

    Output:
        1-D numpy array, same length as x, with noise added
    """
    raise NotImplementedError("Phase 10: implement add_awgn_noise()")


def resample_signal(x: np.ndarray, fs_original: int, fs_new: int) -> np.ndarray:
    """
    TODO(you) — Phase 10

    Resample x from fs_original to fs_new. Used to demonstrate aliasing
    when fs_new drops below the Nyquist rate for DTMF's highest frequency.

    Tip: scipy.signal.resample() handles this in one line.
    """
    raise NotImplementedError("Phase 10: implement resample_signal()")


# ---------------------------------------------------------------------------
# PHASE 11 — Benchmarking
# ---------------------------------------------------------------------------

def benchmark_decoders(x: np.ndarray, fs: int) -> dict:
    """
    TODO(you) — Phase 11

    Run both decoders on the same input and report performance numbers.

    Output: a dict shaped like this (UI expects exactly these keys):
        {
            "goertzel": {"time_ms": float, "memory_kb": float, "result": str},
            "fft":      {"time_ms": float, "memory_kb": float, "result": str},
        }

    Tip: `time.perf_counter()` for timing, `tracemalloc` for memory.
    """
    raise NotImplementedError("Phase 11: implement benchmark_decoders()")


# ---------------------------------------------------------------------------
# Self-test scaffold — run this file directly to sanity-check your work
# without ever opening the GUI.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fs = 8000
    print("DIGIT_TO_FREQS table:", DIGIT_TO_FREQS)

    try:
        tone = generate_dtmf_tone("5", fs=fs, duration=0.25)
        print("generate_dtmf_tone('5') -> shape", tone.shape)
    except NotImplementedError as e:
        print("Not implemented yet:", e)
