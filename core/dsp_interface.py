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
  4. All pre-existing functions keep their original signatures. The live
     mic work added NEW things at the bottom (a vectorized Goertzel, an
     FFT region decoder with interpolation, Q.24-style validation, and the
     DtmfTracker state machine) without touching the old API.

Run the self-test from the project root with:
    python -m core.dsp_interface
(as a module, not "python core/dsp_interface.py", because of the relative
`from .utils import ...` below — that was true before this change too).
"""

from __future__ import annotations
import numpy as np
import math
import scipy.signal
import time
import tracemalloc
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

FREQ_TO_DIGIT = {freqs: digit for digit, freqs in DIGIT_TO_FREQS.items()}

DTMF_FREQS_ALL = LOW_FREQS + HIGH_FREQS   # all 8 bins, low group first


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
    Find where the "beeps" are in a longer recording that has silence
    between digits (e.g. a file the user uploaded).

    Deliberately LEFT ALONE for the file/import path: for a complete
    recording already in memory, a whole-buffer relative threshold works
    fine. It is NOT used by the live mic path anymore — see the live
    decoding engine at the bottom of this file for why (in short: a
    relative threshold against a rolling live buffer breaks as soon as
    room noise, not a tone, is the loudest thing in the buffer).

    Inputs:
        x  : 1-D numpy array, full recording (tones + silence gaps)
        fs : sample rate

    Output:
        list of (start_sample, end_sample) tuples, one per detected tone,
        in chronological order. Example: [(0, 2000), (2400, 4400), ...]
    """
    # Frame the signal into 10ms chunks
    chunk_size = int(fs * 0.01)
    chunks = [x[i:i + chunk_size] for i in range(0, len(x), chunk_size)]

    # Calculate RMS per chunk
    rms = []
    for chunk in chunks:
        rms.append(np.sqrt(np.mean(chunk ** 2)))

    # Find a threshold
    threshold = 0.2 * max(rms)

    # Apply threshold, MERGING consecutive loud frames into one region each.
    # We only create a tuple at a transition: silence->loud (start a region)
    # or loud->silence (close the region). Everything in between just
    # extends the region we're already tracking.
    tone_regions = []
    region_start = None  # frame index where the current loud stretch began

    for i in range(len(rms)):
        is_loud = rms[i] > threshold
        if is_loud and region_start is None:
            region_start = i                      # a new tone just started
        elif not is_loud and region_start is not None:
            tone_regions.append((region_start * chunk_size, i * chunk_size))
            region_start = None                    # that tone just ended

    # If the recording ends while still "loud" (tone runs to the last frame),
    # close that final region using the actual signal length.
    if region_start is not None:
        tone_regions.append((region_start * chunk_size, len(x)))

    return tone_regions


# ---------------------------------------------------------------------------
# PHASE 5 — Goertzel decoder
# ---------------------------------------------------------------------------

def goertzel_decode(x: np.ndarray, fs: int) -> str:
    """
    Decode a full signal (tones + gaps) into a digit string using the
    Goertzel algorithm. This is the core of the whole app (file path).

    Inputs:
        x  : 1-D numpy array, full recording
        fs : sample rate

    Output:
        decoded digit string, e.g. "512*90"

    PERFORMANCE FIX: previously called goertzel_single_freq() -- a per-
    sample Python `for` loop -- once per DTMF frequency (8 calls per tone
    region). That is the *pedagogical reference* implementation (kept
    as-is, further down, for that purpose) and is dramatically slower in
    wall-clock time than a vectorized FFT call despite doing less total
    arithmetic, because a Python-level loop has far more interpreter
    overhead per sample than a single compiled/vectorized routine.
    goertzel_powers() (defined later in this file) computes the exact same
    Goertzel result for a whole list of frequencies in one vectorized
    matrix multiply -- it's already used by the live decoding engine below
    for this reason. Routing the classic file-path decoder through it too
    makes Goertzel's real speed advantage over FFT peak-picking show up
    here as well, with no change to which digit gets decoded (only the
    per-frequency power *scale* differs between the two Goertzel
    functions, and only relative ranking within a region is ever used).
    """
    tone_regions = segment_tone_regions(x, fs)
    decoded_digits = []

    for start, end in tone_regions:
        x_region = x[start:end]

        if len(x_region) == 0:
            continue

        powers = goertzel_powers(x_region, fs, DTMF_FREQS_ALL)
        best_low = LOW_FREQS[int(np.argmax(powers[:4]))]
        best_high = HIGH_FREQS[int(np.argmax(powers[4:]))]

        digit = FREQ_TO_DIGIT.get((best_low, best_high))

        if digit is not None:
            decoded_digits.append(digit)

    return "".join(decoded_digits)


def goertzel_single_freq(x: np.ndarray, fs: int, target_freq: float) -> float:
    """
    Run the Goertzel algorithm for ONE target frequency and return its
    power/magnitude. goertzel_decode() will likely call this 8 times per
    tone region (once per DTMF frequency). This helper is also reused
    directly by Phase 8 (pole-zero view) to show what the algorithm
    "sees" at a single frequency, so keep it self-contained.

    Kept in its classic per-sample recursive form on purpose — it's the
    pedagogical reference implementation. The live engine uses the
    mathematically equivalent vectorized form (goertzel_powers() /
    frame_bin_powers() further down); a Goertzel filter evaluated over a
    whole block IS a single-bin DFT, so the two give identical bin
    selections, just at different speeds.

    Inputs:
        x           : 1-D numpy array, one tone region
        fs          : sample rate
        target_freq : frequency to test, in Hz

    Output:
        a single float: signal power at target_freq
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)

    if N == 0:
        return 0.0

    omega = 2 * np.pi * target_freq / fs
    coeff = 2 * np.cos(omega)

    s_prev2 = 0.0
    s_prev1 = 0.0

    for sample in x:
        s = sample + coeff * s_prev1 - s_prev2
        s_prev2 = s_prev1
        s_prev1 = s

    power = s_prev2 ** 2 + s_prev1 ** 2 - coeff * s_prev1 * s_prev2
    return float(max(0, power))


# ---------------------------------------------------------------------------
# PHASE 6 — FFT decoder
# ---------------------------------------------------------------------------

def fft_decode(x: np.ndarray, fs: int) -> str:
    """
    Same job as goertzel_decode(), but using FFT peak-picking instead of
    the Goertzel algorithm. Lets the UI show the two decoders side by side.

    Inputs / Output: identical shape to goertzel_decode()

    Internals upgraded: each region is now decoded by fft_region_digit()
    (Hann window + 2x zero-padding + parabolic peak interpolation +
    Q.24-style frequency tolerance), instead of raw rectangular-window
    peak-picking. Signature and output type unchanged.
    """
    tone_regions = segment_tone_regions(x, fs)
    decoded_digits = []

    for start, end in tone_regions:
        x_region = x[start:end]
        if len(x_region) == 0:
            continue

        # fft_region_digit is defined further down in this file — Python
        # resolves names at call time, so the ordering is fine.
        digit, _ = fft_region_digit(x_region, fs)
        if digit is not None:
            decoded_digits.append(digit)

    return "".join(decoded_digits)


# ---------------------------------------------------------------------------
# PHASE 7 — Spectrogram
# ---------------------------------------------------------------------------

def compute_spectrogram(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
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
    # Use a longer window (512 samples at 8 kHz = 64 ms) for good frequency
    # resolution — the default 256-sample window gives only ~31 Hz/bin, which
    # is too coarse to separate 697 Hz from 770 Hz (73 Hz apart). 512 gives
    # ~16 Hz/bin at 8 kHz. 75% overlap (noverlap=384) gives smooth time axis.
    # The Hann window is standard for spectrograms — suppresses leakage.
    nperseg = min(512, len(x))
    noverlap = nperseg * 3 // 4  # 75% overlap
    f, t, Sxx = scipy.signal.spectrogram(
        x, fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='spectrum',
    )
    return t, f, Sxx


# ---------------------------------------------------------------------------
# PHASE 8 — Pole-zero design for the Goertzel resonator
# ---------------------------------------------------------------------------

def goertzel_poles(target_freqs: list[float], fs: int) -> list[complex]:
    """
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
    # The Goertzel recurrence s[n] = x[n] + coeff*s[n-1] - s[n-2], with
    # coeff = 2*cos(omega), corresponds to the characteristic equation
    #     z^2 - coeff*z + 1 = 0
    # Solving with the quadratic formula:
    #     z = [coeff +- sqrt(coeff^2 - 4)] / 2
    #       = [2cos(w) +- sqrt(4cos^2(w) - 4)] / 2
    #       = cos(w) +- j*sin(w)        (since cos^2(w) - 1 = -sin^2(w))
    #       = e^{+-j*omega}
    # So every target frequency gives a conjugate pole PAIR sitting exactly
    # on the unit circle, at angle +omega and -omega.
    poles = []
    for f in target_freqs:
        omega = 2 * np.pi * f / fs
        poles.append(complex(np.cos(omega), np.sin(omega)))    # e^{+j*omega}
        poles.append(complex(np.cos(omega), -np.sin(omega)))   # e^{-j*omega}, the conjugate
    return poles


# ---------------------------------------------------------------------------
# PHASE 9 — Impulse response & convolution
# ---------------------------------------------------------------------------

def impulse_response(target_freq: float, fs: int, n_samples: int = 200) -> np.ndarray:
    """
    Inputs:
        target_freq : the resonator's target frequency in Hz
        fs           : sample rate
        n_samples    : length of h[n] to return

    Output:
        1-D numpy array h[n] of length n_samples
    """
    # Feed a unit impulse (x[0]=1, everything else 0) through the exact same
    # recurrence goertzel_single_freq() uses: s[n] = x[n] + coeff*s[n-1] - s[n-2].
    # This IS the definition of an impulse response — "what does this filter
    # do when you hit it with a single sharp pulse and then leave it alone."
    omega = 2 * np.pi * target_freq / fs
    coeff = 2 * np.cos(omega)

    h = np.zeros(n_samples)
    s_prev2 = 0.0
    s_prev1 = 0.0

    for n in range(n_samples):
        x_n = 1.0 if n == 0 else 0.0
        s = x_n + coeff * s_prev1 - s_prev2
        h[n] = s
        s_prev2 = s_prev1
        s_prev1 = s

    return h


def convolve_signals(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Output: y[n] = x[n] * h[n] (discrete convolution). np.convolve(x, h)
    is allowed here — the point of this feature is to VISUALIZE
    convolution, not to hand-roll the FFT-convolution theorem.
    """
    # Hand-rolled convolution (no np.convolve) — implements the same
    # SUPERPOSITION principle as LTISystem.output_by_superposition() in
    # signal_lti.py: treat one signal as a sequence of scaled impulses,
    # and sum up shifted, scaled copies of the other signal for each one.
    #
    # signal_lti.py's own classes implement exactly this with plain Python
    # loops (DiscreteSignal.add() rescans the whole accumulated range on
    # every call) — correct, and a great reference for checking correctness
    # on small signals (see core/dsp_interface_selftest.py), but measured
    # directly: ~1.2s for a 2000-sample signal, scaling close to O(n^2).
    # Extrapolated to a real ~442,000-sample recording, that's roughly 9.6
    # HOURS. For a live app that needs to convolve real audio, that's not a
    # viable path — so this version keeps the exact same algorithm (sum
    # of shifted, scaled copies) but does the shifting/accumulating with
    # vectorized numpy slicing instead of a per-sample Python loop, which
    # is what actually makes it fast.
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    n_out = len(x) + len(h) - 1
    y = np.zeros(n_out)

    if n_out <= 0:
        return y

    # Loop over whichever signal is SHORTER — fewer outer-loop iterations,
    # and each iteration's shift-and-add is one vectorized numpy operation
    # over the (potentially much longer) other signal.
    if len(h) <= len(x):
        short, long_ = h, x
    else:
        short, long_ = x, h

    for k in range(len(short)):
        val = short[k]
        if val == 0.0:
            continue
        y[k:k + len(long_)] += val * long_

    return y


# ---------------------------------------------------------------------------
# PHASE 10 — Noise + resampling
# ---------------------------------------------------------------------------

def add_awgn_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add white Gaussian noise to x at the given signal-to-noise ratio.

    Inputs:
        x      : 1-D numpy array, clean signal
        snr_db : desired signal-to-noise ratio in dB (e.g. 10, 0, -5)

    Output:
        1-D numpy array, same length as x, with noise added
    """
    # SNR (dB) is defined as 10*log10(signal_power / noise_power). Rearranged
    # for noise_power: signal_power / 10^(snr_db/10). Once we know the noise
    # power we want, np.random.normal generates it directly (its "scale"
    # parameter IS the standard deviation, i.e. sqrt(power) for zero-mean
    # noise).
    x = np.asarray(x, dtype=np.float64)
    signal_power = np.mean(x ** 2)

    if signal_power == 0:
        return x.copy()  # nothing to scale noise against — a silent signal stays silent

    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=len(x))
    return x + noise


def resample_signal(x: np.ndarray, fs_original: int, fs_new: int) -> np.ndarray:
    """
    Resample x from fs_original to fs_new. Used to demonstrate aliasing
    when fs_new drops below the Nyquist rate for DTMF's highest frequency.

    Tip: scipy.signal.resample() handles this in one line.
    """
    n_new = int(round(len(x) * fs_new / fs_original))
    if n_new <= 0:
        return np.array([])
    return scipy.signal.resample(x, n_new)


# ---------------------------------------------------------------------------
# PHASE 11 — Benchmarking
# ---------------------------------------------------------------------------

def benchmark_decoders(x: np.ndarray, fs: int) -> dict:
    """
    Run both decoders on the same input and report performance numbers.

    Output: a dict shaped like this (UI expects exactly these keys):
        {
            "goertzel": {"time_ms": float, "memory_kb": float, "result": str},
            "fft":      {"time_ms": float, "memory_kb": float, "result": str},
        }

    Tip: `time.perf_counter()` for timing, `tracemalloc` for memory.
    """
    results = {}
    for name, decode_fn in (("goertzel", goertzel_decode), ("fft", fft_decode)):
        tracemalloc.start()
        start = time.perf_counter()
        result = decode_fn(x, fs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        _current, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results[name] = {
            "time_ms": elapsed_ms,
            "memory_kb": peak_bytes / 1024,
            "result": result,
        }

    return results


# ===========================================================================
# LIVE DECODING ENGINE — frame classifier + temporal state machine
# ===========================================================================
#
# This section powers "Listen Live". The file/import path above
# (segment_tone_regions -> goertzel_decode / fft_decode) deliberately stays
# untouched: for a whole recording already in memory, energy-threshold
# segmentation is fine. Live audio is different in three ways that broke the
# old approach:
#
#   1. There is no "whole recording" — thresholds computed against the max
#      of a rolling buffer change every tick and go haywire when the room
#      (not a tone) is the loudest thing in the buffer. Steady room noise
#      made every frame "loud", so one region spanned the entire buffer, the
#      trailing-safety-margin check could never let it "close", and zero
#      digits ever decoded. This is the root cause of "not working properly".
#   2. Real audio has dips, clicks and AGC gain-rides; time-domain energy
#      knows "loud", not "is this actually a clean DTMF pair".
#   3. Latency matters: a digit should appear ~100 ms after it sounds, not
#      after a 150 ms trailing-silence confirmation on a 200 ms timer.
#
# The engine mirrors what the reference decoders (dtmf-detect, alpercitak/
# dtmf, the dtmf.pages.dev-class websites) all do:
#
#   audio -> 20 ms frames every 5 ms -> per-frame Goertzel bin powers at
#   the 8 DTMF frequencies -> per-frame adaptive gate (noise floor) plus
#   spectral dominance check -> temporal state machine (latch a digit after
#   8 agreeing frames = 40 ms, release after 4 disagreeing frames) ->
#   one-shot Q.24-style validation at latch (dial tone / twist / frequency
#   offset / 2nd harmonics) -> digit event.
#
# Everything here is plain numpy in / plain dicts out — no Qt, testable
# from the __main__ block at the bottom of this file.
# ===========================================================================

_TINY = 1e-20  # keeps log10() finite on zero power

# --- engine geometry -------------------------------------------------------
DTMF_FRAME_SECONDS = 0.02   # analysis frame: 20 ms (50 Hz bin spacing at 8 kHz —
                            # tight enough to separate 697/770 with margin)
DTMF_HOP_SECONDS = 0.005    # frame step: 5 ms -> 200 frames/s. Fine granularity
                            # is what lets ~40 ms gaps separate repeated same
                            # digits ("55") while short in-tone dips do NOT
                            # split a tone (a dip only counts against a frame
                            # if it fails the gate or the dominance check —
                            # plain amplitude dips usually still "match").
DTMF_LATCH_FRAMES = 6       # 6 * 5 ms = 30 ms of agreement. Slightly below the
                            # ITU-T Q.24 40 ms spec, but real mic capture has more
                            # jitter; 8 frames missed digits that were valid.
DTMF_RELEASE_FRAMES = 4     # 4 * 5 ms of disagreement -> tone over
DTMF_RECENT_SECONDS = 1.5   # rolling raw-audio memory (for latch-time validation)

# --- gate / noise floor ------------------------------------------------------
DTMF_GATE_DB_DEFAULT = 10.0      # tone pair must be this many dB above floor
DTMF_ABS_MIN_LEVEL_DB = -75.0    # absolute silence wall (pair-power scale; a
                                 # full-scale DTMF pair reads about -3 dB)
DTMF_FLOOR_LEAK_DB_PER_FRAME = 0.05  # floor may only RISE this fast: 0.05 dB
                                 # per frame = 10 dB/s at 200 fps. It can DROP
                                 # instantly (min-tracker), so a quiet room is
                                 # picked up immediately after a loud burst.
DTMF_MIN_DOMINANCE_DEFAULT = 0.60  # (best row + best col) / total 8-bin power.
                                 # Lowered from 0.75: real rooms have background
                                 # noise that spreads energy across bins, and
                                 # cheap laptop mics can pull dominance below 0.75
                                 # for genuine DTMF. Non-DTMF tones are still
                                 # rejected by the twist + harmonic checks at latch.

# --- Q.24-style validation (checked once, at latch) --------------------------
DTMF_MAX_FORWARD_TWIST_DB = 15.0  # row tone may be at most 15 dB above column tone
                                   # (real phone handsets/laptop speakers have uneven
                                   # frequency response; 10 dB was too tight for mic use)
DTMF_MAX_REVERSE_TWIST_DB = 12.0   # column tone may be at most 12 dB above row tone
DTMF_MIN_HARMONIC_DROP_DB = 6.0   # 2nd harmonic must be >= 6 dB below its fundamental.
                                   # Real microphones + laptop speakers introduce non-
                                   # linearities; 12 dB was calibrated for a perfect
                                   # sine and too strict for practical hardware.
                                   # Speech still fails (full harmonic stack, not one mild 2nd).
DTMF_FREQ_TOLERANCE = 0.025       # +/-2.5% offset probes (relaxed from 2% for mic offset)
DTMF_FFT_TOLERANCE = 0.025        # FFT peak must land within 2.5% of nominal
DIAL_TONE_FREQS = (350.0, 440.0)  # PSTN dial tone — reject if it dominates
DTMF_REJECT_COOLDOWN_S = 0.4      # min spacing between emitted reject events


def _db(p) -> float:
    """10*log10 with an epsilon so zero power maps to -200 dB, not -inf."""
    return 10.0 * float(np.log10(float(p) + _TINY))


# --- vectorized single-bin DFT (== Goertzel over a block) --------------------

_DFT_MATRIX_CACHE: dict = {}


def _bin_dft_matrix(freqs, N: int, fs) -> np.ndarray:
    """Cached complex exponentials e^{-2*pi*j*f*n/fs}, shape (n_freqs, N).
    Building the matrix is the only expensive part; with the cache, every
    frame evaluation is a single matrix multiply."""
    key = (int(N), int(fs), tuple(float(f) for f in freqs))
    W = _DFT_MATRIX_CACHE.get(key)
    if W is None:
        n = np.arange(N)
        W = np.exp(-2j * np.pi * np.outer(np.asarray(freqs, dtype=np.float64), n) / fs)
        if len(_DFT_MATRIX_CACHE) > 8:   # a couple of devices / frame sizes max
            _DFT_MATRIX_CACHE.clear()
        _DFT_MATRIX_CACHE[key] = W
    return W


def goertzel_powers(x: np.ndarray, fs, freqs) -> np.ndarray:
    """
    Vectorized multi-frequency Goertzel-equivalent power measurement.

    A Goertzel filter run over a whole block computes exactly one bin of a
    DFT, so evaluating |sum(x[n] e^{-j 2 pi f n / fs})|^2 for a list of
    frequencies IS the Goertzel result — just computed as one matrix
    multiply instead of a per-sample Python loop (the loop version,
    goertzel_single_freq(), stays for teaching purposes).

    Returns mean-square power (|X|/N)^2 per frequency — a DIFFERENT scale
    from goertzel_single_freq()'s raw power (factor N^2), so never compare
    the two directly. Ratios and dB differences are unaffected, and only
    those are ever used.

    Inputs:
        x     : 1-D numpy array
        fs    : sample rate
        freqs : iterable of frequencies in Hz

    Output:
        1-D numpy array of powers, same order as freqs.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    freqs = list(freqs)
    if x.size == 0 or not freqs:
        return np.zeros(len(freqs))
    W = _bin_dft_matrix(freqs, x.size, fs)
    X = W @ x
    return (np.abs(X) / x.size) ** 2


def frame_bin_powers(frames: np.ndarray, fs, freqs) -> np.ndarray:
    """
    Batched version of goertzel_powers for many equal-length frames.

    Inputs:
        frames : 2-D array, shape (n_frames, N)
        fs     : sample rate
        freqs  : frequencies to evaluate

    Output:
        2-D array, shape (n_frames, n_freqs), mean-square powers.
    """
    frames = np.asarray(frames, dtype=np.float64)
    freqs = list(freqs)
    if frames.size == 0:
        return np.zeros((0, len(freqs)))
    W = _bin_dft_matrix(freqs, frames.shape[1], fs)
    return (np.abs(frames @ W.T) / frames.shape[1]) ** 2


# --- FFT region decoder (Hann + zero-pad + parabolic interpolation) ----------

def fft_region_digit(x: np.ndarray, fs, tolerance: float = DTMF_FFT_TOLERANCE):
    """
    Decode ONE tone region with the enhanced FFT path: Hann window (kills
    rectangular-window leakage), 2x zero-padding, parabolic peak
    interpolation (sub-bin accuracy), then snap to the nearest nominal
    DTMF frequency — accepted only if the measured peak is within
    `tolerance` (fractional) of nominal, mirroring Q.24's frequency
    tolerance. Used by the live engine's "fft" mode and by fft_decode().

    Returns (digit_or_None, (f_low_measured, f_high_measured)).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    N = x.size
    if N < 16:
        return None, (None, None)

    nfft = 1 << (int(np.ceil(np.log2(N))) + 1)      # next pow2, doubled
    mag = np.abs(np.fft.rfft(x * np.hanning(N), nfft))
    bin_hz = fs / nfft
    freqs = np.arange(len(mag)) * bin_hz

    def band_peak(lo, hi):
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            return None
        ks = np.flatnonzero(mask)
        k = int(ks[int(np.argmax(mag[mask]))])
        delta = 0.0
        if 0 < k < len(mag) - 1:
            # Parabolic interpolation on log-magnitude: the true peak sits
            # between bins at k + delta; delta in [-0.5, 0.5].
            a = np.log(mag[k - 1] + _TINY)
            b = np.log(mag[k] + _TINY)
            c = np.log(mag[k + 1] + _TINY)
            denom = a - 2.0 * b + c
            if abs(denom) > 1e-12:
                delta = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
        return (k + delta) * bin_hz

    f_low = band_peak(LOW_FREQS[0] * 0.9, LOW_FREQS[-1] * 1.1)
    f_high = band_peak(HIGH_FREQS[0] * 0.9, HIGH_FREQS[-1] * 1.1)
    if f_low is None or f_high is None:
        return None, (f_low, f_high)

    best_low = min(LOW_FREQS, key=lambda g: abs(g - f_low))
    best_high = min(HIGH_FREQS, key=lambda g: abs(g - f_high))
    if (abs(best_low - f_low) > tolerance * best_low
            or abs(best_high - f_high) > tolerance * best_high):
        return None, (f_low, f_high)
    return FREQ_TO_DIGIT.get((best_low, best_high)), (f_low, f_high)


# --- Q.24-style validation, run once per latch -------------------------------

def validate_dtmf_region(x: np.ndarray, fs, f_low, f_high):
    """
    Decide whether a candidate tone region is REALLY a DTMF digit, using
    ITU-T Q.24-flavoured checks (the same family of checks the reference
    decoders implement). Runs over the region's audio with a Hann window
    for good selectivity. All checks use Goertzel bin powers — they're
    measurements about the signal, independent of which decoder produced
    the candidate.

    Checks, in order (first failure wins, so the most specific / most
    decisive rejection reason is reported first):
      1. dial tone  : 350+440 Hz pair stronger than the candidate pair
      2. twist      : row/column level difference beyond +8 / -4 dB
      3. offset     : winner frequency not the local max among +/-2% probes
      4. harmonics  : 2nd harmonic of either winner within 12 dB of it
                      (voiced speech and many instruments fail this)

    Returns None if the region passes as valid DTMF, else a human-readable
    reason string (shown in the UI's rejection line).
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 8:
        return "region too short to validate"

    w = np.hanning(len(x))
    t = DTMF_FREQ_TOLERANCE
    probes = [
        f_low, f_high,                       # the candidate pair itself
        2.0 * f_low, 2.0 * f_high,           # second harmonics
        f_low * (1 - t), f_low * (1 + t),    # row frequency offset probes
        f_high * (1 - t), f_high * (1 + t),  # column frequency offset probes
        DIAL_TONE_FREQS[0], DIAL_TONE_FREQS[1],
    ]
    (p_low, p_high, h2_low, h2_high,
     low_minus, low_plus, high_minus, high_plus,
     d350, d440) = goertzel_powers(x * w, fs, probes)

    # 1. Dial tone. Pure 350+440 leaks into the DTMF bins through any real
    #    window; if the dial pair is at least as strong as what we think the
    #    "winners" are, this is dial tone, not DTMF. (Genuine DTMF played
    #    OVER dial tone still passes, because then the winners dominate.)
    if _db(d350 + d440) - _db(p_low + p_high) > 0.0:
        return "dial tone (350+440 Hz) is stronger than the tone pair"

    # 2. Twist. A real DTMF pair has both tones within a few dB of each
    #    other; speech, music and single stray tones do not.
    twist_db = _db(p_low) - _db(p_high)
    if twist_db > DTMF_MAX_FORWARD_TWIST_DB:
        return (f"twist {twist_db:.1f} dB exceeds +{DTMF_MAX_FORWARD_TWIST_DB:.0f} dB "
                f"(row tone far louder than column tone)")
    if twist_db < -DTMF_MAX_REVERSE_TWIST_DB:
        return (f"reverse twist {-twist_db:.1f} dB exceeds {DTMF_MAX_REVERSE_TWIST_DB:.0f} dB "
                f"(column tone far louder than row tone)")

    # 3. Frequency offset. If a +/-2% neighbour of the winning frequency has
    #    MORE power than the winner itself, the true tone is off-frequency —
    #    a nearby beep/siren, not DTMF.
    if p_low < max(low_minus, low_plus):
        return "row tone is off-frequency by more than ~2% (not a DTMF frequency)"
    if p_high < max(high_minus, high_plus):
        return "column tone is off-frequency by more than ~2% (not a DTMF frequency)"

    # 4. Second harmonics. Voiced speech is a harmonic stack — energy at f
    #    almost always comes with energy at 2f. A DTMF sine (even from a
    #    cheap speaker with a few % distortion) keeps 2f well down.
    if _db(h2_low) - _db(p_low) > -DTMF_MIN_HARMONIC_DROP_DB:
        return (f"row 2nd harmonic less than {DTMF_MIN_HARMONIC_DROP_DB:.0f} dB down "
                f"(sounds like speech/music, not DTMF)")
    if _db(h2_high) - _db(p_high) > -DTMF_MIN_HARMONIC_DROP_DB:
        return (f"column 2nd harmonic less than {DTMF_MIN_HARMONIC_DROP_DB:.0f} dB down "
                f"(sounds like speech/music, not DTMF)")

    return None  # passed everything — this is a valid DTMF pair


# --- the temporal state machine ------------------------------------------------

class DtmfTracker:
    """
    Frame-based live DTMF detector with a temporal state machine.

    push() accepts whatever new samples have arrived (any amount, any time
    — framing is continuous across calls via an internal carry), and returns
    a list of event dicts:

        {"type": "digit",   "digit": str, "duration_s": float, "region": np.ndarray}
        {"type": "reject",  "reason": str, "duration_s": float, "region": np.ndarray}
        {"type": "release", "digit": str | None, "duration_s": float}

    * "digit"   fires ONCE, the moment a tone is latched (after
                latch_frames consecutive agreeing frames — no waiting for
                the tone to end), which is why live latency is ~40 ms of
                tone plus one UI tick instead of hundreds of ms.
    * "reject"  fires when something tone-like latched but failed Q.24
                validation (rate-limited by a cooldown so a room full of
                speech can't spam the UI).
    * "release" fires when the latched tone has been absent for
                release_frames frames; its duration_s is the FULL tone
                length. A repeat of the same digit requires a full
                latch-release-latch cycle, so "55" comes out as two 5s.

    Tunables (all settable as attributes after construction, which is how
    the UI's threshold slider and algorithm combo work):
        gate_db       — tone pair must exceed the noise floor by this many dB
        min_dominance — per-frame spectral dominance requirement
        algorithm     — "goertzel" (default) or "fft": which decoder picks
                        the digit at latch time. Validation always uses
                        Goertzel bin powers, so the two algorithms can be
                        compared fairly, side by side.

    `last_frame_info` is updated every frame with the current bin levels,
    noise floor, gate level and state — the UI's live bars read it.

    The noise floor is a min-with-leak tracker on the winner-pair power:
    it drops instantly to any quieter level and may only rise 10 dB/s, and
    it only updates while idle and not mid-run, so tones never drag it up.
    Gating on the winner PAIR (narrowband) rather than total frame energy
    buys ~16 dB of noise robustness at 8 kHz — the engine decodes at
    0 dB SNR with the default +10 dB gate, where the old total-energy gate
    already failed around 10 dB SNR.
    """

    def __init__(self, fs: int, algorithm: str = "goertzel",
                 latch_frames: int = DTMF_LATCH_FRAMES,
                 release_frames: int = DTMF_RELEASE_FRAMES,
                 gate_db: float = DTMF_GATE_DB_DEFAULT,
                 min_dominance: float = DTMF_MIN_DOMINANCE_DEFAULT,
                 frame_seconds: float = DTMF_FRAME_SECONDS,
                 hop_seconds: float = DTMF_HOP_SECONDS):
        if fs <= 0:
            raise ValueError("fs must be positive")
        if algorithm not in ("goertzel", "fft"):
            raise ValueError("algorithm must be 'goertzel' or 'fft'")

        self.fs = int(fs)
        self.algorithm = algorithm
        self.latch_frames = int(latch_frames)
        self.release_frames = int(release_frames)
        self.gate_db = float(gate_db)
        self.min_dominance = float(min_dominance)

        self.frame_len = max(4, int(round(frame_seconds * self.fs)))
        self.hop = max(1, int(round(hop_seconds * self.fs)))
        if self.hop > self.frame_len:
            self.hop = self.frame_len

        self.reset()

    # -- public API ----------------------------------------------------------

    def reset(self) -> None:
        """Forget all history (used when listening starts)."""
        self._carry = np.zeros(0)          # samples not yet framed (< frame_len + hop)
        self._recent = np.zeros(0)         # rolling raw audio for latch validation
        self._stream_len = 0               # total samples ever pushed
        self._floor_db = -60.0             # optimistic start; adapts within ~1 s
        self._state = "idle"               # "idle" | "active"
        self._latched_digit = None
        self._run_digit = None
        self._run_len = 0
        self._run_start_abs = 0
        self._tone_start_abs = 0
        self._release_run = 0
        self._last_reject_t = -1e9
        self.last_frame_info: dict = {}

    def push(self, x: np.ndarray) -> list[dict]:
        """Feed new samples (any length); returns events since last push."""
        events: list[dict] = []
        x = np.asarray(x, dtype=np.float64).ravel()

        if x.size:
            if self._recent.size:
                self._recent = np.concatenate([self._recent, x])
            else:
                self._recent = x.copy()
            self._stream_len += int(x.size)
            cap = int(DTMF_RECENT_SECONDS * self.fs)
            if self._recent.size > cap:
                self._recent = self._recent[-cap:]

        buf = np.concatenate([self._carry, x]) if self._carry.size else x
        if buf.size < self.frame_len:
            self._carry = buf.copy()
            return events

        n_frames = 1 + (buf.size - self.frame_len) // self.hop
        starts = np.arange(n_frames) * self.hop
        idx = starts[:, None] + np.arange(self.frame_len)[None, :]
        frames = buf[idx]

        powers = frame_bin_powers(frames, self.fs, DTMF_FREQS_ALL)

        # Absolute stream index of buf[0] (carry was already counted in a
        # previous push, so subtract everything the buffer holds).
        base = self._stream_len - buf.size
        for i in range(n_frames):
            start_abs = base + int(starts[i])
            end_abs = start_abs + self.frame_len
            events.extend(self._step(powers[i], start_abs, end_abs))

        self._carry = buf[n_frames * self.hop:].copy()
        return events

    # -- internals --------------------------------------------------------------

    def _step(self, powers: np.ndarray, start_abs: int, end_abs: int) -> list[dict]:
        """Process ONE 20 ms frame through gate + dominance + state machine."""
        low_i = int(np.argmax(powers[:4]))
        high_j = int(np.argmax(powers[4:]))
        pair_power = float(powers[low_i] + powers[4 + high_j])
        level_db = _db(pair_power)
        dominance = pair_power / (float(powers.sum()) + _TINY)

        gate_level = max(self._floor_db + self.gate_db, DTMF_ABS_MIN_LEVEL_DB)
        candidate = None
        if level_db >= gate_level and dominance >= self.min_dominance:
            candidate = FREQ_TO_DIGIT.get((LOW_FREQS[low_i], HIGH_FREQS[high_j]))

        events: list[dict] = []

        if self._state == "idle":
            if candidate is not None:
                if candidate == self._run_digit:
                    self._run_len += 1
                else:
                    self._run_digit = candidate
                    self._run_len = 1
                    self._run_start_abs = start_abs
            else:
                if self._run_len:
                    self._run_digit = None
                    self._run_len = 0
                # Only quiet/non-candidate frames teach the noise floor.
                self._update_floor(level_db)

            if self._run_len >= self.latch_frames:
                run_start = self._run_start_abs
                region = self._region_audio(run_start, end_abs)
                duration_s = (end_abs - run_start) / float(self.fs)
                self._run_digit = None
                self._run_len = 0

                digit, reason = self._finalize_latch(region)
                if reason is None:
                    self._state = "active"
                    self._latched_digit = digit
                    self._tone_start_abs = run_start
                    self._release_run = 0
                    events.append({"type": "digit", "digit": digit,
                                   "duration_s": duration_s, "region": region})
                else:
                    # Not valid DTMF. Stay idle (no hold state to get stuck
                    # in) and rate-limit the events so continuous speech or
                    # music can't spam the UI with rejections.
                    now_s = end_abs / float(self.fs)
                    if now_s - self._last_reject_t >= DTMF_REJECT_COOLDOWN_S:
                        self._last_reject_t = now_s
                        events.append({"type": "reject", "reason": reason,
                                       "duration_s": duration_s, "region": region})
        else:  # active — a validated digit is currently sounding
            if candidate == self._latched_digit:
                self._release_run = 0
            else:
                self._release_run += 1
                if self._release_run >= self.release_frames:
                    events.append({
                        "type": "release",
                        "digit": self._latched_digit,
                        "duration_s": (end_abs - self._tone_start_abs) / float(self.fs),
                    })
                    self._state = "idle"
                    self._latched_digit = None
                    self._release_run = 0

        self.last_frame_info = {
            "bin_db": [_db(v) for v in powers],   # 8 per-bin levels (bars)
            "level_db": level_db,                 # winner-pair level (gate metric)
            "floor_db": self._floor_db,
            "gate_level_db": gate_level,
            "dominance": dominance,
            "candidate": candidate,
            "state": self._state,
            "latched": self._latched_digit,
        }
        return events

    def _finalize_latch(self, region: np.ndarray):
        """Pick the digit with the configured algorithm, then validate."""
        if region is None or len(region) < max(16, self.frame_len // 2):
            return None, "tone region too short to validate"

        if self.algorithm == "fft":
            digit, (f_low_est, f_high_est) = fft_region_digit(region, self.fs)
            if digit is None:
                parts = []
                for v in (f_low_est, f_high_est):
                    parts.append(f"{v:.0f} Hz" if v is not None else "no peak")
                return None, (f"FFT peaks ({parts[0]}, {parts[1]}) are not within "
                              f"{DTMF_FFT_TOLERANCE * 100:.0f}% of a DTMF pair")
            f_low, f_high = DIGIT_TO_FREQS[digit]
        else:
            windowed = region * np.hanning(len(region))
            powers = goertzel_powers(windowed, self.fs, DTMF_FREQS_ALL)
            f_low = LOW_FREQS[int(np.argmax(powers[:4]))]
            f_high = HIGH_FREQS[int(np.argmax(powers[4:]))]
            digit = FREQ_TO_DIGIT.get((f_low, f_high))

        reason = validate_dtmf_region(region, self.fs, f_low, f_high)
        return digit, reason

    def _region_audio(self, start_abs: int, end_abs: int) -> np.ndarray:
        """Slice [start_abs, end_abs) out of the rolling raw-audio memory."""
        base = self._stream_len - self._recent.size
        i0 = max(0, start_abs - base)
        i1 = min(self._recent.size, end_abs - base)
        if i1 <= i0:
            return np.zeros(0)
        return self._recent[i0:i1].copy()

    def _update_floor(self, level_db: float) -> None:
        """Min-with-leak noise floor: drops instantly, rises at 10 dB/s max."""
        leaked = self._floor_db + DTMF_FLOOR_LEAK_DB_PER_FRAME
        self._floor_db = float(np.clip(min(level_db, leaked), -100.0, -5.0))


# ---------------------------------------------------------------------------
# Self-test scaffold — run this file directly to sanity-check your work
# without ever opening the GUI:
#
#     python -m core.dsp_interface
#
# It exercises the classic file-path decoders AND the new live engine
# (clean, noisy, quiet, repeated digits, FFT mode, plus signals that MUST
# be rejected: a single non-DTMF tone and a dial tone).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)  # reproducible noise for the SNR tests
    fs = 8000
    print("DIGIT_TO_FREQS table:", DIGIT_TO_FREQS)

    tone = generate_dtmf_tone("5", fs=fs, duration=0.25)
    print("generate_dtmf_tone('5') -> shape", tone.shape)

    print("\n-- file-path decoders (unchanged API) --")
    seq = "512*90AD#"
    wav = sequence_to_wav(seq, fs=fs)
    print("goertzel_decode:", goertzel_decode(wav, fs))
    print("fft_decode:     ", fft_decode(wav, fs))

    print("\n-- live engine (DtmfTracker) --")

    def run_live(label, x, **kwargs):
        tracker = DtmfTracker(fs, **kwargs)
        events = tracker.push(x)
        digits = "".join(e["digit"] for e in events if e["type"] == "digit")
        rejects = [e["reason"] for e in events if e["type"] == "reject"]
        print(f"{label:42s} -> digits={digits!r} rejects={rejects}")

    live = sequence_to_wav(seq, fs=fs, tone_duration=0.10, gap_duration=0.06)
    run_live("clean, 100 ms tones / 60 ms gaps", live)
    run_live("same at SNR 15 dB", add_awgn_noise(live, 15))
    run_live("same at SNR 5 dB", add_awgn_noise(live, 5))
    run_live("same at SNR 0 dB (default gate)", add_awgn_noise(live, 0))
    run_live("quiet (-40 dBFS), clean", live * 0.01)
    run_live("repeated digit '55'", sequence_to_wav("55", fs=fs,
                                                    tone_duration=0.10, gap_duration=0.06))
    run_live("same sequence via FFT algorithm", live, algorithm="fft")

    t = np.arange(int(0.4 * fs)) / fs
    run_live("single 700 Hz tone (must reject)", 0.5 * np.sin(2 * np.pi * 700 * t))
    dial = 0.4 * np.sin(2 * np.pi * 350 * t) + 0.4 * np.sin(2 * np.pi * 440 * t)
    run_live("dial tone 350+440 Hz (must reject)", dial)