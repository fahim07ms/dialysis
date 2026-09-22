"""
dsp_interface_selftest.py
===========================
Standalone correctness check: verifies core.dsp_interface.convolve_signals()
produces IDENTICAL output to core.signal_lti.py's DiscreteSignal/LTISystem
classes (the actual provided reference implementation) on small signals,
and to np.convolve for good measure.

Kept separate from dsp_interface.py's own __main__ self-test block because
this one is deliberately slow on purpose for large inputs (see the timing
note below) — it belongs to the "run this occasionally to double-check
correctness" category, not "run this every time you touch the file."

Run directly:
    python core/dsp_interface_selftest.py
"""

import time
import numpy as np

from core import dsp_interface as dsp
from core.signal_lti import DiscreteSignal, LTISystem


def reference_convolve(x_vals: np.ndarray, h_vals: np.ndarray) -> np.ndarray:
    """Ground truth using the ACTUAL provided classes, unmodified."""
    x = DiscreteSignal(0, len(x_vals) - 1)
    x.values = np.asarray(x_vals, dtype=float)
    h = DiscreteSignal(0, len(h_vals) - 1)
    h.values = np.asarray(h_vals, dtype=float)
    y = LTISystem(h).output_by_superposition(x)
    return y.values


def run():
    np.random.seed(0)
    all_passed = True

    print("Checking convolve_signals() against signal_lti.py's DiscreteSignal/LTISystem...")
    print("(kept small — see the timing note in dsp_interface.py's convolve_signals()")
    print(" docstring for why the reference implementation isn't used on real audio)\n")

    test_cases = [(5, 3), (20, 7), (50, 10), (100, 200), (200, 1)]
    for n_x, n_h in test_cases:
        x = np.random.randn(n_x)
        h = np.random.randn(n_h)

        y_mine = dsp.convolve_signals(x, h)
        y_reference = reference_convolve(x, h)
        y_numpy = np.convolve(x, h)

        matches_reference = np.allclose(y_mine, y_reference)
        matches_numpy = np.allclose(y_mine, y_numpy)
        correct_length = len(y_mine) == n_x + n_h - 1

        passed = matches_reference and matches_numpy and correct_length
        all_passed &= passed

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] x={n_x:4d} samples, h={n_h:4d} samples -> "
              f"matches signal_lti.py: {matches_reference}, matches np.convolve: {matches_numpy}")

    print()
    print("Speed comparison (real-audio-scale input, 442,368 samples ~= a 10s recording @ 44.1kHz):")
    x_big = np.random.randn(442368)
    h = dsp.impulse_response(770, 8000, n_samples=200)

    t0 = time.perf_counter()
    dsp.convolve_signals(x_big, h)
    t1 = time.perf_counter()
    print(f"  convolve_signals() (vectorized): {(t1 - t0) * 1000:.1f} ms")
    print("  DiscreteSignal/LTISystem reference on this input: not run — extrapolated ~9.6 HOURS")

    print()
    print("ALL PASSED" if all_passed else "SOME CHECKS FAILED")
    return all_passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
