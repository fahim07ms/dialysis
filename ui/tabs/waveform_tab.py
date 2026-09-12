"""
waveform_tab.py
================
Phase 3 tab: time-domain waveform + magnitude spectrum.

PERFORMANCE NOTES (read this before touching the timing constants below) —
these exist because of a real bug we found: compute_spectrum() on a long
recording can take SECONDS (it's an O(N log N) FFT written in pure Python
loops, not vectorized). Calling it on the whole growing array 30x/sec during
playback froze the UI for seconds at a stretch. Two fixes:

1. LIVE view (update_audio_live, called ~30x/sec during playback): only ever
   runs compute_spectrum() on a small BOUNDED trailing window (a fixed
   number of samples, however long the full recording gets), and only every
   few ticks — not every tick. Cost stays flat no matter how long you play.

2. FULL view (update_audio, called once when playback/import finishes):
   runs compute_spectrum() + segment_tone_regions() on the COMPLETE signal,
   which is the expensive part for a long recording. That now happens on a
   background QThread so the window never freezes, however long it takes.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QThread, pyqtSignal

from core import dsp_interface as dsp
from ui.theme import PRIMARY, SURFACE, TEXT_MUTED

# -- Live-view tuning -----------------------------------------------------
# Window length used for the spectrum DURING playback. Bounded on purpose —
# see the module docstring. The user's compute_spectrum() pads to the next
# power of two, so cost jumps in steps at those boundaries; 0.09s lands just
# under the 4096-sample step (measured ~20ms even at 44.1kHz), comfortably
# inside a single ~33ms tick.
LIVE_SPECTRUM_WINDOW_SECONDS = 0.09
# Recompute the live spectrum every Nth tick (ticks arrive every ~33ms from
# LeftPanel's timer). Every 2nd tick == ~66ms, giving the ~20ms compute
# comfortable headroom without hammering the CPU every single tick.
LIVE_SPECTRUM_EVERY_N_TICKS = 2


def _styled_plot() -> pg.PlotWidget:
    plot = pg.PlotWidget()
    plot.setBackground(SURFACE)
    plot.showGrid(x=True, y=True, alpha=0.15)
    plot.getAxis("left").setTextPen(TEXT_MUTED)
    plot.getAxis("bottom").setTextPen(TEXT_MUTED)
    plot.setMenuEnabled(False)
    return plot


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


class _FullAnalysisThread(QThread):
    """
    Runs the expensive whole-signal analysis (spectrum + segmentation) off
    the UI thread. Both core.dsp_interface functions are pure numpy/Python
    with no Qt calls, so they're safe to run here per the ui/core contract.
    """
    finished_analysis = pyqtSignal(object, object, object, list)
    # (freqs_or_None, mags_or_None, regions_or_None, error_messages)

    def __init__(self, samples, fs, parent=None):
        super().__init__(parent)
        self.samples = samples
        self.fs = fs

    def run(self):
        freqs, mags, regions = None, None, None
        errors = []

        try:
            freqs, mags = dsp.compute_spectrum(self.samples, self.fs)
        except NotImplementedError:
            errors.append("Spectrum needs compute_spectrum() (Phase 3).")

        try:
            regions = dsp.segment_tone_regions(self.samples, self.fs)
        except NotImplementedError:
            errors.append("Segmentation needs segment_tone_regions() (Phase 4).")

        self.finished_analysis.emit(freqs, mags, regions, errors)


class WaveformTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        wf_label = QLabel("TIME-DOMAIN WAVEFORM")
        wf_label.setProperty("role", "section")
        layout.addWidget(wf_label)

        self.wave_plot = _styled_plot()
        self.wave_plot.setLabel("bottom", "Time", units="s")
        self.wave_plot.setLabel("left", "Amplitude")
        self.wave_curve = self.wave_plot.plot(pen=pg.mkPen(PRIMARY, width=2))
        # Large arrays (long recordings) get downsampled for drawing only —
        # the underlying data setData() receives is untouched.
        self.wave_curve.setDownsampling(auto=True, method="peak")
        self.wave_curve.setClipToView(True)
        layout.addWidget(self.wave_plot, 1)

        spec_label = QLabel("MAGNITUDE SPECTRUM")
        spec_label.setProperty("role", "section")
        layout.addWidget(spec_label)

        self.spectrum_plot = _styled_plot()
        self.spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        self.spectrum_plot.setLabel("left", "Magnitude")
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen(PRIMARY, width=2))
        self.spectrum_curve.setDownsampling(auto=True, method="peak")
        self.spectrum_curve.setClipToView(True)
        layout.addWidget(self.spectrum_plot, 1)

        self.status_label = QLabel("Play or import a signal to see it here.")
        self.status_label.setProperty("role", "muted")
        layout.addWidget(self.status_label)

        self._segment_regions = []  # pg.LinearRegionItem list, cleared/redrawn each update
        self._live_tick_counter = 0
        self._analysis_thread = None
        self._analysis_request_id = 0  # guards against a stale/slow analysis overwriting a newer one

    # -- Live view: called ~30x/sec during playback ------------------------

    def update_audio_live(self, samples: np.ndarray, fs: int):
        samples = np.asarray(samples)
        n = len(samples)
        if n == 0:
            return

        # Waveform redraw is cheap (just a slice + setData) — do this every tick.
        t = np.arange(n) / fs
        self.wave_curve.setData(t, samples)
        self.status_label.setText(f"Playing… {n} samples so far.")

        # Spectrum is NOT cheap — bound the window AND throttle how often we do it.
        self._live_tick_counter += 1
        if self._live_tick_counter % LIVE_SPECTRUM_EVERY_N_TICKS != 0:
            return

        window_len = int(LIVE_SPECTRUM_WINDOW_SECONDS * fs)
        windowed = samples[-window_len:] if n > window_len else samples

        try:
            freqs, mags = dsp.compute_spectrum(windowed, fs)
        except NotImplementedError:
            pass  # the final update_audio() call will explain this in status_label
        else:
            self.spectrum_curve.setData(freqs, mags)

    # -- Full view: called once when playback/import finishes --------------

    def update_audio(self, samples: np.ndarray, fs: int):
        samples = np.asarray(samples)
        n = len(samples)
        if n == 0:
            return

        t = np.arange(n) / fs
        self.wave_curve.setData(t, samples)
        self._clear_segment_regions()
        self.status_label.setText(f"{n} samples @ {fs} Hz. Analyzing…")

        # Kick the expensive part to a background thread so a long recording
        # can't freeze the window. Tag the request so that if a newer signal
        # arrives before this finishes, we don't let the stale result win.
        self._analysis_request_id += 1
        request_id = self._analysis_request_id

        self._analysis_thread = _FullAnalysisThread(samples, fs)
        self._analysis_thread.finished_analysis.connect(
            lambda freqs, mags, regions, errors, rid=request_id:
                self._on_full_analysis_done(rid, n, fs, freqs, mags, regions, errors)
        )
        self._analysis_thread.start()

    def _on_full_analysis_done(self, request_id, n, fs, freqs, mags, regions, errors):
        if request_id != self._analysis_request_id:
            return  # a newer signal came in while this was still computing — discard

        if freqs is not None:
            self.spectrum_curve.setData(freqs, mags)
        else:
            self.spectrum_curve.setData([], [])

        status_parts = [f"{n} samples @ {fs} Hz."]

        if regions is not None:
            self._draw_segment_regions(regions, fs)
            status_parts.append(f"Detected {len(regions)} tone region(s).")

        status_parts.extend(errors)
        self.status_label.setText(" ".join(status_parts))

    def _clear_segment_regions(self):
        for region in self._segment_regions:
            self.wave_plot.removeItem(region)
        self._segment_regions = []

    def _draw_segment_regions(self, regions, fs: int):
        r, g, b = _hex_to_rgb(PRIMARY)
        for start, end in regions:
            band = pg.LinearRegionItem(
                values=(start / fs, end / fs),
                brush=pg.mkBrush(r, g, b, 60),
                pen=pg.mkPen(PRIMARY, width=1),
                movable=False,
            )
            band.setZValue(-10)
            self.wave_plot.addItem(band)
            self._segment_regions.append(band)
