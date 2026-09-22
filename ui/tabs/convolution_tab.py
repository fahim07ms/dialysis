"""
convolution_tab.py
====================
Phase 9 tab: two plots that show directly how/why a Goertzel resonator
detects a tone.

Top plot — h[n], the impulse response: what the resonator does all on its
own if you hit it with a single sharp pulse. Because its poles sit exactly
on the unit circle (see the Pole-Zero tab), it never decays — it just rings
forever at its target frequency.

Bottom plot — y[n] = x[n] * h[n], the discrete convolution of that
resonator with whatever signal is currently loaded (keypad, import, or
mic). This IS filtering: convolving with a resonator tuned to frequency f
builds up a strong response wherever the input actually contains
frequency f, and stays small everywhere else — the same underlying
principle goertzel_single_freq() uses internally, made visible.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PyQt6.QtGui import QPainter
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from core import dsp_interface as dsp
from ui.theme import SURFACE, TEXT_MUTED, PRIMARY

N_SAMPLES_H = 200  # length of the impulse response shown/used


def _styled_plot() -> pg.PlotWidget:
    plot = pg.PlotWidget()
    plot.setBackground(SURFACE)
    plot.showGrid(x=True, y=True, alpha=0.15)
    plot.getAxis("left").setTextPen(TEXT_MUTED)
    plot.getAxis("bottom").setTextPen(TEXT_MUTED)
    plot.setMenuEnabled(False)
    plot.setRenderHints(QPainter.RenderHint.Antialiasing)
    return plot


class _ConvolveThread(QThread):
    """Convolving a short h[n] against a long imported/recorded signal is
    O(len(x) * len(h)) — cheap for a short keypad sequence, but potentially
    slow for a long recording. Same lesson as every other 'full analysis'
    in this app: run it off the UI thread so a long signal can't freeze
    the window, whether it turns out fast or slow."""
    finished_convolve = pyqtSignal(object, str)  # (y_or_None, error_message)

    def __init__(self, x, h, parent=None):
        super().__init__(parent)
        self.x = x
        self.h = h

    def run(self):
        try:
            y = dsp.convolve_signals(self.x, self.h)
        except NotImplementedError:
            self.finished_convolve.emit(None, "Needs convolve_signals() (Phase 9).")
            return
        self.finished_convolve.emit(y, "")


class ConvolutionTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._x = None
        self._fs = 8000
        self._thread = None
        self._request_id = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        section_label = QLabel("IMPULSE RESPONSE & CONVOLUTION")
        section_label.setProperty("role", "section")
        layout.addWidget(section_label)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)
        freq_label = QLabel("Resonator frequency:")
        freq_label.setProperty("role", "muted")
        self.freq_combo = QComboBox()
        for f in dsp.LOW_FREQS + dsp.HIGH_FREQS:
            self.freq_combo.addItem(f"{f} Hz", f)
        self.freq_combo.currentIndexChanged.connect(self._recompute)
        controls_row.addWidget(freq_label)
        controls_row.addWidget(self.freq_combo, 1)
        layout.addLayout(controls_row)

        h_label = QLabel("h[n] — IMPULSE RESPONSE (the resonator ringing on its own)")
        h_label.setProperty("role", "section")
        layout.addWidget(h_label)
        self.h_plot = _styled_plot()
        self.h_plot.setLabel("bottom", "n (samples)")
        self.h_plot.setLabel("left", "h[n]")
        self.h_curve = self.h_plot.plot(pen=pg.mkPen(PRIMARY, width=2))
        self.h_curve.setDownsampling(auto=True, method="peak")
        self.h_curve.setClipToView(True)
        layout.addWidget(self.h_plot, 1)

        y_label = QLabel("y[n] = x[n] * h[n] — CONVOLVED WITH THE CURRENT SIGNAL")
        y_label.setProperty("role", "section")
        layout.addWidget(y_label)
        self.y_plot = _styled_plot()
        self.y_plot.setLabel("bottom", "n (samples)")
        self.y_plot.setLabel("left", "y[n]")
        self.y_curve = self.y_plot.plot(pen=pg.mkPen(PRIMARY, width=2))
        self.y_curve.setDownsampling(auto=True, method="peak")
        self.y_curve.setClipToView(True)
        layout.addWidget(self.y_plot, 1)

        self.status_label = QLabel("Play or import a signal to see it convolved with a resonator.")
        self.status_label.setProperty("role", "muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._recompute()

    def update_audio(self, samples: np.ndarray, fs: int):
        self._x = np.asarray(samples)
        self._fs = fs
        self._recompute()

    def _recompute(self):
        freq = self.freq_combo.currentData()
        if freq is None:
            return

        try:
            h = dsp.impulse_response(freq, self._fs, n_samples=N_SAMPLES_H)
        except NotImplementedError:
            self.status_label.setText("Needs impulse_response() implemented in core/dsp_interface.py (Phase 9).")
            self.h_curve.setData([], [])
            self.y_curve.setData([], [])
            return

        self.h_curve.setData(np.arange(len(h)), h)

        if self._x is None or len(self._x) == 0:
            self.status_label.setText(
                f"h[n] shown for {freq} Hz — notice it never decays (marginally stable). "
                f"Play or import a signal to see the convolution."
            )
            self.y_curve.setData([], [])
            return

        self.status_label.setText(f"Convolving {len(self._x)} samples with h[n]…")

        self._request_id += 1
        request_id = self._request_id
        self._thread = _ConvolveThread(self._x, h)
        self._thread.finished_convolve.connect(
            lambda y, err, rid=request_id, f=freq: self._on_convolve_done(rid, f, y, err)
        )
        self._thread.start()

    def _on_convolve_done(self, request_id, freq, y, err):
        if request_id != self._request_id:
            return  # a newer request superseded this one — discard

        if err:
            self.status_label.setText(err)
            self.y_curve.setData([], [])
            return

        y = np.asarray(y)
        self.y_curve.setData(np.arange(len(y)), y)
        self.status_label.setText(
            f"Resonator at {freq} Hz convolved with the current signal. Notice y[n] builds up "
            f"strongly wherever the input actually contains {freq} Hz, and stays small elsewhere — "
            f"this IS the mechanism goertzel_single_freq() uses internally to detect a tone."
        )
