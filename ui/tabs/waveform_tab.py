"""
waveform_tab.py
================
Phase 3 tab: time-domain waveform + magnitude spectrum.

The waveform plot needs no DSP at all — it's just the raw samples, so it
always works. The spectrum plot calls dsp.compute_spectrum(); until you
implement that, it shows a friendly placeholder instead of an empty graph.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

from core import dsp_interface as dsp
from ui.theme import PRIMARY, SURFACE, TEXT_MUTED


def _styled_plot() -> pg.PlotWidget:
    plot = pg.PlotWidget()
    plot.setBackground(SURFACE)
    plot.showGrid(x=True, y=True, alpha=0.15)
    plot.getAxis("left").setTextPen(TEXT_MUTED)
    plot.getAxis("bottom").setTextPen(TEXT_MUTED)
    plot.setMenuEnabled(False)
    return plot


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
        layout.addWidget(self.wave_plot, 1)

        spec_label = QLabel("MAGNITUDE SPECTRUM")
        spec_label.setProperty("role", "section")
        layout.addWidget(spec_label)

        self.spectrum_plot = _styled_plot()
        self.spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        self.spectrum_plot.setLabel("left", "Magnitude")
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen(PRIMARY, width=2))
        layout.addWidget(self.spectrum_plot, 1)

        self.status_label = QLabel("Play or import a signal to see it here.")
        self.status_label.setProperty("role", "muted")
        layout.addWidget(self.status_label)

    def update_audio(self, samples: np.ndarray, fs: int):
        samples = np.asarray(samples)
        n = len(samples)
        if n == 0:
            return

        t = np.arange(n) / fs
        self.wave_curve.setData(t, samples)

        try:
            freqs, mags = dsp.compute_spectrum(samples, fs)
        except NotImplementedError:
            self.spectrum_curve.setData([], [])
            self.status_label.setText(
                "Waveform updated. Spectrum needs compute_spectrum() (Phase 3)."
            )
            return

        self.spectrum_curve.setData(freqs, mags)
        self.status_label.setText(f"Showing {n} samples @ {fs} Hz.")
