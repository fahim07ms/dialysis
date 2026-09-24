"""
noise_sampling_tab.py
========================
Phase 10 tab: two sliders that let you actively break the decoder, on
purpose, to see exactly where and why it breaks.

SNR slider — injects AWGN via add_awgn_noise(). Shows how decode accuracy
degrades as noise increases.

Sample Rate slider — resamples via resample_signal(). Demonstrates
aliasing directly: DTMF's highest frequency is 1633 Hz, so the
Nyquist-Shannon theorem requires fs > 3266 Hz to represent it without
aliasing. Drop below that and decoding breaks — not gracefully degrades,
just breaks, because the signal itself no longer contains the right
information once frequencies fold back on top of each other.

Both sliders use sliderReleased (not valueChanged) to trigger the actual
recompute — recomputing on every intermediate value while dragging would
mean firing add_awgn_noise/resample_signal/decode dozens of times a
second on a signal that could be hundreds of thousands of samples long.
The value LABEL still updates live via valueChanged (cheap), so the
slider doesn't feel unresponsive — only the expensive recompute is
throttled to "once you've picked a value."
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QFrame, QPushButton
)
from PyQt6.QtGui import QPainter
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from core import dsp_interface as dsp
from ui.theme import SURFACE, TEXT_MUTED, PRIMARY

NYQUIST_MIN_FS = int(2 * max(dsp.HIGH_FREQS)) + 1  # 3267 Hz — the true aliasing boundary


def _styled_plot() -> pg.PlotWidget:
    plot = pg.PlotWidget()
    plot.setBackground(SURFACE)
    plot.showGrid(x=True, y=True, alpha=0.15)
    plot.getAxis("left").setTextPen(TEXT_MUTED)
    plot.getAxis("bottom").setTextPen(TEXT_MUTED)
    plot.setMenuEnabled(False)
    plot.setRenderHints(QPainter.RenderHint.Antialiasing)
    return plot


class _ProcessThread(QThread):
    """Noise injection + resampling + decode, off the UI thread — same
    reasoning as every other 'could be slow on a long recording' step in
    this app."""
    finished_processing = pyqtSignal(object, object, object, str, str)
    # (processed_signal, spectrum_freqs, spectrum_mags, decoded_digits, error)

    def __init__(self, x, fs, snr_db, fs_new, parent=None):
        super().__init__(parent)
        self.x = x
        self.fs = fs
        self.snr_db = snr_db
        self.fs_new = fs_new

    def run(self):
        try:
            processed = dsp.add_awgn_noise(self.x, self.snr_db)
            if self.fs_new != self.fs:
                processed = dsp.resample_signal(processed, self.fs, self.fs_new)
        except NotImplementedError as e:
            self.finished_processing.emit(None, None, None, "", str(e))
            return

        try:
            freqs, mags = dsp.compute_spectrum(processed, self.fs_new)
        except NotImplementedError:
            freqs, mags = None, None

        try:
            digits = dsp.goertzel_decode(processed, self.fs_new)
        except NotImplementedError:
            digits = ""

        self.finished_processing.emit(processed, freqs, mags, digits, "")


class NoiseSamplingTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._x = None
        self._fs = 8000
        self._original_digits = ""
        self._thread = None
        self._request_id = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        section_label = QLabel("BREAK THE DECODER, ON PURPOSE")
        section_label.setProperty("role", "section")
        layout.addWidget(section_label)

        # -- SNR slider --
        snr_row = QHBoxLayout()
        snr_title = QLabel("Signal-to-Noise Ratio")
        snr_title.setProperty("role", "muted")
        self.snr_value_label = QLabel("30 dB")
        self.snr_value_label.setProperty("role", "muted")
        snr_row.addWidget(snr_title)
        snr_row.addStretch(1)
        snr_row.addWidget(self.snr_value_label)
        layout.addLayout(snr_row)

        self.snr_slider = QSlider(Qt.Orientation.Horizontal)
        self.snr_slider.setRange(-10, 40)
        self.snr_slider.setValue(30)
        self.snr_slider.valueChanged.connect(self._on_snr_value_changed)
        self.snr_slider.sliderReleased.connect(self._recompute)
        layout.addWidget(self.snr_slider)

        # -- Sample rate slider --
        fs_row = QHBoxLayout()
        fs_title = QLabel("Sample Rate")
        fs_title.setProperty("role", "muted")
        self.fs_value_label = QLabel("8000 Hz")
        self.fs_value_label.setProperty("role", "muted")
        fs_row.addWidget(fs_title)
        fs_row.addStretch(1)
        fs_row.addWidget(self.fs_value_label)
        layout.addLayout(fs_row)

        self.fs_slider = QSlider(Qt.Orientation.Horizontal)
        self.fs_slider.setRange(1500, 8000)
        self.fs_slider.setValue(8000)
        self.fs_slider.valueChanged.connect(self._on_fs_value_changed)
        self.fs_slider.sliderReleased.connect(self._recompute)
        layout.addWidget(self.fs_slider)

        nyquist_note = QLabel(
            f"DTMF's highest tone is {max(dsp.HIGH_FREQS)} Hz, so Nyquist requires "
            f"fs > {NYQUIST_MIN_FS - 1} Hz to avoid aliasing — try dropping below that."
        )
        nyquist_note.setProperty("role", "muted")
        nyquist_note.setWordWrap(True)
        layout.addWidget(nyquist_note)

        # -- Spectrum plot of the processed signal --
        spec_label = QLabel("SPECTRUM AFTER NOISE + RESAMPLING")
        spec_label.setProperty("role", "section")
        layout.addWidget(spec_label)
        self.spectrum_plot = _styled_plot()
        self.spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        self.spectrum_plot.setLabel("left", "Magnitude")
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen(PRIMARY, width=2))
        self.spectrum_curve.setDownsampling(auto=True, method="peak")
        self.spectrum_curve.setClipToView(True)
        layout.addWidget(self.spectrum_plot, 1)

        # -- Result card --
        result_card = QFrame()
        result_card.setProperty("role", "card")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(20, 16, 20, 16)
        result_layout.setSpacing(6)
        self.result_label = QLabel("—")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 24px; font-weight: 800; letter-spacing: 3px;")
        result_layout.addWidget(self.result_label)

        # -- Export Button --
        self.export_btn = QPushButton("Export Processed WAV")
        self.export_btn.setProperty("role", "pill-outline")
        self.export_btn.clicked.connect(self._on_export_clicked)
        result_layout.addWidget(self.export_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(result_card)

        self.status_label = QLabel("Play or import a signal to start breaking it.")
        self.status_label.setProperty("role", "muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _on_snr_value_changed(self, value: int):
        self.snr_value_label.setText(f"{value} dB")

    def _on_fs_value_changed(self, value: int):
        self.fs_value_label.setText(f"{value} Hz")

    def _on_export_clicked(self):
        if not hasattr(self, '_processed_samples') or self._processed_samples is None or len(self._processed_samples) == 0:
            self.status_label.setText("No processed audio to export.")
            return
        from PyQt6.QtWidgets import QFileDialog
        from core import audio_io
        path, _ = QFileDialog.getSaveFileName(self, "Export Processed WAV", "processed.wav", "WAV files (*.wav)")
        if path:
            audio_io.save_wav(path, self._processed_samples, self.fs_slider.value())
            self.status_label.setText(f"Exported processed audio to {path}")

    def update_audio(self, samples: np.ndarray, fs: int):
        self._x = np.asarray(samples)
        self._fs = fs
        self.fs_slider.setRange(1500, fs)
        if self.fs_slider.value() > fs:
            self.fs_slider.setValue(fs)
        try:
            self._original_digits = dsp.goertzel_decode(self._x, fs) if len(self._x) else ""
        except NotImplementedError:
            self._original_digits = ""
        self._recompute()

    def _recompute(self):
        if self._x is None or len(self._x) == 0:
            self.status_label.setText("Play or import a signal to start breaking it.")
            return

        snr_db = self.snr_slider.value()
        fs_new = self.fs_slider.value()
        self.status_label.setText("Processing…")

        self._request_id += 1
        request_id = self._request_id
        self._thread = _ProcessThread(self._x, self._fs, snr_db, fs_new)
        self._thread.finished_processing.connect(
            lambda proc, freqs, mags, digits, err, rid=request_id, snr=snr_db, fsn=fs_new:
                self._on_done(rid, snr, fsn, proc, freqs, mags, digits, err)
        )
        self._thread.start()

    def _on_done(self, request_id, snr_db, fs_new, processed, freqs, mags, digits, err):
        if request_id != self._request_id:
            return

        if err:
            self.status_label.setText(err)
            return

        self._processed_samples = processed

        if freqs is not None:
            self.spectrum_curve.setData(freqs, mags)

        self.result_label.setText(digits if digits else "(nothing decoded)")

        matches = digits == self._original_digits
        aliasing_risk = fs_new < NYQUIST_MIN_FS
        parts = [f"SNR {snr_db} dB, {fs_new} Hz."]
        if self._original_digits:
            parts.append(f"Clean decode was {self._original_digits!r} — {'still matches' if matches else 'BROKEN, no longer matches'}.")
        if aliasing_risk:
            parts.append(f"Below the {NYQUIST_MIN_FS-1} Hz Nyquist boundary — aliasing expected.")
        self.status_label.setText(" ".join(parts))
