"""
spectrogram_tab.py
====================
Phase 7 tab: dynamic spectrogram — time on x, frequency on y, color/brightness
shows intensity. This is basically "watch the magnitude spectrum change over
time" as a picture instead of a movie.

Runs on a background thread, same as the Waveform tab's full analysis.
Lesson learned from the Phase 3 stutter bug: a spectrogram over a long
recording is just as capable of taking seconds to compute as a single big
FFT was, so we don't assume it'll be fast — we just never let it block the
UI thread, whether it turns out fast or slow.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QPainter
from PyQt6.QtCore import QThread, pyqtSignal

from core import dsp_interface as dsp
from ui.theme import SURFACE, TEXT_MUTED


def _teal_colormap() -> pg.ColorMap:
    """Dark background -> deep teal -> the app's PRIMARY teal -> white for
    the hottest spots. Keeps the spectrogram visually part of the same
    theme as everything else instead of a default rainbow/viridis look."""
    positions = [0.0, 0.4, 0.75, 1.0]
    colors = [
        (27, 28, 31, 255),      # near-BG (quiet / no energy)
        (0, 110, 100, 255),     # deep teal
        (84, 224, 208, 255),    # PRIMARY teal
        (255, 255, 255, 255),   # white (loudest)
    ]
    return pg.ColorMap(positions, colors)


class _SpectrogramThread(QThread):
    """Runs dsp.compute_spectrogram() off the UI thread — see module docstring."""
    finished_spectrogram = pyqtSignal(object, object, object, str)
    # (t_or_None, f_or_None, Sxx_or_None, error_message)

    def __init__(self, samples, fs, parent=None):
        super().__init__(parent)
        self.samples = samples
        self.fs = fs

    def run(self):
        try:
            t, f, Sxx = dsp.compute_spectrogram(self.samples, self.fs)
        except NotImplementedError:
            self.finished_spectrogram.emit(
                None, None, None,
                "Spectrogram needs compute_spectrogram() (Phase 7)."
            )
            return
        self.finished_spectrogram.emit(t, f, Sxx, "")


class SpectrogramTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        section_label = QLabel("DYNAMIC SPECTROGRAM")
        section_label.setProperty("role", "section")
        layout.addWidget(section_label)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(SURFACE)
        self.plot.getAxis("left").setTextPen(TEXT_MUTED)
        self.plot.getAxis("bottom").setTextPen(TEXT_MUTED)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Frequency", units="Hz")
        self.plot.setMenuEnabled(False)
        # PlotWidget IS a QGraphicsView under the hood. Without this, the
        # (relatively low-resolution) spectrogram array gets scaled up to
        # fill the plot area using nearest-neighbor sampling — every array
        # cell becomes a hard-edged rectangle, which is exactly the "blocky,
        # not like a typical spectrogram" look. This makes it interpolate
        # smoothly instead, like every other spectrogram viewer does.
        self.plot.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)

        self.image_item = pg.ImageItem()
        self.image_item.setColorMap(_teal_colormap())
        self.plot.addItem(self.image_item)
        layout.addWidget(self.plot, 1)

        self.status_label = QLabel("Play or import a signal to see its spectrogram.")
        self.status_label.setProperty("role", "muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._thread = None
        self._request_id = 0  # guards against a stale result overwriting a newer one

    def update_audio(self, samples: np.ndarray, fs: int):
        samples = np.asarray(samples)
        if len(samples) == 0:
            return

        self.status_label.setText(f"Computing spectrogram for {len(samples)} samples @ {fs} Hz…")

        self._request_id += 1
        request_id = self._request_id

        self._thread = _SpectrogramThread(samples, fs)
        self._thread.finished_spectrogram.connect(
            lambda t, f, Sxx, err, rid=request_id: self._on_done(rid, t, f, Sxx, err)
        )
        self._thread.start()

    def _on_done(self, request_id, t, f, Sxx, err):
        if request_id != self._request_id:
            return  # a newer signal arrived while this was computing — discard

        if err:
            self.image_item.clear()
            self.status_label.setText(err)
            return

        Sxx = np.asarray(Sxx)
        t = np.asarray(t)
        f = np.asarray(f)

        if Sxx.size == 0 or t.size < 2 or f.size < 2:
            self.image_item.clear()
            self.status_label.setText("Spectrogram came back empty — signal may be too short.")
            return

        # Spectrograms are almost always viewed in log magnitude — linear
        # values tend to make everything but the very loudest instant look
        # like a flat blob, since tone energy dwarfs everything else.
        Sxx_db = 10 * np.log10(Sxx + 1e-12)

        # Clip to a fixed dynamic range below the peak, rather than raw
        # autoLevels (literal min/max). A spectrogram's floor is often
        # extremely quiet (near -120dB from the 1e-12 epsilon above), and
        # letting that set the bottom of the color scale compresses all the
        # *meaningful* contrast into a narrow band near the top — this is
        # why real spectrogram viewers show a fixed range (commonly ~60dB)
        # below the loudest point instead.
        DYNAMIC_RANGE_DB = 60
        peak_db = np.max(Sxx_db)
        self.image_item.setImage(Sxx_db.T, levels=(peak_db - DYNAMIC_RANGE_DB, peak_db))
        t0, t1 = float(t[0]), float(t[-1])
        f0, f1 = float(f[0]), float(f[-1])
        self.image_item.setRect(t0, f0, t1 - t0, f1 - f0)

        self.status_label.setText(f"{Sxx.shape[0]} frequency bins × {Sxx.shape[1]} time bins.")
