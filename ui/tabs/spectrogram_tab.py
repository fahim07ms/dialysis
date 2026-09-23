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

Spectrogram appearance notes:
  * We use a standard "plasma" colormap (dark → blue → yellow → white) so
    the plot looks like every other spectrogram the user has seen — soft teal
    was pretty but unreadable for frequency-band identification.
  * dB-scaled: 60 dB dynamic range below peak, so quiet background doesn't
    wash out the tone bands.
  * Frequency-marker lines for all 8 DTMF bins are overlaid so the viewer
    can immediately see which rows/columns the decoder is looking at.
  * Live mic: update_audio_live() accepts the rolling mic buffer (bounded
    window) and re-triggers the spectrogram on every live tick, so the
    spectrogram stays fresh while listening.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QCheckBox, QHBoxLayout
from PyQt6.QtGui import QPainter
from PyQt6.QtCore import QThread, pyqtSignal

from core import dsp_interface as dsp
from ui.theme import SURFACE, TEXT_MUTED, TEXT


# ---------------------------------------------------------------------------
# Colormap — plasma-style (matches what SciPy/matplotlib default looks like)
# ---------------------------------------------------------------------------

def _plasma_colormap() -> pg.ColorMap:
    """
    Approximation of the matplotlib 'plasma' colormap:
    black -> deep purple -> magenta -> orange -> yellow -> white.
    This matches what "a typical spectrogram" looks like in every other
    tool (SciPy, Audacity, Python/matplotlib, etc.).
    """
    positions = [0.0,  0.15, 0.35, 0.55, 0.75, 0.90, 1.0]
    colors = [
        (13,  8,   135, 255),   # near-black / deep indigo
        (84,  2,   163, 255),   # purple
        (163, 0,   121, 255),   # magenta-purple
        (212, 50,  85,  255),   # hot pink / red-orange
        (243, 131, 39,  255),   # orange
        (252, 211, 58,  255),   # yellow
        (240, 249, 33,  255),   # near-white yellow
    ]
    return pg.ColorMap(positions, colors)


# ---------------------------------------------------------------------------
# Background thread — keeps compute_spectrogram() off the UI thread
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class SpectrogramTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        section_label = QLabel("DYNAMIC SPECTROGRAM")
        section_label.setProperty("role", "section")
        layout.addWidget(section_label)

        # Controls row
        controls_row = QHBoxLayout()
        self._show_markers_cb = QCheckBox("Show DTMF frequency markers")
        self._show_markers_cb.setChecked(True)
        self._show_markers_cb.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px;")
        self._show_markers_cb.stateChanged.connect(self._toggle_markers)
        controls_row.addWidget(self._show_markers_cb)
        controls_row.addStretch(1)
        layout.addLayout(controls_row)

        # Main plot
        self.plot = pg.PlotWidget()
        self.plot.setBackground(SURFACE)
        self.plot.getAxis("left").setTextPen(TEXT_MUTED)
        self.plot.getAxis("bottom").setTextPen(TEXT_MUTED)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Frequency", units="Hz")
        self.plot.setMenuEnabled(False)
        # Smooth scaling so the discrete-bin array doesn't look blocky when
        # stretched to fill the widget (same as every other spectrogram viewer).
        self.plot.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )

        self.image_item = pg.ImageItem()
        self.image_item.setColorMap(_plasma_colormap())
        self.plot.addItem(self.image_item)
        layout.addWidget(self.plot, 1)

        # DTMF frequency marker lines
        # Low group (row tones) in teal, high group (col tones) in orange.
        self._freq_lines = []
        for i, f in enumerate(dsp.LOW_FREQS + dsp.HIGH_FREQS):
            color = "#54E0D0" if i < 4 else "#FF9430"
            line = pg.InfiniteLine(
                pos=f,
                angle=0,               # horizontal line at frequency f
                pen=pg.mkPen(color, width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
                label=f"{f} Hz",
                labelOpts={"position": 0.98, "color": color,
                           "fill": (0, 0, 0, 120), "movable": False},
            )
            self.plot.addItem(line)
            self._freq_lines.append(line)

        self.status_label = QLabel("Play or import a signal to see its spectrogram.")
        self.status_label.setProperty("role", "muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._thread = None
        self._request_id = 0  # guards against a stale result overwriting a newer one

        # Live mic: debounce rapid live updates — only re-trigger the
        # spectrogram thread when at least this many new samples arrived.
        self._live_min_new_samples = 2048
        self._live_pending_samples = None
        self._live_pending_fs = None

    # -- Public API ----------------------------------------------------------

    def update_audio(self, samples: np.ndarray, fs: int):
        """Full signal update (import / playback finished)."""
        samples = np.asarray(samples)
        if len(samples) == 0:
            return
        self.status_label.setText(
            f"Computing spectrogram for {len(samples)} samples @ {fs} Hz…"
        )
        self._trigger(samples, fs)

    def update_audio_live(self, samples: np.ndarray, fs: int):
        """
        Rolling live-mic update. Called every ~100 ms from the decoder tick.
        We debounce: only re-trigger when the buffer has grown enough that
        a new spectrogram frame would be meaningfully different.
        """
        samples = np.asarray(samples)
        n = len(samples)
        if n == 0:
            return

        # Check whether we already have a pending live samples reference
        if self._live_pending_samples is not None and \
                n - len(self._live_pending_samples) < self._live_min_new_samples:
            return  # not enough new data yet — skip this tick

        # Keep the last 5 s max so the spectrogram stays readable
        max_live_samples = int(5.0 * fs)
        if n > max_live_samples:
            samples = samples[-max_live_samples:]

        self._live_pending_samples = samples
        self._live_pending_fs = fs
        self.status_label.setText(
            f"Live spectrogram — {len(samples)/fs:.1f} s @ {fs} Hz…"
        )
        self._trigger(samples, fs)

    # -- Internal ------------------------------------------------------------

    def _trigger(self, samples: np.ndarray, fs: int):
        """Start a new background spectrogram computation, tagging it with
        the current request_id so stale results are silently dropped."""
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
        t   = np.asarray(t)
        f   = np.asarray(f)

        if Sxx.size == 0 or t.size < 2 or f.size < 2:
            self.image_item.clear()
            self.status_label.setText(
                "Spectrogram came back empty — signal may be too short."
            )
            return

        # ---- dB conversion with fixed dynamic range -------------------------
        # Log-magnitude display is standard: linear values make all but the
        # very loudest instant look like a flat blob. 60 dB dynamic range
        # below the peak keeps tone bands clearly visible without
        # letting the noise floor swamp the colour scale.
        Sxx_db = 10.0 * np.log10(Sxx + 1e-12)
        DYNAMIC_RANGE_DB = 60
        peak_db = float(np.max(Sxx_db))

        # ---- Image orientation ---------------------------------------------
        # scipy.signal.spectrogram returns Sxx with shape (n_freqs, n_time).
        # pyqtgraph's ImageItem.setImage expects (width, height) = (x, y).
        # With time on X and frequency on Y we need shape (n_time, n_freq),
        # i.e. Sxx.T.  Then setRect maps that (time, freq) space correctly.
        img_data = Sxx_db.T   # shape: (n_time, n_freq)

        self.image_item.setImage(
            img_data,
            levels=(peak_db - DYNAMIC_RANGE_DB, peak_db),
            autoLevels=False,
        )

        # Map pixel array indices to (time, frequency) data coordinates
        t0, t1 = float(t[0]), float(t[-1])
        f0, f1 = float(f[0]), float(f[-1])
        self.image_item.setRect(t0, f0, t1 - t0, f1 - f0)

        # Keep the view range tidy — pyqtgraph may drift after setRect
        self.plot.setXRange(t0, t1, padding=0.02)
        self.plot.setYRange(f0, f1, padding=0.02)

        self.status_label.setText(
            f"{Sxx.shape[0]} frequency bins × {Sxx.shape[1]} time frames  "
            f"(~{(t1 - t0):.2f} s,  freq resolution ~{(f[1]-f[0]):.1f} Hz/bin)"
        )

    def _toggle_markers(self, state):
        visible = bool(state)
        for line in self._freq_lines:
            line.setVisible(visible)
