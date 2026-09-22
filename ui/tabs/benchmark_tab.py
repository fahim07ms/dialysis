"""
benchmark_tab.py
===================
Phase 11 tab: formalizes what the Decoder tab already hinted at informally
(Goertzel ~12ms vs FFT ~55ms on the same signal, back in Phase 6) into a
proper side-by-side comparison — time, memory, and whether the two
algorithms actually agree on the answer.

Runs on a background thread: benchmark_decoders() runs BOTH decoders
back-to-back specifically to measure them, so it's guaranteed to take at
least as long as running each one separately — never something to risk
blocking the UI thread with.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PyQt6.QtGui import QPainter
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from core import dsp_interface as dsp
from ui.theme import SURFACE, TEXT_MUTED, PRIMARY, PRIMARY_DIM


def _styled_plot() -> pg.PlotWidget:
    plot = pg.PlotWidget()
    plot.setBackground(SURFACE)
    plot.showGrid(y=True, alpha=0.15)
    plot.getAxis("left").setTextPen(TEXT_MUTED)
    plot.getAxis("bottom").setTextPen(TEXT_MUTED)
    plot.setMenuEnabled(False)
    plot.setRenderHints(QPainter.RenderHint.Antialiasing)
    return plot


class _BenchmarkThread(QThread):
    finished_benchmark = pyqtSignal(object, str)  # (results_dict_or_None, error)

    def __init__(self, x, fs, parent=None):
        super().__init__(parent)
        self.x = x
        self.fs = fs

    def run(self):
        try:
            results = dsp.benchmark_decoders(self.x, self.fs)
        except NotImplementedError as e:
            self.finished_benchmark.emit(None, str(e))
            return
        self.finished_benchmark.emit(results, "")


class BenchmarkTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._x = None
        self._fs = 8000
        self._thread = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        section_label = QLabel("FFT vs GOERTZEL — HEAD TO HEAD")
        section_label.setProperty("role", "section")
        layout.addWidget(section_label)

        self.run_btn = QPushButton("Run Benchmark")
        self.run_btn.setProperty("role", "pill-primary")
        self.run_btn.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.run_btn)

        charts_row = QHBoxLayout()
        charts_row.setSpacing(16)

        time_col = QVBoxLayout()
        time_title = QLabel("TIME (ms)")
        time_title.setProperty("role", "section")
        time_col.addWidget(time_title)
        self.time_plot = _styled_plot()
        self.time_plot.getAxis("bottom").setTicks([[(0, "Goertzel"), (1, "FFT")]])
        time_col.addWidget(self.time_plot)
        charts_row.addLayout(time_col)

        mem_col = QVBoxLayout()
        mem_title = QLabel("PEAK MEMORY (KB)")
        mem_title.setProperty("role", "section")
        mem_col.addWidget(mem_title)
        self.mem_plot = _styled_plot()
        self.mem_plot.getAxis("bottom").setTicks([[(0, "Goertzel"), (1, "FFT")]])
        mem_col.addWidget(self.mem_plot)
        charts_row.addLayout(mem_col)

        layout.addLayout(charts_row, 1)

        result_card = QFrame()
        result_card.setProperty("role", "card")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(20, 16, 20, 16)
        result_layout.setSpacing(6)
        self.agreement_label = QLabel("—")
        self.agreement_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.agreement_label.setStyleSheet("font-size: 20px; font-weight: 800;")
        result_layout.addWidget(self.agreement_label)
        layout.addWidget(result_card)

        self.status_label = QLabel("Play or import a signal, then press Run Benchmark.")
        self.status_label.setProperty("role", "muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def update_audio(self, samples: np.ndarray, fs: int):
        self._x = np.asarray(samples)
        self._fs = fs
        if len(self._x) > 0:
            self.status_label.setText(
                f"Ready to benchmark {len(self._x)} samples @ {fs} Hz. Press Run Benchmark."
            )

    def _on_run_clicked(self):
        if self._x is None or len(self._x) == 0:
            self.status_label.setText("Play or import a signal first.")
            return

        self.status_label.setText("Running both decoders…")
        self.run_btn.setEnabled(False)

        self._thread = _BenchmarkThread(self._x, self._fs)
        self._thread.finished_benchmark.connect(self._on_done)
        self._thread.start()

    def _on_done(self, results, err):
        self.run_btn.setEnabled(True)

        if err:
            self.status_label.setText(f"Needs benchmark_decoders() implemented (Phase 11): {err}")
            return

        g, f = results["goertzel"], results["fft"]

        self.time_plot.clear()
        self.time_plot.addItem(pg.BarGraphItem(
            x=[0, 1], height=[g["time_ms"], f["time_ms"]], width=0.6,
            brush=pg.mkBrush(PRIMARY), pen=pg.mkPen(PRIMARY_DIM)
        ))

        self.mem_plot.clear()
        self.mem_plot.addItem(pg.BarGraphItem(
            x=[0, 1], height=[g["memory_kb"], f["memory_kb"]], width=0.6,
            brush=pg.mkBrush(PRIMARY), pen=pg.mkPen(PRIMARY_DIM)
        ))

        agree = g["result"] == f["result"]
        self.agreement_label.setText(
            f"Goertzel: {g['result'] or '(empty)'}   |   FFT: {f['result'] or '(empty)'}"
        )

        speedup = f["time_ms"] / g["time_ms"] if g["time_ms"] > 0 else float("inf")
        self.status_label.setText(
            f"Goertzel: {g['time_ms']:.2f} ms, {g['memory_kb']:.1f} KB.  "
            f"FFT: {f['time_ms']:.2f} ms, {f['memory_kb']:.1f} KB.  "
            f"Goertzel was {speedup:.1f}x faster. "
            f"Results {'agree' if agree else 'DISAGREE'}."
        )
