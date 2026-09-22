"""
pole_zero_tab.py
==================
Phase 8 tab: Z-plane pole-zero visualizer for the Goertzel resonators.

Shows WHY Goertzel resonates at exactly the 8 DTMF frequencies: each one
corresponds to a conjugate pole PAIR sitting exactly ON the unit circle
(dsp.goertzel_poles — see its docstring for the derivation), at angle
+-omega = +-2*pi*f/fs. A pole on the unit circle means a marginally
stable resonator: it neither decays away nor blows up on its own — it
just keeps ringing at that exact frequency once excited.

The ZERO plotted alongside each pole pair is NOT from dsp_interface.py.
The classic single-bin Goertzel power filter has a well-known transfer
function with one zero at z = e^{-j*omega} (the same angle as the pole's
conjugate partner). That's a one-line formula, so it's computed directly
here rather than adding a second dsp_interface.py function for it.

WORTH KNOWING: that zero lands EXACTLY on top of the lower-half pole. This
isn't a rendering bug — it's a genuinely well-known property of the
algorithm. Goertzel only cares about ONE frequency, +omega; the conjugate
pole at -omega is a mathematical byproduct of using a real-valued 2nd-order
recursion (avoiding complex arithmetic in the main loop) instead of a
single complex-valued one. The zero exists specifically to cancel that
unwanted pole back out in the final output, leaving one clean resonance.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtGui import QPainter
from PyQt6.QtCore import Qt

from core import dsp_interface as dsp
from ui.theme import SURFACE, TEXT_MUTED, PRIMARY, TEXT

ZERO_COLOR = "#F2C744"  # warm gold, distinct from the teal poles
DEFAULT_FS = 8000


class PoleZeroTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        section_label = QLabel("Z-PLANE — GOERTZEL RESONATOR POLES & ZEROS")
        section_label.setProperty("role", "section")
        layout.addWidget(section_label)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(SURFACE)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.getAxis("left").setTextPen(TEXT_MUTED)
        self.plot.getAxis("bottom").setTextPen(TEXT_MUTED)
        self.plot.setLabel("bottom", "Re(z)")
        self.plot.setLabel("left", "Im(z)")
        self.plot.setAspectLocked(True)  # a circle needs to actually look like a circle
        self.plot.setMenuEnabled(False)
        self.plot.setXRange(-1.3, 1.3)
        self.plot.setYRange(-1.3, 1.3)
        self.plot.setRenderHints(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self.plot, 1)

        legend_row = QHBoxLayout()
        pole_legend = QLabel("✕  Pole")
        pole_legend.setStyleSheet(f"color: {PRIMARY}; font-weight: 700;")
        zero_legend = QLabel("○  Zero")
        zero_legend.setStyleSheet(f"color: {ZERO_COLOR}; font-weight: 700;")
        legend_row.addWidget(pole_legend)
        legend_row.addSpacing(16)
        legend_row.addWidget(zero_legend)
        legend_row.addStretch(1)
        layout.addLayout(legend_row)

        self.status_label = QLabel("")
        self.status_label.setProperty("role", "muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._draw()

    def _draw(self):
        # Unit circle — the stability boundary. Every Goertzel pole sits
        # exactly on it (see module docstring: marginal stability).
        theta = np.linspace(0, 2 * np.pi, 200)
        self.plot.plot(
            np.cos(theta), np.sin(theta),
            pen=pg.mkPen(TEXT_MUTED, width=1, style=Qt.PenStyle.DashLine)
        )
        # Real/imaginary axes for reference
        self.plot.plot([-1.3, 1.3], [0, 0], pen=pg.mkPen(TEXT_MUTED, width=1))
        self.plot.plot([0, 0], [-1.3, 1.3], pen=pg.mkPen(TEXT_MUTED, width=1))

        target_freqs = dsp.LOW_FREQS + dsp.HIGH_FREQS

        try:
            poles = dsp.goertzel_poles(target_freqs, fs=DEFAULT_FS)
        except NotImplementedError:
            self.status_label.setText(
                "Needs goertzel_poles() implemented in core/dsp_interface.py (Phase 8)."
            )
            return

        pole_x = [p.real for p in poles]
        pole_y = [p.imag for p in poles]
        self.plot.plot(
            pole_x, pole_y, pen=None,
            symbol="x", symbolSize=16,
            symbolPen=pg.mkPen(PRIMARY, width=2), symbolBrush=None
        )

        # Zeros (see module docstring — computed directly here, not from
        # dsp_interface.py): one per frequency, at z = e^{-j*omega}.
        zero_x, zero_y = [], []
        for f in target_freqs:
            omega = 2 * np.pi * f / DEFAULT_FS
            zero_x.append(np.cos(omega))
            zero_y.append(-np.sin(omega))
        self.plot.plot(
            zero_x, zero_y, pen=None,
            symbol="o", symbolSize=12,
            symbolPen=pg.mkPen(ZERO_COLOR, width=2), symbolBrush=None
        )

        # Label each frequency near its upper-half pole (the conjugate below
        # mirrors it — one label per frequency is enough, not one per pole).
        for f in target_freqs:
            omega = 2 * np.pi * f / DEFAULT_FS
            label_x, label_y = np.cos(omega) * 1.16, np.sin(omega) * 1.16
            text = pg.TextItem(str(f), color=TEXT, anchor=(0.5, 0.5))
            text.setPos(label_x, label_y)
            self.plot.addItem(text)

        self.status_label.setText(
            f"{len(target_freqs)} DTMF frequencies -> {len(poles)} poles, all exactly on the "
            f"unit circle (marginally stable). Notice each zero sits exactly on top of its "
            f"resonator's lower pole — that's not an overlap glitch, it's how Goertzel cancels "
            f"out the unwanted mirror frequency, leaving one clean resonance per tone."
        )
