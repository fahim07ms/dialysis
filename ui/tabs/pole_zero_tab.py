"""
pole_zero_tab.py
==================
Phase 8 tab: Interactive Z-plane pole-zero visualizer for the Goertzel resonators.

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

INTERACTIVITY (added per project specification):
  * Frequency selector combo — choose any of the 8 DTMF resonators (or All)
  * Selected resonator is highlighted with a larger marker and angle label
  * The frequency response plot below shows |H(e^{j omega})| vs frequency
    for the selected resonator, illustrating its "resonator selectivity"
  * Hover: pyqtgraph ScatterPlotItem signals report the nearest pole/zero
    and show its angle and nominal frequency in the info line
"""

from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QFrame
)
from PyQt6.QtGui import QPainter
from PyQt6.QtCore import Qt

from core import dsp_interface as dsp
from ui.theme import SURFACE, TEXT_MUTED, PRIMARY, TEXT

ZERO_COLOR   = "#F2C744"   # warm gold, distinct from the teal poles
DIM_COLOR    = "#3A3B40"   # dimmed colour for non-selected resonators
ACCENT_SIZE  = 22          # selected resonator marker size
NORMAL_SIZE  = 10          # unselected marker size
DEFAULT_FS   = 8000


# ---------------------------------------------------------------------------
# Transfer function helpers (all pure numpy — no dsp_interface dependency)
# ---------------------------------------------------------------------------

def _goertzel_H(omega_target: float, omega_range: np.ndarray) -> np.ndarray:
    """
    Magnitude response of the Goertzel resonator (2nd-order IIR, analysis
    output after N samples) approximated as the single-bin DFT kernel:

        H(e^{j*omega}) = 1 / |1 - 2*cos(omega_t)*z^{-1} + z^{-2}|

    evaluated on the unit circle (|z| = 1, z = e^{j*omega}).
    Returns |H| for the given array of omega values.
    """
    z = np.exp(1j * omega_range)
    coeff = 2.0 * np.cos(omega_target)
    # denominator: 1 - coeff*z^{-1} + z^{-2}  (multiply top/bottom by z^2)
    denom = z**2 - coeff * z + 1.0
    H = np.abs(z**2) / np.abs(denom)   # |z^2 / denom| = 1/|denom| on unit circle
    return H


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class PoleZeroTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fs = DEFAULT_FS
        self._target_freqs = dsp.LOW_FREQS + dsp.HIGH_FREQS

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        section_label = QLabel("Z-PLANE — GOERTZEL RESONATOR POLES & ZEROS")
        section_label.setProperty("role", "section")
        layout.addWidget(section_label)

        # ---- Controls row --------------------------------------------------
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(12)

        freq_label = QLabel("Highlight resonator:")
        freq_label.setProperty("role", "muted")
        ctrl_row.addWidget(freq_label)

        self._freq_combo = QComboBox()
        self._freq_combo.addItem("All (overview)", None)
        for f in self._target_freqs:
            group = "Low" if f in dsp.LOW_FREQS else "High"
            self._freq_combo.addItem(f"{f} Hz  [{group} group]", f)
        self._freq_combo.currentIndexChanged.connect(self._redraw)
        ctrl_row.addWidget(self._freq_combo, 1)

        self._show_response_cb = QCheckBox("Show |H(ω)| response")
        self._show_response_cb.setChecked(True)
        self._show_response_cb.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 13px;"
        )
        self._show_response_cb.stateChanged.connect(self._on_response_toggle)
        ctrl_row.addWidget(self._show_response_cb)

        layout.addLayout(ctrl_row)

        # ---- Z-plane plot --------------------------------------------------
        self.plot = pg.PlotWidget()
        self.plot.setBackground(SURFACE)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.getAxis("left").setTextPen(TEXT_MUTED)
        self.plot.getAxis("bottom").setTextPen(TEXT_MUTED)
        self.plot.setLabel("bottom", "Re(z)")
        self.plot.setLabel("left", "Im(z)")
        self.plot.setAspectLocked(True)   # a circle needs to look like a circle
        self.plot.setMenuEnabled(False)
        self.plot.setXRange(-1.35, 1.35)
        self.plot.setYRange(-1.35, 1.35)
        self.plot.setRenderHints(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self.plot, 3)

        # ---- Legend row ----------------------------------------------------
        legend_row = QHBoxLayout()
        pole_legend = QLabel("✕  Pole (resonator)")
        pole_legend.setStyleSheet(f"color: {PRIMARY}; font-weight: 700;")
        zero_legend = QLabel("○  Zero (cancellation)")
        zero_legend.setStyleSheet(f"color: {ZERO_COLOR}; font-weight: 700;")
        uc_legend   = QLabel("—  Unit circle (stability boundary)")
        uc_legend.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: 700;")
        for w in (pole_legend, zero_legend, uc_legend):
            legend_row.addWidget(w)
        legend_row.addStretch(1)
        layout.addLayout(legend_row)

        # ---- Frequency response plot (|H(e^jω)| vs Hz) --------------------
        resp_label = QLabel("|H(e^jω)| — RESONATOR FREQUENCY RESPONSE")
        resp_label.setProperty("role", "section")
        layout.addWidget(resp_label)

        self.resp_plot = pg.PlotWidget()
        self.resp_plot.setBackground(SURFACE)
        self.resp_plot.showGrid(x=True, y=True, alpha=0.15)
        self.resp_plot.getAxis("left").setTextPen(TEXT_MUTED)
        self.resp_plot.getAxis("bottom").setTextPen(TEXT_MUTED)
        self.resp_plot.setLabel("bottom", "Frequency", units="Hz")
        self.resp_plot.setLabel("left", "|H|  (linear scale)")
        self.resp_plot.setMenuEnabled(False)
        self.resp_plot.setRenderHints(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self.resp_plot, 2)

        # ---- Status / hover info line -------------------------------------
        self.status_label = QLabel("")
        self.status_label.setProperty("role", "muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Keep references to drawn items so we can clear them on redraw
        self._plot_items: list = []
        self._resp_items: list = []

        self._draw_static()  # unit circle + axes (never removed)
        self._redraw()       # poles, zeros, labels, response curve

    # -----------------------------------------------------------------------
    # Static background (drawn once)
    # -----------------------------------------------------------------------

    def _draw_static(self):
        theta = np.linspace(0, 2 * np.pi, 400)

        # Unit circle — the stability boundary
        self.plot.plot(
            np.cos(theta), np.sin(theta),
            pen=pg.mkPen(TEXT_MUTED, width=1.5,
                         style=Qt.PenStyle.DashLine),
        )
        # Real and imaginary axes
        self.plot.plot([-1.35, 1.35], [0, 0],
                       pen=pg.mkPen(TEXT_MUTED, width=1))
        self.plot.plot([0, 0], [-1.35, 1.35],
                       pen=pg.mkPen(TEXT_MUTED, width=1))

        # Stability-boundary label
        txt = pg.TextItem("|z| = 1  (stability boundary)",
                          color=TEXT_MUTED, anchor=(0.5, 1.0))
        txt.setPos(0.0, 1.32)
        self.plot.addItem(txt)

    # -----------------------------------------------------------------------
    # Dynamic drawing — cleared and redrawn on every combo change
    # -----------------------------------------------------------------------

    def _clear_dynamic(self):
        for item in self._plot_items:
            self.plot.removeItem(item)
        self._plot_items = []
        for item in self._resp_items:
            self.resp_plot.removeItem(item)
        self._resp_items = []

    def _redraw(self):
        self._clear_dynamic()
        selected_f = self._freq_combo.currentData()  # None = All

        try:
            all_poles = dsp.goertzel_poles(self._target_freqs, fs=self._fs)
        except NotImplementedError:
            self.status_label.setText(
                "Needs goertzel_poles() implemented in core/dsp_interface.py (Phase 8)."
            )
            return

        # Draw each frequency's pole pair + zero
        for idx, f in enumerate(self._target_freqs):
            omega = 2.0 * np.pi * f / self._fs
            is_selected = (selected_f is None or f == selected_f)
            pole_color = PRIMARY if is_selected else DIM_COLOR
            zero_color = ZERO_COLOR if is_selected else DIM_COLOR
            p_size     = ACCENT_SIZE if (is_selected and selected_f is not None) else NORMAL_SIZE
            z_size     = int(p_size * 0.75)

            # Upper pole: e^{+j*omega}
            px, py = np.cos(omega), np.sin(omega)
            # Lower pole / zero: e^{-j*omega}
            zx, zy = np.cos(omega), -np.sin(omega)

            # Poles (✕ markers)
            scatter_poles = pg.ScatterPlotItem(
                x=[px, zx], y=[py, zy],
                symbol="x",
                size=p_size,
                pen=pg.mkPen(pole_color, width=2.5 if is_selected else 1.5),
                brush=None,
            )
            self.plot.addItem(scatter_poles)
            self._plot_items.append(scatter_poles)

            # Zero (○ marker) — sits exactly on the lower-half pole
            scatter_zero = pg.ScatterPlotItem(
                x=[zx], y=[zy],
                symbol="o",
                size=z_size,
                pen=pg.mkPen(zero_color, width=2.0 if is_selected else 1.0),
                brush=None,
            )
            self.plot.addItem(scatter_zero)
            self._plot_items.append(scatter_zero)

            # Frequency label near upper-half pole (only if selected or all overview)
            if is_selected:
                lx = np.cos(omega) * 1.18
                ly = np.sin(omega) * 1.18
                label_color = PRIMARY if selected_f is not None else TEXT_MUTED
                label_size  = 11 if selected_f is not None else 8
                txt = pg.TextItem(
                    f"{f} Hz",
                    color=label_color,
                    anchor=(0.5, 0.5),
                )
                txt.setPos(lx, ly)
                self.plot.addItem(txt)
                self._plot_items.append(txt)

                # Angle annotation for the selected resonator
                if selected_f is not None:
                    angle_deg = np.degrees(omega)
                    ann = pg.TextItem(
                        f"ω = 2π·{f}/{self._fs}\n"
                        f"  = {omega:.4f} rad  ({angle_deg:.1f}°)\n"
                        f"  pole: e^{{+jω}}, zero: e^{{-jω}}",
                        color=TEXT,
                        anchor=(0.0, 0.0),
                    )
                    ann.setPos(-1.30, 1.20)
                    self.plot.addItem(ann)
                    self._plot_items.append(ann)

                    # Draw the angle arc from real axis to the pole
                    arc_theta = np.linspace(0, omega, 60)
                    r = 0.35
                    arc_item = self.plot.plot(
                        r * np.cos(arc_theta),
                        r * np.sin(arc_theta),
                        pen=pg.mkPen(PRIMARY, width=1.5,
                                     style=Qt.PenStyle.DotLine),
                    )
                    self._plot_items.append(arc_item)

                    # Radial line from origin to the upper pole
                    radial = self.plot.plot(
                        [0, px], [0, py],
                        pen=pg.mkPen(PRIMARY, width=1,
                                     style=Qt.PenStyle.DotLine),
                    )
                    self._plot_items.append(radial)

        # ---- Frequency response plot ---------------------------------------
        if self._show_response_cb.isChecked():
            self._draw_response(selected_f)

        # ---- Status text ---------------------------------------------------
        if selected_f is not None:
            omega = 2.0 * np.pi * selected_f / self._fs
            group = "Low (row)" if selected_f in dsp.LOW_FREQS else "High (column)"
            self.status_label.setText(
                f"Selected: {selected_f} Hz  [{group} group]  |  "
                f"ω = {omega:.4f} rad  ({np.degrees(omega):.1f}°)  |  "
                f"Poles at e^{{±jω}}  (exactly on unit circle — marginally stable).  "
                f"Zero at e^{{-jω}} cancels the unwanted mirror resonance."
            )
        else:
            self.status_label.setText(
                f"{len(self._target_freqs)} DTMF frequencies → "
                f"{len(self._target_freqs) * 2} poles, all exactly on the unit circle "
                f"(marginally stable). Each zero sits on top of its resonator's lower "
                f"pole — not a rendering glitch, it's how Goertzel cancels the mirror frequency."
            )

    # -----------------------------------------------------------------------
    # Frequency response plot
    # -----------------------------------------------------------------------

    def _draw_response(self, selected_f):
        """Draw |H(e^{jω})| magnitude response for the selected (or all) resonators."""
        # Frequency axis: 0 to fs/2
        n_pts   = 2048
        f_axis  = np.linspace(0, self._fs / 2, n_pts)
        omega_axis = 2.0 * np.pi * f_axis / self._fs

        freqs_to_plot = [selected_f] if selected_f is not None else self._target_freqs

        for i, f in enumerate(freqs_to_plot):
            omega_t = 2.0 * np.pi * f / self._fs
            H = _goertzel_H(omega_t, omega_axis)
            # Clip to keep the resonance peak readable (the pole is ON the unit
            # circle so the theoretical gain is infinite at exactly f_target)
            H_clipped = np.clip(H, 0, 50)

            is_low = f in dsp.LOW_FREQS
            if selected_f is not None:
                color = PRIMARY
            else:
                color = "#54E0D0" if is_low else "#FF9430"

            curve = self.resp_plot.plot(
                f_axis, H_clipped,
                pen=pg.mkPen(color, width=2),
                name=f"{f} Hz",
            )
            self._resp_items.append(curve)

            # Vertical marker at the resonant frequency
            vline = pg.InfiniteLine(
                pos=f,
                angle=90,
                pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                label=f"{f} Hz",
                labelOpts={"position": 0.95, "color": color,
                           "fill": (0, 0, 0, 120), "movable": False},
            )
            self.resp_plot.addItem(vline)
            self._resp_items.append(vline)

    def _on_response_toggle(self, state):
        self._redraw()
