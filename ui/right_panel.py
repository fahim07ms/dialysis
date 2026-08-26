"""
right_panel.py
===============
Right side of the app: the tabbed analysis area. Each tab is a placeholder
today — we'll fill them in one at a time as we work through the phases
(Phase 3: Waveform + Spectrum, Phase 5/6: Decoder, Phase 7: Spectrogram,
Phase 8: Pole-Zero, Phase 9: Impulse/Convolution, Phase 11: Benchmark).

Keeping every tab as its own small widget class (built later) means we can
drop pyqtgraph plots in without touching this file's layout logic again.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel
from PyQt6.QtCore import Qt


def _placeholder_tab(message: str) -> QWidget:
    w = QWidget()
    layout = QVBoxLayout(w)
    label = QLabel(message)
    label.setProperty("role", "muted")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setWordWrap(True)
    layout.addWidget(label)
    return w


class RightPanel(QTabWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDocumentMode(True)

        self.addTab(
            _placeholder_tab("Waveform + magnitude spectrum arrive in Phase 3."),
            "Waveform"
        )
        self.addTab(
            _placeholder_tab("Dynamic spectrogram arrives in Phase 7."),
            "Spectrogram"
        )
        self.addTab(
            _placeholder_tab("Goertzel / FFT decoder output arrives in Phase 5–6."),
            "Decoder"
        )
        self.addTab(
            _placeholder_tab("Z-plane pole-zero view arrives in Phase 8."),
            "Pole-Zero"
        )
        self.addTab(
            _placeholder_tab("Impulse response + convolution view arrives in Phase 9."),
            "Convolution"
        )
        self.addTab(
            _placeholder_tab("Noise / sample-rate sliders arrive in Phase 10."),
            "Noise && Sampling"
        )
        self.addTab(
            _placeholder_tab("FFT vs Goertzel benchmark arrives in Phase 11."),
            "Benchmark"
        )
