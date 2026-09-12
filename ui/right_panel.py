"""
right_panel.py
===============
Right side of the app: the tabbed analysis area. Each remaining tab is a
placeholder today — we'll fill them in one at a time as we work through
the phases (Phase 7: Spectrogram, Phase 8: Pole-Zero, Phase 9:
Impulse/Convolution, Phase 10: Noise & Sampling, Phase 11: Benchmark).

The Decoder used to live here as its own tab — it's been moved into the
LEFT panel (see ui/decoder_panel.py) so decoding sits right next to the
controls that produced the signal, no tab-switching required.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel
from PyQt6.QtCore import Qt

from ui.tabs.waveform_tab import WaveformTab
from ui.tabs.spectrogram_tab import SpectrogramTab


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

        self.waveform_tab = WaveformTab()
        self.addTab(self.waveform_tab, "Waveform")
        self.spectrogram_tab = SpectrogramTab()
        self.addTab(self.spectrogram_tab, "Spectrogram")
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
