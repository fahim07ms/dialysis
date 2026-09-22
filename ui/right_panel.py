"""
right_panel.py
===============
Right side of the app: the tabbed analysis area. All phases (0-11) are
now built out — Waveform, Spectrogram, Decoder, Pole-Zero, Convolution,
Noise & Sampling, and Benchmark are all real, functioning tabs.
"""

from PyQt6.QtWidgets import QTabWidget

from ui.tabs.waveform_tab import WaveformTab
from ui.tabs.spectrogram_tab import SpectrogramTab
from ui.tabs.pole_zero_tab import PoleZeroTab
from ui.tabs.convolution_tab import ConvolutionTab
from ui.tabs.noise_sampling_tab import NoiseSamplingTab
from ui.tabs.benchmark_tab import BenchmarkTab
from ui.decoder_panel import DecoderPanel


class RightPanel(QTabWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDocumentMode(True)

        self.waveform_tab = WaveformTab()
        self.addTab(self.waveform_tab, "Waveform")
        self.spectrogram_tab = SpectrogramTab()
        self.addTab(self.spectrogram_tab, "Spectrogram")
        self.decoder_tab = DecoderPanel()
        self.addTab(self.decoder_tab, "Decoder")
        self.pole_zero_tab = PoleZeroTab()
        self.addTab(self.pole_zero_tab, "Pole-Zero")
        self.convolution_tab = ConvolutionTab()
        self.addTab(self.convolution_tab, "Convolution")
        self.noise_sampling_tab = NoiseSamplingTab()
        self.addTab(self.noise_sampling_tab, "Noise && Sampling")
        self.benchmark_tab = BenchmarkTab()
        self.addTab(self.benchmark_tab, "Benchmark")
