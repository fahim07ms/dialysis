"""
decoder_tab.py
================
Phase 5/6 tab: pick an algorithm, hit Decode, see the digit string come out.

This tab holds onto the most recent full signal (whatever the Waveform tab
last received via audio_ready) and only runs the decoder when you press the
button — decoding can be a bit of work, so we don't want it firing 30x/sec
during the live-playback view like the waveform/spectrum do.
"""

import time
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QComboBox, QPushButton
)
from PyQt6.QtCore import Qt

from core import dsp_interface as dsp

ALGORITHMS = {
    "Goertzel": dsp.goertzel_decode,
    "FFT (Phase 6)": dsp.fft_decode,
}


class DecoderTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples = None
        self._fs = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        section_label = QLabel("DECODE THE CURRENT SIGNAL")
        section_label.setProperty("role", "section")
        layout.addWidget(section_label)

        # -- Controls row: algorithm picker + Decode button --
        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)

        self.algo_combo = QComboBox()
        self.algo_combo.addItems(list(ALGORITHMS.keys()))

        self.decode_btn = QPushButton("Decode")
        self.decode_btn.setProperty("role", "pill-primary")
        self.decode_btn.clicked.connect(self._on_decode_clicked)

        controls_row.addWidget(self.algo_combo, 1)
        controls_row.addWidget(self.decode_btn, 0)
        layout.addLayout(controls_row)

        # -- Result card --
        result_card = QFrame()
        result_card.setProperty("role", "card")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(24, 24, 24, 24)
        result_layout.setSpacing(8)

        result_title = QLabel("DECODED DIGITS")
        result_title.setProperty("role", "section")
        result_layout.addWidget(result_title)

        self.result_label = QLabel("—")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 40px; font-weight: 800; letter-spacing: 6px;")
        result_layout.addWidget(self.result_label)

        self.timing_label = QLabel("")
        self.timing_label.setProperty("role", "muted")
        self.timing_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(self.timing_label)

        layout.addWidget(result_card)

        self.status_label = QLabel("Play or import a signal, then press Decode.")
        self.status_label.setProperty("role", "muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch(1)

    def update_audio(self, samples: np.ndarray, fs: int):
        """Connected to LeftPanel.audio_ready — just remembers the latest
        FULL signal. Doesn't decode automatically; that's what the button
        is for."""
        self._samples = np.asarray(samples)
        self._fs = fs
        if len(self._samples) > 0:
            self.status_label.setText(
                f"Ready to decode {len(self._samples)} samples @ {fs} Hz. Press Decode."
            )

    def _on_decode_clicked(self):
        if self._samples is None or len(self._samples) == 0:
            self.status_label.setText("Play or import a signal first.")
            return

        algo_name = self.algo_combo.currentText()
        decode_fn = ALGORITHMS[algo_name]

        start = time.perf_counter()
        try:
            digits = decode_fn(self._samples, self._fs)
        except NotImplementedError:
            self.result_label.setText("—")
            self.timing_label.setText("")
            fn_name = "goertzel_decode()" if algo_name == "Goertzel" else "fft_decode()"
            self.status_label.setText(f"{algo_name} needs {fn_name} implemented first.")
            return
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.result_label.setText(digits if digits else "(empty)")
        self.timing_label.setText(f"{algo_name} · {elapsed_ms:.2f} ms")
        self.status_label.setText(f"Decoded {len(digits)} digit(s) from {len(self._samples)} samples.")
