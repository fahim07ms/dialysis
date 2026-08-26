"""
left_panel.py
=============
Left side of the app: keypad, digit text field, Play/Export/Import controls.

This file calls into core.dsp_interface for any math, and core.audio_io for
any device I/O — it never does the math itself. If a DSP function isn't
implemented yet, we catch NotImplementedError and show a friendly status
message instead of crashing, so the UI stays usable while you're still
building out core/dsp_interface.py phase by phase.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QLineEdit,
    QPushButton, QFileDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ui.keypad import Keypad
from core import dsp_interface as dsp
from core import audio_io

SAMPLE_RATE = 8000
TONE_DURATION = 0.25


class _PlaybackThread(QThread):
    """Runs sd.play/sd.wait off the UI thread so the window never freezes."""
    failed = pyqtSignal(str)

    def __init__(self, samples, fs, parent=None):
        super().__init__(parent)
        self.samples = samples
        self.fs = fs

    def run(self):
        try:
            audio_io.play_array(self.samples, self.fs, blocking=True)
        except Exception as e:
            self.failed.emit(str(e))


class LeftPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._playback_thread = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(16)

        # ---- Title ----
        title = QLabel("Dialysis")
        title.setProperty("role", "title")
        subtitle = QLabel("DTMF Signal Analysis Studio")
        subtitle.setProperty("role", "muted")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        # ---- Card: keypad + entry field ----
        card = QFrame()
        card.setProperty("role", "card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        card_layout.setSpacing(14)

        section_label = QLabel("KEYPAD INPUT")
        section_label.setProperty("role", "section")
        card_layout.addWidget(section_label)

        self.entry = QLineEdit()
        self.entry.setPlaceholderText("Tap keys or type here…")
        self.entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.entry)

        self.keypad = Keypad()
        self.keypad.digit_pressed.connect(self._on_digit_pressed)
        card_layout.addWidget(self.keypad)

        # ---- Row: Play / Clear ----
        button_row = QHBoxLayout()
        button_row.setSpacing(10)

        self.play_btn = QPushButton("▶  Play Sequence")
        self.play_btn.setProperty("role", "pill-primary")
        self.play_btn.clicked.connect(self._on_play_clicked)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setProperty("role", "pill-outline")
        self.clear_btn.clicked.connect(self.entry.clear)

        button_row.addWidget(self.play_btn, 2)
        button_row.addWidget(self.clear_btn, 1)
        card_layout.addLayout(button_row)

        outer.addWidget(card)

        # ---- Card: file import / export ----
        io_card = QFrame()
        io_card.setProperty("role", "card")
        io_layout = QVBoxLayout(io_card)
        io_layout.setContentsMargins(20, 20, 20, 20)
        io_layout.setSpacing(10)

        io_section = QLabel("AUDIO FILE I/O")
        io_section.setProperty("role", "section")
        io_layout.addWidget(io_section)

        io_row = QHBoxLayout()
        io_row.setSpacing(10)

        self.export_btn = QPushButton("Export WAV")
        self.export_btn.setProperty("role", "pill-outline")
        self.export_btn.clicked.connect(self._on_export_clicked)

        self.import_btn = QPushButton("Import WAV")
        self.import_btn.setProperty("role", "pill-outline")
        self.import_btn.clicked.connect(self._on_import_clicked)

        io_row.addWidget(self.export_btn)
        io_row.addWidget(self.import_btn)
        io_layout.addLayout(io_row)

        outer.addWidget(io_card)

        # ---- Status line (feedback instead of crashing/blank clicks) ----
        self.status_label = QLabel("Ready.")
        self.status_label.setProperty("role", "muted")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        outer.addStretch(1)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    # -- signal / callback handlers -----------------------------------

    def _on_digit_pressed(self, digit: str):
        self.entry.setText(self.entry.text() + digit)
        self._play_single_digit(digit)

    def _play_single_digit(self, digit: str):
        try:
            samples = dsp.generate_dtmf_tone(digit, fs=SAMPLE_RATE, duration=TONE_DURATION)
        except NotImplementedError:
            self._set_status(
                f"Key '{digit}' pressed — waiting on generate_dtmf_tone() "
                f"in core/dsp_interface.py (Phase 1)."
            )
            return
        self._play(samples, SAMPLE_RATE)
        self._set_status(f"Playing '{digit}'…")

    def _on_play_clicked(self):
        digits = self.entry.text().strip()
        if not digits:
            self._set_status("Type or tap a digit sequence first.")
            return
        try:
            samples = dsp.sequence_to_wav(digits, fs=SAMPLE_RATE, tone_duration=TONE_DURATION)
        except NotImplementedError:
            self._set_status(
                "sequence_to_wav() isn't implemented yet (Phase 2) — "
                "playing each digit individually instead."
            )
            self._play_each_digit(digits)
            return
        self._play(samples, SAMPLE_RATE)
        self._set_status(f"Playing sequence: {digits}")

    def _play_each_digit(self, digits: str):
        # Fallback so Play still does *something* useful before Phase 2 exists.
        import numpy as np
        chunks = []
        for d in digits:
            try:
                chunks.append(dsp.generate_dtmf_tone(d, fs=SAMPLE_RATE, duration=TONE_DURATION))
                chunks.append(np.zeros(int(SAMPLE_RATE * 0.05)))
            except NotImplementedError:
                self._set_status("generate_dtmf_tone() isn't implemented yet (Phase 1).")
                return
        if chunks:
            full = np.concatenate(chunks)
            self._play(full, SAMPLE_RATE)

    def _on_export_clicked(self):
        digits = self.entry.text().strip()
        if not digits:
            self._set_status("Type a digit sequence before exporting.")
            return
        try:
            samples = dsp.sequence_to_wav(digits, fs=SAMPLE_RATE, tone_duration=TONE_DURATION)
        except NotImplementedError:
            self._set_status("Export needs sequence_to_wav() first (Phase 2).")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export WAV", f"{digits}.wav", "WAV files (*.wav)")
        if path:
            audio_io.save_wav(path, samples, SAMPLE_RATE)
            self._set_status(f"Exported to {path}")

    def _on_import_clicked(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import WAV", "", "WAV files (*.wav)")
        if not path:
            return
        samples, fs = audio_io.load_wav(path)
        self._set_status(f"Loaded {path} ({len(samples)} samples @ {fs} Hz). "
                          f"Decoding arrives in Phase 4/5.")
        # NOTE for Phase 4/5: this is where we will hand `samples, fs` off to
        # segment_tone_regions() / goertzel_decode() and push results to the
        # right-hand analysis tabs.

    # -- helpers ---------------------------------------------------------

    def _play(self, samples, fs):
        self._playback_thread = _PlaybackThread(samples, fs)
        self._playback_thread.failed.connect(
            lambda msg: self._set_status(f"Playback error: {msg}")
        )
        self._playback_thread.start()

    def _set_status(self, text: str):
        self.status_label.setText(text)
