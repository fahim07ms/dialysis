"""
left_panel.py
=============
Left side of the app: keypad, digit text field, Play/Export/Import controls.

This file calls into core.dsp_interface for any math, and core.audio_io for
any device I/O — it never does the math itself. If a DSP function isn't
implemented yet, we catch NotImplementedError and show a friendly status
message instead of crashing, so the UI stays usable while you're still
building out core/dsp_interface.py phase by phase.

LIVE PLAYBACK VIEW
-------------------
While audio plays, a QTimer ticks roughly 30 times a second. Each tick, we
estimate "how many samples have played so far" from wall-clock elapsed time
(samples_played = elapsed_seconds * fs), and emit just that much of the
array via `playback_progress`. The Waveform tab redraws with that partial
slice, so the plot appears to "grow" in sync with what you're hearing. When
playback finishes, we emit the FULL array via `audio_ready`, which triggers
the complete analysis (spectrum + segmentation) on the whole signal.
"""

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QLineEdit,
    QPushButton, QFileDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, QTimer, QElapsedTimer, pyqtSignal

from ui.keypad import Keypad
from ui.decoder_panel import DecoderPanel
from core import dsp_interface as dsp
from core import audio_io

SAMPLE_RATE = 8000
TONE_DURATION = 0.25
PROGRESS_TICK_MS = 33  # ~30 fps for the live-growing waveform/spectrum


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

    # Fired ~30x/sec DURING playback with the samples played so far — lets
    # the Waveform tab draw a "growing" waveform + spectrum in sync with audio.
    playback_progress = pyqtSignal(object, int)  # (partial numpy array, fs)

    # Fired once playback finishes (or immediately for Export/Import-without-
    # play) with the COMPLETE array — triggers full spectrum + segmentation.
    audio_ready = pyqtSignal(object, int)  # (full numpy array, fs)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._playback_thread = None

        # State for whatever is currently playing / was last loaded
        self._current_samples = None
        self._current_fs = SAMPLE_RATE
        self._imported_samples = None
        self._imported_fs = SAMPLE_RATE
        self._imported_path = None

        self._playback_clock = QElapsedTimer()
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(PROGRESS_TICK_MS)
        self._progress_timer.timeout.connect(self._on_progress_tick)

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

        # -- Loaded-file row: filename + its own Play button. This is what
        # was missing before — importing a file loaded it, but nothing ever
        # called _play() on it.
        loaded_row = QHBoxLayout()
        loaded_row.setSpacing(10)

        self.loaded_file_label = QLabel("No file loaded")
        self.loaded_file_label.setProperty("role", "muted")
        self.loaded_file_label.setWordWrap(True)

        self.play_imported_btn = QPushButton("▶ Play")
        self.play_imported_btn.setProperty("role", "pill-outline")
        self.play_imported_btn.setEnabled(False)
        self.play_imported_btn.clicked.connect(self._on_play_imported_clicked)

        loaded_row.addWidget(self.loaded_file_label, 1)
        loaded_row.addWidget(self.play_imported_btn, 0)
        io_layout.addLayout(loaded_row)

        outer.addWidget(io_card)

        # ---- Card: decoder (moved here from a separate tab — decoding now
        # sits right next to whatever produced the signal) ----
        self.decoder_panel = DecoderPanel()
        self.audio_ready.connect(self.decoder_panel.update_audio)
        outer.addWidget(self.decoder_panel)

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
        samples = dsp.generate_dtmf_tone(digit, fs=SAMPLE_RATE, duration=TONE_DURATION)
        self._start_playback(samples, SAMPLE_RATE, status=f"Playing '{digit}'…")

    def _on_play_clicked(self):
        digits = self.entry.text().strip()
        if not digits:
            self._set_status("Type or tap a digit sequence first.")
            return
        samples = dsp.sequence_to_wav(digits, fs=SAMPLE_RATE, tone_duration=TONE_DURATION)
        self._start_playback(samples, SAMPLE_RATE, status=f"Playing sequence: {digits}")

    def _on_export_clicked(self):
        digits = self.entry.text().strip()
        if not digits:
            self._set_status("Type a digit sequence before exporting.")
            return
        samples = dsp.sequence_to_wav(digits, fs=SAMPLE_RATE, tone_duration=TONE_DURATION)
        path, _ = QFileDialog.getSaveFileName(self, "Export WAV", f"{digits}.wav", "WAV files (*.wav)")
        if path:
            audio_io.save_wav(path, samples, SAMPLE_RATE)
            self._set_status(f"Exported to {path}")

    def _on_import_clicked(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import WAV", "", "WAV files (*.wav)")
        if not path:
            return
        samples, fs = audio_io.load_wav(path)

        self._imported_samples = samples
        self._imported_fs = fs
        self._imported_path = path
        self.play_imported_btn.setEnabled(True)
        self.loaded_file_label.setText(os.path.basename(path))

        # Show the full waveform/spectrum right away (static), and separately
        # let the person press Play to hear it with the live-growing view.
        self.audio_ready.emit(samples, fs)
        self._set_status(f"Loaded {os.path.basename(path)} "
                          f"({len(samples)} samples @ {fs} Hz). Press Play to hear it.")

    def _on_play_imported_clicked(self):
        if self._imported_samples is None or len(self._imported_samples) == 0:
            self._set_status("Import a WAV file first.")
            return
        self._start_playback(
            self._imported_samples, self._imported_fs,
            status=f"Playing {os.path.basename(self._imported_path)}…"
        )

    # -- playback + live view ---------------------------------------------

    def _start_playback(self, samples, fs, status: str):
        if samples is None or len(samples) == 0:
            self._set_status("Nothing to play.")
            return

        audio_io.stop_playback()  # don't let two playbacks overlap

        self._current_samples = samples
        self._current_fs = fs

        self._play(samples, fs)
        self._playback_clock.start()
        self._progress_timer.start()
        self._set_status(status)

    def _on_progress_tick(self):
        if self._current_samples is None:
            self._progress_timer.stop()
            return

        elapsed_s = self._playback_clock.elapsed() / 1000.0
        pos = int(elapsed_s * self._current_fs)
        total = len(self._current_samples)

        if pos >= total:
            self._progress_timer.stop()
            self.audio_ready.emit(self._current_samples, self._current_fs)
            return

        self.playback_progress.emit(self._current_samples[:max(pos, 1)], self._current_fs)

    def _play(self, samples, fs):
        self._playback_thread = _PlaybackThread(samples, fs)
        self._playback_thread.failed.connect(self._on_playback_failed)
        self._playback_thread.start()

    def _on_playback_failed(self, msg: str):
        # No audio device, driver issue, etc. Stop faking a live progress bar
        # and just jump straight to showing the complete waveform/spectrum.
        self._progress_timer.stop()
        self._set_status(f"Playback error: {msg}")
        if self._current_samples is not None:
            self.audio_ready.emit(self._current_samples, self._current_fs)

    # -- helpers ---------------------------------------------------------

    def _set_status(self, text: str):
        self.status_label.setText(text)
