"""
decoder_panel.py
=================
Decoder controls — its own tab in the right panel: pick an algorithm,
decode whatever's currently loaded (from the keypad, an imported file, or
live through the microphone), see the result.

LIVE MIC DECODING — HOW IT WORKS NOW (frame engine, v2)
---------------------------------------------------------
The old live path segmented the rolling buffer by time-domain energy
first (segment_tone_regions + merge/filter + a trailing-silence safety
margin) and only then asked "is this DTMF?". That ordering is exactly
backwards for live audio, and it was the root cause of live decoding
failing:

  * segment_tone_regions thresholds against max(RMS) of whatever is in the
    buffer. In a normal room the loudest thing in the buffer IS the room
    noise, so every frame counts as "loud", one region spans the whole
    buffer, the safety margin never lets it close, and zero digits ever
    decode. (The Detection Threshold slider couldn't help — the confidence
    gate it adjusted never even ran.)
  * A transient click raised the global threshold and swallowed the next
    real tone; the threshold itself changed every 200 ms tick.
  * The 150 ms safety margin + 200 ms timer meant digits appeared ~0.5 s
    late even when everything worked.

The new path (all the math lives in core/dsp_interface.py, Qt-free):

1. core.audio_io.MicStream captures at the device's NATIVE rate (no forced
   8 kHz), DC-blocked, with a bounded drop-oldest queue.
2. A QTimer ticks every 100 ms, drains the queue, and feeds the samples to
   dsp.DtmfTracker — a 20 ms frame / 5 ms hop engine that measures the 8
   DTMF bin powers per frame (vectorized Goertzel), gates each frame
   against an ADAPTIVE noise floor (min-with-leak tracker), and requires
   spectral dominance, all per frame.
3. A temporal state machine latches a digit after 8 agreeing frames
   (40 ms — the ITU-T Q.24 minimum tone duration) and emits it
   IMMEDIATELY; it releases after 4 disagreeing frames. No merge gaps, no
   safety margins, no re-segmentation of already-cleaned regions.
4. At latch time, one Q.24-style validation pass (dial tone, twist,
   frequency offset, 2nd harmonics) decides "real DTMF" vs "speech, music,
   dial tone, stray beep" — and the UI shows the rejection REASON.
5. The tracker is fully self-contained (its own carry/ring buffers), so
   the display buffer here is just for the waveform tab and the debug WAV
   dump, trimmed by time, independent of detection state.

The "Detection Threshold" slider now maps to something physically
meaningful: how many dB above the measured noise floor a tone pair must
be (the live bars + floor/gate lines show exactly what the gate sees).
The one-shot Decode button still runs the classic file-path decoders
(goertzel_decode / fft_decode) unchanged.
"""

import time
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QComboBox,
    QPushButton, QSlider
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import QColor, QPainter, QPen

from core import dsp_interface as dsp
from core import audio_io

ALGORITHMS = {
    "Goertzel": dsp.goertzel_decode,
    "FFT": dsp.fft_decode,
}

MIC_TICK_MS = 100                    # drain + detect cadence (engine adds ~40 ms)
MIC_DISPLAY_SECONDS = 5.0            # rolling waveform/debug-dump buffer

GATE_DB_MIN = 3                      # slider range: dB above the noise floor
GATE_DB_MAX = 30
GATE_DB_DEFAULT = 10


class LiveBinsWidget(QWidget):
    """
    Real-time per-frequency meter for the 8 DTMF bins: four blue bars for
    the low (row) group, four orange bars for the high (column) group,
    with the tracker's noise floor (grey) and current gate level (red
    dashed) drawn on the same dB axis.

    A real DTMF tone shows as exactly two bars (one per group) popping
    well above the gate line; noise shows as a low, flat comb. This is the
    fastest way to answer "why isn't it decoding" — quiet mic, gate set
    too high, or a room that's too noisy are all visible at a glance.

    Note: the bars are per-BIN power while the gate/floor lines are on the
    winner-PAIR scale the gate actually uses (a pure tone reads ~3 dB
    higher on the pair scale) — close enough for a visual aid.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bin_db = None
        self._floor_db = None
        self._gate_db = None
        self.setMinimumHeight(170)

    def set_data(self, info: dict):
        if not info or not info.get("bin_db"):
            self._bin_db = None
            self._floor_db = None
            self._gate_db = None
        else:
            self._bin_db = list(info["bin_db"])
            self._floor_db = info.get("floor_db")
            self._gate_db = info.get("gate_level_db")
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        label_h = 18
        plot_top = 8.0
        plot_h = h - label_h - plot_top

        if not self._bin_db or len(self._bin_db) != 8 or plot_h <= 4:
            p.setPen(QColor("#888888"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "DTMF bin levels appear here while listening")
            p.end()
            return

        floor = self._floor_db if self._floor_db is not None else -60.0
        lo = floor - 20.0
        hi = floor + 45.0
        span = max(hi - lo, 1.0)

        def y_of(db):
            frac = (min(max(db, lo), hi) - lo) / span
            return plot_top + plot_h * (1.0 - frac)

        freqs = dsp.LOW_FREQS + dsp.HIGH_FREQS
        bw = w / 8.0
        font = p.font()
        font.setPointSizeF(7.5)
        p.setFont(font)

        for i, db in enumerate(self._bin_db):
            x0 = i * bw + 3.0
            x1 = (i + 1) * bw - 3.0
            top_y = y_of(db)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#3D8BFF") if i < 4 else QColor("#FF9430"))
            p.drawRect(QRectF(x0, top_y, x1 - x0, plot_top + plot_h - top_y))
            p.setPen(QColor("#9A9A9A"))
            p.drawText(QRectF(x0, h - label_h, x1 - x0, label_h),
                       Qt.AlignmentFlag.AlignCenter, str(freqs[i]))

        for level, color, style, name in (
            (self._floor_db, QColor("#777777"), Qt.PenStyle.SolidLine, "floor"),
            (self._gate_db, QColor("#E05555"), Qt.PenStyle.DashLine, "gate"),
        ):
            if level is None:
                continue
            yy = y_of(level)
            p.setPen(QPen(color, 1, style))
            p.drawLine(QPointF(0.0, yy), QPointF(float(w), yy))
            p.drawText(QRectF(4.0, yy - 14.0, 90.0, 12.0),
                       Qt.AlignmentFlag.AlignLeft, name)
        p.end()


class DecoderPanel(QWidget):

    # Emitted every mic tick with the current rolling buffer, so the
    # Waveform tab can show live audio the same way it shows playback.
    live_audio_ready = pyqtSignal(object, int)  # (numpy array, fs)

    # Emitted every mic tick with the tracker's latest per-frame info
    # (bin levels, floor, gate, state) for any spectrum/bars visualization.
    live_bins_ready = pyqtSignal(object)        # (dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples = None
        self._fs = None

        # -- Live mic state --
        self._mic_stream = None
        self._mic_fs = 8000
        self._mic_display = np.zeros(0)
        self._tracker = None
        self._mic_transcript = ""
        self._last_event_text = ""
        self._gate_db = float(GATE_DB_DEFAULT)
        self._flash_seq = 0
        self._mic_timer = QTimer(self)
        self._mic_timer.setInterval(MIC_TICK_MS)
        self._mic_timer.timeout.connect(self._on_mic_tick)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)

        card = QFrame()
        card.setProperty("role", "card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        section_label = QLabel("DECODER")
        section_label.setProperty("role", "section")
        layout.addWidget(section_label)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)

        self.algo_combo = QComboBox()
        self.algo_combo.addItems(list(ALGORITHMS.keys()))
        self.algo_combo.currentTextChanged.connect(self._on_algo_changed)

        self.decode_btn = QPushButton("Decode")
        self.decode_btn.setProperty("role", "pill-primary")
        self.decode_btn.clicked.connect(self._on_decode_clicked)

        controls_row.addWidget(self.algo_combo, 1)
        controls_row.addWidget(self.decode_btn, 0)
        layout.addLayout(controls_row)

        self.mic_btn = QPushButton("Listen Live")
        self.mic_btn.setProperty("role", "pill-outline")
        self.mic_btn.clicked.connect(self._on_mic_toggle_clicked)
        layout.addWidget(self.mic_btn)

        # -- Detection Threshold slider — how many dB above the measured
        # noise floor a tone pair must be to count. Higher = stricter
        # (fewer false positives, but a quiet tone in a noisy room may be
        # missed). The floor/gate lines on the bars below show exactly
        # what this slider does.
        threshold_row = QHBoxLayout()
        threshold_label = QLabel("Detection Threshold")
        threshold_label.setProperty("role", "muted")
        self.threshold_value_label = QLabel(f"+{GATE_DB_DEFAULT} dB above floor")
        self.threshold_value_label.setProperty("role", "muted")
        threshold_row.addWidget(threshold_label)
        threshold_row.addStretch(1)
        threshold_row.addWidget(self.threshold_value_label)
        layout.addLayout(threshold_row)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(GATE_DB_MIN, GATE_DB_MAX)
        self.threshold_slider.setValue(GATE_DB_DEFAULT)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        layout.addWidget(self.threshold_slider)

        self.bins_widget = LiveBinsWidget()
        layout.addWidget(self.bins_widget)

        # Last-detected digit flash + debug dump button
        flash_row = QHBoxLayout()
        self.flash_label = QLabel("")
        self.flash_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.flash_label.setStyleSheet(
            "font-size: 40px; font-weight: 800; color: #3D8BFF; min-height: 48px;"
        )
        self.save_debug_btn = QPushButton("💾 Save Recording")
        self.save_debug_btn.setProperty("role", "pill-outline")
        self.save_debug_btn.clicked.connect(self._on_save_debug_clicked)
        flash_row.addWidget(self.save_debug_btn, 0)
        flash_row.addStretch(1)
        flash_row.addWidget(self.flash_label, 1)
        flash_row.addStretch(1)
        layout.addLayout(flash_row)

        self.result_label = QLabel("—")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(
            "font-size: 26px; font-weight: 800; letter-spacing: 3px;"
        )
        layout.addWidget(self.result_label)

        self.status_label = QLabel("Play or import a signal, then press Decode.")
        self.status_label.setProperty("role", "muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        outer.addWidget(card)

    # -- Controls ------------------------------------------------------------

    def _on_threshold_changed(self, value: int):
        self._gate_db = float(value)
        if self._tracker is not None:
            self._tracker.gate_db = self._gate_db
        self.threshold_value_label.setText(f"+{value} dB above floor")

    def _on_algo_changed(self, algo_name: str):
        # Applies to the live engine immediately; the one-shot Decode
        # button reads the combo again when clicked.
        if self._tracker is not None:
            self._tracker.algorithm = algo_name.lower()

    # -- One-shot decode (file path, Phase 5/6 behavior) ---------------------

    def update_audio(self, samples: np.ndarray, fs: int):
        """Connected to LeftPanel.audio_ready — just remembers the latest
        FULL signal. Doesn't decode automatically; that's what the button
        is for (decoding is fast, but no reason to run it on every tick)."""
        self._samples = np.asarray(samples)
        self._fs = fs
        if len(self._samples) > 0:
            self.status_label.setText(
                f"Ready to decode {len(self._samples)} samples @ {fs} Hz."
            )

    def _on_decode_clicked(self):
        if self._samples is None or len(self._samples) == 0:
            self.status_label.setText("Play or import a signal first.")
            return

        algo_name = self.algo_combo.currentText()
        start = time.perf_counter()
        digits, events = dsp.decode_file_events(
            self._samples, self._fs,
            algorithm=algo_name.lower(),
            gate_db=self._gate_db,   # the Detection Threshold slider now
                                     # applies to file decoding too
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.result_label.setText(digits if digits else "(empty)")

        rejects = [e["reason"] for e in events if e["type"] == "reject"]
        msg = f"{algo_name} · {elapsed_ms:.2f} ms · {len(digits)} digit(s)"
        if not digits:
            msg += " — no valid DTMF found"
        if rejects:
            msg += f" · {len(rejects)} region(s) rejected (last: {rejects[-1]})"
        self.status_label.setText(msg)

    # -- Live microphone decoding -------------------------------------------

    def _on_mic_toggle_clicked(self):
        if self._mic_stream is None:
            self._start_mic()
        else:
            self._stop_mic()

    def _start_mic(self):
        try:
            stream = audio_io.MicStream()   # native device rate — read stream.fs
            stream.start()
        except Exception as e:
            self.status_label.setText(f"Couldn't open microphone: {e}")
            return

        self._mic_stream = stream
        self._mic_fs = stream.fs
        self._mic_display = np.zeros(0)
        self._mic_recording = np.zeros(0)
        self._mic_transcript = ""
        self._last_event_text = ""
        self._tracker = dsp.DtmfTracker(
            self._mic_fs,
            algorithm=self.algo_combo.currentText().lower(),
            gate_db=self._gate_db,
        )

        self.result_label.setText("—")
        self.flash_label.setText("")
        self.bins_widget.set_data({})
        self.mic_btn.setText("⏹  Stop Listening")
        self.decode_btn.setEnabled(False)  # avoid a one-shot decode racing the live loop
        self.status_label.setText(
            f"Listening… {stream.device_name} @ {stream.fs} Hz"
        )
        self._mic_timer.start()

    def _stop_mic(self):
        self._mic_timer.stop()
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop()
            except Exception:
                pass  # already gone / device unplugged — nothing more we can do
        self._mic_stream = None
        self._tracker = None

        self.mic_btn.setText("Listen Live")
        self.decode_btn.setEnabled(True)
        n = len(self._mic_transcript)
        self.status_label.setText(
            f"Stopped listening. {n} digit(s) detected."
        )

    def _on_mic_tick(self):
        if self._mic_stream is None or self._tracker is None:
            return

        try:
            new_samples = self._mic_stream.read_available()
            if len(new_samples) > 0:
                cap = int(MIC_DISPLAY_SECONDS * self._mic_fs)
                self._mic_display = np.concatenate([self._mic_display, new_samples])[-cap:]
                self._mic_recording = np.concatenate([self._mic_recording, new_samples])
                for event in self._tracker.push(new_samples):
                    self._handle_event(event)

            # Feed the Waveform tab (fs is whatever the device actually runs at).
            self.live_audio_ready.emit(self._mic_display, self._mic_fs)

            info = dict(self._tracker.last_frame_info)
            self.live_bins_ready.emit(info)
            self.bins_widget.set_data(info)
            self._refresh_mic_status()
        except Exception as e:
            self._stop_mic()
            self.status_label.setText(f"Live decoding error: {e}")

    def _handle_event(self, event: dict):
        kind = event.get("type")

        if kind == "digit":
            self._mic_transcript += event["digit"]
            self.result_label.setText(self._mic_transcript)
            self._flash_digit(event["digit"])

            note = self._cross_check_note(event.get("region"))
            self._last_event_text = (
                f"Detected {event['digit']!r} "
                f"(latched in {event['duration_s'] * 1000:.0f} ms{note})"
            )

        elif kind == "reject":
            self._last_event_text = f"Rejected: {event['reason']}"

        elif kind == "release":
            self._last_event_text = (
                f"Released {event['digit']!r} after {event['duration_s']:.2f} s"
            )

    def _cross_check_note(self, region) -> str:
        """Run the OTHER decoder on the same region for an agreement badge —
        this app's two side-by-side decoders are a feature none of the
        reference websites have, so show it off."""
        other = self._other_algorithm_digit(region)
        if other is None:
            return ""
        if other == self._current_event_digit_context(region):
            return " · other decoder agrees ✓"
        return f" · other decoder says {other!r} ✗"

    def _current_event_digit_context(self, region):
        # The digit the event carried is the one the active algorithm
        # produced; recover it from the tracker's last frame info latch.
        info = self._tracker.last_frame_info if self._tracker else {}
        return info.get("latched")

    def _other_algorithm_digit(self, region):
        if region is None or len(region) < 16:
            return None
        if self.algo_combo.currentText() == "Goertzel":
            digit, _ = dsp.fft_region_digit(region, self._mic_fs)
            return digit
        windowed = region * np.hanning(len(region))
        powers = dsp.goertzel_powers(windowed, self._mic_fs, dsp.DTMF_FREQS_ALL)
        f_low = dsp.LOW_FREQS[int(np.argmax(powers[:4]))]
        f_high = dsp.HIGH_FREQS[int(np.argmax(powers[4:]))]
        return dsp.FREQ_TO_DIGIT.get((f_low, f_high))

    def _flash_digit(self, digit: str):
        self._flash_seq += 1
        seq = self._flash_seq
        self.flash_label.setText(digit)

        def _clear():
            if seq == self._flash_seq:
                self.flash_label.setText("")

        QTimer.singleShot(350, _clear)

    def _refresh_mic_status(self):
        info = self._tracker.last_frame_info if self._tracker else {}
        parts = [f"Listening @ {self._mic_fs} Hz"]
        floor = info.get("floor_db")
        if floor is not None:
            parts.append(f"noise floor {floor:.0f} dB")
            parts.append(f"gate +{self._gate_db:.0f} dB")
        if self._mic_stream is not None:
            dropped = self._mic_stream.overflow_count + self._mic_stream.pa_status_count
            if dropped:
                parts.append(f"⚠ {dropped} dropped chunk(s)")
        text = " · ".join(parts)
        if self._last_event_text:
            text += f"\n{self._last_event_text}"
        self.status_label.setText(text)

    def _on_save_debug_clicked(self):
        """Phase-0 style instrumentation: when something doesn't decode,
        save exactly what the mic delivered and inspect it offline."""
        if not hasattr(self, '_mic_recording') or self._mic_recording is None or len(self._mic_recording) == 0:
            self.status_label.setText("No mic audio to save yet.")
            return
        path = f"mic_recording_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        audio_io.save_wav(path, self._mic_recording, self._mic_fs)
        self.status_label.setText(
            f"Saved {len(self._mic_recording) / self._mic_fs:.1f} s @ "
            f"{self._mic_fs} Hz to {path}"
        )