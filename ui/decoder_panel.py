"""
decoder_panel.py
=================
Decoder controls, living in the LEFT panel — decoding sits right next to
the controls that produced the signal: type/play a sequence, import a
file, OR now, listen live through the microphone.

LIVE MIC DECODING — HOW IT WORKS
----------------------------------
No new core/dsp_interface.py functions were needed for this. The existing
pipeline (segment_tone_regions -> goertzel_decode/fft_decode) already does
everything required — the only new work here is orchestration:

1. core.audio_io.MicStream captures audio continuously in the background
   (via sounddevice's own internal thread) and queues it up.
2. A QTimer here ticks every MIC_TICK_MS, drains whatever's queued, and
   appends it to a rolling buffer (self._mic_buffer).
3. Each tick, we run segment_tone_regions() on that buffer to find tone
   regions — but we only TRUST a region as "finished" if there's at least
   MIC_SAFETY_MARGIN_SECONDS of buffer AFTER its end. Without that margin,
   a region's "end" might just be a brief dip mid-tone, not the tone
   actually finishing — trusting it too early risks cutting a tone off
   and decoding garbage.
4. Confirmed (safe) regions get an extra CONFIDENCE CHECK before being
   trusted as real DTMF (see _is_valid_dtmf_region below) — segmentation
   only knows "this was loud", not "this was actually two clean tones".
   A real mic picks up room noise, and noise can easily be loud enough to
   cross the energy threshold without being DTMF at all. Only regions that
   pass get decoded and appended to a running transcript; the buffer is
   then trimmed to drop everything already confirmed, keeping memory and
   per-tick work bounded no matter how long you leave the mic listening
   (same "bounded window" lesson from the Phase 3 stutter fix).
"""

import time
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QComboBox, QPushButton
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from core import dsp_interface as dsp
from core import audio_io

ALGORITHMS = {
    "Goertzel": dsp.goertzel_decode,
    "FFT": dsp.fft_decode,
}

MIC_FS = 8000                        # standard telephony rate, plenty above DTMF's Nyquist need
MIC_TICK_MS = 200                    # how often we drain the mic queue and try to decode
MIC_SAFETY_MARGIN_SECONDS = 0.15     # required trailing silence before trusting a region as "done"
MIC_MAX_BUFFER_SECONDS = 6.0         # hard cap so a stuck/failed decode can't leak memory forever

# Confidence-gate thresholds — measured empirically (see conversation notes):
# pure room noise tops out around 0.58 dominance across many trials, while
# real DTMF tones (even noisy ones) never drop below ~0.99. 0.75 sits
# comfortably in the middle of that gap with a lot of margin either way.
MIC_MIN_DOMINANCE = 0.75


class DecoderPanel(QWidget):

    # Emitted every mic tick with the current rolling buffer, so the
    # Waveform tab can show live audio the same way it shows playback.
    live_audio_ready = pyqtSignal(object, int)  # (numpy array, fs)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples = None
        self._fs = None

        # -- Live mic state --
        self._mic_stream = None
        self._mic_buffer = np.array([])
        self._mic_transcript = ""
        self._mic_timer = QTimer(self)
        self._mic_timer.setInterval(MIC_TICK_MS)
        self._mic_timer.timeout.connect(self._on_mic_tick)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

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

        self.decode_btn = QPushButton("Decode")
        self.decode_btn.setProperty("role", "pill-primary")
        self.decode_btn.clicked.connect(self._on_decode_clicked)

        controls_row.addWidget(self.algo_combo, 1)
        controls_row.addWidget(self.decode_btn, 0)
        layout.addLayout(controls_row)

        self.mic_btn = QPushButton("🎤  Listen Live")
        self.mic_btn.setProperty("role", "pill-outline")
        self.mic_btn.clicked.connect(self._on_mic_toggle_clicked)
        layout.addWidget(self.mic_btn)

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

    # -- One-shot decode (existing Phase 5/6 behavior) ---------------------

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
        digits, elapsed_ms, error = self._run_decode(algo_name, self._samples, self._fs)
        if error:
            self.result_label.setText("—")
            self.status_label.setText(error)
            return

        self.result_label.setText(digits if digits else "(empty)")
        self.status_label.setText(
            f"{algo_name} · {elapsed_ms:.2f} ms · {len(digits)} digit(s)"
        )

    def _run_decode(self, algo_name: str, samples: np.ndarray, fs: int):
        """Shared by both one-shot Decode and the live mic loop. Returns
        (digits, elapsed_ms, error_message_or_None)."""
        decode_fn = ALGORITHMS[algo_name]
        start = time.perf_counter()
        try:
            digits = decode_fn(samples, fs)
        except NotImplementedError:
            fn_name = "goertzel_decode()" if algo_name == "Goertzel" else "fft_decode()"
            return "", 0.0, f"{algo_name} needs {fn_name} implemented first."
        elapsed_ms = (time.perf_counter() - start) * 1000
        return digits, elapsed_ms, None

    # -- Live microphone decoding -------------------------------------------

    def _on_mic_toggle_clicked(self):
        if self._mic_stream is None:
            self._start_mic()
        else:
            self._stop_mic()

    def _start_mic(self):
        try:
            stream = audio_io.MicStream(fs=MIC_FS)
            stream.start()
        except Exception as e:
            self.status_label.setText(f"Couldn't open microphone: {e}")
            return

        self._mic_stream = stream
        self._mic_buffer = np.array([])
        self._mic_transcript = ""
        self.result_label.setText("—")

        self.mic_btn.setText("⏹  Stop Listening")
        self.decode_btn.setEnabled(False)  # avoid a one-shot decode racing the live loop
        self.status_label.setText("Listening… dial some digits.")
        self._mic_timer.start()

    def _stop_mic(self):
        self._mic_timer.stop()
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop()
            except Exception:
                pass  # already gone / device unplugged — nothing more we can do about it
        self._mic_stream = None

        self.mic_btn.setText("🎤  Listen Live")
        self.decode_btn.setEnabled(True)
        self.status_label.setText("Stopped listening.")

    def _on_mic_tick(self):
        if self._mic_stream is None:
            return

        new_samples = self._mic_stream.read_available()
        if len(new_samples) > 0:
            self._mic_buffer = np.concatenate([self._mic_buffer, new_samples])

        # Hard safety cap — if decoding somehow stalls, don't leak memory forever.
        max_len = int(MIC_MAX_BUFFER_SECONDS * MIC_FS)
        if len(self._mic_buffer) > max_len:
            self._mic_buffer = self._mic_buffer[-max_len:]

        # Feed the Waveform tab so you can see what the mic is hearing live.
        self.live_audio_ready.emit(self._mic_buffer, MIC_FS)

        # Nothing captured yet (e.g. the very first tick, before the OS has
        # delivered any audio) — segment_tone_regions() has nothing to work
        # with, so just wait for the next tick instead of calling it on an
        # empty array.
        if len(self._mic_buffer) == 0:
            return

        regions = dsp.segment_tone_regions(self._mic_buffer, MIC_FS)
        margin = int(MIC_SAFETY_MARGIN_SECONDS * MIC_FS)
        algo_name = self.algo_combo.currentText()

        # Regions come back in chronological order. Walk forward; only trust
        # a region once there's real silence (the margin) after it — the
        # first region without enough trailing buffer might still be an
        # in-progress tone, so stop there and leave it for next tick.
        confirmed_end = 0
        new_digits = ""

        for start, end in regions:
            if end + margin > len(self._mic_buffer):
                break  # not safely closed yet

            confirmed_end = end
            region_audio = self._mic_buffer[start:end]

            # segment_tone_regions only knows "this was loud" — it can't
            # tell real DTMF apart from a loud cough or a door closing.
            # This extra check can, so anything that doesn't look like a
            # genuine two-tone signal gets silently skipped here instead of
            # turning into a bogus digit.
            if not self._is_valid_dtmf_region(region_audio, MIC_FS):
                continue

            digits, _, error = self._run_decode(algo_name, region_audio, MIC_FS)
            if error:
                self.status_label.setText(error)
                return
            new_digits += digits

        if new_digits:
            self._mic_transcript += new_digits
            self.result_label.setText(self._mic_transcript)
            self.status_label.setText(f"Listening… {len(self._mic_transcript)} digit(s) so far.")

        if confirmed_end > 0:
            # Advance the buffer past everything we just looked at — this is
            # what keeps memory and per-tick work bounded forever, whether
            # those regions turned out to be real tones or rejected noise.
            self._mic_buffer = self._mic_buffer[confirmed_end:]

    def _is_valid_dtmf_region(self, x_region: np.ndarray, fs: int) -> bool:
        """
        Confidence gate for the mic path. Lives here, not in
        dsp_interface.py — it's a validation layer built on top of the
        already-exposed goertzel_single_freq(), not new core math.

        The idea: a genuine DTMF tone concentrates almost all its energy
        into exactly 2 of the 8 possible frequencies (measured: >99.9% for
        real tones, even noisy ones). Room noise spreads energy out across
        all 8 fairly evenly instead (measured: pure noise tops out around
        58% "dominance" across many trials). MIC_MIN_DOMINANCE sits well
        inside that gap.
        """
        if len(x_region) == 0:
            return False

        low_powers = [dsp.goertzel_single_freq(x_region, fs, f) for f in dsp.LOW_FREQS]
        high_powers = [dsp.goertzel_single_freq(x_region, fs, f) for f in dsp.HIGH_FREQS]

        best_low = max(low_powers)
        best_high = max(high_powers)
        total_power = sum(low_powers) + sum(high_powers)

        if total_power <= 0:
            return False

        dominance = (best_low + best_high) / total_power
        return dominance >= MIC_MIN_DOMINANCE
