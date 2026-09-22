"""
audio_io.py
===========
Thin wrapper around `sounddevice` for playing back numpy arrays and writing
WAV files. This is plumbing (device I/O), not signal-processing math, so it
lives outside dsp_interface.py — you shouldn't need to touch this file.

Kept Qt-free on purpose so it can be reused or tested outside the GUI too.

MicStream changes for the live DTMF decoder:
  * Capture runs at the input device's DEFAULT sample rate instead of a
    forced 8 kHz. Some devices / OS mixers resample poorly or reject 8 kHz
    outright, and nothing in the detection math needs 8 kHz — the frame
    engine sizes its windows from whatever fs it is given. The actual rate
    is reported via `stream.fs`; open the stream and READ it, don't assume.
  * A one-pole DC blocker (~80 Hz high-pass), with filter state carried
    across callback chunks, keeps mic DC offset / bias out of the level
    estimates downstream.
  * The internal queue is bounded (drop-oldest) and overflows are COUNTED
    (`overflow_count`, `pa_status_count`) so the UI can warn about dropped
    audio instead of silently losing it.
"""

from __future__ import annotations
import queue
import numpy as np
import sounddevice as sd
import scipy.signal
from scipy.io import wavfile


def play_array(x: np.ndarray, fs: int, blocking: bool = False) -> None:
    """Play a 1-D numpy float array through the default output device."""
    x = np.asarray(x, dtype=np.float32)
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak > 1.0:
        x = x / peak  # simple safety normalization so we never clip/distort
    sd.play(x, samplerate=fs)
    if blocking:
        sd.wait()


def stop_playback() -> None:
    sd.stop()


def save_wav(path: str, x: np.ndarray, fs: int) -> None:
    """Write a 1-D numpy float array out as a 16-bit PCM WAV file."""
    x = np.asarray(x, dtype=np.float64)
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    pcm16 = (x * 32767).astype(np.int16)
    wavfile.write(path, fs, pcm16)


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file back as (float64 array in [-1, 1], sample_rate)."""
    fs, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float64) - 128) / 128.0
    else:
        data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)  # collapse stereo to mono
    return data, fs


class MicStream:
    """
    Continuous microphone capture, push-based: sounddevice calls our
    callback from its own internal audio thread whenever a new chunk is
    ready; we just drop each chunk into a thread-safe queue. Whoever wants
    the audio (a QTimer on the UI thread, typically) calls read_available()
    to drain whatever has arrived since the last check — non-blocking, so
    it's safe to call from a UI timer without risking a freeze.

    Deliberately Qt-free, same as the rest of this file.
    """

    def __init__(self, fs: int | None = None, channels: int = 1,
                 blocksize: int | None = None, queue_chunks: int = 64):
        # Ask the OS for the default input device and its native rate first.
        try:
            info = sd.query_devices(kind="input")
            default_rate = int(info.get("default_samplerate", 48000))
            self.device_name = str(info.get("name", "default input device"))
        except Exception:
            default_rate = 48000
            self.device_name = "default input device"

        self.requested_fs = int(fs) if fs is not None else default_rate
        self.fs = self.requested_fs
        self.overflow_count = 0    # chunks we dropped because our queue was full
        self.pa_status_count = 0   # PortAudio-side input overflows reported in callbacks

        self._queue: queue.Queue = queue.Queue(maxsize=queue_chunks)
        try:
            self._stream = self._open_stream(self.fs, channels, blocksize)
        except Exception:
            if self.fs == default_rate:
                raise  # nothing left to fall back to
            # The requested rate isn't supported by this device — fall back
            # to its native default rate rather than failing. The detection
            # engine adapts to whatever fs it gets, so this is transparent.
            self.fs = default_rate
            self._stream = self._open_stream(self.fs, channels, blocksize)

        # DC blocker: y[n] = x[n] - x[n-1] + R*y[n-1]  (pole at R).
        # Cutoff ~ (1-R)*fs/(2*pi) ~ 80 Hz — far below DTMF's lowest tone
        # (697 Hz passes at ~unity gain), while DC offset / rumble is removed.
        r = 1.0 - 2.0 * np.pi * 80.0 / self.fs
        self._hp_b = [1.0, -1.0]
        self._hp_a = [1.0, -r]
        self._hp_zi = np.zeros(1)

    def _open_stream(self, samplerate: int, channels: int, blocksize):
        return sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            blocksize=blocksize,
            dtype="float32",
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        if status:
            self.pa_status_count += 1
        mono = indata[:, 0].astype(np.float64)
        # DC-block on the audio thread. A few hundred samples through a
        # 2-tap IIR is microseconds — safe to do here.
        mono, self._hp_zi = scipy.signal.lfilter(
            self._hp_b, self._hp_a, mono, zi=self._hp_zi
        )
        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            # The consumer isn't draining fast enough. Drop the OLDEST
            # queued chunk to make room (keeping newest = keeping the live
            # edge), and count the loss so the UI can warn the user.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(mono)
            except queue.Full:
                pass
            self.overflow_count += 1

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()

    def read_available(self) -> np.ndarray:
        """Pop everything currently queued and return it as one array.
        Returns an empty array (not an error) if nothing new has arrived.
        Samples are float64, mono, DC-blocked, at the rate given by .fs."""
        chunks = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return np.array([])
        return np.concatenate(chunks)