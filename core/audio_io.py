"""
audio_io.py
===========
Thin wrapper around `sounddevice` for playing back numpy arrays and writing
WAV files. This is plumbing (device I/O), not signal-processing math, so it
lives outside dsp_interface.py — you shouldn't need to touch this file.

Kept Qt-free on purpose so it can be reused or tested outside the GUI too.
"""

from __future__ import annotations
import queue
import numpy as np
import sounddevice as sd
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

    def __init__(self, fs: int = 8000, channels: int = 1, blocksize: int = 800):
        self.fs = fs
        self._queue: queue.Queue = queue.Queue()
        self._stream = sd.InputStream(
            samplerate=fs,
            channels=channels,
            blocksize=blocksize,
            dtype="float32",
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        # indata arrives as float32 (the format devices actually support);
        # convert to float64 here so everything downstream (segmentation,
        # Goertzel, FFT) sees the same dtype it always has.
        self._queue.put(indata[:, 0].astype(np.float64))

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()

    def read_available(self) -> np.ndarray:
        """Pop everything currently queued and return it as one array.
        Returns an empty array (not an error) if nothing new has arrived."""
        chunks = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return np.array([])
        return np.concatenate(chunks)
