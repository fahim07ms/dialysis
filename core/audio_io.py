"""
audio_io.py
===========
Thin wrapper around `sounddevice` for playing back numpy arrays and writing
WAV files. This is plumbing (device I/O), not signal-processing math, so it
lives outside dsp_interface.py — you shouldn't need to touch this file.

Kept Qt-free on purpose so it can be reused or tested outside the GUI too.
"""

from __future__ import annotations
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
