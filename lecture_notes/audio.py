"""Decode any audio/video file to 16 kHz mono PCM with PyAV (no separate ffmpeg install)."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Callable

import av
import numpy as np

SAMPLE_RATE = 16000


def extract_wav(
    src: str | Path,
    dst: str | Path,
    on_progress: Callable[[float], None] | None = None,
) -> float:
    """Write `src`'s first audio track to `dst` as 16 kHz mono 16-bit WAV.

    Streams frame by frame, so multi-GB videos never sit in memory.
    Returns the audio duration in seconds.
    """
    src, dst = Path(src), Path(dst)
    written = 0
    next_report = 0
    with av.open(str(src)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError(f"No audio track found in {src.name}")
        stream.thread_type = "AUTO"
        total = container.duration / av.time_base if container.duration else None
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)

        with wave.open(str(dst), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)

            def write(frames) -> None:
                nonlocal written
                for out in frames:
                    w.writeframes(out.to_ndarray().astype(np.int16, copy=False).tobytes())
                    written += out.samples

            for frame in container.decode(stream):
                write(resampler.resample(frame))
                if on_progress and total and written >= next_report:
                    on_progress(min(written / SAMPLE_RATE / total, 1.0))
                    next_report = written + 60 * SAMPLE_RATE
            write(resampler.resample(None))

    if written == 0:
        raise ValueError(f"The audio track in {src.name} is empty")
    return written / SAMPLE_RATE


def load_wav(path: str | Path) -> np.ndarray:
    """Load a 16 kHz mono 16-bit WAV as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16 kHz mono 16-bit PCM")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0
