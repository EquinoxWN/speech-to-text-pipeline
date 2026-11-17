"""Decode any audio file and resample it to 16 kHz mono float32 (Whisper's input format)."""

from __future__ import annotations

import io
import itertools
import wave
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import numpy.typing as npt

from speech_to_text_pipeline.errors import PipelineError
from speech_to_text_pipeline.files import write_bytes

SAMPLE_RATE = 16_000
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_SECONDS = 4 * 3600.0

Samples = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Audio:
    """Decoded audio at 16 kHz mono, with what the source file was."""

    samples: Samples
    source_rate: int
    source_channels: int

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return len(self.samples) / SAMPLE_RATE


def _decode(path: Path, max_seconds: float) -> tuple[Samples, int, int]:
    """Decode the first audio stream through FFmpeg's resampler to 16 kHz mono float32."""
    limit = int(max_seconds * SAMPLE_RATE)
    parts: list[Samples] = []
    total = 0
    try:
        with av.open(str(path), mode="r") as container:
            if not container.streams.audio:
                raise PipelineError(f"{path}: no audio stream")
            stream = container.streams.audio[0]
            rate, channels = int(stream.rate or 0), int(stream.channels or 0)
            if rate <= 0 or channels <= 0:
                raise PipelineError(f"{path}: audio stream has no sample rate or channels")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
            for frame in itertools.chain(container.decode(stream), [None]):
                for out in resampler.resample(frame):
                    block = out.to_ndarray().reshape(-1).astype(np.float32)
                    total += block.size
                    if total > limit:
                        raise PipelineError(f"{path}: longer than the {max_seconds:.0f} s limit")
                    parts.append(block)
    except av.error.FFmpegError as exc:
        raise PipelineError(f"{path}: not a readable audio file ({exc})") from exc
    samples = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return samples, rate, channels


def load_audio(
    path: Path, max_bytes: int = MAX_FILE_BYTES, max_seconds: float = MAX_SECONDS
) -> Audio:
    """Read a file of any FFmpeg-supported format as 16 kHz mono float32, size and length capped."""
    if not path.is_file():
        raise PipelineError(f"{path}: not a file")
    size = path.stat().st_size
    if size == 0:
        raise PipelineError(f"{path}: empty file")
    if size > max_bytes:
        raise PipelineError(f"{path}: {size} bytes is over the {max_bytes} byte limit")
    samples, rate, channels = _decode(path, max_seconds)
    if samples.size == 0:
        raise PipelineError(f"{path}: no audio samples decoded")
    if not np.isfinite(samples).all():
        raise PipelineError(f"{path}: decoded samples are not finite")
    return Audio(samples=np.ascontiguousarray(samples), source_rate=rate, source_channels=channels)


def wav_bytes(samples: npt.NDArray[np.floating], rate: int, channels: int = 1) -> bytes:
    """Encode float samples in [-1, 1] (interleaved if stereo) as 16-bit PCM WAV."""
    if rate <= 0 or channels <= 0:
        raise PipelineError("rate and channels must be positive")
    if samples.size % channels:
        raise PipelineError("sample count is not a multiple of the channel count")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).round().astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()


def write_wav(path: Path, samples: npt.NDArray[np.floating], rate: int, channels: int = 1) -> None:
    """Write a 16-bit PCM WAV file atomically."""
    write_bytes(path, wav_bytes(samples, rate, channels))
