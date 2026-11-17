"""Voice-activity detection (Silero VAD) and the map from silence-free time back to original time."""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps

from speech_to_text_pipeline.audio import SAMPLE_RATE, Samples
from speech_to_text_pipeline.errors import PipelineError


@dataclass(frozen=True)
class SpeechSpan:
    """A stretch of speech in the original audio, in samples at 16 kHz (end exclusive)."""

    start: int
    end: int

    @property
    def start_s(self) -> float:
        """Start in seconds."""
        return self.start / SAMPLE_RATE

    @property
    def end_s(self) -> float:
        """End in seconds."""
        return self.end / SAMPLE_RATE

    @property
    def length(self) -> int:
        """Length in samples."""
        return self.end - self.start


@dataclass(frozen=True)
class VadConfig:
    """Silero VAD settings (probability threshold, gaps, padding)."""

    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 500
    speech_pad_ms: int = 200

    def __post_init__(self) -> None:
        """Refuse settings Silero cannot use."""
        if not 0.0 < self.threshold < 1.0:
            raise PipelineError("VAD threshold must be between 0 and 1")
        if min(self.min_speech_ms, self.min_silence_ms, self.speech_pad_ms) < 0:
            raise PipelineError("VAD durations must not be negative")


def validate_spans(spans: list[SpeechSpan], total: int) -> None:
    """Spans must be non-empty, sorted, non-overlapping and inside the audio."""
    previous_end = 0
    for span in spans:
        if span.start < previous_end or span.end <= span.start or span.end > total:
            raise PipelineError(f"invalid speech span {span} (audio has {total} samples)")
        previous_end = span.end


def detect_speech(samples: Samples, config: VadConfig | None = None) -> list[SpeechSpan]:
    """Speech spans found by Silero VAD (the ONNX model shipped inside faster-whisper)."""
    config = config or VadConfig()
    if samples.ndim != 1:
        raise PipelineError("VAD needs mono audio")
    if samples.size == 0:
        return []
    options = VadOptions(
        threshold=config.threshold,
        min_speech_duration_ms=config.min_speech_ms,
        min_silence_duration_ms=config.min_silence_ms,
        speech_pad_ms=config.speech_pad_ms,
    )
    raw = get_speech_timestamps(samples, options, sampling_rate=SAMPLE_RATE)
    spans = [SpeechSpan(int(r["start"]), int(r["end"])) for r in raw]
    validate_spans(spans, samples.size)
    return spans


class TimeMap:
    """Maps a time in the silence-free audio back to the original recording."""

    def __init__(self, spans: list[SpeechSpan]) -> None:
        """Build cumulative offsets for sorted, non-overlapping spans."""
        if not spans:
            raise PipelineError("a time map needs at least one speech span")
        validate_spans(spans, spans[-1].end)
        self.spans = list(spans)
        self.compact_starts: list[int] = []
        total = 0
        for span in spans:
            self.compact_starts.append(total)
            total += span.length
        self.total = total

    @property
    def total_s(self) -> float:
        """Length of the silence-free audio in seconds."""
        return self.total / SAMPLE_RATE

    def to_original(self, seconds: float, is_end: bool = False) -> float:
        """Original time of a compact time; an end exactly on a join stays in the earlier span."""
        sample = min(max(seconds * SAMPLE_RATE, 0.0), float(self.total))
        if is_end:
            index = bisect.bisect_left(self.compact_starts, sample) - 1
        else:
            index = bisect.bisect_right(self.compact_starts, sample) - 1
        index = max(index, 0)
        span = self.spans[index]
        original = span.start + (sample - self.compact_starts[index])
        return min(original, float(span.end)) / SAMPLE_RATE


def remove_silence(samples: Samples, spans: list[SpeechSpan]) -> tuple[Samples, TimeMap]:
    """Concatenate the speech spans and return the map back to original time."""
    validate_spans(spans, samples.size)
    if not spans:
        raise PipelineError("no speech to keep")
    compact = np.concatenate([samples[s.start : s.end] for s in spans]).astype(np.float32)
    return compact, TimeMap(spans)
