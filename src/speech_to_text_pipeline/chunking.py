"""Pack speech spans into chunks of at most one Whisper window, cutting at silence where possible."""

from __future__ import annotations

from dataclasses import dataclass

from speech_to_text_pipeline.audio import SAMPLE_RATE
from speech_to_text_pipeline.errors import PipelineError

WHISPER_WINDOW_S = 30.0


@dataclass(frozen=True)
class Chunk:
    """A slice of the silence-free audio, in samples (end exclusive)."""

    start: int
    end: int

    @property
    def start_s(self) -> float:
        """Start in seconds of silence-free time."""
        return self.start / SAMPLE_RATE

    @property
    def end_s(self) -> float:
        """End in seconds of silence-free time."""
        return self.end / SAMPLE_RATE


def plan_chunks(span_lengths: list[int], max_seconds: float = WHISPER_WINDOW_S) -> list[Chunk]:
    """Greedy packing of whole spans; a span longer than the limit is cut into equal-limit pieces."""
    if max_seconds <= 0:
        raise PipelineError("chunk length must be positive")
    if any(length <= 0 for length in span_lengths):
        raise PipelineError("span lengths must be positive")
    limit = int(max_seconds * SAMPLE_RATE)
    chunks: list[Chunk] = []
    start = 0
    end = 0
    for length in span_lengths:
        if end - start + length <= limit:
            end += length
            continue
        if end > start:
            chunks.append(Chunk(start, end))
        start = end
        while length > limit:
            chunks.append(Chunk(start, start + limit))
            start += limit
            length -= limit
        end = start + length
    if end > start:
        chunks.append(Chunk(start, end))
    return chunks
