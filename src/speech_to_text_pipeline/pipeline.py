"""The M1 pipeline: decode and resample, cut silence, detect language, transcribe chunks, map times."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from faster_whisper.tokenizer import _LANGUAGE_CODES

from speech_to_text_pipeline.audio import SAMPLE_RATE, Samples, load_audio
from speech_to_text_pipeline.chunking import WHISPER_WINDOW_S, plan_chunks
from speech_to_text_pipeline.engine import Engine, RawSegment
from speech_to_text_pipeline.errors import PipelineError
from speech_to_text_pipeline.transcript import LanguageGuess, Segment, Stats, Transcript, Word
from speech_to_text_pipeline.vad import (
    SpeechSpan,
    TimeMap,
    VadConfig,
    detect_speech,
    remove_silence,
)

# A resumed pass starts at a span join no earlier than 0.3 s before the last transcribed end,
# and only when at least 1 s of the chunk is left.
RESUME_TOLERANCE_S = 0.3
MIN_REST_S = 1.0
LANGUAGE_CODES: frozenset[str] = frozenset(str(code) for code in _LANGUAGE_CODES)


@dataclass(frozen=True)
class Options:
    """Run settings: forced language (None detects), VAD on or off, chunk length."""

    language: str | None = None
    vad: bool = True
    vad_config: VadConfig = field(default_factory=VadConfig)
    chunk_seconds: float = WHISPER_WINDOW_S

    def __post_init__(self) -> None:
        """Refuse unknown language codes and bad chunk lengths."""
        if self.language is not None and self.language not in LANGUAGE_CODES:
            raise PipelineError(f"unknown language code {self.language!r}")
        if not 1.0 <= self.chunk_seconds <= WHISPER_WINDOW_S:
            raise PipelineError(f"chunk length must be between 1 and {WHISPER_WINDOW_S:.0f} s")


def transcribe_file(path: Path, engine: Engine, options: Options | None = None) -> Transcript:
    """Run the whole pipeline on one audio file."""
    started = time.perf_counter()
    audio = load_audio(path)
    decoded = time.perf_counter()
    return transcribe_samples(
        audio.samples, engine, options, source=str(path), decode_seconds=decoded - started
    )


def transcribe_samples(
    samples: Samples,
    engine: Engine,
    options: Options | None = None,
    source: str = "<samples>",
    decode_seconds: float = 0.0,
) -> Transcript:
    """Run VAD, language detection and chunked transcription on 16 kHz mono samples."""
    options = options or Options()
    if samples.ndim != 1 or samples.size == 0:
        raise PipelineError("expected non-empty mono samples")
    t0 = time.perf_counter()
    if options.vad:
        spans = detect_speech(samples, options.vad_config)
    else:
        spans = [SpeechSpan(0, samples.size)]
    t1 = time.perf_counter()
    audio_seconds = samples.size / SAMPLE_RATE
    if not spans:
        stats = Stats(audio_seconds, 0.0, 0, decode_seconds, t1 - t0, 0.0)
        forced = _forced(options)
        return Transcript(source, engine.name, forced, forced is not None, (), stats)
    compact, time_map = remove_silence(samples, spans)
    language = _forced(options) or engine.detect_language(
        compact[: int(WHISPER_WINDOW_S * SAMPLE_RATE)]
    )
    chunks = plan_chunks([s.length for s in spans], options.chunk_seconds)
    segments: list[Segment] = []
    passes = 0
    for chunk in chunks:
        start: int | None = chunk.start
        while start is not None:
            raw = engine.transcribe(compact[start : chunk.end], language.language)
            passes += 1
            begin = start / SAMPLE_RATE
            segments.extend(_place(seg, begin, chunk.end_s, time_map) for seg in raw)
            covered = start + int(max((seg.end for seg in raw), default=0.0) * SAMPLE_RATE)
            start = _resume_point(time_map.compact_starts, covered, start, chunk.end)
    t2 = time.perf_counter()
    segments.sort(key=lambda s: (s.start, s.end))
    stats = Stats(
        audio_seconds, time_map.total_s, len(chunks), decode_seconds, t1 - t0, t2 - t1, passes
    )
    return Transcript(
        source, engine.name, language, options.language is not None, tuple(segments), stats
    )


def _resume_point(joins: list[int], covered: int, start: int, end: int) -> int | None:
    """Where to transcribe again when the engine stopped early: the next span join, if any.

    Whisper can end a window after one utterance when two utterances are joined with no pause
    (found on the 2-clip demo, see docs/results/m1.md); the rest of the chunk is then sent again,
    starting at the first span join at or after the covered point (never inside a span), so each
    join is used at most once and the loop always moves forward.
    """
    if covered <= start:
        return None
    tolerance = int(RESUME_TOLERANCE_S * SAMPLE_RATE)
    for join in joins:
        if join > start and join >= covered - tolerance:
            return join if end - join >= int(MIN_REST_S * SAMPLE_RATE) else None
    return None


def _forced(options: Options) -> LanguageGuess | None:
    """The language the caller fixed, if any."""
    return LanguageGuess(options.language, 1.0) if options.language else None


def _place(seg: RawSegment, chunk_start: float, chunk_end: float, time_map: TimeMap) -> Segment:
    """Move a chunk-relative segment and its words to original-recording time."""

    def to_original(t: float, is_end: bool) -> float:
        """Chunk time to compact time (clamped to the chunk) to original time."""
        compact = min(max(chunk_start + t, chunk_start), chunk_end)
        return round(time_map.to_original(compact, is_end=is_end), 3)

    words = []
    for w in seg.words:
        start = to_original(w.start, is_end=False)
        end = max(to_original(w.end, is_end=True), start)
        words.append(Word(start, end, w.text, round(w.probability, 4)))
    start = to_original(seg.start, is_end=False)
    end = max(to_original(seg.end, is_end=True), start)
    return Segment(start, end, seg.text, tuple(words))
