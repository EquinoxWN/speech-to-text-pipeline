"""Transcript data model and its JSON and text forms."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Word:
    """One word with its time in the original recording."""

    start: float
    end: float
    text: str
    probability: float


@dataclass(frozen=True)
class Segment:
    """A run of words Whisper emitted together."""

    start: float
    end: float
    text: str
    words: tuple[Word, ...]


@dataclass(frozen=True)
class LanguageGuess:
    """Detected language, its probability and the runners-up."""

    language: str
    probability: float
    top: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class Stats:
    """Durations and timings of one run (wall times from a monotonic clock)."""

    audio_seconds: float
    speech_seconds: float
    chunks: int
    decode_seconds: float
    vad_seconds: float
    transcribe_seconds: float
    engine_calls: int = 0

    @property
    def removed_seconds(self) -> float:
        """Silence cut before transcription."""
        return self.audio_seconds - self.speech_seconds

    @property
    def real_time_factor(self) -> float:
        """Total processing time divided by audio length (below 1 is faster than real time)."""
        total = self.decode_seconds + self.vad_seconds + self.transcribe_seconds
        return total / self.audio_seconds if self.audio_seconds > 0 else 0.0


@dataclass(frozen=True)
class Transcript:
    """Everything one run produced."""

    source: str
    model: str
    language: LanguageGuess | None
    language_forced: bool
    segments: tuple[Segment, ...]
    stats: Stats
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def text(self) -> str:
        """Whole transcript as one string."""
        return " ".join(s.text for s in self.segments if s.text).strip()

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dictionary (schema version 1)."""
        data = asdict(self)
        data["schema_version"] = SCHEMA_VERSION
        data["text"] = self.text
        data["stats"]["removed_seconds"] = round(self.stats.removed_seconds, 3)
        data["stats"]["real_time_factor"] = round(self.stats.real_time_factor, 4)
        return data

    def to_json(self) -> str:
        """Pretty JSON, UTF-8 characters kept as they are."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"

    def to_text(self) -> str:
        """One line per segment: [start - end] text, times in the original recording."""
        lines = [f"[{clock(s.start)} - {clock(s.end)}] {s.text}" for s in self.segments]
        return "\n".join(lines) + ("\n" if lines else "")


def clock(seconds: float) -> str:
    """Seconds as HH:MM:SS.mmm."""
    millis = round(max(seconds, 0.0) * 1000)
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
