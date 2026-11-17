"""Transcript output formats and atomic writes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from speech_to_text_pipeline.files import write_text
from speech_to_text_pipeline.transcript import (
    LanguageGuess,
    Segment,
    Stats,
    Transcript,
    Word,
    clock,
)


def sample() -> Transcript:
    """A two-segment transcript."""
    words = (Word(2.5, 2.9, "Hello", 0.9), Word(3.0, 3.4, "world", 0.8))
    segments = (
        Segment(2.5, 3.4, "Hello world", words),
        Segment(3661.25, 3662.0, "これは", (Word(3661.25, 3662.0, "これは", 0.7),)),
    )
    stats = Stats(4000.0, 1000.0, 2, 1.0, 2.0, 37.0)
    return Transcript("a.wav", "fake", LanguageGuess("en", 0.95), False, segments, stats)


def test_clock_formats_hours_minutes_seconds_millis() -> None:
    assert clock(0) == "00:00:00.000"
    assert clock(3661.25) == "01:01:01.250"
    assert clock(59.9996) == "00:01:00.000"
    assert clock(-3) == "00:00:00.000"


def test_text_form_has_one_line_per_segment() -> None:
    assert sample().to_text() == (
        "[00:00:02.500 - 00:00:03.400] Hello world\n[01:01:01.250 - 01:01:02.000] これは\n"
    )


def test_json_form_keeps_words_stats_and_utc_time() -> None:
    data = json.loads(sample().to_json())
    assert data["schema_version"] == 1
    assert data["text"] == "Hello world これは"
    assert data["segments"][0]["words"][1] == {
        "start": 3.0, "end": 3.4, "text": "world", "probability": 0.8
    }  # fmt: skip
    assert data["stats"]["removed_seconds"] == 3000.0
    assert data["stats"]["real_time_factor"] == 0.01
    assert data["created_utc"].endswith("+00:00")
    assert "これは" in sample().to_json()


def test_real_time_factor_of_empty_audio_is_zero() -> None:
    assert Stats(0.0, 0.0, 0, 0.1, 0.1, 0.0).real_time_factor == 0.0


def test_a_failed_write_keeps_the_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "out.json"
    target.write_text("old", encoding="utf-8")

    def broken(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError, match="disk full"):
        write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


def test_write_creates_folders_and_replaces(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b.txt"
    write_text(target, "one")
    write_text(target, "two")
    assert target.read_text(encoding="utf-8") == "two"
