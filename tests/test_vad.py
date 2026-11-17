"""Silence removal and the map from silence-free time back to original time."""

from __future__ import annotations

import numpy as np
import pytest

from speech_to_text_pipeline.audio import SAMPLE_RATE
from speech_to_text_pipeline.errors import PipelineError
from speech_to_text_pipeline.vad import (
    SpeechSpan,
    TimeMap,
    VadConfig,
    detect_speech,
    remove_silence,
    validate_spans,
)

SR = SAMPLE_RATE


def spans_s(*pairs: tuple[float, float]) -> list[SpeechSpan]:
    """Spans from (start, end) pairs in seconds."""
    return [SpeechSpan(int(a * SR), int(b * SR)) for a, b in pairs]


def test_time_map_adds_back_the_silence_before_each_span() -> None:
    # Speech 2-5 s and 9-10 s: compact 0-3 s is 2-5 s, compact 3-4 s is 9-10 s.
    m = TimeMap(spans_s((2, 5), (9, 10)))
    assert m.total_s == pytest.approx(4.0)
    assert m.to_original(0.0) == pytest.approx(2.0)
    assert m.to_original(1.25) == pytest.approx(3.25)
    assert m.to_original(3.0) == pytest.approx(9.0)
    assert m.to_original(3.5) == pytest.approx(9.5)
    assert m.to_original(4.0, is_end=True) == pytest.approx(10.0)


def test_an_end_on_a_join_stays_in_the_earlier_span() -> None:
    m = TimeMap(spans_s((2, 5), (9, 10)))
    assert m.to_original(3.0, is_end=True) == pytest.approx(5.0)
    assert m.to_original(3.0, is_end=False) == pytest.approx(9.0)


def test_times_outside_the_compact_audio_are_clamped() -> None:
    m = TimeMap(spans_s((2, 5), (9, 10)))
    assert m.to_original(-1.0) == pytest.approx(2.0)
    assert m.to_original(99.0) == pytest.approx(10.0)
    assert m.to_original(99.0, is_end=True) == pytest.approx(10.0)


def test_three_spans_keep_order_and_offsets() -> None:
    m = TimeMap(spans_s((0, 1), (3, 4), (10, 12)))
    assert [m.to_original(t) for t in (0.5, 1.5, 2.0, 3.0)] == pytest.approx([0.5, 3.5, 10.0, 11.0])


def test_remove_silence_concatenates_exactly_the_speech_samples() -> None:
    samples = np.arange(10 * SR, dtype=np.float32)
    spans = spans_s((1, 2), (4, 4.5))
    compact, m = remove_silence(samples, spans)
    assert compact.size == int(1.5 * SR)
    assert compact[0] == SR
    assert compact[SR - 1] == 2 * SR - 1
    assert compact[SR] == 4 * SR
    assert compact[-1] == int(4.5 * SR) - 1
    assert m.to_original(1.0) == pytest.approx(4.0)


@pytest.mark.parametrize(
    "spans",
    [
        [SpeechSpan(10, 5)],
        [SpeechSpan(5, 5)],
        [SpeechSpan(0, 10), SpeechSpan(5, 20)],
        [SpeechSpan(20, 30), SpeechSpan(0, 10)],
        [SpeechSpan(-5, 10)],
        [SpeechSpan(0, 101)],
    ],
)
def test_invalid_spans_are_refused(spans: list[SpeechSpan]) -> None:
    with pytest.raises(PipelineError, match="invalid speech span"):
        validate_spans(spans, 100)


def test_touching_spans_are_allowed() -> None:
    validate_spans([SpeechSpan(0, 10), SpeechSpan(10, 20)], 20)


def test_empty_inputs_are_refused_or_empty() -> None:
    with pytest.raises(PipelineError, match="at least one speech span"):
        TimeMap([])
    with pytest.raises(PipelineError, match="no speech"):
        remove_silence(np.zeros(100, dtype=np.float32), [])
    assert detect_speech(np.zeros(0, dtype=np.float32)) == []


def test_silero_finds_no_speech_in_silence_or_a_steady_tone() -> None:
    assert detect_speech(np.zeros(5 * SR, dtype=np.float32)) == []
    t = np.arange(5 * SR) / SR
    assert detect_speech((0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)) == []


def test_vad_needs_mono() -> None:
    with pytest.raises(PipelineError, match="mono"):
        detect_speech(np.zeros((2, 100), dtype=np.float32))


@pytest.mark.parametrize(
    "kwargs",
    [{"threshold": 0.0}, {"threshold": 1.0}, {"min_silence_ms": -1}, {"speech_pad_ms": -5}],
)
def test_bad_vad_settings_are_refused(kwargs: dict[str, float]) -> None:
    with pytest.raises(PipelineError):
        VadConfig(**kwargs)  # type: ignore[arg-type]
