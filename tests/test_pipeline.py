"""The pipeline with a scripted engine: chunk timing mapped back to the original recording."""

from __future__ import annotations

import numpy as np
import pytest

from speech_to_text_pipeline.audio import SAMPLE_RATE
from speech_to_text_pipeline.engine import RawSegment, RawWord
from speech_to_text_pipeline.errors import PipelineError
from speech_to_text_pipeline.pipeline import Options, _resume_point, transcribe_samples
from speech_to_text_pipeline.vad import SpeechSpan
from tests.conftest import FakeEngine

SR = SAMPLE_RATE


def patch_vad(monkeypatch: pytest.MonkeyPatch, spans: list[tuple[float, float]]) -> None:
    """Make VAD return fixed spans (seconds) so timing can be checked exactly."""
    found = [SpeechSpan(int(a * SR), int(b * SR)) for a, b in spans]
    monkeypatch.setattr("speech_to_text_pipeline.pipeline.detect_speech", lambda s, c: found)


def test_word_times_come_back_in_original_time(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_vad(monkeypatch, [(3, 7), (12, 15)])
    engine = FakeEngine()
    t = transcribe_samples(np.zeros(20 * SR, dtype=np.float32), engine)
    # One chunk of 7 s compact audio: first word 0.5-1.0 -> 3.5-4.0; last word 6.5-7.0 -> 14.5-15.0.
    assert engine.calls == [(7 * SR, "en")]
    words = t.segments[0].words
    assert (words[0].start, words[0].end) == (3.5, 4.0)
    assert (words[1].start, words[1].end) == (14.5, 15.0)
    assert (t.segments[0].start, t.segments[0].end) == (3.5, 15.0)
    assert t.stats.audio_seconds == 20.0
    assert t.stats.speech_seconds == 7.0
    assert t.stats.removed_seconds == 13.0


def test_each_chunk_is_offset_by_its_own_start(monkeypatch: pytest.MonkeyPatch) -> None:
    # 20 s + 20 s of speech do not fit one 30 s window: two chunks, cut at the silence.
    patch_vad(monkeypatch, [(1, 21), (40, 60)])
    engine = FakeEngine()
    t = transcribe_samples(np.zeros(61 * SR, dtype=np.float32), engine)
    assert [n for n, _ in engine.calls] == [20 * SR, 20 * SR]
    assert t.stats.chunks == 2
    assert [(s.start, s.end) for s in t.segments] == [(1.5, 21.0), (40.5, 60.0)]
    assert [w.text for s in t.segments for w in s.words] == ["first1", "last1", "first2", "last2"]


def test_language_is_detected_on_at_most_30_s_of_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_vad(monkeypatch, [(0, 50)])
    engine = FakeEngine(language="ja")
    t = transcribe_samples(np.zeros(50 * SR, dtype=np.float32), engine)
    assert engine.detected_on == [30 * SR]
    assert t.language is not None
    assert t.language.language == "ja"
    assert not t.language_forced
    assert {lang for _, lang in engine.calls} == {"ja"}


def test_a_forced_language_skips_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_vad(monkeypatch, [(0, 5)])
    engine = FakeEngine(language="en")
    t = transcribe_samples(np.zeros(5 * SR, dtype=np.float32), engine, Options(language="id"))
    assert engine.detected_on == []
    assert engine.calls == [(5 * SR, "id")]
    assert t.language_forced
    assert t.language is not None
    assert t.language.language == "id"


def test_no_speech_means_no_engine_call(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_vad(monkeypatch, [])
    engine = FakeEngine()
    t = transcribe_samples(np.zeros(5 * SR, dtype=np.float32), engine)
    assert engine.calls == []
    assert engine.detected_on == []
    assert t.segments == ()
    assert t.language is None
    assert t.stats.removed_seconds == 5.0
    assert t.text == ""


def test_without_vad_the_whole_recording_is_kept() -> None:
    engine = FakeEngine()
    t = transcribe_samples(np.zeros(4 * SR, dtype=np.float32), engine, Options(vad=False))
    assert engine.calls == [(4 * SR, "en")]
    assert t.stats.removed_seconds == 0.0
    assert t.segments[0].words[0].start == 0.5


def test_real_vad_on_pure_silence_finds_nothing() -> None:
    engine = FakeEngine()
    t = transcribe_samples(np.zeros(3 * SR, dtype=np.float32), engine)
    assert engine.calls == []
    assert t.stats.speech_seconds == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [{"language": "xx"}, {"language": "english"}, {"chunk_seconds": 0.5}, {"chunk_seconds": 31}],
)
def test_bad_options_are_refused(kwargs: dict[str, object]) -> None:
    with pytest.raises(PipelineError):
        Options(**kwargs)  # type: ignore[arg-type]


def test_bad_samples_are_refused() -> None:
    with pytest.raises(PipelineError, match="non-empty mono"):
        transcribe_samples(np.zeros(0, dtype=np.float32), FakeEngine())
    with pytest.raises(PipelineError, match="non-empty mono"):
        transcribe_samples(np.zeros((2, 10), dtype=np.float32), FakeEngine())


class StopsAtFirstPause(FakeEngine):
    """Mimics Whisper ending a window after the first utterance: covers only the first 2 s."""

    def transcribe(self, samples: np.ndarray, language: str) -> list[RawSegment]:
        """Return one segment for the first 2 s of whatever it is given."""
        self.calls.append((samples.size, language))
        n = len(self.calls)
        return [RawSegment(0.0, 2.0, f"part{n}", (RawWord(0.0, 2.0, f"part{n}", 0.9),))]


def test_speech_after_an_early_stop_is_transcribed_from_the_next_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Spans of 2 s, 3 s and 2 s (compact joins at 2 s and 5 s); one chunk of 7 s.
    patch_vad(monkeypatch, [(1, 3), (10, 13), (20, 22)])
    engine = StopsAtFirstPause()
    t = transcribe_samples(np.zeros(25 * SR, dtype=np.float32), engine)
    assert [n for n, _ in engine.calls] == [7 * SR, 5 * SR, 2 * SR]
    assert [(s.start, s.text) for s in t.segments] == [
        (1.0, "part1"),
        (10.0, "part2"),
        (20.0, "part3"),
    ]
    assert t.stats.engine_calls == 3
    assert t.stats.chunks == 1


def test_a_full_pass_is_not_repeated(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_vad(monkeypatch, [(1, 3), (10, 13)])
    engine = FakeEngine()
    t = transcribe_samples(np.zeros(15 * SR, dtype=np.float32), engine)
    assert len(engine.calls) == 1
    assert t.stats.engine_calls == 1


@pytest.mark.parametrize(
    ("covered", "start", "end", "expected"),
    [
        (2 * SR, 0, 10 * SR, 2 * SR),  # stopped exactly at a join
        (2 * SR + 100, 0, 10 * SR, 2 * SR),  # just past a join, within 0.3 s
        (int(2.5 * SR), 0, 10 * SR, 5 * SR),  # 0.5 s past a join: that join is skipped
        (int(1.5 * SR), 0, 10 * SR, 2 * SR),  # stopped inside a span: resume at its end
        (3 * SR, 0, 10 * SR, 5 * SR),  # inside a span: resume at the next join
        (6 * SR, 0, 10 * SR, None),  # past the last join
        (0, 0, 10 * SR, None),  # nothing transcribed: stop
        (2 * SR, 2 * SR, 10 * SR, None),  # no progress: stop
        (5 * SR, 2 * SR, int(5.5 * SR), None),  # less than 1 s left
    ],
)
def test_resume_point(covered: int, start: int, end: int, expected: int | None) -> None:
    assert _resume_point([0, 2 * SR, 5 * SR], covered, start, end) == expected
