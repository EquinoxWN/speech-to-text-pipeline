"""Packing speech spans into Whisper-sized chunks."""

from __future__ import annotations

import itertools
import random

import pytest

from speech_to_text_pipeline.audio import SAMPLE_RATE
from speech_to_text_pipeline.chunking import Chunk, plan_chunks
from speech_to_text_pipeline.errors import PipelineError

SR = SAMPLE_RATE


def test_short_spans_share_one_chunk() -> None:
    assert plan_chunks([5 * SR, 10 * SR, 15 * SR]) == [Chunk(0, 30 * SR)]


def test_a_span_that_does_not_fit_starts_a_new_chunk_at_the_join() -> None:
    assert plan_chunks([10 * SR, 25 * SR]) == [Chunk(0, 10 * SR), Chunk(10 * SR, 35 * SR)]


def test_a_long_span_is_cut_at_the_limit() -> None:
    assert plan_chunks([70 * SR]) == [
        Chunk(0, 30 * SR),
        Chunk(30 * SR, 60 * SR),
        Chunk(60 * SR, 70 * SR),
    ]


def test_a_long_span_after_a_short_one() -> None:
    assert plan_chunks([5 * SR, 40 * SR], max_seconds=20) == [
        Chunk(0, 5 * SR),
        Chunk(5 * SR, 25 * SR),
        Chunk(25 * SR, 45 * SR),
    ]


def test_no_spans_no_chunks() -> None:
    assert plan_chunks([]) == []


def test_random_layouts_cover_everything_without_gaps_and_respect_the_limit() -> None:
    rng = random.Random(7)
    for _ in range(300):
        lengths = [rng.randint(1, 50 * SR) for _ in range(rng.randint(1, 12))]
        limit = rng.choice([5.0, 12.5, 30.0])
        chunks = plan_chunks(lengths, limit)
        assert chunks[0].start == 0
        assert chunks[-1].end == sum(lengths)
        assert all(a.end == b.start for a, b in itertools.pairwise(chunks))
        assert all(0 < c.end - c.start <= int(limit * SR) for c in chunks)
        # Every cut is either at a span join or inside a span longer than the limit.
        joins = {sum(lengths[: i + 1]) for i in range(len(lengths))}
        long_spans = any(n > int(limit * SR) for n in lengths)
        assert long_spans or all(c.end in joins for c in chunks)


@pytest.mark.parametrize(
    ("lengths", "limit"), [([SR], 0.0), ([SR], -1.0), ([0], 30.0), ([-5], 30.0)]
)
def test_bad_inputs_are_refused(lengths: list[int], limit: float) -> None:
    with pytest.raises(PipelineError):
        plan_chunks(lengths, limit)
