"""Word and character error rates on worked examples."""

from __future__ import annotations

import pytest

from speech_to_text_pipeline.errors import PipelineError
from speech_to_text_pipeline.metrics import cer, edit_distance, normalize, score, wer


def test_edit_distance_counts_each_kind_of_edit() -> None:
    assert edit_distance(list("kitten"), list("sitting")) == 3
    assert edit_distance([], list("abc")) == 3
    assert edit_distance(list("abc"), []) == 3
    assert edit_distance(list("abc"), list("abc")) == 0


def test_wer_worked_example() -> None:
    # One substitution (sat -> sit), one deletion (the), one insertion (down) over 6 words.
    assert wer("the cat sat on the mat", "the cat sit on mat down") == pytest.approx(3 / 6)


def test_wer_ignores_case_and_punctuation() -> None:
    assert wer("Tim-tim virtual, OK!", "tim tim virtual ok") == 0.0


def test_wer_can_exceed_one() -> None:
    assert wer("yes", "no no no") == pytest.approx(3.0)


def test_cer_ignores_spaces_and_counts_characters() -> None:
    assert cer("这 并 不 是", "这并不是") == 0.0
    assert cer("这并不是", "這並不是") == pytest.approx(2 / 4)


def test_normalize_unifies_width_and_punctuation() -> None:
    # Full-width ABC, an ideographic comma and full stop, and curly quotes.
    assert normalize("ＡＢＣ、 def。  “x”") == "abc def x"  # noqa: RUF001


def test_score_picks_cer_for_languages_without_word_spaces() -> None:
    assert score("zh", "金属", "金屬") == ("CER", 0.5)
    assert score("ja", "東京", "東京") == ("CER", 0.0)
    assert score("id", "laut jerman", "laut cerman") == ("WER", 0.5)


def test_an_empty_reference_is_refused() -> None:
    with pytest.raises(PipelineError, match="reference is empty"):
        wer(" ... ", "hello")
    with pytest.raises(PipelineError, match="reference is empty"):
        cer("", "x")
