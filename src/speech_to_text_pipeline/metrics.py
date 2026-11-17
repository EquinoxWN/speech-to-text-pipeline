"""Word and character error rates with a small, documented text normalization."""

from __future__ import annotations

import re
import unicodedata

from speech_to_text_pipeline.errors import PipelineError

# Languages written without spaces between words: scored by characters (CER), not words.
CHARACTER_LANGUAGES = frozenset({"zh", "ja", "th", "lo", "my", "km", "yue"})


def normalize(text: str) -> str:
    """NFKC, lower case, punctuation and symbols to spaces, runs of spaces collapsed."""
    text = unicodedata.normalize("NFKC", text).lower()
    kept = [" " if unicodedata.category(ch)[0] in {"P", "S", "Z", "C"} else ch for ch in text]
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance between two token lists (substitutions, insertions, deletions)."""
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
        previous = current
    return previous[-1]


def error_rate(ref_tokens: list[str], hyp_tokens: list[str]) -> float:
    """Edit distance over reference length."""
    if not ref_tokens:
        raise PipelineError("reference is empty after normalization")
    return edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate after normalization."""
    return error_rate(normalize(reference).split(), normalize(hypothesis).split())


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate after normalization, spaces ignored."""
    return error_rate(
        list(normalize(reference).replace(" ", "")), list(normalize(hypothesis).replace(" ", ""))
    )


def score(language: str, reference: str, hypothesis: str) -> tuple[str, float]:
    """CER for languages without word spaces, WER otherwise; returns (metric name, value)."""
    if language in CHARACTER_LANGUAGES:
        return "CER", cer(reference, hypothesis)
    return "WER", wer(reference, hypothesis)
