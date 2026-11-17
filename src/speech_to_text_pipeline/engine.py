"""Speech recognition engines: the faster-whisper one and the interface the pipeline needs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from speech_to_text_pipeline.audio import Samples
from speech_to_text_pipeline.errors import PipelineError
from speech_to_text_pipeline.transcript import LanguageGuess

# Model sizes this project allows, with the Hugging Face repository and pinned revision of each.
# Only "tiny" has been run locally; the others are listed so the CLI refuses unknown names early.
MODELS: dict[str, tuple[str, str]] = {
    "tiny": ("Systran/faster-whisper-tiny", "d90ca5fe260221311c53c58e660288d3deb8d356"),
    "base": ("Systran/faster-whisper-base", "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"),
    "small": ("Systran/faster-whisper-small", "536b0662742c02347bc0e980a01041f333bce120"),
}


@dataclass(frozen=True)
class RawWord:
    """A word as the engine returns it, timed within the chunk it was given."""

    start: float
    end: float
    text: str
    probability: float


@dataclass(frozen=True)
class RawSegment:
    """A segment as the engine returns it, timed within its chunk."""

    start: float
    end: float
    text: str
    words: tuple[RawWord, ...]


class Engine(Protocol):
    """What the pipeline needs from a recognizer."""

    name: str

    def detect_language(self, samples: Samples) -> LanguageGuess:
        """Language of up to the first 30 s of the samples."""
        ...

    def transcribe(self, samples: Samples, language: str) -> list[RawSegment]:
        """Segments with word timestamps for one chunk of at most 30 s."""
        ...


def download_model(size: str) -> str:
    """Fetch the pinned model files into the Hugging Face cache (HF_HOME) and return the folder."""
    if size not in MODELS:
        raise PipelineError(f"unknown model size {size!r}; choose from {sorted(MODELS)}")
    from faster_whisper.utils import download_model as fetch

    repo, revision = MODELS[size]
    try:
        return str(fetch(repo, revision=revision))
    except (OSError, ValueError) as exc:
        raise PipelineError(f"could not download {repo}: {exc}") from exc


class WhisperEngine:
    """faster-whisper (CTranslate2) on CPU with int8 weights."""

    def __init__(self, size: str = "tiny", threads: int = 2, beam_size: int = 5) -> None:
        """Load (and on first use download into HF_HOME) the pinned model."""
        if size not in MODELS:
            raise PipelineError(f"unknown model size {size!r}; choose from {sorted(MODELS)}")
        if threads < 1 or beam_size < 1:
            raise PipelineError("threads and beam size must be at least 1")
        from faster_whisper import WhisperModel  # imported late: loading CTranslate2 is slow

        repo, revision = MODELS[size]
        self.name = f"{repo}@{revision[:7]}"
        self.beam_size = beam_size
        try:
            self.model: Any = WhisperModel(
                repo,
                device="cpu",
                compute_type="int8",
                cpu_threads=threads,
                revision=revision,
                local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise PipelineError(f"could not load model {repo}: {exc}") from exc

    def detect_language(self, samples: Samples) -> LanguageGuess:
        """Whisper's language-token probabilities on the first 30 s."""
        language, probability, ranked = self.model.detect_language(audio=samples)
        top = tuple((str(code), float(p)) for code, p in ranked[:5])
        return LanguageGuess(str(language), float(probability), top)

    def transcribe(self, samples: Samples, language: str) -> list[RawSegment]:
        """Beam search, word timestamps on, no internal VAD (already done)."""
        segments, _info = self.model.transcribe(
            samples,
            language=language,
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        out: list[RawSegment] = []
        for seg in segments:
            words = tuple(
                RawWord(float(w.start), float(w.end), str(w.word).strip(), float(w.probability))
                for w in (seg.words or [])
            )
            out.append(RawSegment(float(seg.start), float(seg.end), str(seg.text).strip(), words))
        return out
