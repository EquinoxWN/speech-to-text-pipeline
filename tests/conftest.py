"""Shared fixtures: a scripted fake engine, and the real Whisper tiny engine when it is available.

Tests marked `model` need the 8 sample clips (`stt fetch-samples`) and the pinned Whisper tiny model
(`stt fetch-model`). Without them they are skipped, unless STT_REQUIRE_MODEL=1 is set (as in
`.tmp/check.sh` and CI), which turns a missing clip or model into a test failure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from speech_to_text_pipeline.audio import SAMPLE_RATE, Samples
from speech_to_text_pipeline.engine import RawSegment, RawWord, WhisperEngine
from speech_to_text_pipeline.errors import PipelineError
from speech_to_text_pipeline.samples import CLIPS, clip_path
from speech_to_text_pipeline.transcript import LanguageGuess

SAMPLES_DIR = Path(__file__).resolve().parents[1] / ".tmp" / "samples"


@dataclass
class FakeEngine:
    """Returns one segment per chunk: a word at 0.5-1.0 s and a word ending at the chunk end."""

    language: str = "en"
    name: str = "fake"
    detected_on: list[int] = field(default_factory=list)
    calls: list[tuple[int, str]] = field(default_factory=list)

    def detect_language(self, samples: Samples) -> LanguageGuess:
        """Record the length it was given and return the configured language."""
        self.detected_on.append(samples.size)
        return LanguageGuess(self.language, 0.9, ((self.language, 0.9), ("xx", 0.1)))

    def transcribe(self, samples: Samples, language: str) -> list[RawSegment]:
        """Record the call and return a scripted segment."""
        self.calls.append((samples.size, language))
        n = len(self.calls)
        length = samples.size / SAMPLE_RATE
        words = (
            RawWord(0.5, 1.0, f"first{n}", 0.9),
            RawWord(length - 0.5, length, f"last{n}", 0.8),
        )
        return [RawSegment(0.5, length, f"first{n} last{n}", words)]


def tone(seconds: float, freq: float = 440.0, rate: int = SAMPLE_RATE, amp: float = 0.5) -> Samples:
    """A sine wave."""
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture(scope="session")
def require_model() -> bool:
    """True when missing clips or model must fail instead of skip."""
    return os.environ.get("STT_REQUIRE_MODEL") == "1"


@pytest.fixture(scope="session")
def clips_dir(require_model: bool) -> Path:
    """Folder holding the 8 verified sample clips."""
    missing = [c.clip_id for c in CLIPS if not clip_path(c, SAMPLES_DIR).is_file()]
    if missing:
        if require_model:
            pytest.fail(f"sample clips missing: {missing}; run `stt fetch-samples`")
        pytest.skip("sample clips not downloaded; run `stt fetch-samples` (see conftest docstring)")
    return SAMPLES_DIR


@pytest.fixture(scope="session")
def whisper(require_model: bool) -> WhisperEngine:
    """The pinned Whisper tiny model on CPU, loaded once, from the local cache only."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return WhisperEngine("tiny", threads=2)
    except PipelineError as exc:
        if require_model:
            pytest.fail(f"Whisper tiny not available: {exc}; run `stt fetch-model`")
        pytest.skip("Whisper tiny not downloaded; run `stt fetch-model` (see conftest docstring)")
