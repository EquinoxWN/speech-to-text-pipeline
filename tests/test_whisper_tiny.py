"""End to end with the real Whisper tiny model and Silero VAD on the FLEURS sample clips (CPU)."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from speech_to_text_pipeline.audio import SAMPLE_RATE, load_audio, write_wav
from speech_to_text_pipeline.cli import main
from speech_to_text_pipeline.engine import WhisperEngine
from speech_to_text_pipeline.metrics import wer
from speech_to_text_pipeline.pipeline import Options, transcribe_file, transcribe_samples
from speech_to_text_pipeline.samples import CLIPS, Clip, clip_path
from speech_to_text_pipeline.vad import detect_speech

pytestmark = pytest.mark.model
SR = SAMPLE_RATE
EN = CLIPS[0]


def silence(seconds: float) -> np.ndarray:
    """Digital silence."""
    return np.zeros(int(seconds * SR), dtype=np.float32)


@pytest.mark.parametrize("clip", CLIPS, ids=[c.clip_id for c in CLIPS])
def test_language_is_detected(clip_path_dir: Path, whisper: WhisperEngine, clip: Clip) -> None:
    samples = load_audio(clip_path(clip, clip_path_dir)).samples
    guess = whisper.detect_language(samples)
    assert guess.language == clip.language
    assert guess.probability > 0.9
    assert guess.top[0][0] == clip.language


def test_vad_finds_the_speech_and_drops_added_silence(clip_path_dir: Path) -> None:
    speech = load_audio(clip_path(EN, clip_path_dir)).samples
    padded = np.concatenate([silence(4.0), speech, silence(3.0), speech, silence(4.0)])
    spans = detect_speech(padded)
    assert spans
    assert spans[0].start_s >= 4.0 - 0.25
    assert spans[-1].end_s <= padded.size / SR - 4.0 + 0.25
    kept = sum(s.length for s in spans) / SR
    assert kept <= 2 * speech.size / SR - 2.0
    assert kept >= 2 * 5.0
    # The 3 s gap between the two copies is removed: no span covers its middle.
    middle = (4.0 + speech.size / SR + 1.5) * SR
    assert not any(s.start <= middle < s.end for s in spans)


def test_word_times_shift_by_exactly_the_added_silence(
    clip_path_dir: Path, whisper: WhisperEngine
) -> None:
    speech = load_audio(clip_path(EN, clip_path_dir)).samples
    plain = transcribe_samples(speech, whisper, Options(language="en"))
    shifted = transcribe_samples(
        np.concatenate([silence(5.0), speech]), whisper, Options(language="en")
    )
    a = [w for s in plain.segments for w in s.words]
    b = [w for s in shifted.segments for w in s.words]
    assert [w.text for w in a] == [w.text for w in b]
    assert len(a) >= 15
    for x, y in zip(a, b, strict=True):
        assert y.start - x.start == pytest.approx(5.0, abs=0.05)
    assert all(w.start <= w.end for w in b)
    assert all(p.start <= q.start for p, q in itertools.pairwise(b))


def test_english_clip_is_transcribed_closely(clip_path_dir: Path, whisper: WhisperEngine) -> None:
    t = transcribe_samples(load_audio(clip_path(EN, clip_path_dir)).samples, whisper)
    assert t.language is not None
    assert t.language.language == "en"
    assert wer(EN.reference, t.text) <= 0.15


def test_a_44k_stereo_file_goes_through_the_whole_pipeline(
    clip_path_dir: Path, whisper: WhisperEngine, tmp_path: Path
) -> None:
    speech = load_audio(clip_path(EN, clip_path_dir)).samples
    signal = np.concatenate([silence(2.0), speech, silence(2.0)])
    t44 = np.arange(int(signal.size * 44_100 / SR)) / 44_100
    up = np.interp(t44, np.arange(signal.size) / SR, signal)
    path = tmp_path / "stereo44k.wav"
    write_wav(path, np.stack([up, up], axis=1).reshape(-1), 44_100, channels=2)
    t = transcribe_file(path, whisper)
    assert t.language is not None
    assert t.language.language == "en"
    assert t.stats.removed_seconds >= 4.0 - 0.5
    words = [w for s in t.segments for w in s.words]
    assert words[0].start >= 2.0
    assert words[-1].end <= t.stats.audio_seconds - 2.0 + 0.3
    assert wer(EN.reference, t.text) <= 0.15


def test_run_samples_command(
    clip_path_dir: Path, whisper: WhisperEngine, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "r.json"
    code = main(
        ["run-samples", "--dest", str(clip_path_dir), "--json", str(report)], engine=whisper
    )
    assert code == 0
    assert "language detected correctly: 8/8" in capsys.readouterr().out
    assert report.is_file()


@pytest.fixture
def clip_path_dir(clips_dir: Path) -> Path:
    """Alias for the verified clip folder."""
    return clips_dir


def test_two_utterances_joined_after_vad_are_both_transcribed(
    clip_path_dir: Path, whisper: WhisperEngine
) -> None:
    first = load_audio(clip_path(CLIPS[1], clip_path_dir)).samples
    second = load_audio(clip_path(EN, clip_path_dir)).samples
    signal = np.concatenate([silence(3.0), first, silence(4.0), second, silence(2.0)])
    t = transcribe_samples(signal, whisper, Options(language="en"))
    assert t.stats.engine_calls >= 2
    assert wer(CLIPS[1].reference + " " + EN.reference, t.text) <= 0.25
    # "slow" occurs once, in the second utterance, which starts after the 4 s gap.
    starts = [w.start for s in t.segments for w in s.words if w.text.strip(",.").lower() == "slow"]
    gap_end = 3.0 + first.size / SR + 4.0
    assert len(starts) == 1
    assert gap_end <= starts[0] <= gap_end + 3.0
