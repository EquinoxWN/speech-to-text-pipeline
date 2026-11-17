"""Decoding and resampling to 16 kHz mono, and refusal of bad files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from speech_to_text_pipeline.audio import SAMPLE_RATE, load_audio, wav_bytes, write_wav
from speech_to_text_pipeline.errors import PipelineError
from tests.conftest import tone


def dominant_frequency(samples: np.ndarray, rate: int) -> float:
    """Frequency of the largest FFT bin."""
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    return float(np.fft.rfftfreq(samples.size, 1 / rate)[int(np.argmax(spectrum))])


def test_stereo_44k_becomes_16k_mono_with_the_same_pitch(tmp_path: Path) -> None:
    left = tone(2.0, 440.0, rate=44_100)
    stereo = np.stack([left, left], axis=1).reshape(-1)
    write_wav(tmp_path / "a.wav", stereo, 44_100, channels=2)
    audio = load_audio(tmp_path / "a.wav")
    assert (audio.source_rate, audio.source_channels) == (44_100, 2)
    assert audio.samples.dtype == np.float32
    assert abs(audio.samples.size - 2 * SAMPLE_RATE) <= 16
    assert audio.duration == pytest.approx(2.0, abs=0.001)
    assert dominant_frequency(audio.samples, SAMPLE_RATE) == pytest.approx(440.0, abs=1.0)


def test_upsampling_8k_keeps_duration_and_pitch(tmp_path: Path) -> None:
    write_wav(tmp_path / "a.wav", tone(1.5, 1000.0, rate=8_000), 8_000)
    audio = load_audio(tmp_path / "a.wav")
    assert abs(audio.samples.size - 24_000) <= 16
    assert dominant_frequency(audio.samples, SAMPLE_RATE) == pytest.approx(1000.0, abs=1.0)


def test_content_above_8k_is_filtered_not_folded(tmp_path: Path) -> None:
    write_wav(tmp_path / "a.wav", tone(1.0, 12_000.0, rate=48_000), 48_000)
    out = load_audio(tmp_path / "a.wav").samples[1000:-1000]
    # A naive decimation would fold 12 kHz to 4 kHz at nearly full level (RMS about 0.35).
    assert float(np.sqrt(np.mean(out**2))) < 0.01


def test_downmix_averages_channels(tmp_path: Path) -> None:
    s = tone(1.0, 300.0, rate=16_000)
    write_wav(tmp_path / "opposite.wav", np.stack([s, -s], axis=1).reshape(-1), 16_000, 2)
    write_wav(tmp_path / "same.wav", np.stack([s, s], axis=1).reshape(-1), 16_000, 2)
    assert float(np.abs(load_audio(tmp_path / "opposite.wav").samples).max()) < 1e-3
    assert float(np.abs(load_audio(tmp_path / "same.wav").samples).max()) > 0.3


def test_16k_mono_passes_through_unchanged(tmp_path: Path) -> None:
    s = tone(0.5, 200.0)
    write_wav(tmp_path / "a.wav", s, SAMPLE_RATE)
    out = load_audio(tmp_path / "a.wav").samples
    assert out.size == s.size
    assert float(np.abs(out - s).max()) < 1e-3


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("empty.wav", b"", "empty file"),
        ("junk.wav", b"this is not audio at all" * 50, "not a readable audio file"),
    ],
)
def test_bad_files_are_refused(tmp_path: Path, name: str, content: bytes, message: str) -> None:
    (tmp_path / name).write_bytes(content)
    with pytest.raises(PipelineError, match=message):
        load_audio(tmp_path / name)


def test_missing_file_and_directory_are_refused(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="not a file"):
        load_audio(tmp_path / "nope.wav")
    with pytest.raises(PipelineError, match="not a file"):
        load_audio(tmp_path)


def test_wav_without_frames_is_refused(tmp_path: Path) -> None:
    (tmp_path / "a.wav").write_bytes(wav_bytes(np.zeros(0, dtype=np.float32), SAMPLE_RATE))
    with pytest.raises(PipelineError, match="no audio samples"):
        load_audio(tmp_path / "a.wav")


def test_size_and_length_caps(tmp_path: Path) -> None:
    write_wav(tmp_path / "a.wav", tone(3.0), SAMPLE_RATE)
    size = (tmp_path / "a.wav").stat().st_size
    with pytest.raises(PipelineError, match="byte limit"):
        load_audio(tmp_path / "a.wav", max_bytes=size - 1)
    assert load_audio(tmp_path / "a.wav", max_bytes=size).samples.size == 3 * SAMPLE_RATE
    with pytest.raises(PipelineError, match="2 s limit"):
        load_audio(tmp_path / "a.wav", max_seconds=2.0)
    assert load_audio(tmp_path / "a.wav", max_seconds=3.0).samples.size == 3 * SAMPLE_RATE


def test_wav_writer_refuses_bad_shapes() -> None:
    with pytest.raises(PipelineError, match="positive"):
        wav_bytes(np.zeros(4), 0)
    with pytest.raises(PipelineError, match="multiple of the channel count"):
        wav_bytes(np.zeros(5), SAMPLE_RATE, channels=2)


def test_wav_writer_clips_out_of_range_samples(tmp_path: Path) -> None:
    write_wav(tmp_path / "a.wav", np.array([2.0, -2.0, 0.5], dtype=np.float32), SAMPLE_RATE)
    out = load_audio(tmp_path / "a.wav").samples
    assert out[:3] == pytest.approx([1.0, -1.0, 0.5], abs=1e-3)
