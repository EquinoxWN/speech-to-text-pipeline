"""Command line with a scripted engine: outputs, argument errors and exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from speech_to_text_pipeline import __version__
from speech_to_text_pipeline.audio import SAMPLE_RATE, write_wav
from speech_to_text_pipeline.cli import main
from tests.conftest import FakeEngine


def silent_wav(tmp_path: Path, seconds: float = 2.0) -> Path:
    """A silent 16 kHz file."""
    path = tmp_path / "in.wav"
    write_wav(path, np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)
    return path


def test_transcribe_without_vad_writes_json_and_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audio = silent_wav(tmp_path)
    out_json, out_txt = tmp_path / "t.json", tmp_path / "t.txt"
    code = main(
        ["transcribe", str(audio), "--no-vad", "--json", str(out_json), "--text", str(out_txt)],
        engine=FakeEngine(language="id"),
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "language: id (0.90)" in printed
    assert "[00:00:00.500 - 00:00:02.000] first1 last1" in printed
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["language"]["language"] == "id"
    assert data["segments"][0]["words"][1]["end"] == 2.0
    assert out_txt.read_text(encoding="utf-8").startswith("[00:00:00.500")


def test_transcribe_of_silence_reports_no_speech(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = FakeEngine()
    assert main(["transcribe", str(silent_wav(tmp_path)), "--language", "en"], engine=engine) == 0
    printed = capsys.readouterr().out
    assert "removed 2.00 s" in printed
    assert engine.calls == []


@pytest.mark.parametrize(
    "extra",
    [
        ["--model", "huge"],
        ["--threads", "0"],
        ["--threads", "99"],
        ["--json", "out.txt"],
        ["--text", "out.json"],
    ],
)
def test_bad_arguments_exit_2(tmp_path: Path, extra: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["transcribe", str(silent_wav(tmp_path)), *extra], engine=FakeEngine())
    assert exc.value.code == 2


def test_unknown_language_and_missing_file_exit_2_before_loading_a_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # No engine is passed: if validation came after model loading, this would try to load Whisper.
    assert main(["transcribe", str(silent_wav(tmp_path)), "--language", "klingon"]) == 2
    assert "unknown language code" in capsys.readouterr().err
    assert main(["transcribe", str(tmp_path / "missing.wav")]) == 2
    assert "not a file" in capsys.readouterr().err


def test_unreadable_audio_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"\x00" * 1000)
    assert main(["transcribe", str(bad)], engine=FakeEngine()) == 2
    assert "error:" in capsys.readouterr().err


def test_run_samples_needs_the_clips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run-samples", "--dest", str(tmp_path)], engine=FakeEngine()) == 2
    assert "stt fetch-samples" in capsys.readouterr().err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"stt {__version__}"
