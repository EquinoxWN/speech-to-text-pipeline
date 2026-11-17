"""Command line: stt transcribe | fetch-samples | fetch-model | run-samples."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from speech_to_text_pipeline import __version__
from speech_to_text_pipeline.audio import SAMPLE_RATE, load_audio
from speech_to_text_pipeline.engine import MODELS, Engine, WhisperEngine, download_model
from speech_to_text_pipeline.errors import PipelineError
from speech_to_text_pipeline.files import write_text
from speech_to_text_pipeline.metrics import score
from speech_to_text_pipeline.pipeline import Options, transcribe_file, transcribe_samples
from speech_to_text_pipeline.samples import CLIPS, clip_path, fetch_all

DEFAULT_SAMPLES = Path(".tmp/samples")
PAD_SECONDS = 2.0


def _threads(value: str) -> int:
    """argparse type: thread count 1 to 32."""
    n = int(value)
    if not 1 <= n <= 32:
        raise argparse.ArgumentTypeError("threads must be between 1 and 32")
    return n


def _output(value: str, suffix: str) -> Path:
    """argparse helper: an output path with the expected suffix."""
    path = Path(value)
    if path.suffix != suffix:
        raise argparse.ArgumentTypeError(f"output file must end in {suffix}")
    return path


def build_parser() -> argparse.ArgumentParser:
    """All subcommands and their options."""
    parser = argparse.ArgumentParser(prog="stt", description=__doc__)
    parser.add_argument("--version", action="version", version=f"stt {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="transcribe one audio file")
    t.add_argument("audio", type=Path)
    t.add_argument("--model", choices=sorted(MODELS), default="tiny")
    t.add_argument("--language", default=None, help="Whisper language code; detected if omitted")
    t.add_argument("--no-vad", action="store_true", help="keep silence (no voice detection)")
    t.add_argument("--threads", type=_threads, default=2)
    t.add_argument("--json", type=lambda v: _output(v, ".json"), default=None)
    t.add_argument("--text", type=lambda v: _output(v, ".txt"), default=None)

    f = sub.add_parser("fetch-samples", help="download the 8 FLEURS sample clips (SHA-256 checked)")
    f.add_argument("--dest", type=Path, default=DEFAULT_SAMPLES)

    m = sub.add_parser("fetch-model", help="download a pinned Whisper model into HF_HOME")
    m.add_argument("--model", choices=sorted(MODELS), default="tiny")

    r = sub.add_parser("run-samples", help="language, VAD and error rate on the sample clips")
    r.add_argument("--dest", type=Path, default=DEFAULT_SAMPLES)
    r.add_argument("--model", choices=sorted(MODELS), default="tiny")
    r.add_argument("--threads", type=_threads, default=2)
    r.add_argument("--json", type=lambda v: _output(v, ".json"), default=None)
    return parser


def main(argv: Sequence[str] | None = None, engine: Engine | None = None) -> int:
    """Entry point; returns the process exit code (2 for bad input)."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "fetch-samples":
            for path in fetch_all(args.dest):
                print(f"ok {path}")
            return 0
        if args.command == "fetch-model":
            print(f"ok {download_model(args.model)}")
            return 0
        if args.command == "transcribe":
            options = Options(language=args.language, vad=not args.no_vad)
            if not args.audio.is_file():
                raise PipelineError(f"{args.audio}: not a file")
            return _transcribe(
                args, engine or WhisperEngine(args.model, threads=args.threads), options
            )
        return _run_samples(args, engine or WhisperEngine(args.model, threads=args.threads))
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _transcribe(args: argparse.Namespace, engine: Engine, options: Options) -> int:
    """Transcribe one file and print or write the result."""
    transcript = transcribe_file(args.audio, engine, options)
    lang = transcript.language
    s = transcript.stats
    detected = f"{lang.language} ({lang.probability:.2f})" if lang else "none (no speech)"
    print(f"language: {detected}{' forced' if transcript.language_forced else ''}")
    print(
        f"audio {s.audio_seconds:.2f} s, speech kept {s.speech_seconds:.2f} s, "
        f"removed {s.removed_seconds:.2f} s, {s.chunks} chunk(s), {s.engine_calls} engine call(s), "
        f"RTF {s.real_time_factor:.3f}"
    )
    print(transcript.to_text(), end="")
    if args.json:
        write_text(args.json, transcript.to_json())
    if args.text:
        write_text(args.text, transcript.to_text())
    return 0


def _run_samples(args: argparse.Namespace, engine: Engine) -> int:
    """Each clip padded with 2 s of silence on both sides, language detected, error rate scored."""
    missing = [c.clip_id for c in CLIPS if not clip_path(c, args.dest).is_file()]
    if missing:
        raise PipelineError(f"missing clips {missing}; run `stt fetch-samples` first")
    pad = np.zeros(int(PAD_SECONDS * SAMPLE_RATE), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    print("| clip | lang | detected (p) | audio s | removed s | first word s | metric | RTF |")
    print("|---|---|---|---|---|---|---|---|")
    for clip in CLIPS:
        audio = load_audio(clip_path(clip, args.dest))
        padded = np.concatenate([pad, audio.samples, pad])
        t = transcribe_samples(padded, engine, Options(), source=clip.clip_id)
        lang = t.language
        words = [w for s in t.segments for w in s.words]
        metric, value = score(clip.language, clip.reference, t.text)
        row = {
            "clip": clip.clip_id,
            "language": clip.language,
            "detected": lang.language if lang else None,
            "probability": round(lang.probability, 3) if lang else None,
            "audio_seconds": round(t.stats.audio_seconds, 2),
            "removed_seconds": round(t.stats.removed_seconds, 2),
            "first_word_seconds": words[0].start if words else None,
            "metric": metric,
            "value": round(value, 3),
            "real_time_factor": round(t.stats.real_time_factor, 3),
            "text": t.text,
        }
        rows.append(row)
        print(
            f"| {clip.clip_id} | {clip.language} | {row['detected']} ({row['probability']}) "
            f"| {row['audio_seconds']} | {row['removed_seconds']} | {row['first_word_seconds']} "
            f"| {metric} {row['value']:.3f} | {row['real_time_factor']:.3f} |"
        )
    correct = sum(r["detected"] == r["language"] for r in rows)
    print(f"language detected correctly: {correct}/{len(rows)} (model {engine.name}, CPU)")
    if args.json:
        report = {
            "created_utc": datetime.now(UTC).isoformat(),
            "model": engine.name,
            "pad_seconds": PAD_SECONDS,
            "rows": rows,
        }
        write_text(args.json, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
