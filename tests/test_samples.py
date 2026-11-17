"""Sample clip downloads: pinned checksum, size cap, bounded retries, no partial files."""

from __future__ import annotations

import hashlib
import io
import urllib.error
from pathlib import Path
from typing import BinaryIO

import pytest

from speech_to_text_pipeline import samples
from speech_to_text_pipeline.errors import PipelineError
from speech_to_text_pipeline.samples import CLIPS, Clip, clip_path, fetch_clip

PAYLOAD = b"RIFF" + bytes(range(256)) * 40
CLIP = Clip("t", "en", "en_us/t.wav", hashlib.sha256(PAYLOAD).hexdigest(), len(PAYLOAD), "ref")


class Server:
    """Fake opener: fails a given number of times, then serves the given body."""

    def __init__(self, body: bytes, failures: int = 0, error: Exception | None = None) -> None:
        """Script the responses."""
        self.body = body
        self.failures = failures
        self.error = error or urllib.error.URLError("down")
        self.urls: list[tuple[str, float]] = []

    def __call__(self, url: str, timeout: float) -> BinaryIO:
        """Serve or fail."""
        self.urls.append((url, timeout))
        if len(self.urls) <= self.failures:
            raise self.error
        return io.BytesIO(self.body)


def no_sleep(_: float) -> None:
    """Skip backoff waits in tests."""


def leftovers(dest: Path) -> list[str]:
    """Partial files left behind."""
    return [p.name for p in dest.glob("*.part")]


def test_a_matching_download_is_kept(tmp_path: Path) -> None:
    server = Server(PAYLOAD)
    path = fetch_clip(CLIP, tmp_path, server, no_sleep)
    assert path.read_bytes() == PAYLOAD
    assert server.urls == [(CLIP.url, samples.TIMEOUT_S)]
    assert CLIP.url.startswith(
        "https://huggingface.co/datasets/FluidInference/fleurs-full/resolve/"
    )


def test_a_verified_file_is_not_downloaded_again(tmp_path: Path) -> None:
    clip_path(CLIP, tmp_path).write_bytes(PAYLOAD)
    server = Server(b"")
    fetch_clip(CLIP, tmp_path, server, no_sleep)
    assert server.urls == []


def test_a_corrupt_local_copy_is_replaced(tmp_path: Path) -> None:
    clip_path(CLIP, tmp_path).write_bytes(b"old and wrong")
    fetch_clip(CLIP, tmp_path, Server(PAYLOAD), no_sleep)
    assert clip_path(CLIP, tmp_path).read_bytes() == PAYLOAD


def test_a_checksum_mismatch_is_refused_and_not_retried(tmp_path: Path) -> None:
    tampered = PAYLOAD[:-1] + b"X"
    server = Server(tampered)
    with pytest.raises(PipelineError, match="SHA-256"):
        fetch_clip(CLIP, tmp_path, server, no_sleep)
    assert len(server.urls) == 1
    assert not clip_path(CLIP, tmp_path).exists()
    assert leftovers(tmp_path) == []


def test_oversized_responses_are_cut_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(samples, "MAX_CLIP_BYTES", 100)
    with pytest.raises(PipelineError, match="larger than 100 bytes"):
        fetch_clip(CLIP, tmp_path, Server(PAYLOAD), no_sleep)
    assert leftovers(tmp_path) == []


def test_transient_failures_are_retried_with_backoff(tmp_path: Path) -> None:
    waits: list[float] = []
    server = Server(PAYLOAD, failures=2, error=TimeoutError("slow"))
    fetch_clip(CLIP, tmp_path, server, waits.append)
    assert len(server.urls) == 3
    assert waits == [1.0, 2.0]


def test_retries_are_bounded_and_leave_no_partial_file(tmp_path: Path) -> None:
    waits: list[float] = []
    server = Server(PAYLOAD, failures=99)
    with pytest.raises(PipelineError, match="after 3 attempts"):
        fetch_clip(CLIP, tmp_path, server, waits.append)
    assert len(server.urls) == 3
    assert waits == [1.0, 2.0]
    assert leftovers(tmp_path) == []
    assert not clip_path(CLIP, tmp_path).exists()


def test_plain_http_is_refused() -> None:
    with pytest.raises(PipelineError, match="non-HTTPS"):
        samples._open_url("http://example.com/a.wav", 1.0)


def test_the_manifest_is_pinned_and_covers_four_languages() -> None:
    assert {c.language for c in CLIPS} == {"en", "id", "ja", "zh"}
    assert all(len(c.sha256) == 64 and int(c.sha256, 16) >= 0 for c in CLIPS)
    assert all(0 < c.size <= samples.MAX_CLIP_BYTES and c.reference for c in CLIPS)
    assert len(samples.SOURCE_REVISION) == 40
    assert len({c.clip_id for c in CLIPS}) == len(CLIPS)
