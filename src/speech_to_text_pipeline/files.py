"""Atomic file writes: a temporary sibling renamed over the target."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically; the old file stays if anything fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text atomically."""
    write_bytes(path, text.encode("utf-8"))
