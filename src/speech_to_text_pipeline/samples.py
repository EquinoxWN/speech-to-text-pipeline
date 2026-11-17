"""Public sample clips (FLEURS test split, CC BY 4.0): pinned URLs, SHA-256 checked downloads."""

from __future__ import annotations

import hashlib
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from speech_to_text_pipeline.errors import PipelineError

# FLEURS (Conneau et al., 2022, arXiv:2205.12446), test split, CC BY 4.0, as single 16 kHz mono WAV
# files in the Hugging Face dataset FluidInference/fleurs-full, pinned to one commit. The SHA-256
# values are the Git LFS object ids the Hub publishes for each file (checked 2026-11-17).
SOURCE_REPO = "FluidInference/fleurs-full"
SOURCE_REVISION = "1cca811bb8ea4d370345f108f00518167040282c"
BASE_URL = f"https://huggingface.co/datasets/{SOURCE_REPO}/resolve/{SOURCE_REVISION}"
MAX_CLIP_BYTES = 2 * 1024 * 1024
TIMEOUT_S = 60.0
ATTEMPTS = 3


@dataclass(frozen=True)
class Clip:
    """One sample clip: where it comes from, its checksum and its reference transcript."""

    clip_id: str
    language: str
    path: str
    sha256: str
    size: int
    reference: str

    @property
    def url(self) -> str:
        """Pinned download URL."""
        return f"{BASE_URL}/{self.path}"


CLIPS: tuple[Clip, ...] = (
    Clip("en_us_0000", "en", "en_us/en_us_0000.wav",
         "7835bd6ffb54ce38a2a9bcde3905ba424faed94d50a474f21a9cbe9209b869df", 337964,
         "however due to the slow communication channels styles in the west could lag behind by 25 to 30 year"),
    Clip("en_us_0001", "en", "en_us/en_us_0001.wav",
         "ffd32e9953ffb14bed7f4e3a2d037ea7ac7de1ba5ad25d4c0f678fd06b24aab0", 280364,
         "all nouns alongside the word sie for you always begin with a capital letter even in the middle of a sentence"),
    Clip("id_id_0000", "id", "id_id/id_id_0000.wav",
         "483dea3628adc905d86bfed5dbf983d80bccdbee0fe63afb371190748dc24401", 274604,
         "tim-tim virtual memiliki standar keunggulan yang sama dengan tim konvensional tetapi ada sedikit perbedaan"),
    Clip("id_id_0001", "id", "id_id/id_id_0001.wav",
         "be5c98250e75cbf7439450a7b6a3dbdc088f3b9310ea793fb26443035245e175", 245804,
         "segera setelah permusuhan pecah britania memulai blokade laut jerman"),
    Clip("ja_jp_0000", "ja", "ja_jp/ja_jp_0000.wav",
         "26b6cd491b5d52cf05d89207cb88499cdaba89749aebc29c8f216e834e3d9071", 334124,
         "インターネットで 敵対的環境コース について検索すると おそらく現地企業の住所が出てくるでしょう"),
    Clip("ja_jp_0001", "ja", "ja_jp/ja_jp_0001.wav",
         "258fe8b9954d74c44bcfeb93317fff647f0a0f9135c472f4af32f6db5b02281e", 455084,
         "また 北側に行くなら世界的に有名なマリア像の聖地であるファティマの聖母の聖域神社を訪れましょう"),
    Clip("cmn_hans_cn_0000", "zh", "cmn_hans_cn/cmn_hans_cn_0000.wav",
         "5214e16584b6498ddbd22f321738c0f62b19117ab1880b21e405fba983986b6a", 332204,
         "这 并 不 是 告 别 这 是 一 个 篇 章 的 结 束 也 是 新 篇 章 的 开 始"),
    Clip("cmn_hans_cn_0001", "zh", "cmn_hans_cn/cmn_hans_cn_0001.wav",
         "d2ff70aa3a9b965bd2cab664852f2b5eb90d02dca4ed9066686d926152ade514", 265004,
         "钙 钾 等 元 素 属 于 金 属 银 和 金 等 元 素 当 然 也 是 金 属"),
)  # fmt: skip

Opener = Callable[[str, float], BinaryIO]


def _open_url(url: str, timeout: float) -> BinaryIO:
    """HTTPS GET with a timeout."""
    if not url.startswith("https://"):
        raise PipelineError(f"refusing non-HTTPS URL {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "speech-to-text-pipeline"})  # noqa: S310
    response: BinaryIO = urllib.request.urlopen(request, timeout=timeout)  # noqa: S310
    return response


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def clip_path(clip: Clip, dest: Path) -> Path:
    """Where a clip is stored under the destination folder."""
    return dest / f"{clip.clip_id}.wav"


def fetch_clip(
    clip: Clip,
    dest: Path,
    opener: Opener = _open_url,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download one clip unless a verified copy exists; bounded retries, no partial files left."""
    target = clip_path(clip, dest)
    if target.is_file() and sha256_of(target) == clip.sha256:
        return target
    if not clip.url.startswith("https://"):
        raise PipelineError(f"refusing non-HTTPS URL {clip.url}")
    dest.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.{os.getpid()}.part")
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            _download(clip, partial, opener)
            os.replace(partial, target)
            return target
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt + 1 < ATTEMPTS:
                sleep(2.0**attempt)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    raise PipelineError(f"{clip.clip_id}: download failed after {ATTEMPTS} attempts: {last_error}")


def _download(clip: Clip, partial: Path, opener: Opener) -> None:
    """Stream to the partial file with a size cap, then check size and SHA-256."""
    digest = hashlib.sha256()
    written = 0
    with opener(clip.url, TIMEOUT_S) as response, partial.open("wb") as out:
        while block := response.read(1 << 16):
            written += len(block)
            if written > MAX_CLIP_BYTES:
                raise PipelineError(f"{clip.clip_id}: larger than {MAX_CLIP_BYTES} bytes")
            digest.update(block)
            out.write(block)
    if written != clip.size or digest.hexdigest() != clip.sha256:
        raise PipelineError(f"{clip.clip_id}: size or SHA-256 does not match the pinned value")


def fetch_all(dest: Path, opener: Opener = _open_url) -> list[Path]:
    """Download every sample clip (about 2.5 MB in total)."""
    return [fetch_clip(clip, dest, opener) for clip in CLIPS]
