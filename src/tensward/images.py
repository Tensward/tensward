"""What the engine's image processor will accept, checked from the headers alone, so that a bad
image is refused at ``init`` and never sent. No imaging library is needed: only the format and
the displayed size are read, and nothing is decoded."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import struct
from collections import OrderedDict
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from .artifacts import open_regular, read_chunks
from .errors import PROJECT_INPUTS_INVALID, PreflightError
from .files import load_strict_json, write_json

if TYPE_CHECKING:
    from .inputs import PromptEntry

MAX_IMAGE_BYTES = 20 * 2**20
MAX_IMAGE_PIXELS = 40_000_000
MAX_WORKLOAD_IMAGE_BYTES = 2**30
MAX_IMAGES_PER_PROMPT = 16

_MALFORMED = "the image is truncated or malformed"
_ANIMATED = "the image is animated; the engine would use one frame, so give a still image"
# JPEG frame headers: every start-of-frame marker but the DHT, JPG and DAC ones in between.
_JPEG_FRAMES = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
_JPEG_SCAN, _JPEG_END = 0xDA, 0xD9
_EXIF_ORIENTATION = 0x0112


@dataclass(frozen=True, slots=True)
class ImageInfo:
    """An image's MIME type and its size as displayed (after any EXIF rotation)."""

    mime: str
    width: int
    height: int


def probe_image(data: bytes) -> ImageInfo:
    """The format and displayed size of a PNG, JPEG or WebP; ``ValueError`` says why not."""
    try:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            mime, (width, height) = "image/png", _png_size(data)
        elif data.startswith(b"\xff\xd8\xff"):
            mime, (width, height) = "image/jpeg", _jpeg_size(data)
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            mime, (width, height) = "image/webp", _webp_size(data)
        else:
            raise ValueError("the image is not a PNG, JPEG or WebP file")
    except (struct.error, IndexError):
        raise ValueError(_MALFORMED) from None
    if not width or not height:
        raise ValueError(_MALFORMED)
    if width * height > MAX_IMAGE_PIXELS:
        megapixels = width * height / 1e6
        raise ValueError(
            f"the image is {megapixels:.0f} megapixels; the limit is {MAX_IMAGE_PIXELS // 10**6}"
        )
    return ImageInfo(mime, width, height)


def _png_size(data: bytes) -> tuple[int, int]:
    """The IHDR size; an ``acTL`` chunk before the first IDAT makes it an animation."""
    size = None
    position = 8
    while True:
        length, kind = struct.unpack_from(">I4s", data, position)
        if kind == b"IDAT":
            break
        if kind == b"acTL":
            raise ValueError(_ANIMATED)
        if size is None:
            if kind != b"IHDR":
                raise ValueError(_MALFORMED)
            size = struct.unpack_from(">II", data, position + 8)
        position += 12 + length
    if size is None:
        raise ValueError(_MALFORMED)
    return size


def _jpeg_size(data: bytes) -> tuple[int, int]:
    """The frame header's size, swapped when the EXIF orientation turns the image by 90 degrees."""
    size, orientation = None, 1
    position = 2
    while True:
        if data[position] != 0xFF:
            raise ValueError(_MALFORMED)
        marker = data[position + 1]
        if marker == 0xFF:  # fill byte
            position += 1
            continue
        if marker in (_JPEG_SCAN, _JPEG_END):
            break
        (length,) = struct.unpack_from(">H", data, position + 2)
        segment = data[position + 4 : position + 2 + length]
        if marker in _JPEG_FRAMES:
            height, width = struct.unpack_from(">xHH", segment)
            size = (width, height)
        elif marker == 0xE1 and segment.startswith(b"Exif\x00\x00"):
            orientation = _exif_orientation(segment[6:]) or orientation
        position += 2 + length
    if size is None:
        raise ValueError(_MALFORMED)
    return size[::-1] if orientation in (5, 6, 7, 8) else size


def _exif_orientation(tiff: bytes) -> int | None:
    """The orientation tag of the first IFD of an EXIF block, if it has one."""
    if tiff[:2] not in (b"II", b"MM"):
        raise ValueError(_MALFORMED)
    order = "<" if tiff[:2] == b"II" else ">"
    (offset,) = struct.unpack_from(f"{order}I", tiff, 4)
    (entries,) = struct.unpack_from(f"{order}H", tiff, offset)
    for index in range(entries):
        tag, _, _, value = struct.unpack_from(f"{order}HHIH", tiff, offset + 2 + 12 * index)
        if tag == _EXIF_ORIENTATION:
            return int(value)
    return None


def _webp_size(data: bytes) -> tuple[int, int]:
    """The size from the first chunk: a VP8X canvas, or a lossless or lossy bitstream header."""
    kind = data[12:16]
    if kind == b"VP8X":
        if data[20] & 0x02:
            raise ValueError(_ANIMATED)
        canvas = struct.unpack_from("<6B", data, 24)
        return (
            1 + int.from_bytes(bytes(canvas[:3]), "little"),
            1 + int.from_bytes(bytes(canvas[3:]), "little"),
        )
    if kind == b"VP8L":
        signature, bits = struct.unpack_from("<BI", data, 20)
        if signature == 0x2F:
            return (bits & 0x3FFF) + 1, (bits >> 14 & 0x3FFF) + 1
    elif kind == b"VP8 ":
        start, width, height = struct.unpack_from("<3sHH", data, 23)
        if start == b"\x9d\x01\x2a":
            return width & 0x3FFF, height & 0x3FFF
    raise ValueError(_MALFORMED)


@dataclass(frozen=True, slots=True)
class PromptImage:
    """One registered image: its path relative to the prompts file, and what its bytes were."""

    url: str
    sha256: str
    size: int
    mime: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class PromptImages:
    """The workload's images by relative path, and the directory they are relative to."""

    root: Path
    images: Mapping[str, PromptImage]


def load_prompt_images(
    prompts_path: Path,
    entries: Iterable[PromptEntry],
    *,
    cache: Path | None,
    refresh: bool,
) -> PromptImages:
    """Read, check and hash every image the prompts name, all below the prompts file's directory.

    With ``cache`` (a private file), an image whose (inode, size, mtime_ns) is what it was when
    last hashed is not read again, so an image edited in place is noticed unless it kept its
    size and mtime. ``refresh`` reads every image anyway and rewrites the cache.
    """
    root = prompts_path.parent
    known = {} if cache is None or refresh else _read_cache(cache)
    images: dict[str, PromptImage] = {}
    rows: dict[str, list[Any]] = {}
    total = 0
    for url in dict.fromkeys(url for entry in entries for url in entry.image_urls):
        try:
            images[url], rows[url] = _load_image(root, url, known.get(url))
            total += images[url].size
            if total > MAX_WORKLOAD_IMAGE_BYTES:
                raise ValueError(
                    f"the images together exceed {MAX_WORKLOAD_IMAGE_BYTES // 2**30} GiB"
                )
        except (PreflightError, ValueError) as error:
            raise PreflightError(PROJECT_INPUTS_INVALID, f"workload image {url}: {error}") from None
    if cache is not None and rows != known:
        write_json(cache, rows)
    return PromptImages(root, images)


def _load_image(root: Path, url: str, cached: list[Any] | None) -> tuple[PromptImage, list[Any]]:
    """One image and its cache row ``[inode, size, mtime_ns, sha256, mime, width, height]``.
    Each directory above the file may be a link only if it stays inside ``root``; the file
    itself may not be one."""
    path = root / url
    directory = Path(os.path.realpath(path.parent))
    if not directory.is_relative_to(root):
        raise ValueError("it is outside the prompts directory")
    with open_regular(directory / path.name) as (descriptor, size):
        if size > MAX_IMAGE_BYTES:
            raise ValueError(f"it is over the {MAX_IMAGE_BYTES // 2**20} MiB limit")
        info = os.fstat(descriptor)
        stamp = [info.st_ino, size, info.st_mtime_ns]
        if cached and cached[:3] == stamp:
            sha256, mime, width, height = cached[3:]
        else:
            data = b"".join(read_chunks(descriptor, size, url))
            sha256 = hashlib.sha256(data).hexdigest()
            mime, width, height = astuple(probe_image(data))
    row = [*stamp, sha256, mime, width, height]
    return PromptImage(url, sha256, size, mime, width, height), row


class ImageChanged(ValueError):
    """An image's file is no longer the one that was registered."""

    def __init__(self, url: str) -> None:
        super().__init__(f"workload image {url} changed since registration")


class ImageSource:
    """The ``data:`` URLs the requests carry, made when a request needs them. The engine never
    sees a path, and a file that is not byte for byte the registered one is never sent.

    The most recently used URLs are kept up to ``max_cache_bytes`` of text, so a large workload
    does not hold every encoded image at once; a file is read and hashed only on a miss.
    """

    def __init__(self, images: PromptImages, *, max_cache_bytes: int = 256 * 2**20) -> None:
        self._images = images
        self._max_cache_bytes = max_cache_bytes
        self._cache: OrderedDict[str, str] = OrderedDict()
        self.cached_bytes = 0

    async def data_url(self, url: str) -> str:
        if url in self._cache:
            self._cache.move_to_end(url)
            return self._cache[url]
        data_url = await asyncio.to_thread(self._read, url)  # disk reads must not stall streams
        self.cached_bytes -= len(self._cache.pop(url, ""))  # another request may have read it too
        self._cache[url] = data_url
        self.cached_bytes += len(data_url)
        while self.cached_bytes > self._max_cache_bytes:
            self.cached_bytes -= len(self._cache.popitem(last=False)[1])
        return data_url

    def _read(self, url: str) -> str:
        image = self._images.images[url]
        try:
            with open_regular(self._images.root / url) as (descriptor, size):
                if size != image.size:  # before reading: a replacement may be any size
                    raise ImageChanged(url)
                data = b"".join(read_chunks(descriptor, size, url))
        except PreflightError:
            raise ImageChanged(url) from None
        if hashlib.sha256(data).hexdigest() != image.sha256:
            raise ImageChanged(url)
        return f"data:{image.mime};base64,{base64.b64encode(data).decode()}"


def _read_cache(cache: Path) -> dict[str, list[Any]]:
    """The rows saved by the last load, by url; empty when there is no usable cache."""
    try:
        known = load_strict_json(cache.read_bytes(), "the images cache", PROJECT_INPUTS_INVALID)
    except (OSError, PreflightError):
        return {}
    if not isinstance(known, dict):
        return {}
    return {url: row for url, row in known.items() if isinstance(row, list) and len(row) == 7}
