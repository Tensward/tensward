"""Tiny image byte builders for the tests: just enough of each format to be read by its header.

Only ``png`` is a complete file (the example images are made with it); the JPEG and WebP
builders stop after the header, which is all ``probe_image`` reads.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

from tensward.images import PromptImage, PromptImages


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload))
    )


def png(width: int, height: int, *, animated: bool = False) -> bytes:
    """A decodable black RGB PNG; ``animated`` adds the ``acTL`` chunk that makes it an APNG."""
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    compressor, row = zlib.compressobj(), b"\x00" + bytes(3 * width)
    pixels = b"".join(compressor.compress(row) for _ in range(height)) + compressor.flush()
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", header),
            _png_chunk(b"acTL", struct.pack(">II", 2, 0)) if animated else b"",
            _png_chunk(b"IDAT", pixels),
            _png_chunk(b"IEND", b""),
        ]
    )


def _exif_orientation(orientation: int) -> bytes:
    """An APP1 segment whose little-endian TIFF IFD holds only the orientation tag."""
    entry = struct.pack("<HHIHH", 0x0112, 3, 1, orientation, 0)
    tiff = b"II*\x00" + struct.pack("<IH", 8, 1) + entry + struct.pack("<I", 0)
    payload = b"Exif\x00\x00" + tiff
    return b"\xff\xe1" + struct.pack(">H", 2 + len(payload)) + payload


def jpeg(
    width: int,
    height: int,
    *,
    progressive: bool = False,
    components: int = 3,
    orientation: int | None = None,
) -> bytes:
    """A JPEG that stops after its frame header."""
    frame = struct.pack(">BHHB", 8, height, width, components)
    frame += b"".join(bytes((index + 1, 0x11, 0)) for index in range(components))
    marker = b"\xff\xc2" if progressive else b"\xff\xc0"
    exif = b"" if orientation is None else _exif_orientation(orientation)
    return b"\xff\xd8" + exif + marker + struct.pack(">H", 2 + len(frame)) + frame + b"\xff\xd9"


def webp(width: int, height: int, *, kind: str = "VP8X", animated: bool = False) -> bytes:
    """A WebP RIFF container holding only the header of a ``VP8X``, ``VP8L`` or ``VP8`` chunk."""
    if kind == "VP8X":
        canvas = (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
        payload = bytes((0x02 if animated else 0, 0, 0, 0)) + canvas
    elif kind == "VP8L":
        payload = b"\x2f" + struct.pack("<I", (width - 1) | (height - 1) << 14)
    else:
        payload = b"\x10\x00\x00\x9d\x01\x2a" + struct.pack("<HH", width, height)
    chunk = kind.ljust(4).encode() + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def write_images(root: Path, *, count: int, size: int) -> PromptImages:
    """``count`` files of ``size`` bytes each below ``root``, registered as they are now. They
    are not decodable: only what reads the bytes (not the header) can use them."""
    images = {}
    for index in range(count):
        data = bytes([index]) * size
        (root / f"{index}.png").write_bytes(data)
        images[f"{index}.png"] = PromptImage(
            f"{index}.png", hashlib.sha256(data).hexdigest(), size, "image/png", 1, 1
        )
    return PromptImages(root, images)
