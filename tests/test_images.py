"""Image header checks: what the engine would accept, read without decoding."""

import asyncio
from pathlib import Path

import pytest
from image_fixtures import jpeg, png, webp, write_images

from tensward.images import ImageInfo, ImageSource, probe_image


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (png(640, 480), ImageInfo("image/png", 640, 480)),
        (jpeg(1536, 768), ImageInfo("image/jpeg", 1536, 768)),
        (jpeg(1536, 768, orientation=6), ImageInfo("image/jpeg", 768, 1536)),  # rotated
        (jpeg(800, 600, progressive=True), ImageInfo("image/jpeg", 800, 600)),
        (jpeg(800, 600, components=4), ImageInfo("image/jpeg", 800, 600)),  # CMYK
        (webp(768, 1536), ImageInfo("image/webp", 768, 1536)),
        (webp(300, 200, kind="VP8L"), ImageInfo("image/webp", 300, 200)),
        (webp(300, 200, kind="VP8"), ImageInfo("image/webp", 300, 200)),
    ],
)
def test_supported_images_report_their_displayed_size(data: bytes, expected: ImageInfo) -> None:
    assert probe_image(data) == expected


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (png(10, 10, animated=True), "animated"),
        (webp(10, 10, animated=True), "animated"),
        (png(8000, 6000), "megapixels"),
        (b"GIF89a" + bytes(20), "PNG, JPEG or WebP"),
        (png(10, 10)[:20], "truncated"),
        (b"\x89PNG\r\n\x1a\n" + bytes(30), "malformed"),
    ],
)
def test_unsupported_images_say_why(data: bytes, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        probe_image(data)


def test_the_image_cache_is_bounded(tmp_path: Path) -> None:
    images = write_images(tmp_path, count=5, size=600_000)
    source = ImageSource(images, max_cache_bytes=1_000_000)
    for url in images.images:
        asyncio.run(source.data_url(url))
    assert source.cached_bytes <= 1_000_000
