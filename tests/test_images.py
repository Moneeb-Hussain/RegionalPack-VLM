"""Image validation and the public-image downloader. No model weights."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from src.errors import ImageValidationError
from src.images import (
    fetch_first_available_image,
    image_from_bytes,
    label_for_filename,
    load_rgb_image,
    pixel_sha256,
)


def _png_bytes(size: tuple[int, int] = (32, 24), color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _Body:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self, _limit: int = -1) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_load_rgb_image_accepts_a_real_png(tmp_path: Path):
    path = tmp_path / "pack.png"
    path.write_bytes(_png_bytes())
    image = load_rgb_image(path)
    assert image.mode == "RGB"
    assert image.size == (32, 24)
    assert len(pixel_sha256(image)) == 64


def test_load_rgb_image_rejects_corrupt_and_tiny_files(tmp_path: Path):
    corrupt = tmp_path / "bad.png"
    corrupt.write_bytes(b"this is not an image")
    with pytest.raises(ImageValidationError, match="corrupted"):
        load_rgb_image(corrupt)
    tiny = tmp_path / "tiny.png"
    tiny.write_bytes(_png_bytes((8, 8)))
    with pytest.raises(ImageValidationError, match="at least"):
        load_rgb_image(tiny)
    with pytest.raises(ImageValidationError, match="not found"):
        load_rgb_image(tmp_path / "missing.png")


def test_fetch_skips_a_failed_url_and_saves_the_next(tmp_path: Path):
    good = _png_bytes()

    def opener(url: str, _timeout: int):
        if "fail" in url:
            raise OSError("network down")
        return _Body(good)

    dest = tmp_path / "demo.jpg"
    path, used = fetch_first_available_image(
        ["https://example.test/fail.jpg", "https://example.test/ok.jpg"],
        dest,
        opener=opener,
    )
    assert used.endswith("ok.jpg")
    assert path.is_file()
    assert load_rgb_image(path).size == (32, 24)


def test_fetch_reports_every_failed_url(tmp_path: Path):
    def opener(url: str, _timeout: int):
        raise OSError(f"down:{url}")

    with pytest.raises(ImageValidationError, match="fail-a") as raised:
        fetch_first_available_image(
            ["https://example.test/fail-a.jpg", "https://example.test/fail-b.jpg"],
            tmp_path / "demo.jpg",
            opener=opener,
        )
    assert "fail-b" in str(raised.value)


def test_fetch_rejects_non_http_urls(tmp_path: Path):
    with pytest.raises(ImageValidationError, match="http"):
        fetch_first_available_image(["file:///tmp/secret.jpg"], tmp_path / "demo.jpg")


def test_image_from_bytes_rejects_garbage():
    with pytest.raises(ImageValidationError):
        image_from_bytes(b"not-bytes-of-an-image", "demo")


def test_labels_prefer_an_explicit_map_over_the_filename():
    assert label_for_filename("urdu_01.jpg") == ("urdu_script", "filename_prefix")
    assert label_for_filename("latin_02.png") == ("latin_script", "filename_prefix")
    assert label_for_filename("photo.jpg") == ("unlabeled", "missing")
    assert label_for_filename("urdu_01.jpg", {"urdu_01.jpg": "latin_script"}) == (
        "latin_script",
        "manual",
    )
    with pytest.raises(ImageValidationError):
        label_for_filename("photo.jpg", {"photo.jpg": "maybe"})
