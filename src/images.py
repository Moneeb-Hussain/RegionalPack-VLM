"""Validate images and download the public smoke-test photograph."""

from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps, UnidentifiedImageError

from src.errors import ImageValidationError

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
MIN_SIDE_PX = 16


def pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    payload = f"{rgb.width}x{rgb.height}|".encode("ascii") + rgb.tobytes()
    return hashlib.sha256(payload).hexdigest()


def load_rgb_image(path: Path) -> Image.Image:
    """Open an image file and return an RGB copy the model can consume."""
    if not path.is_file():
        raise ImageValidationError(f"Image not found: {path}")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ImageValidationError(
            f"Unsupported image type {path.suffix!r}. Use JPG, PNG, WEBP, or BMP."
        )
    try:
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened)
            width, height = image.size
            if width < MIN_SIDE_PX or height < MIN_SIDE_PX:
                raise ImageValidationError(
                    f"{path.name} is {width}x{height}. Both sides must be at least {MIN_SIDE_PX}px."
                )
            return image.convert("RGB")
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(
            f"Could not read {path.name}. The file may be corrupted. Original error: {exc}"
        ) from exc


def image_from_bytes(payload: bytes, name: str) -> Image.Image:
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise ImageValidationError(f"{name} exceeds the {MAX_DOWNLOAD_BYTES} byte download limit.")
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened)
            width, height = image.size
            if width < MIN_SIDE_PX or height < MIN_SIDE_PX:
                raise ImageValidationError(
                    f"{name} is {width}x{height}. Both sides must be at least {MIN_SIDE_PX}px."
                )
            return image.convert("RGB")
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(
            f"Downloaded bytes for {name} are not a readable image. Original error: {exc}"
        ) from exc


def _default_opener(url: str, timeout: int) -> urllib.request.addinfourl:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "RegionalPack-VLM-phase0"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def fetch_first_available_image(
    urls: list[str],
    dest: Path,
    *,
    opener: Callable[[str, int], object] | None = None,
    timeout: int = 60,
) -> tuple[Path, str]:
    """Download the first URL that returns a readable image.

    The caller passes the returned file through ``load_rgb_image``. A failed URL
    is recorded and the next URL is tried. This does not substitute a different model.
    """
    if not urls:
        raise ImageValidationError("No public image URLs were configured.")
    opener = opener or _default_opener
    errors: list[str] = []
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        scheme = urllib.parse.urlparse(url).scheme
        if scheme not in {"http", "https"}:
            errors.append(f"{url}: only http and https URLs are allowed")
            continue
        try:
            with opener(url, timeout) as response:
                payload = response.read(MAX_DOWNLOAD_BYTES + 1)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{url}: {exc}")
            continue
        try:
            image = image_from_bytes(payload, url)
        except ImageValidationError as exc:
            errors.append(f"{url}: {exc}")
            continue
        image.save(dest, format="JPEG", quality=95)
        return dest, url
    joined = "; ".join(errors)
    raise ImageValidationError(
        "Could not download a public smoke-test image. "
        f"Tried {len(urls)} URL(s). Errors: {joined}"
    )


def label_for_filename(name: str, labels: dict[str, str] | None = None) -> tuple[str, str]:
    """Return ``(label, source)`` for an uploaded file.

    An explicit ``labels`` entry wins. Otherwise a ``urdu_`` or ``latin_``
    filename prefix is treated as the user's label. Anything else is unlabeled.
    """
    labels = labels or {}
    if name in labels:
        label = labels[name]
        if label not in {"urdu_script", "latin_script"}:
            raise ImageValidationError(
                f"Label for {name} must be 'urdu_script' or 'latin_script', got {label!r}."
            )
        return label, "manual"
    stem = Path(name).name.lower()
    if stem.startswith("urdu_"):
        return "urdu_script", "filename_prefix"
    if stem.startswith("latin_"):
        return "latin_script", "filename_prefix"
    return "unlabeled", "missing"
