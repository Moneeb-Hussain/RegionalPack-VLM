"""Copy the approved JPEG bytes from raw.zip into data/raw.

The zip still uses juice_intl and juice_local for the apple-drink photographs.
Those bytes are stored under the approved ids apple_fruita and apple_fruitien.
Files are not re-encoded. EXIF is left in the file.
"""

from __future__ import annotations

import csv
import hashlib
import io
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.dataset import ANGLES

# Folder names inside raw.zip that differ from the approved manifest ids.
ZIP_FOLDER_TO_PRODUCT = {
    "juice_intl": "apple_fruita",
    "juice_local": "apple_fruitien",
}

PROVENANCE_COLUMNS = (
    "product_id",
    "angle",
    "dest_path",
    "zip_member",
    "sha256",
    "bytes",
    "readable",
    "exif_orientation",
    "prior_sha256",
    "action",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _exif_orientation(payload: bytes) -> str:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            value = image.getexif().get(274)
            return "" if value is None else str(value)
    except (UnidentifiedImageError, OSError, ValueError):
        return ""


def _readable(payload: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def benchmark_members(zip_path: Path) -> dict[tuple[str, str], str]:
    """Map (product_id, angle) to the zip member path. Extras are omitted."""
    found: dict[tuple[str, str], str] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if name.startswith("__MACOSX") or name.endswith("/") or "/._" in name:
                continue
            parts = Path(name).parts
            if len(parts) != 3 or parts[0] != "raw":
                continue
            folder, filename = parts[1], parts[2]
            if filename not in {f"{angle}.jpg" for angle in ANGLES}:
                continue
            product_id = ZIP_FOLDER_TO_PRODUCT.get(folder, folder)
            angle = filename.replace(".jpg", "")
            found[(product_id, angle)] = name
    return found


def integrate_raw_zip(root: Path, zip_path: Path) -> list[dict[str, str]]:
    """Byte-copy the 60 benchmark JPEGs. Existing different files are backed up."""
    root = Path(root)
    members = benchmark_members(zip_path)
    backup_root = root / "data" / "raw_backup" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for (product_id, angle), member in sorted(members.items()):
            payload = archive.read(member)
            dest = root / "data" / "raw" / product_id / f"{angle}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            digest = _sha256(payload)
            prior = ""
            action = "copied"
            if dest.is_file():
                current = dest.read_bytes()
                prior = _sha256(current)
                if current == payload:
                    action = "unchanged"
                else:
                    backup = backup_root / product_id / dest.name
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dest, backup)
                    dest.write_bytes(payload)
                    action = "replaced"
            else:
                dest.write_bytes(payload)
            rows.append(
                {
                    "product_id": product_id,
                    "angle": angle,
                    "dest_path": dest.relative_to(root).as_posix(),
                    "zip_member": member,
                    "sha256": digest,
                    "bytes": str(len(payload)),
                    "readable": "yes" if _readable(payload) else "no",
                    "exif_orientation": _exif_orientation(payload),
                    "prior_sha256": prior,
                    "action": action,
                }
            )
    path = root / "data" / "image_provenance.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PROVENANCE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return rows
