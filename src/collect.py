"""Import phone photographs into the dataset folders. This module does not score a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.dataset import (
    ANGLES,
    ANNOTATION_COLUMNS,
    MANIFEST_COLUMNS,
    render_validation_report,
    validate_dataset,
)

JPEG_SUFFIXES = {".jpg", ".jpeg"}
RECOGNIZED_SUFFIXES = JPEG_SUFFIXES | {".png", ".webp", ".tif", ".tiff", ".heic", ".heif"}
LOG_COLUMNS = (
    "recorded_at_utc",
    "product_id",
    "angle",
    "dest_path",
    "source_filename",
    "sha256",
    "bytes",
)


def _read_manifest(root: Path) -> list[dict[str, str]]:
    path = root / "data" / "product_manifest.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != list(MANIFEST_COLUMNS):
            raise SystemExit(f"{path} header does not match the manifest schema.")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readable_jpeg(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image_format = image.format
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        return f"unreadable ({exc})"
    if image_format != "JPEG":
        return f"{image_format or 'unknown'} rather than JPEG"
    return None


def prepare_directories(root: Path) -> list[Path]:
    """Create one empty folder per manifest product. Does not create photographs."""
    root = Path(root)
    created: list[Path] = []
    for row in _read_manifest(root):
        folder = root / "data" / "raw" / row["product_id"]
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder / ".gitkeep"
        if not marker.exists():
            marker.write_text("", encoding="utf-8")
        created.append(folder)
    return created


def annotation_template_rows(root: Path) -> list[dict[str, str]]:
    """Identity rows from the manifest. Text columns stay empty."""
    rows: list[dict[str, str]] = []
    for product in _read_manifest(root):
        for angle in ANGLES:
            product_id = product["product_id"]
            rows.append(
                {
                    "image_path": f"data/raw/{product_id}/{angle}.jpg",
                    "product_id": product_id,
                    "category": product["category"],
                    "variant": product["variant"],
                    "brand_true": product["brand"],
                    "brand_group": product["brand_group"],
                    "origin": product["origin"],
                    "script_present": "",
                    "text_true_latin": "",
                    "text_true_urdu": "",
                    "angle": angle,
                    "occlusion_note": "",
                    "brand_visible": "",
                    "ambiguous_text": "",
                    "notes": "",
                    "review_status": "",
                }
            )
    return rows


def write_annotation_template(root: Path) -> Path:
    path = Path(root) / "data" / "annotations_template.csv"
    rows = annotation_template_rows(root)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ANNOTATION_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _ensure_log(root: Path) -> Path:
    path = root / "data" / "image_import_log.csv"
    if not path.is_file():
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=list(LOG_COLUMNS)).writeheader()
    return path


def _append_log(root: Path, row: dict[str, str]) -> None:
    path = _ensure_log(root)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LOG_COLUMNS))
        writer.writerow(row)


def _assignments_from_tree(source: Path, known: set[str]) -> tuple[list[tuple[Path, str, str]], list[str]]:
    messages: list[str] = []
    assignments: list[tuple[Path, str, str]] = []
    if not source.is_dir():
        return [], [f"{source} is not a directory."]
    for product_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        product_id = product_dir.name
        if product_id not in known:
            messages.append(f"Skipped {product_dir.name}: not a manifest product_id.")
            continue
        found: dict[str, Path] = {}
        for path in sorted(product_dir.iterdir()):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            angle = path.stem.lower()
            if suffix not in RECOGNIZED_SUFFIXES or angle not in ANGLES:
                messages.append(f"Skipped {path.name}: name must be straight, tilt, or occluded.")
                continue
            if angle in found:
                messages.append(f"Skipped {path.name}: {product_id} already has a {angle} file.")
                continue
            found[angle] = path
        for angle, path in found.items():
            assignments.append((path, product_id, angle))
    return assignments, messages


def _assignments_from_map(
    source: Path,
    map_path: Path,
    known: set[str],
) -> tuple[list[tuple[Path, str, str]], list[str]]:
    messages: list[str] = []
    assignments: list[tuple[Path, str, str]] = []
    with map_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        if header != ["source_filename", "product_id", "angle"]:
            return [], ["Map header must be source_filename, product_id, angle."]
        for row in reader:
            name = (row.get("source_filename") or "").strip()
            product_id = (row.get("product_id") or "").strip()
            angle = (row.get("angle") or "").strip().lower()
            if not name or Path(name).is_absolute() or ".." in Path(name).parts:
                messages.append(f"Skipped map row with unsafe filename {name!r}.")
                continue
            if product_id not in known:
                messages.append(f"Skipped {name}: {product_id} is not in the manifest.")
                continue
            if angle not in ANGLES:
                messages.append(f"Skipped {name}: angle must be straight, tilt, or occluded.")
                continue
            assignments.append((source / name, product_id, angle))
    return assignments, messages


def import_photographs(
    root: Path,
    source: Path,
    *,
    map_path: Path | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> list[str]:
    """Copy JPEG bytes into data/raw. Does not resize, crop, or re-encode."""
    root = Path(root)
    source = Path(source)
    raw = (root / "data" / "raw").resolve()
    if source.resolve() == raw or raw in source.resolve().parents:
        return ["Refusing to import from inside data/raw."]
    known = {row["product_id"] for row in _read_manifest(root)}
    if map_path is None:
        assignments, messages = _assignments_from_tree(source, known)
    else:
        assignments, messages = _assignments_from_map(source, map_path, known)
    for source_path, product_id, angle in assignments:
        dest = root / "data" / "raw" / product_id / f"{angle}.jpg"
        label = f"{product_id}/{angle}.jpg"
        if not source_path.is_file():
            messages.append(f"Missing source file for {label}: {source_path.name}")
            continue
        suffix = source_path.suffix.lower()
        if suffix not in JPEG_SUFFIXES:
            messages.append(
                f"{source_path.name} is {suffix or 'without a suffix'}. "
                "Export a camera JPEG. This tool does not convert or re-encode."
            )
            continue
        problem = _readable_jpeg(source_path)
        if problem is not None:
            messages.append(f"{source_path.name} was not copied: {problem}.")
            continue
        digest = _sha256(source_path)
        if dest.exists() and not overwrite:
            messages.append(
                f"Warning: {dest.relative_to(root).as_posix()} already exists. "
                "Skipped. Re-run with --overwrite to replace it."
            )
            continue
        action = "would copy" if dry_run else "copied"
        if dest.exists() and overwrite:
            action = "would replace" if dry_run else "replaced"
            messages.append(f"Warning: replacing {dest.relative_to(root).as_posix()}.")
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest)
            if _sha256(dest) != digest:
                messages.append(f"{label} hash changed during copy. Left the file in place.")
                continue
            _append_log(
                root,
                {
                    "recorded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "product_id": product_id,
                    "angle": angle,
                    "dest_path": dest.relative_to(root).as_posix(),
                    "source_filename": source_path.name,
                    "sha256": digest,
                    "bytes": str(dest.stat().st_size),
                },
            )
        messages.append(f"{action} {source_path.name} -> {label} sha256={digest}")
    if not assignments and not messages:
        messages.append("No photograph assignments were found.")
    return messages


def validation_report(root: Path, *, require_complete: bool = False) -> str:
    result = validate_dataset(root, require_complete=require_complete)
    report = render_validation_report(result)
    provenance = Path(root) / "data" / "image_provenance.csv"
    if not provenance.is_file():
        return report
    rows = list(csv.DictReader(provenance.open(encoding="utf-8")))
    readable = sum(row.get("readable") == "yes" for row in rows)
    rotated = sum(row.get("exif_orientation") not in ("", "1") for row in rows)
    extra = [
        "",
        "## Image provenance",
        "",
        f"Authoritative source: raw.zip. Rows: {len(rows)}. Readable: {readable}.",
        "JPEG bytes were copied. They were not re-encoded.",
        f"Files with a non-upright EXIF orientation: {rotated}. The loader rotates those in memory only.",
        "juice_intl and juice_local in the zip are stored as apple_fruita and apple_fruitien.",
        "",
    ]
    return report + "\n".join(extra)


def _command_prepare(root: Path) -> int:
    folders = prepare_directories(root)
    _ensure_log(root)
    print(f"Prepared {len(folders)} product folders under data/raw/. No photographs were created.")
    return 0


def _command_template(root: Path) -> int:
    path = write_annotation_template(root)
    print(f"Wrote {path.relative_to(root).as_posix()} with empty transcription fields.")
    print("data/annotations.csv was not changed.")
    return 0


def _command_import(args: argparse.Namespace, root: Path) -> int:
    messages = import_photographs(
        root,
        Path(args.source),
        map_path=Path(args.map) if args.map else None,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    for message in messages:
        print(message)
    failed = ("Skipped", "Missing source", "Refusing", "was not copied", "hash changed", "already exists")
    return 1 if any(any(marker in message for marker in failed) for message in messages) else 0


def _command_validate(args: argparse.Namespace, root: Path) -> int:
    text = validation_report(root, require_complete=args.complete)
    report = Path(args.report)
    if not report.is_absolute():
        report = root / report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {report.relative_to(root).as_posix()}")
    return 0 if text.startswith("# Dataset validation") and "\nOK: yes\n" in text else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and check the RegionalPack photograph set.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Create empty data/raw/<product_id>/ folders from the manifest.")
    sub.add_parser("prepare-annotations", help="Write data/annotations_template.csv. Does not invent text.")
    import_parser = sub.add_parser("import", help="Copy camera JPEGs into the product folders.")
    import_parser.add_argument("--source", required=True, help="Folder of phone photographs.")
    import_parser.add_argument("--map", help="CSV with source_filename, product_id, angle.")
    import_parser.add_argument("--dry-run", action="store_true", help="Print the copies without writing.")
    import_parser.add_argument("--overwrite", action="store_true", help="Replace an existing angle after a warning.")
    validate_parser = sub.add_parser("validate", help="Print a collection report. Does not run the model.")
    validate_parser.add_argument(
        "--complete",
        action="store_true",
        help="Treat every missing photograph as an error.",
    )
    validate_parser.add_argument("--report", default="data/validation_report.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if args.command == "prepare":
        return _command_prepare(root)
    if args.command == "prepare-annotations":
        return _command_template(root)
    if args.command == "import":
        return _command_import(args, root)
    if args.command == "validate":
        return _command_validate(args, root)
    parser.error(f"Unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
