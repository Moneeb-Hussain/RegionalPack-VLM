"""CPU checks for photograph import. Fixtures stay in temporary directories."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from src.collect import import_photographs, prepare_directories, write_annotation_template
from src.dataset import validate_dataset
from test_dataset import _scaffold, _save_jpg

ROOT = Path(__file__).resolve().parents[1]


def _jpeg(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), color).save(path, format="JPEG")


def test_prepare_creates_folders_without_photographs(tmp_path: Path):
    root = _scaffold(tmp_path)
    folders = prepare_directories(root)
    assert [path.name for path in folders] == ["cola_intl", "cola_local"]
    assert list((root / "data" / "raw").glob("*/*.jpg")) == []


def test_template_leaves_transcription_blank(tmp_path: Path):
    root = _scaffold(tmp_path)
    path = write_annotation_template(root)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 6
    assert all(row["text_true_latin"] == "" and row["text_true_urdu"] == "" for row in rows)
    assert all(row["occlusion_note"] == "" and row["script_present"] == "" for row in rows)
    assert rows[0]["brand_true"] == "Coca-Cola"
    assert rows[0]["variant"] == "test variant"
    assert rows[0]["brand_group"] == "international"


def test_import_copies_jpeg_bytes_and_records_the_source_name(tmp_path: Path):
    root = _scaffold(tmp_path)
    source = tmp_path / "inbox" / "cola_intl"
    original = source / "straight.jpg"
    _jpeg(original, (8, 9, 10))
    messages = import_photographs(root, tmp_path / "inbox")
    dest = root / "data" / "raw" / "cola_intl" / "straight.jpg"
    assert dest.read_bytes() == original.read_bytes()
    assert any("straight.jpg -> cola_intl/straight.jpg" in message for message in messages)
    log = list(csv.DictReader((root / "data" / "image_import_log.csv").open(encoding="utf-8")))
    assert log[0]["source_filename"] == "straight.jpg"
    assert log[0]["sha256"]
    assert log[0]["dest_path"] == "data/raw/cola_intl/straight.jpg"


def test_dry_run_and_png_do_not_write(tmp_path: Path):
    root = _scaffold(tmp_path)
    inbox = tmp_path / "inbox" / "cola_intl"
    _jpeg(inbox / "tilt.jpeg", (2, 3, 4))
    png = inbox / "occluded.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (6, 6), (5, 5, 5)).save(png, format="PNG")
    messages = import_photographs(root, tmp_path / "inbox", dry_run=True)
    assert not (root / "data" / "raw" / "cola_intl" / "tilt.jpg").exists()
    assert not (root / "data" / "image_import_log.csv").exists()
    assert any("would copy" in message for message in messages)
    assert any("does not convert" in message for message in messages)


def test_existing_image_is_kept_until_overwrite(tmp_path: Path):
    root = _scaffold(tmp_path)
    dest = root / "data" / "raw" / "cola_intl" / "straight.jpg"
    _save_jpg(dest, (1, 1, 1))
    original = dest.read_bytes()
    inbox = tmp_path / "inbox" / "cola_intl"
    _jpeg(inbox / "straight.jpg", (9, 9, 9))
    skipped = import_photographs(root, tmp_path / "inbox")
    assert dest.read_bytes() == original
    assert any("already exists" in message for message in skipped)
    replaced = import_photographs(root, tmp_path / "inbox", overwrite=True)
    assert dest.read_bytes() == (inbox / "straight.jpg").read_bytes()
    assert any("replaced" in message for message in replaced)


def test_map_file_places_a_camera_filename(tmp_path: Path):
    root = _scaffold(tmp_path)
    source = tmp_path / "flat"
    _jpeg(source / "IMG_0099.JPG", (4, 5, 6))
    map_path = tmp_path / "map.csv"
    map_path.write_text(
        "source_filename,product_id,angle\nIMG_0099.JPG,cola_local,occluded\n",
        encoding="utf-8",
    )
    import_photographs(root, source, map_path=map_path)
    assert (root / "data" / "raw" / "cola_local" / "occluded.jpg").is_file()
    log = list(csv.DictReader((root / "data" / "image_import_log.csv").open(encoding="utf-8")))
    assert log[0]["source_filename"] == "IMG_0099.JPG"


def test_repository_collection_slots_match_the_manifest():
    result = validate_dataset(ROOT)
    assert result["status"] == "collected"
    assert result["present_count"] == 60
    manifest = list(csv.DictReader((ROOT / "data" / "product_manifest.csv").open(encoding="utf-8")))
    ids = [row["product_id"] for row in manifest]
    assert "chocolate_intl" in ids
    assert "chocolate_local" in ids
    assert "milk_intl" not in ids
    assert len(ids) == 20
    for product_id in ids:
        folder = ROOT / "data" / "raw" / product_id
        assert folder.is_dir()
        assert {path.name for path in folder.glob("*.jpg")} == {"straight.jpg", "tilt.jpg", "occluded.jpg"}
    template = list(csv.DictReader((ROOT / "data" / "annotations_template.csv").open(encoding="utf-8")))
    assert len(template) == 60
    assert all(row["text_true_latin"] == "" and row["text_true_urdu"] == "" for row in template)
    assert {row["product_id"] for row in template} == set(ids)
    live = list(csv.DictReader((ROOT / "data" / "annotations.csv").open(encoding="utf-8")))
    assert len(live) == 60
    assert all(row["review_status"] == "verified" and row["text_true_latin"] for row in live)
