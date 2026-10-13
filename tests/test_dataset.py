"""CPU checks for the Phase 1A dataset plan. No evaluation images are stored here."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml
from PIL import Image

from src.dataset import (
    ANGLES,
    ANNOTATION_COLUMNS,
    MANIFEST_COLUMNS,
    SEEN_COLUMNS,
    validate_dataset,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _manifest_row(product_id: str, category: str, brand: str, group: str, origin: str) -> dict[str, str]:
    return {
        "product_id": product_id,
        "category": category,
        "brand": brand,
        "product_line": brand,
        "variant": "test variant",
        "package_type": "test pack",
        "brand_group": group,
        "origin": origin,
        "membership_basis": "test",
        "group_rule": {
            "international": "sold_under_same_name_outside_pakistan",
            "exploratory": "excluded_from_primary",
        }.get(group, "pakistan_market_line"),
        "availability_status": "pending_shelf_check",
        "shelf_status": "pending_approval",
        "selection_basis": "not from phase 0",
        "source_note": "test",
        "script_note": "",
        "comparison_role": "exploratory" if group == "exploratory" else "primary",
        "limitation": "outside the primary comparison" if group == "exploratory" else "",
        "photo_location": "",
    }


def _annotation(product_id: str, angle: str, **overrides: str) -> dict[str, str]:
    row = {
        "image_path": f"data/raw/{product_id}/{angle}.jpg",
        "product_id": product_id,
        "category": "cola",
        "variant": "test variant",
        "brand_true": "Example",
        "brand_group": "international",
        "origin": "international",
        "script_present": "latin",
        "text_true_latin": "Example Cola",
        "text_true_urdu": "",
        "angle": angle,
        "occlusion_note": "",
        "brand_visible": "",
        "ambiguous_text": "",
        "notes": "",
    }
    row.update(overrides)
    return row


def _scaffold(tmp_path: Path, manifest: list[dict[str, str]] | None = None) -> Path:
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    (data / "raw" / ".gitkeep").write_text("", encoding="utf-8")
    if manifest is None:
        manifest = [
            _manifest_row("cola_intl", "cola", "Coca-Cola", "international", "international"),
            _manifest_row("cola_local", "cola", "Next Cola", "pakistan_origin", "local"),
        ]
    _write_csv(data / "product_manifest.csv", MANIFEST_COLUMNS, manifest)
    _write_csv(data / "annotations.csv", ANNOTATION_COLUMNS, [])
    _write_csv(data / "phase0_seen_products.csv", SEEN_COLUMNS, [])
    alias_dir = tmp_path / "configs"
    alias_dir.mkdir(parents=True, exist_ok=True)
    products = {
        row["product_id"]: {"brand": row["brand"], "aliases": [row["brand"]]}
        for row in manifest
    }
    (alias_dir / "brand_aliases.yaml").write_text(
        yaml.safe_dump({"products": products}, sort_keys=False),
        encoding="utf-8",
    )
    (alias_dir / "collection_config.yaml").write_text(
        (ROOT / "configs" / "collection_config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return tmp_path


def _save_jpg(path: Path, color: tuple[int, int, int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if color is None:
        shift = sum(path.as_posix().encode()) % 200
        color = (20 + shift, 40, 60)
    Image.new("RGB", (8, 8), color).save(path, format="JPEG")


def test_checklist_matches_the_manifest():
    from src.checklist import render_checklist

    text = (ROOT / "data" / "availability_checklist.md").read_text(encoding="utf-8")
    assert text == render_checklist(ROOT)
    for product_id in (
        "cola_intl",
        "biscuit_local",
        "chocolate_intl",
        "chocolate_local",
        "soap_local",
    ):
        assert product_id in text
    assert "milk_intl" not in text
    assert "in_hand" in text
    assert "Cadbury Dairy Milk" in text
    assert "CandyLand YOU" in text
    assert "rich chocolate biscuits with vanilla cream" in text
    assert "Yellow Label, loose black tea" in text
    assert "chicken instant noodles" in text
    assert "Super Crisp BBQ" in text
    assert "exploratory" in text
    assert "juice_intl" not in text


def test_repository_photographs_and_verified_text_are_complete():
    result = validate_dataset(ROOT)
    assert result["errors"] == []
    assert result["ok"] is True
    assert result["status"] == "collected"
    assert result["present_count"] == 60
    assert result["annotation_gaps"] == []


def test_complete_product_with_three_angles_passes(tmp_path: Path):
    root = _scaffold(tmp_path)
    rows = [_annotation("cola_intl", angle) for angle in ANGLES]
    _write_csv(root / "data" / "annotations.csv", ANNOTATION_COLUMNS, rows)
    for angle in ANGLES:
        _save_jpg(root / "data" / "raw" / "cola_intl" / f"{angle}.jpg")
    result = validate_dataset(root)
    assert result["ok"] is True
    assert result["status"] == "collected"


def test_missing_tilt_file_is_reported(tmp_path: Path):
    root = _scaffold(tmp_path)
    rows = [_annotation("cola_intl", angle) for angle in ANGLES]
    _write_csv(root / "data" / "annotations.csv", ANNOTATION_COLUMNS, rows)
    _save_jpg(root / "data" / "raw" / "cola_intl" / "straight.jpg")
    _save_jpg(root / "data" / "raw" / "cola_intl" / "occluded.jpg")
    result = validate_dataset(root)
    assert result["ok"] is False
    assert any("Missing image file" in error and "tilt.jpg" in error for error in result["errors"])


def test_duplicate_annotation_path_is_reported(tmp_path: Path):
    root = _scaffold(tmp_path)
    rows = [_annotation("cola_intl", "straight"), _annotation("cola_intl", "straight")]
    _write_csv(root / "data" / "annotations.csv", ANNOTATION_COLUMNS, rows)
    result = validate_dataset(root)
    assert any("Duplicate annotation image_path" in error for error in result["errors"])


def test_visible_text_may_differ_when_a_corner_is_covered(tmp_path: Path):
    root = _scaffold(tmp_path)
    rows = [_annotation("cola_intl", angle) for angle in ANGLES]
    rows[2]["text_true_latin"] = "Example"
    rows[2]["occlusion_note"] = "Lower-left tagline covered by a hand."
    _write_csv(root / "data" / "annotations.csv", ANNOTATION_COLUMNS, rows)
    for angle in ANGLES:
        _save_jpg(root / "data" / "raw" / "cola_intl" / f"{angle}.jpg")
    result = validate_dataset(root)
    assert result["ok"] is True
    assert result["errors"] == []


def test_inconsistent_brand_is_reported(tmp_path: Path):
    root = _scaffold(tmp_path)
    rows = [_annotation("cola_intl", angle) for angle in ANGLES]
    rows[2]["brand_true"] = "Other brand"
    _write_csv(root / "data" / "annotations.csv", ANNOTATION_COLUMNS, rows)
    for angle in ANGLES:
        _save_jpg(root / "data" / "raw" / "cola_intl" / f"{angle}.jpg")
    result = validate_dataset(root)
    assert any("inconsistent brand_true" in error for error in result["errors"])


def test_brand_group_and_origin_must_agree(tmp_path: Path):
    root = _scaffold(tmp_path)
    row = _annotation("cola_intl", "straight", brand_group="pakistan_origin", origin="international")
    _write_csv(root / "data" / "annotations.csv", ANNOTATION_COLUMNS, [row])
    result = validate_dataset(root)
    assert any("disagree" in error for error in result["errors"])


def test_extra_image_and_wrong_count_are_reported(tmp_path: Path):
    root = _scaffold(tmp_path)
    _save_jpg(root / "data" / "raw" / "cola_intl" / "straight.jpg")
    _save_jpg(root / "data" / "raw" / "cola_intl" / "extra.jpg")
    result = validate_dataset(root)
    assert any("no annotation row" in error for error in result["errors"])
    assert any("Unexpected evaluation image name" in error for error in result["errors"])


def test_phase0_path_is_rejected(tmp_path: Path):
    root = _scaffold(tmp_path)
    row = _annotation("cola_intl", "straight")
    row["image_path"] = "results/uploads/urdu_01.jpg"
    _write_csv(root / "data" / "annotations.csv", ANNOTATION_COLUMNS, [row])
    result = validate_dataset(root)
    assert any("Phase 0 path" in error for error in result["errors"])


def test_phase0_identity_must_not_choose_the_manifest(tmp_path: Path):
    root = _scaffold(tmp_path)
    _write_csv(
        root / "data" / "phase0_seen_products.csv",
        SEEN_COLUMNS,
        [
            {
                "tested_product": "Example",
                "filename": "not recorded",
                "on_manifest": "yes",
                "manifest_product_id": "cola_intl",
                "used_to_choose_manifest": "yes",
                "overlap_note": "test",
            }
        ],
    )
    result = validate_dataset(root)
    assert any("used to choose the manifest" in error for error in result["errors"])


def test_group_rule_cannot_assign_both_groups(tmp_path: Path):
    root = _scaffold(tmp_path)
    manifest_path = root / "data" / "product_manifest.csv"
    text = manifest_path.read_text(encoding="utf-8")
    text = text.replace(
        "sold_under_same_name_outside_pakistan",
        "pakistan_market_line",
        1,
    )
    manifest_path.write_text(text, encoding="utf-8")
    result = validate_dataset(root)
    assert any("one group only" in error for error in result["errors"])


def test_shared_brand_alias_is_rejected(tmp_path: Path):
    root = _scaffold(tmp_path)
    alias_path = root / "configs" / "brand_aliases.yaml"
    loaded = yaml.safe_load(alias_path.read_text(encoding="utf-8"))
    loaded["products"]["cola_local"]["aliases"].append("Coca-Cola")
    alias_path.write_text(yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")
    result = validate_dataset(root)
    assert any("both cola_intl and cola_local" in error or "both cola_local and cola_intl" in error for error in result["errors"])


def test_repository_aliases_and_phase0_filenames():
    result = validate_dataset(ROOT)
    assert result["errors"] == []
    assert result["status"] == "collected"
    aliases = yaml.safe_load((ROOT / "configs" / "brand_aliases.yaml").read_text(encoding="utf-8"))
    rite = aliases["products"]["biscuit_local"]["aliases"]
    cadbury = aliases["products"]["chocolate_intl"]["aliases"]
    candyland = aliases["products"]["chocolate_local"]["aliases"]
    assert "Bisconni" not in rite
    assert "Cadbury Dairy Milk" in cadbury
    assert "Mondelēz" not in cadbury
    assert "YOU" in candyland
    assert "CandyLand YOU" in candyland
    assert "CandyLand" not in candyland
    assert "Shan" not in aliases["products"]["noodles_local"]["aliases"]
    assert "Nestlé" not in aliases["products"]["apple_fruita"]["aliases"]
    seen = (ROOT / "data" / "phase0_seen_products.csv").read_text(encoding="utf-8")
    for name in ("latin_01.jpg", "latin_02.jpg", "latin_03.jpg", "urdu_01.jpg", "urdu_02.jpg", "urdu_03.jpg"):
        assert name in seen
    assert "not recorded" not in seen
    manifest = (ROOT / "data" / "product_manifest.csv").read_text(encoding="utf-8")
    assert "rich chocolate biscuits with vanilla cream" in manifest
    assert "chocolate_intl" in manifest
    assert "milk_intl" not in manifest
    assert "juice_intl" not in manifest
    assert "Knorr" in manifest
    assert "BBQ" in manifest
    assert "exploratory" in manifest
    assert "Fruita Vitals" in manifest


def test_unknown_product_and_manifest_mismatch_are_reported(tmp_path: Path):
    root = _scaffold(tmp_path)
    row = _annotation("cola_intl", "straight", category="tea", brand_group="pakistan_origin", origin="local")
    _write_csv(root / "data" / "annotations.csv", ANNOTATION_COLUMNS, [row])
    (root / "data" / "raw" / "not_a_product").mkdir()
    result = validate_dataset(root)
    assert any("not a manifest product_id" in error for error in result["errors"])
    assert any("category is 'tea'" in error for error in result["errors"])
    assert any("brand_group does not match the manifest" in error for error in result["errors"])
    assert any("data/raw/not_a_product" in error for error in result["errors"])


def test_duplicate_and_unreadable_images_are_reported(tmp_path: Path):
    root = _scaffold(tmp_path)
    rows = [_annotation("cola_intl", angle) for angle in ANGLES]
    _write_csv(root / "data" / "annotations.csv", ANNOTATION_COLUMNS, rows)
    _save_jpg(root / "data" / "raw" / "cola_intl" / "straight.jpg", (1, 2, 3))
    _save_jpg(root / "data" / "raw" / "cola_intl" / "tilt.jpg", (1, 2, 3))
    (root / "data" / "raw" / "cola_intl" / "occluded.jpg").write_bytes(b"not-a-jpeg")
    result = validate_dataset(root)
    assert any("Duplicate image hash" in error for error in result["errors"])
    assert any("Unreadable image file" in error for error in result["errors"])


def test_complete_flag_lists_missing_photographs(tmp_path: Path):
    root = _scaffold(tmp_path)
    result = validate_dataset(root, require_complete=True)
    assert result["ok"] is False
    assert any("Missing photograph: data/raw/cola_intl/straight.jpg" in error for error in result["errors"])


def test_exploratory_pair_is_outside_the_primary_comparison(tmp_path: Path):
    root = _scaffold(
        tmp_path,
        [
            _manifest_row("apple_fruita", "apple", "Fruita Vitals", "exploratory", "exploratory"),
            _manifest_row("apple_fruitien", "apple", "Fruitien", "exploratory", "exploratory"),
        ],
    )
    result = validate_dataset(root)
    assert result["ok"] is True
    assert result["errors"] == []


def test_a_pair_must_share_one_photo_location(tmp_path: Path):
    root = _scaffold(tmp_path)
    path = root / "data" / "product_manifest.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["photo_location"] = "kitchen"
    rows[1]["photo_location"] = "bathroom"
    _write_csv(path, MANIFEST_COLUMNS, rows)
    result = validate_dataset(root)
    assert any("more than one photo location" in error for error in result["errors"])


def test_category_must_pair_both_brand_groups(tmp_path: Path):
    root = _scaffold(
        tmp_path,
        [_manifest_row("cola_intl", "cola", "Coca-Cola", "international", "international")],
    )
    result = validate_dataset(root)
    assert any("one international brand and one pakistan_origin brand" in error for error in result["errors"])
