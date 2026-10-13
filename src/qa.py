"""Phase A dataset and annotation audit. This module does not call a model.

It does not mark a draft as verified. A blank transcription is not treated as
proof that the script is absent.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.artifacts import write_text
from src.dataset import validate_dataset

CASE_NOT_REVIEWED = "not_reviewed"
CASE_LEGIBLE = "legible"
CASE_ABSENT = "absent"
CASE_UNREADABLE = "unreadable_or_ambiguous"


def classify_row(row: dict[str, str]) -> dict[str, object]:
    """One of four script cases. Unverified rows stay not_reviewed."""
    review = (row.get("review_status") or "").strip()
    reviewed = review == "verified"
    latin = (row.get("text_true_latin") or "").strip()
    urdu = (row.get("text_true_urdu") or "").strip()
    script = (row.get("script_present") or "").strip()
    ambiguous = (row.get("ambiguous_text") or "").strip()
    ambiguous_lower = ambiguous.lower()
    signals: list[str] = []
    if not reviewed and not urdu:
        signals.append("urdu_blank_is_not_recorded_as_absent")
    if script in ("mixed", "urdu", "roman_urdu") and not urdu:
        signals.append("script_label_includes_urdu_but_urdu_cell_is_blank")
    if "urdu" in ambiguous_lower:
        signals.append("draft_note_says_urdu_is_present")
        if any(word in ambiguous_lower for word in ("not clear", "unreadable", "not legible")):
            signals.append("draft_note_says_urdu_is_unreadable")
    if ambiguous:
        signals.append("ambiguous_note")
    if not reviewed and latin:
        signals.append("latin_draft_present")
    if not reviewed and not latin:
        signals.append("latin_blank")
    if reviewed:
        latin_case = CASE_LEGIBLE if latin else CASE_ABSENT
        if urdu:
            urdu_case = CASE_LEGIBLE
        elif "draft_note_says_urdu_is_unreadable" in signals:
            urdu_case = CASE_UNREADABLE
        else:
            urdu_case = CASE_ABSENT
    else:
        latin_case = CASE_NOT_REVIEWED
        urdu_case = CASE_NOT_REVIEWED
    return {
        "image_path": row.get("image_path", ""),
        "product_id": row.get("product_id", ""),
        "angle": row.get("angle", ""),
        "review_status": review,
        "latin_case": latin_case,
        "urdu_case": urdu_case,
        "signals": signals,
        "action": _action(row, signals),
    }


def _action(row: dict[str, str], signals: list[str]) -> str:
    path = row.get("image_path", "")
    parts = [f"Open {path} by itself."]
    if (row.get("review_status") or "").strip() != "verified":
        parts.append("Do not mark it verified until you have looked at this photograph.")
    if "latin_draft_present" in signals:
        parts.append("Check the Latin draft against the front of the package in this frame.")
    if "script_label_includes_urdu_but_urdu_cell_is_blank" in signals:
        parts.append(
            "The script label says Urdu is involved and the Urdu cell is blank. "
            "Transcribe legible Urdu, or record that it is present but unreadable. "
            "Do not leave the blank cell as if the script were absent."
        )
    elif "draft_note_says_urdu_is_present" in signals:
        parts.append("The draft note says Urdu is present. The Urdu cell is still blank.")
    if "ambiguous_note" in signals:
        parts.append("Resolve the uncertain-text note: " + (row.get("ambiguous_text") or "").strip())
    if (row.get("occlusion_note") or "").strip():
        parts.append(
            "Confirm the occlusion note, and delete any word that this frame does not actually show."
        )
    return " ".join(parts)


def _read_annotations(root: Path) -> list[dict[str, str]]:
    with (root / "data" / "annotations.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_images(root: Path) -> dict[str, object]:
    from PIL import Image, ImageOps

    files = sorted(path for path in (root / "data" / "raw").rglob("*.jpg") if path.is_file())
    hashes: dict[str, str] = {}
    duplicates: list[str] = []
    unreadable: list[str] = []
    exif_non_upright = 0
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            duplicates.append(f"{path.as_posix()} duplicates {hashes[digest]}")
        hashes[digest] = path.as_posix()
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                orientation = image.getexif().get(274)
                if orientation not in (None, 1):
                    exif_non_upright += 1
                ImageOps.exif_transpose(image)
        except OSError as exc:
            unreadable.append(f"{path.as_posix()}: {exc}")
    phase0_in_raw = [
        path.name
        for path in files
        if "phase0" in path.name.lower()
    ]
    return {
        "photograph_count": len(files),
        "unique_sha256": len(hashes),
        "duplicate_hashes": duplicates,
        "unreadable": unreadable,
        "exif_non_upright": exif_non_upright,
        "phase0_files_in_raw": phase0_in_raw,
    }


def _product_flags(rows: list[dict[str, str]]) -> list[str]:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[row["product_id"]][row["angle"]] = row
    lines = []
    for product_id, angles in grouped.items():
        latin = {angle: angles[angle].get("text_true_latin", "").strip() for angle in angles}
        flags = []
        if len(set(latin.values())) == 1:
            flags.append("Latin draft is identical on all three angles")
        if "occluded" in angles and "straight" in angles:
            if latin.get("occluded") and latin["occluded"] == latin.get("straight"):
                flags.append("occluded Latin matches straight Latin")
        if len(set(latin.values())) > 1:
            flags.append("Latin draft differs by angle")
        if flags:
            lines.append(f"- `{product_id}`: " + "; ".join(flags) + ".")
    return lines


def write_reports(root: Path) -> dict[str, object]:
    root = Path(root)
    validation = validate_dataset(root)
    images = audit_images(root)
    rows = _read_annotations(root)
    classified = [classify_row(row) for row in rows]
    review_counts = Counter((row.get("review_status") or "").strip() or "blank" for row in rows)
    latin_cases = Counter(str(item["latin_case"]) for item in classified)
    urdu_cases = Counter(str(item["urdu_case"]) for item in classified)
    urdu_nonempty = sum(1 for row in rows if (row.get("text_true_urdu") or "").strip())
    predictions = root / "results" / "benchmark" / "benchmark_predictions.jsonl"
    completed_predictions = 0
    if predictions.is_file():
        for line in predictions.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("inference_completed") is True:
                completed_predictions += 1
    payload = {
        "status": validation["status"],
        "ok": validation["ok"],
        "errors": validation["errors"],
        "annotation_gap_count": len(validation["annotation_gaps"]),
        "present_count": validation["present_count"],
        "expected_count": validation["expected_count"],
        "products": validation["expected_count"] // 3 if validation["expected_count"] else 0,
        "images": images,
        "review_status_counts": dict(review_counts),
        "latin_case_counts": dict(latin_cases),
        "urdu_case_counts": dict(urdu_cases),
        "urdu_nonempty": urdu_nonempty,
        "verified_count": review_counts.get("verified", 0),
        "completed_benchmark_predictions": completed_predictions,
        "phase0_separated": not images["phase0_files_in_raw"],
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "dataset_validation.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    write_text(reports / "dataset_qa.md", _dataset_md(payload, validation))
    write_text(reports / "annotation_review_summary.md", _annotation_md(rows, classified, payload))
    _write_unresolved(reports / "unresolved_annotations.csv", classified)
    write_text(root / "data" / "annotation_review.html", _review_html(rows, classified))
    return payload


def _dataset_md(payload: dict[str, object], validation: dict[str, object]) -> str:
    images = payload["images"]
    errors = list(validation.get("errors") or [])
    lines = [
        "# Dataset QA",
        "",
        f"Status: {payload['status']}.",
        f"Structural errors: {len(errors)}.",
        f"Photographs: {payload['present_count']} present, {payload['expected_count']} expected.",
        f"Products: {payload['products']}.",
        f"Unique SHA-256 hashes: {images['unique_sha256']}.",
        f"Duplicate hashes: {len(images['duplicate_hashes'])}.",
        f"Unreadable files: {len(images['unreadable'])}.",
        f"Photographs with non-upright EXIF orientation: {images['exif_non_upright']}.",
        "EXIF is applied in memory by `load_rgb_image`. The JPEG bytes are not rewritten.",
        f"Phase 0 filenames inside `data/raw`: {len(images['phase0_files_in_raw'])}.",
        "`RegionalPack-VLM-60-Renamed-Photos/` is a provenance copy, including `juice_intl` and `juice_local` folder names and two extra biscuit frames. Those files are not the evaluation set. The evaluation paths are `data/raw/apple_fruita` and `data/raw/apple_fruitien`.",
        f"Saved completed benchmark predictions: {payload['completed_benchmark_predictions']}.",
        "",
        "## Errors",
        "",
    ]
    lines.extend(f"- {error}" for error in errors) if errors else lines.append("None.")
    lines.extend(
        [
            "",
            "## Annotation gate",
            "",
            f"Rows still needing a person: {payload['annotation_gap_count']}.",
            "This gate means the text is not ground truth. It does not mean a photograph is missing.",
            "",
        ]
    )
    return "\n".join(lines)


def _annotation_md(
    rows: list[dict[str, str]],
    classified: list[dict[str, object]],
    payload: dict[str, object],
) -> str:
    lines = [
        "# Annotation review summary",
        "",
        "No draft was marked verified by this audit.",
        f"Verified: {payload['verified_count']}.",
        f"needs_review: {payload['review_status_counts'].get('needs_review', 0)}.",
        f"Blank review_status: {payload['review_status_counts'].get('blank', 0)}.",
        f"Urdu cells with text: {payload['urdu_nonempty']}.",
        "",
        "Script cases are `not_reviewed`, `legible`, `unreadable_or_ambiguous`, or `absent`.",
        "An unverified row is `not_reviewed` for both scripts. A blank cell on that row is not `absent`.",
        "",
        f"Latin cases: {payload['latin_case_counts']}.",
        f"Urdu cases: {payload['urdu_case_counts']}.",
        "",
        "## Same-product drafts",
        "",
        "Identical text across angles is a review flag, not a finding that the text was copied.",
        "The occlusion note has to be checked on that photograph.",
        "",
    ]
    lines.extend(_product_flags(rows))
    lines.extend(["", "## Unresolved actions", ""])
    for item in classified:
        if item["review_status"] == "verified":
            continue
        lines.append(f"- {item['action']}")
    lines.append("")
    return "\n".join(lines)


def _write_unresolved(path: Path, classified: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "product_id", "angle", "review_status", "latin_case", "urdu_case", "signals", "action"],
            lineterminator="\n",
        )
        writer.writeheader()
        for item in classified:
            if item["review_status"] == "verified":
                continue
            writer.writerow(
                {
                    "image_path": item["image_path"],
                    "product_id": item["product_id"],
                    "angle": item["angle"],
                    "review_status": item["review_status"],
                    "latin_case": item["latin_case"],
                    "urdu_case": item["urdu_case"],
                    "signals": ";".join(item["signals"]),
                    "action": item["action"],
                }
            )


def _review_html(rows: list[dict[str, str]], classified: list[dict[str, object]]) -> str:
    by_path = {str(item["image_path"]): item for item in classified}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["product_id"]].append(row)
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Annotation review</title>",
        "<style>body{font-family:sans-serif;max-width:960px;margin:24px} img{max-width:280px;height:auto} .card{border:1px solid #ccc;padding:12px;margin:12px 0}</style>",
        "</head><body>",
        "<h1>Annotation review</h1>",
        "<p>Drafts are not verified. Start the local review page with <code>.venv/bin/python -m src.review</code> to correct and verify one photograph at a time. This file is a snapshot.</p>",
    ]
    for product_id, product_rows in grouped.items():
        parts.append(f"<h2>{_escape(product_id)}</h2>")
        for row in product_rows:
            item = by_path[row["image_path"]]
            parts.append("<div class='card'>")
            parts.append(f"<p><strong>{_escape(row['angle'])}</strong> review_status={_escape(row.get('review_status') or 'blank')}</p>")
            parts.append(f"<img src='{_escape(row['image_path'].removeprefix('data/'))}' alt=''>")
            parts.append(f"<p>Latin case: {_escape(str(item['latin_case']))}. Urdu case: {_escape(str(item['urdu_case']))}.</p>")
            parts.append(f"<pre>{_escape(row.get('text_true_latin') or '')}</pre>")
            parts.append(f"<p>Urdu: {_escape(row.get('text_true_urdu') or '(blank)')}</p>")
            parts.append(f"<p>Brand visible: {_escape(row.get('brand_visible') or '')}</p>")
            parts.append(f"<p>Occlusion: {_escape(row.get('occlusion_note') or '')}</p>")
            parts.append(f"<p>Uncertain: {_escape(row.get('ambiguous_text') or '')}</p>")
            parts.append(f"<p>{_escape(str(item['action']))}</p>")
            parts.append("</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def main() -> int:
    payload = write_reports(Path("."))
    print(
        f"status={payload['status']} photographs={payload['present_count']} "
        f"verified={payload['verified_count']} gaps={payload['annotation_gap_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
