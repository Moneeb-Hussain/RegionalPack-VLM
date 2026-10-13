"""Score a finished benchmark run for the Version 0.1 pilot.

Brand and category use the locked manifest and aliases. OCR is computed
only for the pre-registered subset, and only where review_status is verified.
This module does not load a model and does not invent responses.
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from src.artifacts import read_jsonl, write_text
from src.evaluate import load_eval_config, output_paths
from src.scoring import score_predictions

RELEASE = "0.1"


def _aliases(root: Path) -> dict[str, list[str]]:
    loaded = yaml.safe_load((root / "configs" / "brand_aliases.yaml").read_text(encoding="utf-8"))
    return {
        product_id: list(entry.get("aliases") or [])
        for product_id, entry in loaded["products"].items()
    }


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pct(value: float | None, numerator: int | None = None, denominator: int | None = None) -> str:
    if value is None or denominator == 0:
        return "not measured"
    if numerator is None or denominator is None:
        return f"{value:.0%}"
    return f"{numerator}/{denominator} ({value:.0%})"


def _image_rate(images: list[dict[str, object]], field: str) -> tuple[float | None, int, int]:
    decided = [
        row
        for row in images
        if row.get("comparison_role") == "primary" and isinstance(row.get(field), bool)
    ]
    if not decided:
        return None, 0, 0
    hits = sum(row[field] is True for row in decided)
    return hits / len(decided), hits, len(decided)


def _quoted(predictions: list[dict[str, object]], prompt_id: str) -> dict[str, str]:
    return {
        str(row.get("image_id")): str(row.get("response") or "")
        for row in predictions
        if row.get("prompt_id") == prompt_id and row.get("inference_completed") is True
    }


def _quote_lines(
    scored: dict[str, object],
    responses: dict[str, str],
    *,
    want_hit: bool,
    limit: int,
) -> list[str]:
    lines = []
    for row in scored["images"]:
        if row.get("comparison_role") != "primary":
            continue
        if row.get("brand_correct") is not want_hit:
            continue
        text = responses.get(str(row["image_id"]))
        if not text:
            continue
        lines.append(f"- `{row['image_id']}`: {text.strip().splitlines()[0][:240]}")
        if len(lines) >= limit:
            break
    return lines


def qualitative_transcriptions(predictions: list[dict[str, object]], limit: int = 4) -> list[str]:
    """Quote saved transcriptions. These are observations, not an accuracy."""
    lines = []
    for row in predictions:
        if row.get("prompt_id") != "transcription" or row.get("inference_completed") is not True:
            continue
        text = str(row.get("response") or "").strip()
        first = text.splitlines()[0][:240] if text else "(empty response)"
        lines.append(
            f"- `{row.get('image_id')}` (qualitative observation, not an accuracy): {first}"
        )
        if len(lines) >= limit:
            break
    return lines


def write_release_summary(root: Path) -> Path:
    root = Path(root)
    config = load_eval_config(root)
    paths = output_paths(root, config)
    annotations = _rows(root / "data" / "annotations.csv")
    manifest = _rows(root / "data" / "product_manifest.csv")
    ocr_config = config["ocr_subset"]
    if ocr_config.get("scope") == "all_verified":
        subset: list[str] | None = None
    else:
        subset = [str(item) for item in ocr_config["image_ids"]]
    predictions: list[dict[str, object]] = []
    if paths["jsonl"].is_file():
        predictions = [row for row in read_jsonl(paths["jsonl"]) if row.get("record_type", "inference") == "inference"]
    scored = score_predictions(
        annotations=annotations,
        predictions=predictions,
        aliases_by_product=_aliases(root),
        manifest=manifest,
        ocr_image_ids=None if subset is None else set(subset),
        require_verified_ocr=True,
    )
    images = list(scored["images"])
    brand_value, brand_hits, brand_n = _image_rate(images, "brand_correct")
    category_value, category_hits, category_n = _image_rate(images, "category_correct")
    primary = [row for row in scored["products"] if row["comparison_role"] == "primary"]
    brand_decided = [row for row in primary if row["brand"] in ("correct", "incorrect")]
    category_decided = [row for row in primary if row["category"] in ("correct", "incorrect")]
    product_brand_hits = sum(row["brand"] == "correct" for row in brand_decided)
    product_category_hits = sum(row["category"] == "correct" for row in category_decided)
    if subset is None:
        verified_in_subset = [row["image_path"] for row in annotations if row.get("review_status") == "verified"]
    else:
        allowed = set(subset)
        verified_in_subset = [
            row["image_path"]
            for row in annotations
            if f"{row['product_id']}/{row['angle']}" in allowed and row.get("review_status") == "verified"
        ]
    drafts = sum(row.get("review_status") == "needs_review" for row in annotations)
    pilot_ids = [str(item) for item in (config.get("inference_subset") or {}).get("image_ids") or []]
    expected_calls = (len(pilot_ids) or 60) * 3
    measured = (
        f"Pre-registered model inputs: {len(pilot_ids)}."
        if pilot_ids
        else "Frozen model inputs: all 60 photographs, three angles each."
    )
    completed = sum(row.get("inference_completed") is True for row in predictions)
    identification = _quoted(predictions, "identification")
    hits = _quote_lines(scored, identification, want_hit=True, limit=3)
    misses = _quote_lines(scored, identification, want_hit=False, limit=3)
    observations = qualitative_transcriptions(predictions)
    lines = [
        f"# RegionalPack-VLM {config.get('release_label', 'Version 0.1 preliminary pilot')}",
        "",
        "This file is descriptive. It is not a significance test and it is not a causal claim about geography.",
        "",
        "## What was measured",
        "",
        measured,
        f"Saved inference rows: {completed} of {expected_calls}.",
        f"Primary brand accuracy, product majority: {_pct(scored['primary_brand_accuracy'], product_brand_hits, len(brand_decided))}.",
        f"Primary products with 3 decided brand photographs: {len(brand_decided)}.",
        f"Primary category accuracy, product majority: {_pct(scored['primary_category_accuracy'], product_category_hits, len(category_decided))}.",
        f"Primary products with 3 decided category photographs: {len(category_decided)}.",
        f"Primary brand accuracy, image-level: {_pct(brand_value, brand_hits, brand_n)}.",
        f"Primary brand image sample size: {brand_n}.",
        f"Primary category accuracy, image-level: {_pct(category_value, category_hits, category_n)}.",
        f"Primary category image sample size: {category_n}.",
        "Product-level brand and category use a strict majority of the three photographs.",
        "The apple-drink pair is exploratory and is not in those two rates.",
        f"Exploratory product ids: {', '.join(scored['exploratory_product_ids']) or 'none'}.",
        "",
        "## OCR",
        "",
        ocr_config["rule"],
        "OCR scope: every verified row." if subset is None else f"Listed OCR images: {len(subset)}.",
        f"Human-verified transcriptions eligible for OCR: {len(verified_in_subset)}.",
        f"OCR images evaluated: {scored['ocr_evaluated_count']}.",
        "Unverified draft text was not used. No OCR accuracy is reported from it.",
        f"Draft rows left as needs_review: {drafts}.",
        "",
        "### Qualitative transcriptions",
        "",
    ]
    if observations:
        lines.extend(observations)
    else:
        lines.append("No transcription responses have been saved yet.")
    lines.extend(["", "## Brand examples", ""])
    if not predictions:
        lines.append("No model responses have been saved yet, so there are no brand examples.")
    else:
        lines.append("Correct brand, from saved identification responses:")
        lines.extend(hits or ["- None in the saved primary responses."])
        lines.append("Missed brand, from saved identification responses:")
        lines.extend(misses or ["- None in the saved primary responses."])
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "Version 0.1 is a preliminary pilot. One model, one annotator, and 18 primary products.",
            (
                "The frozen run uses all three photographs of each product. "
                f"Saved responses: {completed}."
                if completed
                else "The frozen run uses all three photographs of each product. No benchmark response has been saved."
            ),
            "straight, tilt, and occluded are phone capture conditions, not a controlled 3D viewpoint experiment.",
            "Flavor, filling, gel-claim, and fragrance differences are recorded on the manifest and are not treated as matched variants.",
            (
                "All transcription rows used for OCR are human-verified."
                if drafts == 0
                else "Draft OCR labels remain needs_review. They are not ground truth."
            ),
            "",
        ]
    )
    paths["dir"].mkdir(parents=True, exist_ok=True)
    dest = paths["dir"] / "v0.1_summary.md"
    write_text(dest, "\n".join(lines))
    return dest


def main() -> int:
    path = write_release_summary(Path("."))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
