"""Score saved predictions against human annotations.

This module does not load a model and does not invent text.
Text hidden in one photograph is omitted from that photograph's ground truth.
An empty ground-truth string is not counted as a missed transcription.
"""

from __future__ import annotations

import unicodedata
from collections import Counter

from src.dataset import _alias_key

ANGLES_NOTE = (
    "straight, tilt, and occluded are phone capture conditions. "
    "This is not a controlled 3D viewpoint experiment."
)

CATEGORY_ACCEPT = {
    "cola": ("cola", "beverage", "soft drink", "drink"),
    "biscuit": ("biscuit", "cookie", "snack"),
    "tea": ("tea", "beverage"),
    "chocolate": ("chocolate", "confectionery", "candy", "snack"),
    "chips": ("chip", "crisp", "snack"),
    "detergent": ("detergent", "cleaning", "laundry"),
    "toothpaste": ("toothpaste", "oral", "dental"),
    "apple": ("juice", "nectar", "beverage", "drink"),
    "noodles": ("noodle", "instant noodle"),
    "soap": ("soap", "bath"),
}


def brand_match(response: str, aliases: list[str]) -> bool:
    """True when an accepted alias appears as its own words."""
    padded = f" {_alias_key(response)} "
    for alias in aliases:
        key = _alias_key(alias)
        if key and f" {key} " in padded:
            return True
    return False


def category_match(response: str, category: str) -> bool:
    text = _alias_key(response)
    accepted = CATEGORY_ACCEPT.get(category, (category,))
    return any(_alias_key(phrase) in text for phrase in accepted)


def normalize_ocr(text: str, *, script: str) -> str:
    """NFC and collapsed whitespace. Latin is casefolded. Punctuation stays.

    Punctuation is not stripped. The specification allows stripping only
    punctuation that is not part of the brand, and that exception is not
    automatic. Brand matching uses ``_alias_key`` instead.
    """
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = " ".join(normalized.split())
    if script == "latin":
        normalized = normalized.casefold()
    return normalized


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, start=1):
        current = [i]
        for j, right in enumerate(hypothesis, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (left != right)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _cer(reference: str, hypothesis: str, *, script: str) -> float | None:
    reference = normalize_ocr(reference, script=script)
    if not reference:
        return None
    hypothesis = normalize_ocr(hypothesis, script=script)
    if reference == hypothesis:
        return 0.0
    return _edit_distance(list(reference), list(hypothesis)) / len(reference)


def _wer(reference: str, hypothesis: str) -> float | None:
    """Latin and Roman-Urdu word error rate. Not used for Urdu script."""
    ref_words = normalize_ocr(reference, script="latin").split()
    if not ref_words:
        return None
    hyp_words = normalize_ocr(hypothesis, script="latin").split()
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def _majority(flags: list[bool | None]) -> str:
    decided = [flag for flag in flags if flag is not None]
    if len(decided) != 3:
        return "incomplete"
    correct = sum(decided)
    if correct >= 2:
        return "correct"
    return "incorrect"


def _ocr_is_allowed(
    annotation: dict[str, str],
    image_id: str,
    *,
    ocr_image_ids: set[str] | None,
    require_verified: bool,
) -> bool:
    if not require_verified:
        return True
    if annotation.get("review_status") != "verified":
        return False
    if ocr_image_ids is not None and image_id not in ocr_image_ids:
        return False
    return True


def score_predictions(
    *,
    annotations: list[dict[str, str]],
    predictions: list[dict[str, str]],
    aliases_by_product: dict[str, list[str]],
    manifest: list[dict[str, str]],
    ocr_image_ids: set[str] | None = None,
    require_verified_ocr: bool = False,
) -> dict[str, object]:
    """Per-image rows plus a product majority. Exploratory products stay out of the primary mean."""
    by_key = {
        (row.get("image_id", ""), row.get("prompt_id", "")): row
        for row in predictions
        if row.get("inference_completed") in (True, "true", "True")
    }
    roles = {row["product_id"]: row.get("comparison_role", "") for row in manifest}
    image_rows: list[dict[str, object]] = []
    for annotation in annotations:
        product_id = annotation["product_id"]
        image_id = f"{product_id}/{annotation['angle']}"
        identification = by_key.get((image_id, "identification"), {})
        category = by_key.get((image_id, "category"), {})
        transcription = by_key.get((image_id, "transcription"), {})
        ocr_ok = _ocr_is_allowed(
            annotation,
            image_id,
            ocr_image_ids=ocr_image_ids,
            require_verified=require_verified_ocr,
        )
        if ocr_ok:
            response = str(transcription.get("response") or "")
            latin = _cer(annotation.get("text_true_latin", ""), response, script="latin")
            urdu = _cer(annotation.get("text_true_urdu", ""), response, script="urdu")
            latin_wer = _wer(annotation.get("text_true_latin", ""), response)
        else:
            latin = None
            urdu = None
            latin_wer = None
        image_rows.append(
            {
                "image_id": image_id,
                "product_id": product_id,
                "angle": annotation["angle"],
                "comparison_role": roles.get(product_id, ""),
                "brand_correct": (
                    brand_match(
                        str(identification.get("response") or ""),
                        aliases_by_product.get(product_id, []),
                    )
                    if identification
                    else None
                ),
                "category_correct": (
                    category_match(
                        str(category.get("response") or ""),
                        annotation.get("category", ""),
                    )
                    if category
                    else None
                ),
                "latin_cer": latin,
                "latin_wer": latin_wer,
                "urdu_cer": urdu,
                "latin_scored": latin is not None,
                "urdu_scored": urdu is not None,
                "ocr_evaluated": bool(ocr_ok and transcription),
            }
        )

    product_rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in image_rows:
        grouped.setdefault(str(row["product_id"]), []).append(row)
    for product_id, rows in grouped.items():
        latin_scores = [float(row["latin_cer"]) for row in rows if row["latin_cer"] is not None]
        wer_scores = [float(row["latin_wer"]) for row in rows if row["latin_wer"] is not None]
        product_rows.append(
            {
                "product_id": product_id,
                "comparison_role": roles.get(product_id, ""),
                "brand": _majority([row["brand_correct"] if isinstance(row["brand_correct"], bool) else None for row in rows]),
                "category": _majority([row["category_correct"] if isinstance(row["category_correct"], bool) else None for row in rows]),
                "latin_cer_mean": None if not latin_scores else sum(latin_scores) / len(latin_scores),
                "latin_cer_min": None if not latin_scores else min(latin_scores),
                "latin_cer_max": None if not latin_scores else max(latin_scores),
                "latin_wer_mean": None if not wer_scores else sum(wer_scores) / len(wer_scores),
                "latin_wer_min": None if not wer_scores else min(wer_scores),
                "latin_wer_max": None if not wer_scores else max(wer_scores),
                "images_with_latin_ground_truth": len(latin_scores),
            }
        )

    def _rate(rows: list[dict[str, object]], field: str) -> float | None:
        decided = [row for row in rows if row[field] in ("correct", "incorrect")]
        if not decided:
            return None
        return sum(row[field] == "correct" for row in decided) / len(decided)

    primary = [row for row in product_rows if row["comparison_role"] == "primary"]
    brands = [str(row["brand"]) for row in product_rows]
    return {
        "angles_note": ANGLES_NOTE,
        "images": image_rows,
        "products": product_rows,
        "primary_brand_accuracy": _rate(primary, "brand"),
        "primary_category_accuracy": _rate(primary, "category"),
        "exploratory_product_ids": [
            row["product_id"] for row in product_rows if row["comparison_role"] == "exploratory"
        ],
        "assertion_counts": dict(Counter(brands)),
        "ocr_evaluated_count": sum(1 for row in image_rows if row["ocr_evaluated"]),
    }
