"""Post-hoc brand adjudication. Does not change the frozen score.

Rules written on 2026-10-12, after the frozen 13/18 result existed:

- If the identification says Bisconni or Biscotti is the brand, and Rite is
  the flavor or is not itself named as the brand, that image is not a
  product-brand success.
- An identification that names Bonus Tristar is a product-brand success in
  the adjudicated set, including when the same sentence calls it gum.
- An identification that names Shoop as the brand is a product-brand success
  in the adjudicated set.
- An explicit abstention stays a failure.

The second rate is the majority of these image-level decisions. Totals are
counted in code. Frozen columns in results/scored_predictions.csv are read
and not rewritten.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCORED = ROOT / "results" / "scored_predictions.csv"
PROTECTED = (
    SCORED,
    ROOT / "results" / "metrics_summary.json",
    ROOT / "results" / "benchmark" / "benchmark_predictions.jsonl",
    ROOT / "data" / "annotations.csv",
    ROOT / "prompts" / "prompts.yaml",
    ROOT / "configs" / "scoring_config.yaml",
    ROOT / "configs" / "brand_aliases.yaml",
)
PARENT_BRAND = re.compile(
    r"brand(?: name)? is biscotti|"
    r"brand(?: name)? is bisconni|"
    r"from the brand biscotti|"
    r"from the brand bisconni|"
    r"specific brand name is biscotti|"
    r"specific brand name is bisconni",
    re.IGNORECASE,
)
RITE_FLAVOR = re.compile(r"rite[\"']?\s+flavor", re.IGNORECASE)
SHOOP_BRAND = re.compile(r"brand name is [\"']?shoop", re.IGNORECASE)


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """Descriptive 95% Wilson interval. Not a hypothesis test."""
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _bool(value: object) -> bool:
    return value is True or value == "True"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_scored(path: Path = SCORED) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def adjudicate_brand(row: dict[str, str]) -> tuple[bool, str, str]:
    """Return adjudicated success, reviewer status, and rationale."""
    text = row["identification_response"]
    frozen = _bool(row["brand_headline_success"])
    if _bool(row["explicit_abstention"]):
        return (
            False,
            "confirmed_abstention",
            "The identification says the specific brand cannot be identified. "
            "A name inside quotation marks is packaging text, not a brand identification.",
        )
    if row["product_id"] == "biscuit_local" and PARENT_BRAND.search(text):
        flavor = " It also calls Rite a flavor." if RITE_FLAVOR.search(text) else ""
        product_name = " It also calls Rite the product name." if re.search(r"product name is rite", text, re.IGNORECASE) else ""
        return (
            False,
            "adjudicated_not_product_brand",
            "The response assigns the brand role to Bisconni or Biscotti. "
            "Rite is not named as the brand."
            + flavor
            + product_name
            + " Parent-as-brand is not product-brand identification.",
        )
    if row["product_id"] == "detergent_local" and "bonus tristar" in text.casefold():
        gum = ""
        if re.search(r"sugar-free gum", text, re.IGNORECASE):
            gum = " The same answer calls the product sugar-free gum. That is a product-type error. It does not cancel the explicit Bonus Tristar brand identification."
        return (
            True,
            "adjudicated_product_brand",
            "The response explicitly identifies Bonus Tristar." + gum,
        )
    if row["product_id"] == "noodles_local" and SHOOP_BRAND.search(text):
        return (
            True,
            "adjudicated_product_brand",
            "The response says the specific brand name is Shoop.",
        )
    return frozen, "frozen_unchanged", "No post-hoc brand rule applies. The frozen result is kept."


def category_reading(row: dict[str, str]) -> tuple[str, bool]:
    """Frozen category result, plus a labeled post-hoc broad reading.

    The post-hoc flag does not replace frozen category accuracy.
    """
    frozen = _bool(row["frozen_category_match"])
    response = row["category_response"].casefold()
    if frozen:
        return "frozen_correct", True
    if row["category"] in {"soap", "toothpaste"} and "cleaning product" in response:
        return "broad_cleaning_product", True
    return "other_miss", False


def _majority(flags: list[bool]) -> bool:
    if len(flags) != 3:
        raise ValueError("A product majority needs three photographs")
    return sum(flags) >= 2


def product_rates(rows: list[dict[str, object]], field: str) -> dict[str, object]:
    primary = [row for row in rows if row["comparison_role"] == "primary"]
    grouped: dict[str, list[bool]] = {}
    for row in primary:
        grouped.setdefault(str(row["product_id"]), []).append(bool(row[field]))
    hits = sum(_majority(flags) for flags in grouped.values())
    return {"hits": hits, "n": len(grouped), "label": f"{hits}/{len(grouped)}"}


def decide(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    decided = []
    for row in rows:
        success, status, rationale = adjudicate_brand(row)
        frozen = _bool(row["brand_headline_success"])
        interpretation, broad_success = category_reading(row)
        if frozen and not success:
            difference = "frozen_hit_adjudicated_miss"
        elif success and not frozen:
            difference = "frozen_miss_adjudicated_hit"
        else:
            difference = "unchanged"
        decided.append(
            {
                "image_id": row["image_id"],
                "product_id": row["product_id"],
                "angle": row["angle"],
                "comparison_role": row["comparison_role"],
                "origin": row["origin"],
                "category": row["category"],
                "frozen_brand_success": frozen,
                "adjudicated_brand_success": success,
                "reviewer_status": status,
                "difference_from_frozen": difference,
                "post_hoc": status != "frozen_unchanged",
                "identification_response": row["identification_response"],
                "rationale": rationale,
                "frozen_category_match": _bool(row["frozen_category_match"]),
                "category_interpretation": interpretation,
                "posthoc_broad_category_success": broad_success,
                "category_response": row["category_response"],
                "latin_cer": row["latin_cer"],
                "urdu_cer": row["urdu_cer"],
                "repeat_count": row["repeat_count"],
            }
        )
    return decided


def _fmt_interval(hits: int, total: int) -> str:
    interval = wilson(hits, total)
    if interval is None:
        return "not computed"
    return f"{interval[0]:.0%}–{interval[1]:.0%}"


def write_reports(decisions: list[dict[str, object]]) -> None:
    frozen = product_rates(decisions, "frozen_brand_success")
    adjudicated = product_rates(decisions, "adjudicated_brand_success")
    frozen_category = product_rates(decisions, "frozen_category_match")
    broad_category = product_rates(decisions, "posthoc_broad_category_success")
    if frozen["label"] != "13/18":
        raise SystemExit(f"Frozen brand majority changed: {frozen['label']}")

    primary = [row for row in decisions if row["comparison_role"] == "primary"]
    frozen_images = sum(bool(row["frozen_brand_success"]) for row in primary)
    adjudicated_images = sum(bool(row["adjudicated_brand_success"]) for row in primary)
    reviewed = [row for row in decisions if row["post_hoc"]]

    results = ROOT / "results"
    reports = ROOT / "reports"
    fieldnames = list(decisions[0].keys())
    with (results / "adjudication_decisions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decisions)

    _write_adjudication(reports / "ADJUDICATION_REPORT.md", reviewed, frozen, adjudicated)
    _write_revised(
        reports / "REVISED_BENCHMARK_ANALYSIS.md",
        decisions,
        frozen,
        adjudicated,
        frozen_category,
        broad_category,
        frozen_images,
        adjudicated_images,
    )
    _write_ocr(reports / "OCR_METRIC_INTERPRETATION.md", decisions)


def _write_adjudication(path: Path, reviewed: list[dict[str, object]], frozen: dict[str, object], adjudicated: dict[str, object]) -> None:
    lines = [
        "# Brand adjudication",
        "",
        "This file is post hoc. It was written after the frozen score existed. It does not replace that score.",
        "",
        f"Frozen automatic brand majority, recomputed from image rows: {frozen['label']}.",
        f"Adjudicated brand majority, recomputed from the image decisions below: {adjudicated['label']}.",
        "",
        "The adjudicated total is the number of primary products with at least 2 of 3 photographs marked successful in `results/adjudication_decisions.csv`. It is not typed in by hand.",
        "",
        "## Rules applied",
        "",
        "| Rule | Decision |",
        "|---|---|",
        "| The response says Bisconni or Biscotti is the brand | Not a Rite product-brand success, including when Rite is called the flavor or the product name |",
        "| The response names Bonus Tristar | Product-brand success in the adjudicated set. A gum description in the same answer is noted and does not cancel the brand decision |",
        "| The response says the brand name is Shoop | Product-brand success in the adjudicated set |",
        "| The response says the specific brand cannot be identified | Failure, even if the name is quoted |",
        "",
        "## Reviewed responses",
        "",
    ]
    for row in reviewed:
        lines.extend(
            [
                f"### `{row['image_id']}`",
                "",
                f"Frozen success: {row['frozen_brand_success']}. Adjudicated success: {row['adjudicated_brand_success']}. Status: {row['reviewer_status']}. Difference: {row['difference_from_frozen']}.",
                "",
                "Response:",
                "",
                "```text",
                str(row["identification_response"]).strip(),
                "```",
                "",
                f"Rationale: {row['rationale']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _origin_rate(rows: list[dict[str, object]], field: str, origin: str) -> str:
    chosen = [row for row in rows if row["comparison_role"] == "primary" and row["origin"] == origin]
    grouped: dict[str, list[bool]] = {}
    for row in chosen:
        grouped.setdefault(str(row["product_id"]), []).append(bool(row[field]))
    hits = sum(_majority(flags) for flags in grouped.values())
    return f"{hits}/{len(grouped)}"


def _write_revised(
    path: Path,
    decisions: list[dict[str, object]],
    frozen: dict[str, object],
    adjudicated: dict[str, object],
    frozen_category: dict[str, object],
    broad_category: dict[str, object],
    frozen_images: int,
    adjudicated_images: int,
) -> None:
    primary_products = []
    seen = []
    for row in decisions:
        if row["comparison_role"] != "primary" or row["product_id"] in seen:
            continue
        seen.append(row["product_id"])
        images = [item for item in decisions if item["product_id"] == row["product_id"]]
        primary_products.append(
            {
                "product_id": row["product_id"],
                "category": row["category"],
                "origin": row["origin"],
                "frozen": _majority([bool(item["frozen_brand_success"]) for item in images]),
                "adjudicated": _majority([bool(item["adjudicated_brand_success"]) for item in images]),
                "frozen_category": _majority([bool(item["frozen_category_match"]) for item in images]),
                "broad_category": _majority([bool(item["posthoc_broad_category_success"]) for item in images]),
            }
        )
    broad_images = sum(row["category_interpretation"] == "broad_cleaning_product" for row in decisions)
    other_misses = sum(row["category_interpretation"] == "other_miss" for row in decisions)
    lines = [
        "# Revised benchmark analysis",
        "",
        "Two scores are reported. Set A is the frozen automatic score. Set B is a post-hoc reading of the same saved answers. Set B does not replace Set A.",
        "",
        "The study has 20 products. The primary comparison uses 18. The two apple drinks stay exploratory. Eighteen products is a small sample. A Wilson interval describes that sample. It is not a test that origin caused the difference.",
        "",
        "## Set A — frozen automatic score",
        "",
        "| Rate | Result | Wilson 95% interval |",
        "|---|---|---|",
        f"| Brand, product majority | {frozen['label']} | {_fmt_interval(int(frozen['hits']), int(frozen['n']))} |",
        f"| Brand, primary images | {frozen_images}/54 | not used |",
        f"| Category, product majority | {frozen_category['label']} | {_fmt_interval(int(frozen_category['hits']), int(frozen_category['n']))} |",
        "",
        f"International brand majority: {_origin_rate(decisions, 'frozen_brand_success', 'international')}. Pakistan-market brand majority: {_origin_rate(decisions, 'frozen_brand_success', 'local')}. These are counts for this shelf sample, not a geographic effect.",
        "",
        "## Set B — post-hoc product-brand reading",
        "",
        "Counted from `adjudicated_brand_success` with the same three-photo majority. Abstentions stay failures. Rite responses that name Bisconni or Biscotti as the brand stay failures. Explicit Bonus Tristar and Shoop identifications count.",
        "",
        "| Rate | Result | Wilson 95% interval |",
        "|---|---|---|",
        f"| Brand, product majority | {adjudicated['label']} | {_fmt_interval(int(adjudicated['hits']), int(adjudicated['n']))} |",
        f"| Brand, primary images | {adjudicated_images}/54 | not used |",
        "",
        f"International brand majority: {_origin_rate(decisions, 'adjudicated_brand_success', 'international')}. Pakistan-market brand majority: {_origin_rate(decisions, 'adjudicated_brand_success', 'local')}. The change from Set A is the Bonus Tristar and Shoop decisions. It is still not evidence that origin caused the remaining misses.",
        "",
        "## Product table",
        "",
        "| Product | Origin | Set A brand | Set B brand | Frozen category |",
        "|---|---|---|---|---|",
    ]
    for row in primary_products:
        lines.append(
            f"| {row['product_id']} | {row['origin']} | {row['frozen']} | {row['adjudicated']} | {row['frozen_category']} |"
        )
    lines.extend(
        [
            "",
            "Exploratory apple drinks are not in the table. Their frozen brand results were not changed.",
            "",
            "## Category reading, post hoc",
            "",
            f"Frozen category majority remains {frozen_category['label']}.",
            "",
            f"Every frozen category miss is the sentence “This is a cleaning product.” There are {broad_images} such images, all soap or toothpaste. Other category-prompt misses: {other_misses}.",
            "",
            "“Cleaning product” is broad. It is not a claim that the item is food, a drink, or another unrelated category. The category prompt itself offers that phrase as an example. The frozen accept list includes it for detergent and not for soap or toothpaste. That list was not edited.",
            "",
            f"If those broad answers were also treated as acceptable, the post-hoc category majority would be {broad_category['label']}. That figure is not the frozen category accuracy.",
            "",
            "The Bonus tilt identification calls the packet sugar-free gum, while the category prompt for the same photo says cleaning product. That is a cross-prompt contradiction. The category score uses only the category prompt.",
            "",
            "## What would change the product rate",
            "",
            "Set B stays at its counted majority if the straight Rite photo is read alone as a product-line mention, because the other two Rite photos still name Bisconni or Biscotti as the brand and a majority needs two photographs. Keeping the frozen Rite tilt hit and also accepting the straight Rite photo would be a different rule. It is not Set B.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_ocr(path: Path, decisions: list[dict[str, object]]) -> None:
    latin = []
    for row in decisions:
        value = row["latin_cer"]
        if value not in ("", None):
            latin.append((str(row["image_id"]), float(value), int(row["repeat_count"])))
    latin_values = [item[1] for item in latin]
    mean = sum(latin_values) / len(latin_values)
    ordered = sorted(latin_values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    without = [item[1] for item in latin if item[2] < 4]
    highest = sorted(latin, key=lambda item: item[1], reverse=True)[:5]
    lines = [
        "# OCR metric interpretation",
        "",
        "The Latin and Urdu character error rates in `results/scored_predictions.csv` are unchanged. Nothing in the model output was deleted, and the human transcriptions were not edited.",
        "",
        "## Why character error rate can exceed 1",
        "",
        "Character error rate is the edit distance from the human transcription to the model transcription, divided by the number of characters in the human transcription. Insertions add to the distance. They do not add to the denominator. A long extra string, a price, a nutrition line, or a repeated sentence can make the distance larger than the reference, so the rate is greater than 1.",
        "",
        "Punctuation is kept. Latin text is casefolded after Unicode NFC, and whitespace is collapsed. Urdu is not casefolded. An empty human transcription is not scored. There is no Urdu word-error rate.",
        "",
        "## Extra packaging text",
        "",
        "The human transcription stops at the brand, the product or flavor, and the prominent tagline. Prices, badges, and regulatory lines were left out on purpose. If the model transcribes those lines, every extra character is an insertion. The original rate counts them. They were not stripped after the run.",
        "",
        "## Repeated generation and the mean",
        "",
        f"Latin mean character error rate: {mean:.2f} on {len(latin_values)} images. Median: {median:.2f}.",
        "",
        f"The mean without images whose transcription repeats a line at least four times: {sum(without) / len(without):.2f} on {len(without)} images. The Knorr tilt transcription is that repeated output. Removing it from the mean is a sensitivity. The original mean still includes it.",
        "",
        "Highest Latin rates in this run:",
        "",
        "| Image | Latin CER | Repeated-line count |",
        "|---|---|---|",
    ]
    for image_id, value, repeats in highest:
        lines.append(f"| `{image_id}` | {value:.2f} | {repeats} |")
    lines.extend(
        [
            "",
            "## Proposed secondary split",
            "",
            "This split was written after the run, once extra text and the repeated Knorr line were visible. It is not a replacement endpoint. It is a uniform rule for a later reading of the same strings.",
            "",
            "Using the same normalization as the character error rate:",
            "",
            "1. Reading: a reference word is read if it occurs in the model transcription. Coverage is the share of reference words that are read. A reading error is a reference word that is absent or replaced.",
            "2. Scope: extra words are model words that are not reference words. Prices, nutrition lines, and repeated sentences land here. A scope error can exist even when every reference word was read.",
            "",
            "The two counts stay separate. Incorrect text is not deleted to improve either count, and the human reference is not shortened to match the model.",
            "",
            "The original character error rates remain the OCR record for this submission until that secondary count is computed on every image under the rule above.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    hashes = {path: _sha256(path) for path in PROTECTED}
    rows = load_scored()
    if len(rows) != 60:
        raise SystemExit(f"Expected 60 scored images, found {len(rows)}")
    decisions = decide(rows)
    primary = [row for row in decisions if row["comparison_role"] == "primary"]
    if len({row["product_id"] for row in primary}) != 18:
        raise SystemExit("Primary product count is not 18")
    if len({row["product_id"] for row in decisions}) != 20:
        raise SystemExit("Product count is not 20")
    if any(row["reviewer_status"] == "confirmed_abstention" and row["adjudicated_brand_success"] for row in decisions):
        raise SystemExit("An abstention was marked successful")
    write_reports(decisions)
    for path, digest in hashes.items():
        if _sha256(path) != digest:
            raise SystemExit(f"Protected file changed: {path}")
    adjudicated = product_rates(decisions, "adjudicated_brand_success")
    print(adjudicated["label"])


if __name__ == "__main__":
    main()
