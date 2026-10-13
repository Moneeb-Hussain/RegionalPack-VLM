"""Score the saved Qwen3-VL benchmark without changing frozen rules.

The headline brand and category rates call ``src.scoring`` unchanged.
An explicit abstention is recorded separately and is not a brand success.
A name blocked only by a comma or quotation mark is flagged for a person.
That flag does not change the headline rate.

This script does not load a model and does not edit annotations, predictions,
prompts, or scoring configs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.artifacts import read_jsonl
from src.scoring import brand_match, category_match, normalize_ocr, _cer, _wer

PREDICTIONS = ROOT / "results" / "benchmark" / "benchmark_predictions.jsonl"
PROTECTED = (
    "data/annotations.csv",
    "data/product_manifest.csv",
    "results/benchmark/benchmark_predictions.jsonl",
    "prompts/prompts.yaml",
    "configs/brand_aliases.yaml",
    "configs/scoring_config.yaml",
    "configs/inference_config.yaml",
    "configs/eval_config.yaml",
)
ABSTAIN = re.compile(
    r"not (?:clearly )?(?:identifiable|visible)|"
    r"cannot be identified|can not be identified|"
    r"cannot identify|can't identify|unable to identify",
    re.IGNORECASE,
)
ANGLES = ("straight", "tilt", "occluded")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """Descriptive 95% Wilson interval. Not a hypothesis test."""
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _pct(successes: int, total: int) -> str:
    if total == 0:
        return "0/0"
    return f"{successes}/{total} ({successes / total:.0%})"


def _loose_match(response: str, aliases: list[str]) -> bool:
    """Diagnostic only. Quotes and commas become spaces. Not the headline rule."""
    loosened = response.replace('"', " ").replace("“", " ").replace("”", " ").replace(",", " ")
    return brand_match(loosened, aliases)


def _repeat(text: str) -> tuple[int, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0, ""
    phrase, count = Counter(lines).most_common(1)[0]
    return count, phrase


def _urdu_chars(text: str) -> bool:
    return any("\u0600" <= char <= "\u06ff" for char in text)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def load_inputs(root: Path = ROOT) -> dict[str, object]:
    annotations = list(csv.DictReader((root / "data" / "annotations.csv").open(encoding="utf-8", newline="")))
    manifest = list(csv.DictReader((root / "data" / "product_manifest.csv").open(encoding="utf-8", newline="")))
    aliases = yaml.safe_load((root / "configs" / "brand_aliases.yaml").read_text(encoding="utf-8"))["products"]
    predictions = [
        row
        for row in read_jsonl(root / "results" / "benchmark" / "benchmark_predictions.jsonl")
        if row.get("record_type", "inference") == "inference"
    ]
    return {
        "annotations": annotations,
        "manifest": manifest,
        "aliases": aliases,
        "predictions": predictions,
    }


def score_rows(inputs: dict[str, object]) -> list[dict[str, object]]:
    annotations = inputs["annotations"]
    manifest = {row["product_id"]: row for row in inputs["manifest"]}
    aliases = inputs["aliases"]
    by_key = {(row["image_id"], row["prompt_id"]): row for row in inputs["predictions"]}
    scored = []
    for annotation in annotations:
        product_id = annotation["product_id"]
        image_id = f"{product_id}/{annotation['angle']}"
        identification = str(by_key[(image_id, "identification")]["response"] or "")
        category_response = str(by_key[(image_id, "category")]["response"] or "")
        transcription = str(by_key[(image_id, "transcription")]["response"] or "")
        accepted = list(aliases[product_id]["aliases"])
        frozen = brand_match(identification, accepted)
        abstention = bool(ABSTAIN.search(identification))
        loose = _loose_match(identification, accepted)
        if abstention:
            outcome = "abstention"
            adjudication = "abstention_not_a_hit"
        elif frozen:
            outcome = "asserted_alias"
            adjudication = "none"
        elif loose:
            outcome = "punctuation_blocked_assertion"
            adjudication = "pending_human"
        else:
            outcome = "no_accepted_alias"
            adjudication = "none"
        headline = bool(frozen) and not abstention
        flags = []
        if abstention and loose:
            flags.append("alias_quoted_inside_abstention")
        if outcome == "punctuation_blocked_assertion":
            flags.append("alias_present_but_comma_or_quotes_block_frozen_match")
        if re.search(r"sugar-free gum", identification, re.IGNORECASE):
            flags.append("identification_calls_it_gum")
        if re.search(r"brand (?:name )?is biscotti|brand biscotti", identification, re.IGNORECASE):
            flags.append("wrong_parent_spelling_biscotti")
        if re.search(r"specific brand name is bisconni", identification, re.IGNORECASE):
            flags.append("parent_named_as_the_brand")
        if "dairy" in category_response.casefold() and annotation["category"] == "chocolate":
            flags.append("category_also_says_dairy")
        if (
            annotation["category"] in {"soap", "toothpaste"}
            and "cleaning" in category_response.casefold()
            and not category_match(category_response, annotation["category"])
        ):
            flags.append("category_uses_prompt_example_cleaning_product")
        latin_ref = annotation.get("text_true_latin", "")
        urdu_ref = annotation.get("text_true_urdu", "")
        repeats, phrase = _repeat(transcription)
        if repeats >= 4:
            flags.append("repeated_transcription_line")
        scored.append(
            {
                "image_id": image_id,
                "product_id": product_id,
                "category": annotation["category"],
                "origin": manifest[product_id]["origin"],
                "comparison_role": manifest[product_id]["comparison_role"],
                "angle": annotation["angle"],
                "brand_true": annotation["brand_true"],
                "identification_response": identification,
                "category_response": category_response,
                "transcription_response": transcription,
                "frozen_brand_match": frozen,
                "explicit_abstention": abstention,
                "alias_if_quotes_and_commas_ignored": loose,
                "brand_outcome": outcome,
                "brand_headline_success": headline,
                "adjudication": adjudication,
                "frozen_category_match": category_match(category_response, annotation["category"]),
                "latin_cer": _cer(latin_ref, transcription, script="latin"),
                "latin_wer": _wer(latin_ref, transcription),
                "urdu_cer": _cer(urdu_ref, transcription, script="urdu"),
                "urdu_ground_truth_nonempty": bool(urdu_ref.strip()),
                "urdu_in_response": _urdu_chars(transcription),
                "latin_ref_chars": len(normalize_ocr(latin_ref, script="latin")),
                "latin_hyp_chars": len(normalize_ocr(transcription, script="latin")),
                "repeat_count": repeats,
                "repeat_phrase": phrase,
                "flags": "|".join(flags),
            }
        )
    return scored


def _majority(flags: list[bool]) -> str:
    if len(flags) != 3:
        return "incomplete"
    return "correct" if sum(flags) >= 2 else "incorrect"


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    primary = [row for row in rows if row["comparison_role"] == "primary"]
    exploratory = [row for row in rows if row["comparison_role"] == "exploratory"]

    def product_table(pool: list[dict[str, object]]) -> list[dict[str, object]]:
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in pool:
            grouped.setdefault(str(row["product_id"]), []).append(row)
        table = []
        for product_id, items in grouped.items():
            items = sorted(items, key=lambda item: ANGLES.index(str(item["angle"])))
            table.append(
                {
                    "product_id": product_id,
                    "category": items[0]["category"],
                    "origin": items[0]["origin"],
                    "comparison_role": items[0]["comparison_role"],
                    "brand_true": items[0]["brand_true"],
                    "brand_hits": sum(bool(item["brand_headline_success"]) for item in items),
                    "brand": _majority([bool(item["brand_headline_success"]) for item in items]),
                    "category_hits": sum(bool(item["frozen_category_match"]) for item in items),
                    "category_result": _majority([bool(item["frozen_category_match"]) for item in items]),
                    "outcomes": {str(item["angle"]): item["brand_outcome"] for item in items},
                    "pending": any(item["adjudication"] == "pending_human" for item in items),
                }
            )
        return table

    products = product_table(rows)
    primary_products = [row for row in products if row["comparison_role"] == "primary"]
    brand_hits = sum(row["brand"] == "correct" for row in primary_products)
    category_hits = sum(row["category_result"] == "correct" for row in primary_products)
    image_brand_hits = sum(bool(row["brand_headline_success"]) for row in primary)
    image_category_hits = sum(bool(row["frozen_category_match"]) for row in primary)

    by_angle = {}
    for angle in ANGLES:
        chosen = [row for row in primary if row["angle"] == angle]
        by_angle[angle] = {
            "brand": _pct(sum(bool(row["brand_headline_success"]) for row in chosen), len(chosen)),
            "brand_hits": sum(bool(row["brand_headline_success"]) for row in chosen),
            "category": _pct(sum(bool(row["frozen_category_match"]) for row in chosen), len(chosen)),
            "category_hits": sum(bool(row["frozen_category_match"]) for row in chosen),
            "n": len(chosen),
        }

    pairs = []
    categories = []
    for row in primary_products:
        if row["category"] not in categories:
            categories.append(row["category"])
    for category in categories:
        members = [row for row in primary_products if row["category"] == category]
        intl = next(row for row in members if row["origin"] == "international")
        local = next(row for row in members if row["origin"] == "local")
        pairs.append(
            {
                "category": category,
                "international_brand": intl["brand"],
                "local_brand": local["brand"],
                "international_category": intl["category_result"],
                "local_category": local["category_result"],
                "international_id": intl["product_id"],
                "local_id": local["product_id"],
            }
        )

    latin = [float(row["latin_cer"]) for row in rows if row["latin_cer"] is not None]
    latin_wer = [float(row["latin_wer"]) for row in rows if row["latin_wer"] is not None]
    urdu = [float(row["urdu_cer"]) for row in rows if row["urdu_cer"] is not None]
    degenerate = [row for row in rows if int(row["repeat_count"]) >= 4]
    latin_without_degenerate = [
        float(row["latin_cer"])
        for row in rows
        if row["latin_cer"] is not None and int(row["repeat_count"]) < 4
    ]
    outcomes = Counter(str(row["brand_outcome"]) for row in primary)
    return {
        "n_predictions_expected": 180,
        "n_images": len(rows),
        "primary_products": 18,
        "exploratory_products": [row["product_id"] for row in products if row["comparison_role"] == "exploratory"],
        "product_brand": {"hits": brand_hits, "n": len(primary_products), "label": _pct(brand_hits, len(primary_products))},
        "product_category": {
            "hits": category_hits,
            "n": len(primary_products),
            "label": _pct(category_hits, len(primary_products)),
        },
        "image_brand": {"hits": image_brand_hits, "n": len(primary), "label": _pct(image_brand_hits, len(primary))},
        "image_category": {
            "hits": image_category_hits,
            "n": len(primary),
            "label": _pct(image_category_hits, len(primary)),
        },
        "product_brand_wilson95": wilson(brand_hits, len(primary_products)),
        "product_category_wilson95": wilson(category_hits, len(primary_products)),
        "by_angle": by_angle,
        "pairs": pairs,
        "products": products,
        "primary_brand_outcomes": dict(outcomes),
        "abstentions_primary": outcomes["abstention"],
        "pending_adjudication_images": sum(row["adjudication"] == "pending_human" for row in primary),
        "exploratory_brand": {
            row["product_id"]: row["brand"] for row in products if row["comparison_role"] == "exploratory"
        },
        "exploratory_category": {
            row["product_id"]: row["category_result"] for row in products if row["comparison_role"] == "exploratory"
        },
        "latin_cer_mean": sum(latin) / len(latin),
        "latin_cer_median": _median(latin),
        "latin_cer_n": len(latin),
        "latin_wer_mean": sum(latin_wer) / len(latin_wer),
        "latin_wer_median": _median(latin_wer),
        "latin_cer_mean_without_repeated_line": sum(latin_without_degenerate) / len(latin_without_degenerate),
        "latin_cer_n_without_repeated_line": len(latin_without_degenerate),
        "urdu_cer_mean": None if not urdu else sum(urdu) / len(urdu),
        "urdu_cer_median": _median(urdu),
        "urdu_cer_n": len(urdu),
        "urdu_omitted_when_present": sum(
            1 for row in rows if row["urdu_ground_truth_nonempty"] and not row["urdu_in_response"]
        ),
        "degenerate_images": [str(row["image_id"]) for row in degenerate],
        "origin_brand_products": {
            "international": _pct(
                sum(row["brand"] == "correct" for row in primary_products if row["origin"] == "international"),
                sum(row["origin"] == "international" for row in primary_products),
            ),
            "local": _pct(
                sum(row["brand"] == "correct" for row in primary_products if row["origin"] == "local"),
                sum(row["origin"] == "local" for row in primary_products),
            ),
        },
        "exploratory_image_count": len(exploratory),
    }


def _bar_svg(path: Path, title: str, categories: list[str], series: list[tuple[str, list[int]]], ymax: int) -> None:
    width, height = 760, 320
    left, right, top, bottom = 48, 16, 36, 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    group = plot_w / max(len(categories), 1)
    bar_w = group / (len(series) + 1)
    colors = ("#3d5a40", "#8a5a44")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f5f2"/>',
        f'<text x="{left}" y="22" font-family="Georgia, serif" font-size="16">{title}</text>',
    ]
    for tick in range(0, ymax + 1, max(ymax // 4, 1)):
        y = top + plot_h - (tick / ymax) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif">{tick}</text>'
        )
    for index, category in enumerate(categories):
        for series_index, (_name, values) in enumerate(series):
            value = values[index]
            bar_h = 0 if ymax == 0 else (value / ymax) * plot_h
            x = left + index * group + series_index * bar_w + 6
            y = top + plot_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 4:.1f}" height="{bar_h:.1f}" fill="{colors[series_index % 2]}"/>'
            )
        parts.append(
            f'<text x="{left + index * group + group / 2:.1f}" y="{height - 36}" text-anchor="middle" font-size="11" font-family="sans-serif">{category}</text>'
        )
    legend_x = left
    for series_index, (name, _values) in enumerate(series):
        parts.append(f'<rect x="{legend_x}" y="{height - 18}" width="12" height="12" fill="{colors[series_index % 2]}"/>')
        parts.append(
            f'<text x="{legend_x + 16}" y="{height - 8}" font-size="11" font-family="sans-serif">{name}</text>'
        )
        legend_x += 180
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_outputs(rows: list[dict[str, object]], summary: dict[str, object], hashes: dict[str, str]) -> None:
    results = ROOT / "results"
    reports = ROOT / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with (results / "scored_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (results / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    brand_intl = []
    brand_local = []
    categories = []
    for pair in summary["pairs"]:
        categories.append(str(pair["category"]))
        brand_intl.append(1 if pair["international_brand"] == "correct" else 0)
        brand_local.append(1 if pair["local_brand"] == "correct" else 0)
    _bar_svg(
        figures / "brand_by_pair.svg",
        "Primary product-level brand success by pair (1 = majority of 3 photos)",
        categories,
        [("International", brand_intl), ("Pakistan-market", brand_local)],
        1,
    )
    angles = list(ANGLES)
    _bar_svg(
        figures / "brand_by_capture.svg",
        "Primary image-level brand successes by capture condition (of 18)",
        angles,
        [
            ("Brand successes", [summary["by_angle"][angle]["brand_hits"] for angle in angles]),
            ("Category successes", [summary["by_angle"][angle]["category_hits"] for angle in angles]),
        ],
        18,
    )
    _write_analysis(summary)
    _write_failures(rows)
    _write_qa(rows, summary, hashes)


def _write_analysis(summary: dict[str, object]) -> None:
    brand = summary["product_brand"]
    category = summary["product_category"]
    brand_interval = summary["product_brand_wilson95"]
    category_interval = summary["product_category_wilson95"]
    lines = [
        "# RegionalPack-VLM benchmark analysis",
        "",
        "One saved run of Qwen3-VL-2B-Instruct. Descriptive counts only. This is not a test that geography, or pretraining frequency, caused the misses.",
        "",
        "## Headline rates",
        "",
        "Brand and category use the frozen alias list and the frozen category phrases. A product is correct when at least 2 of its 3 photographs succeed. The apple-drink pair is excluded.",
        "",
        "| Rate | Result | Wilson 95% interval |",
        "|---|---|---|",
        f"| Primary brand, product majority | {brand['label']} | {brand_interval[0]:.0%}–{brand_interval[1]:.0%} |",
        f"| Primary category, product majority | {category['label']} | {category_interval[0]:.0%}–{category_interval[1]:.0%} |",
        f"| Primary brand, image | {summary['image_brand']['label']} | not used |",
        f"| Primary category, image | {summary['image_category']['label']} | not used |",
        "",
        "The interval treats 18 products as independent draws. It describes uncertainty for this sample. It is not a significance test. Image-level intervals are omitted because the three photographs of one product are not independent.",
        "",
        f"International products, brand majority: {summary['origin_brand_products']['international']}. Pakistan-market products, brand majority: {summary['origin_brand_products']['local']}.",
        "",
        "Five local product-brand majorities fail under the frozen rule. Zero international product-brand majorities fail. That split is a description of these 18 products. It is not evidence of a geographic mechanism. Three of the five local misses still contain the product-brand name, blocked by a comma or quotation marks, and are waiting for human adjudication.",
        "",
        "## Pairs",
        "",
        "| Category | International brand | Pakistan-market brand | International category | Pakistan-market category |",
        "|---|---|---|---|---|",
    ]
    for pair in summary["pairs"]:
        lines.append(
            f"| {pair['category']} | {pair['international_brand']} | {pair['local_brand']} | {pair['international_category']} | {pair['local_category']} |"
        )
    lines.extend(
        [
            "",
            "Exploratory apple drinks, not in the rates above: "
            + ", ".join(
                f"{product} brand {summary['exploratory_brand'][product]}, category {summary['exploratory_category'][product]}"
                for product in summary["exploratory_products"]
            )
            + ".",
            "",
            "## Capture condition",
            "",
            "straight, tilt, and occluded are phone conditions, not a controlled viewpoint experiment.",
            "",
            "| Condition | Brand | Category |",
            "|---|---|---|",
        ]
    )
    for angle in ANGLES:
        item = summary["by_angle"][angle]
        lines.append(f"| {angle} | {item['brand']} | {item['category']} |")
    lines.extend(
        [
            "",
            "## Abstention and adjudication",
            "",
            f"Primary identification responses: {summary['primary_brand_outcomes']}.",
            "",
            "An explicit statement that the specific brand cannot be identified is an abstention. A name quoted inside that statement is not a brand success. On this run the frozen matcher already rejected those quoted names, because a quotation mark stays attached to the word. The abstention label records why. It did not flip any frozen hit to a miss.",
            "",
            f"Images waiting for a person: {summary['pending_adjudication_images']}. The model names the accepted brand, but a comma or quotation mark stops the frozen word match. Those images stay misses in the headline rate.",
            "",
            "If a reviewer later accepts those assertions, and still rejects abstentions, biscuit Rite, Bonus Tristar, and Shoop would become product-level brand successes. CandyLand YOU and Super Crisp would not. That sensitivity is not the reported rate.",
            "",
            "## Category note",
            "",
            "Soap and toothpaste fail the frozen category list when the answer is “cleaning product.” That phrase is an example in the category prompt, and it is an accepted phrase only for detergent. The list was not expanded after seeing these answers.",
            "",
            "Chocolate answers that also say “dairy” still match, because the frozen list accepts “chocolate” and the word is present. The dairy/snack wording is flagged, not rescored.",
            "",
            "## OCR",
            "",
            "Normalization is NFC, collapsed whitespace, and Latin casefold. Punctuation stays. Urdu is not casefolded. Empty ground truth is not a miss. There is no Urdu word-error rate. Extra model text raises character error rate, and the rate can exceed 1.",
            "",
            f"| Script | Images | Mean CER | Median CER | Mean WER |",
            f"|---|---|---|---|---|",
            f"| Latin | {summary['latin_cer_n']} | {summary['latin_cer_mean']:.2f} | {summary['latin_cer_median']:.2f} | {summary['latin_wer_mean']:.2f} |",
            f"| Urdu | {summary['urdu_cer_n']} | {summary['urdu_cer_mean']:.2f} | {summary['urdu_cer_median']:.2f} | not computed |",
            "",
            f"Latin mean CER without the repeated Knorr line: {summary['latin_cer_mean_without_repeated_line']:.2f} on {summary['latin_cer_n_without_repeated_line']} images. That exclusion is a sensitivity, not the headline OCR number.",
            "",
            f"Urdu ground truth with no Urdu characters in the response: {summary['urdu_omitted_when_present']} images, all three Shoop photographs.",
            "",
            f"Repeated-line transcriptions: {', '.join(summary['degenerate_images']) or 'none'}.",
            "",
            "Prices, badges, and regulatory lines were left out of the human transcription on purpose. When the model adds them, character error rate counts the extra characters. They were not removed after the run.",
            "",
            "Figures: `reports/figures/brand_by_pair.svg`, `reports/figures/brand_by_capture.svg`.",
            "",
        ]
    )
    (ROOT / "reports" / "benchmark_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def _excerpt(text: str, limit: int = 500) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit].rstrip() + "…"


def _write_failures(rows: list[dict[str, object]]) -> None:
    chosen = []
    for image_id in (
        "chocolate_local/straight",
        "chips_local/straight",
        "biscuit_local/tilt",
        "biscuit_local/straight",
        "detergent_local/tilt",
        "detergent_local/straight",
        "noodles_local/straight",
        "noodles_intl/tilt",
        "soap_intl/straight",
        "toothpaste_intl/straight",
        "cola_intl/straight",
    ):
        chosen.append(next(row for row in rows if row["image_id"] == image_id))
    lines = [
        "# Failure cases and contrasting responses",
        "",
        "Excerpts are copied from `benchmark_predictions.jsonl`. The selection rule is: every abstention, every punctuation-blocked assertion, the gum/cleaning contradiction, the Biscotti spelling, the repeated Knorr line, one soap and one toothpaste category miss, and one clear international hit.",
        "",
    ]
    for row in chosen:
        lines.extend(
            [
                f"## `{row['image_id']}`",
                "",
                f"Frozen brand match: {row['frozen_brand_match']}. Outcome: {row['brand_outcome']}. Adjudication: {row['adjudication']}. Category match: {row['frozen_category_match']}.",
                f"Flags: {row['flags'] or 'none'}.",
                "",
                "Identification:",
                "",
                "```text",
                _excerpt(str(row["identification_response"])),
                "```",
                "",
                "Category:",
                "",
                "```text",
                _excerpt(str(row["category_response"]), 300),
                "```",
                "",
                "Transcription:",
                "",
                "```text",
                _excerpt(str(row["transcription_response"]), 400),
                "```",
                "",
                f"Latin CER {row['latin_cer']:.2f}. Urdu CER {row['urdu_cer'] if row['urdu_cer'] is not None else 'not scored'}. Repeat count {row['repeat_count']}.",
                "",
            ]
        )
    (ROOT / "reports" / "failure_cases.md").write_text("\n".join(lines), encoding="utf-8")


def _write_qa(rows: list[dict[str, object]], summary: dict[str, object], hashes: dict[str, str]) -> None:
    keys = [(row["image_id"], prompt) for row in rows for prompt in ("identification", "transcription", "category")]
    lines = [
        "# Scoring QA",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Image rows scored | {len(rows)} |",
        f"| Prediction slots represented | {len(keys)} unique image/prompt pairs in the scored table |",
        f"| Primary brand headline | {summary['product_brand']['label']} |",
        f"| Abstention override flipped a frozen hit | no |",
        f"| Punctuation-blocked images left out of the headline | {summary['pending_adjudication_images']} |",
        "",
        "Protected files, SHA-256 recorded before the report files were written:",
        "",
        "| File | SHA-256 |",
        "|---|---|",
    ]
    for path, digest in hashes.items():
        lines.append(f"| `{path}` | `{digest}` |")
    lines.extend(
        [
            "",
            "The analyzer reads those files and writes only `results/scored_predictions.csv`, `results/metrics_summary.json`, `reports/benchmark_analysis.md`, `reports/failure_cases.md`, `reports/scoring_qa.md`, and `reports/figures/`.",
            "",
            "Frozen pieces that were not edited for this score: brand aliases, category accept phrases, OCR normalization, prompts, decoding, and the pinned revision.",
            "",
            "Headline brand success is `brand_match` from `src/scoring.py`, except an explicit abstention cannot be a success. On these 180 responses that exception changed no frozen hit.",
            "",
        ]
    )
    (ROOT / "reports" / "scoring_qa.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    hashes = {relative: sha256(ROOT / relative) for relative in PROTECTED}
    inputs = load_inputs(ROOT)
    predictions = inputs["predictions"]
    if len(predictions) != 180:
        raise SystemExit(f"Expected 180 predictions, found {len(predictions)}")
    pairs = {(row["image_id"], row["prompt_id"]) for row in predictions}
    if len(pairs) != 180:
        raise SystemExit("Prediction pairs are not unique")
    rows = score_rows(inputs)
    if len(rows) != 60:
        raise SystemExit(f"Expected 60 images, found {len(rows)}")
    if any(row["frozen_brand_match"] and row["explicit_abstention"] for row in rows):
        raise SystemExit("An abstention was also a frozen hit. Review before publishing a rate.")
    summary = summarize(rows)
    write_outputs(rows, summary, hashes)
    for relative, digest in hashes.items():
        if sha256(ROOT / relative) != digest:
            raise SystemExit(f"Protected file changed: {relative}")
    print(json.dumps({"product_brand": summary["product_brand"], "product_category": summary["product_category"]}, indent=2))


if __name__ == "__main__":
    main()
