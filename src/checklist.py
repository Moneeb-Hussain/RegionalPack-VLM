"""Printable shelf checklist rendered from the product manifest.

This module does not record availability and does not score a model.
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml


def _cell(value: str) -> str:
    return value.replace("|", "/").replace("\n", " ").strip()


def _exact_variant(row: dict[str, str]) -> str:
    brand = row["brand"].strip()
    line = row["product_line"].strip()
    variant = row["variant"].strip()
    parts: list[str] = []
    if line and line.casefold() != brand.casefold() and line.casefold() not in variant.casefold():
        parts.append(line)
    parts.append(variant)
    body = ", ".join(parts)
    if brand.casefold() not in body.casefold():
        return f"{brand}: {body}"
    return body


def render_checklist(root: Path) -> str:
    """Build the shelf sheet from product_manifest.csv and brand_aliases.yaml."""
    root = Path(root)
    manifest_path = root / "data" / "product_manifest.csv"
    alias_path = root / "configs" / "brand_aliases.yaml"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with alias_path.open("r", encoding="utf-8") as handle:
        aliases = yaml.safe_load(handle)["products"]

    lines = [
        "# Availability checklist",
        "",
        "Print this page. Tick it in the store. Do not photograph the benchmark set on this visit.",
        "",
        "| | |",
        "|---|---|",
        "| Store | |",
        "| Date | |",
        "| Checker | |",
        "",
        "Every row below is copied from `data/product_manifest.csv`.",
        "These packs are photographed. Visible text is not filled in yet.",
        "",
        "## Limitations and the exploratory pair",
        "",
        "- Biscuits: Oreo chocolate cream and Rite chocolate biscuits with vanilla cream are both sandwich biscuits. The fillings are not identical. Bisconni is the parent mark, not an alias for Rite.",
        "- Chips: Lay's Classic and Super Crisp BBQ are both potato-chip bags. The flavors are not identical.",
        "- Noodles: Knorr Chicken and Shoop Chattpata are both instant noodles. The flavors are not identical. Shoop is the printed brand. Shan alone is not a product-brand alias.",
        "- Toothpaste: Colgate Fresh Gel and Medicam UltraFresh Gel 3-in-1 are both gels. The gel claims are not identical.",
        "- Soap: Lux Vibrant Glow Tangerine and Capri Soothing Glow are both beauty-soap bars. The fragrances are not identical.",
        "- Apple drinks: `apple_fruita` and `apple_fruitien` are photographed and excluded from the primary international-versus-Pakistan comparison. Fruita Vitals is a Nestlé Pakistan sub-brand, and a sale under that name outside Pakistan is not supported.",
        "",
        "A category may be photographed in its own place. Both products in a pair use that same place, the same light, and the same background.",
        "",
        "## Products",
        "",
        "| Shot | Product ID | Group | Role | Exact variant | Package type | Availability | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        note = row.get("limitation", "").strip() or row["source_note"].strip()
        script = row["script_note"].strip()
        if script:
            note = f"{note} {script}".strip()
        variant = _exact_variant(row)
        lines.append(
            "| ☐ | {pid} | {group} | {role} | {variant} | {package} | {availability} | {note} |".format(
                pid=_cell(row["product_id"]),
                group=_cell(row["origin"]),
                role=_cell(row.get("comparison_role", "")),
                variant=_cell(variant),
                package=_cell(row["package_type"]),
                availability=_cell(row["availability_status"]),
                note=_cell(note),
            )
        )
    lines.extend(
        [
            "",
            "Photograph only the pack named in the variant column.",
            "A flavor or gel-claim difference stays in the limitation. It is not edited into a match.",
            "",
            "## Accepted brand aliases",
            "",
            "Fixed in `configs/brand_aliases.yaml` before inference.",
            "Prompt 1 may use these strings for the brand. A parent company is not an alias.",
            "",
            "| Product ID | Accepted names |",
            "|---|---|",
        ]
    )
    for row in rows:
        product_id = row["product_id"]
        names = aliases[product_id]["aliases"]
        lines.append(f"| {_cell(product_id)} | {_cell('; '.join(names))} |")
    lines.extend(
        [
            "",
            "Photography and annotation, after this list is locked: `data/photography_guide.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_checklist(root: Path) -> Path:
    path = Path(root) / "data" / "availability_checklist.md"
    path.write_text(render_checklist(root), encoding="utf-8")
    return path
