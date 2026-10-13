"""Draft figures and, only when needed, the adjudication approval checklist.

Reads the saved score tables. Does not edit predictions, annotations,
prompts, or scoring configs, and does not upload anything.

An existing reports/ADJUDICATION_APPROVAL.md is left untouched, including
during figure regeneration. A checked Approve or Reject box is never
overwritten. The unsigned template is written only when that file is
missing, or when this script is run with --write-approval-template and
the file has no checked decision.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
DECISIONS = ROOT / "results" / "adjudication_decisions.csv"

EXPECTED_CHANGES = (
    "biscuit_local/tilt",
    "detergent_local/straight",
    "detergent_local/occluded",
    "noodles_local/straight",
    "noodles_local/tilt",
)


def _load() -> list[dict[str, str]]:
    with DECISIONS.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _svg(path: Path, body: str, width: int, height: int) -> None:
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="#fbfaf7"/>',
                body,
                "</svg>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _grouped_bars() -> None:
    # Counts out of 9 primary products in each origin group.
    series = (("Set A, frozen", (9, 4), "#1f4e79"), ("Set B, post hoc", (9, 6), "#b85c38"))
    labels = ("International", "Pakistan-market")
    width, height = 720, 420
    left, top, plot_w, plot_h = 70, 70, 560, 250
    ymax = 9
    parts = [
        '<text x="70" y="32" font-family="Georgia, serif" font-size="18">Brand successes by origin</text>',
        '<text x="70" y="52" font-family="sans-serif" font-size="12" fill="#444">Products correct, majority of 3 photographs. Each group has 9 products.</text>',
    ]
    for tick in (0, 3, 6, 9):
        y = top + plot_h - (tick / ymax) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e4e0d8"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif">{tick}</text>'
        )
    parts.append(
        f'<text x="24" y="{top + plot_h / 2:.0f}" font-size="12" font-family="sans-serif" transform="rotate(-90 24 {top + plot_h / 2:.0f})">Products correct (of 9)</text>'
    )
    group = plot_w / 2
    bar_w = 70
    for group_index, label in enumerate(labels):
        for series_index, (name, values, color) in enumerate(series):
            value = values[group_index]
            bar_h = (value / ymax) * plot_h
            x = left + group_index * group + 80 + series_index * (bar_w + 16)
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}"/>')
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif">{value}/9</text>'
            )
        parts.append(
            f'<text x="{left + group_index * group + group / 2:.1f}" y="{top + plot_h + 28}" text-anchor="middle" font-size="13" font-family="sans-serif">{label}</text>'
        )
    parts.append(f'<rect x="70" y="{height - 36}" width="14" height="14" fill="#1f4e79"/>')
    parts.append(f'<text x="90" y="{height - 25}" font-size="12" font-family="sans-serif">Set A, frozen (13/18 overall)</text>')
    parts.append(f'<rect x="320" y="{height - 36}" width="14" height="14" fill="#b85c38"/>')
    parts.append(f'<text x="340" y="{height - 25}" font-size="12" font-family="sans-serif">Set B, post hoc (15/18 overall)</text>')
    _svg(FIGURES / "brand_by_origin.svg", "\n".join(parts), width, height)


def _capture_bars() -> None:
    # Set A primary image hits. 18 photographs in each capture condition.
    values = (("straight", 13), ("tilt", 15), ("occluded", 14))
    width, height = 720, 380
    left, top, plot_w, plot_h = 70, 70, 560, 220
    ymax = 18
    parts = [
        '<text x="70" y="32" font-family="Georgia, serif" font-size="18">Set A brand successes by capture condition</text>',
        '<text x="70" y="52" font-family="sans-serif" font-size="12" fill="#444">Frozen matcher. 18 primary photographs in each condition. Apple drinks excluded.</text>',
    ]
    for tick in (0, 6, 12, 18):
        y = top + plot_h - (tick / ymax) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e4e0d8"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif">{tick}</text>'
        )
    parts.append(
        f'<text x="22" y="{top + plot_h / 2:.0f}" font-size="12" font-family="sans-serif" transform="rotate(-90 22 {top + plot_h / 2:.0f})">Photographs correct (of 18)</text>'
    )
    slot = plot_w / 3
    for index, (label, value) in enumerate(values):
        bar_h = (value / ymax) * plot_h
        x = left + index * slot + slot / 2 - 36
        y = top + plot_h - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="72" height="{bar_h:.1f}" fill="#1f4e79"/>')
        parts.append(
            f'<text x="{x + 36:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif">{value}/18</text>'
        )
        parts.append(
            f'<text x="{left + index * slot + slot / 2:.1f}" y="{top + plot_h + 28}" text-anchor="middle" font-size="13" font-family="sans-serif">{label}</text>'
        )
    _svg(FIGURES / "brand_by_capture_set_a.svg", "\n".join(parts), width, height)


def checklist_path() -> Path:
    return ROOT / "reports" / "ADJUDICATION_APPROVAL.md"


def approval_is_signed(text: str) -> bool:
    return "- [x] Approve this image decision" in text or "- [x] Reject this image decision" in text


def ensure_checklist(rows: list[dict[str, str]], path: Path, *, write_template: bool = False) -> str:
    """Keep a checklist that already exists. Write the unsigned template otherwise.

    Returns ``kept``, ``kept-signed``, or ``written``.
    """
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if approval_is_signed(current):
            return "kept-signed"
        if not write_template:
            return "kept"
    _write_checklist(rows, path)
    return "written"


def _write_checklist(rows: list[dict[str, str]], path: Path) -> None:
    changes = [row for row in rows if row["difference_from_frozen"] != "unchanged"]
    found = tuple(row["image_id"] for row in changes)
    if found != EXPECTED_CHANGES:
        raise SystemExit(f"Unexpected score changes: {found}")
    lines = [
        "# Human adjudication approval checklist",
        "",
        "Status: pending. The boxes below were not signed by the reviewer. An earlier draft marked them approved from the build script. That signature is withdrawn.",
        "",
        "Set A, the frozen automatic brand majority, is 13/18 and does not depend on these boxes. Set B, the post-hoc majority, is 15/18 if every decision below is kept. Approval is pending.",
        "",
        "The Rite tilt change, by itself, does not move the product majority: the other two Rite photographs are already misses. Rejecting both Bonus upgrades, or both Shoop upgrades, would drop that product back to a miss and move Set B from 15/18 to 14/18.",
        "",
        "Reviewer: pending  Date: pending",
        "",
    ]
    for index, row in enumerate(changes, start=1):
        lines.extend(
            [
                f"## {index}. `{row['image_id']}`",
                "",
                f"Frozen result: {row['frozen_brand_success']}. Proposed adjudicated result: {row['adjudicated_brand_success']}. Difference: {row['difference_from_frozen']}.",
                "",
                "Exact identification response:",
                "",
                "```text",
                row["identification_response"].strip(),
                "```",
                "",
                f"Rationale: {row['rationale']}",
                "",
                "- [ ] Approve this image decision",
                "- [ ] Reject this image decision",
                "",
                "Reviewer note:",
                "",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_captions() -> None:
    text = "\n".join(
        [
            "# Figure captions",
            "",
            "## `brand_by_origin.svg`",
            "",
            "Brand successes by origin. Each bar is a count out of 9 primary products. A product counts as correct when at least 2 of its 3 photographs succeed. Apple drinks are excluded. Set A is the frozen matcher: 9/9 international and 4/9 Pakistan-market, 13/18 overall. Set B is the post-hoc reading: 9/9 and 6/9, 15/18 overall. The bars are descriptive counts, not a test that origin caused the difference.",
            "",
            "## `brand_by_capture_set_a.svg`",
            "",
            "Set A brand successes by phone capture condition. Each bar is a count out of 18 primary photographs. Straight 13/18, tilt 15/18, occluded 14/18. The same 18 products appear in every bar, so these are not 54 independent products. Apple drinks are excluded.",
            "",
        ]
    )
    (FIGURES / "CAPTIONS.md").write_text(text, encoding="utf-8")


def main(write_template: bool = False, checklist: Path | None = None) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = _load()
    _grouped_bars()
    _capture_bars()
    outcome = ensure_checklist(rows, checklist or checklist_path(), write_template=write_template)
    _write_captions()
    print(f"draft figures written; approval checklist {outcome}")
    return outcome


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate publication figures.")
    parser.add_argument(
        "--write-approval-template",
        action="store_true",
        help=(
            "Write the unsigned approval template when reports/ADJUDICATION_APPROVAL.md "
            "is missing or still unsigned. A file with a checked decision is kept."
        ),
    )
    arguments = parser.parse_args()
    main(write_template=arguments.write_approval_template)
