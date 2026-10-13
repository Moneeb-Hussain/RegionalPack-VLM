"""Fill visible-text fields for one photograph at a time.

This does not call the model. Blank cells stay blank until you pass them.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.dataset import ANNOTATION_COLUMNS


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ANNOTATION_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pending(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if not row.get("script_present", "").strip()]


def apply_annotation(
    rows: list[dict[str, str]],
    *,
    image_path: str,
    latin: str,
    urdu: str,
    script: str,
    brand_visible: str,
    occlusion: str,
    ambiguous: str,
    notes: str | None = None,
    review_status: str | None = None,
) -> None:
    matches = [row for row in rows if row["image_path"] == image_path]
    if len(matches) != 1:
        raise ValueError(f"Expected one annotation row for {image_path}.")
    row = matches[0]
    row["text_true_latin"] = latin
    row["text_true_urdu"] = urdu
    row["script_present"] = script
    row["brand_visible"] = brand_visible
    row["occlusion_note"] = occlusion
    row["ambiguous_text"] = ambiguous
    if notes is not None:
        row["notes"] = notes
    if review_status is not None:
        if review_status not in ("", "needs_review", "verified"):
            raise ValueError(f"review_status must be blank, needs_review, or verified, not {review_status!r}.")
        row["review_status"] = review_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record visible text for one benchmark photograph.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--image", default="")
    parser.add_argument("--latin", default="")
    parser.add_argument("--urdu", default="")
    parser.add_argument("--script", default="", choices=["", "latin", "urdu", "roman_urdu", "mixed"])
    parser.add_argument("--brand-visible", default="", choices=["", "yes", "no"])
    parser.add_argument("--occlusion", default="")
    parser.add_argument("--ambiguous", default="")
    parser.add_argument("--notes", default=None)
    parser.add_argument(
        "--verified",
        action="store_true",
        help="Set review_status to verified. Omit this flag to leave the existing status unchanged.",
    )
    args = parser.parse_args(argv)
    root = Path(args.root)
    path = root / "data" / "annotations.csv"
    rows = _load(path)
    remaining = pending(rows)
    if not args.image:
        print(f"{len(remaining)} photographs still need visible text.")
        if remaining:
            nxt = remaining[0]
            print("Next image:")
            print(nxt["image_path"])
            print("Open that file, then run:")
            print(
                "python3 -m src.annotate --image "
                + nxt["image_path"]
                + " --script latin --brand-visible yes --latin \"...\" --urdu \"\" --occlusion \"\" --ambiguous \"\""
            )
        return 0
    if not args.script:
        raise SystemExit("Pass --script after you have looked at the photograph. Do not guess it.")
    apply_annotation(
        rows,
        image_path=args.image,
        latin=args.latin,
        urdu=args.urdu,
        script=args.script,
        brand_visible=args.brand_visible,
        occlusion=args.occlusion,
        ambiguous=args.ambiguous,
        notes=args.notes,
        review_status="verified" if args.verified else None,
    )
    _save(path, rows)
    print(f"Saved {args.image}. {len(pending(rows))} photographs still need visible text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
