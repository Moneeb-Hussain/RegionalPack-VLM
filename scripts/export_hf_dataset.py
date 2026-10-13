"""Build release/v0.1/hf_dataset from the verified photographs and annotations.

The script copies each JPEG byte for byte. It does not resize images, edit
annotations, or read model outputs.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "release" / "v0.1" / "hf_dataset"
ANNOTATIONS = ROOT / "data" / "annotations.csv"
MANIFEST = ROOT / "data" / "product_manifest.csv"
RAW = ROOT / "data" / "raw"

ANGLE_ORDER = {"straight": 0, "tilt": 1, "occluded": 2}
EXPLORATORY = {"apple_fruita", "apple_fruitien"}

METADATA_FIELDS = [
    "file_name",
    "product_id",
    "angle",
    "comparison_role",
    "origin",
    "brand_true",
    "text_true_latin",
    "text_true_urdu",
    "review_status",
    "category",
    "variant",
    "brand_group",
    "script_present",
    "occlusion_note",
    "brand_visible",
    "ambiguous_text",
    "notes",
    "image_path",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows() -> list[dict[str, str]]:
    annotations = _read_csv(ANNOTATIONS)
    manifest = {row["product_id"]: row for row in _read_csv(MANIFEST)}
    if len(annotations) != 60:
        raise SystemExit(f"expected 60 annotations, found {len(annotations)}")
    if len(manifest) != 20:
        raise SystemExit(f"expected 20 manifest rows, found {len(manifest)}")

    built = []
    seen = set()
    for row in annotations:
        product_id = row["product_id"]
        angle = row["angle"]
        key = (product_id, angle)
        if key in seen:
            raise SystemExit(f"duplicate photograph: {product_id}/{angle}")
        seen.add(key)
        if product_id not in manifest:
            raise SystemExit(f"missing manifest row: {product_id}")
        if row["review_status"] != "verified":
            raise SystemExit(f"unverified annotation: {product_id}/{angle}")
        if angle not in ANGLE_ORDER:
            raise SystemExit(f"unexpected angle: {angle}")
        source = ROOT / row["image_path"]
        if source != RAW / product_id / f"{angle}.jpg":
            raise SystemExit(f"unexpected image path: {row['image_path']}")
        if not source.is_file():
            raise SystemExit(f"missing photograph: {source}")
        role = manifest[product_id]["comparison_role"]
        if product_id in EXPLORATORY and role != "exploratory":
            raise SystemExit(f"{product_id} is not exploratory in the manifest")
        if product_id not in EXPLORATORY and role != "primary":
            raise SystemExit(f"{product_id} is not primary in the manifest")
        record = {
            "file_name": f"images/{product_id}_{angle}.jpg",
            "comparison_role": role,
        }
        for field in METADATA_FIELDS:
            if field in record:
                continue
            if field not in row:
                raise SystemExit(f"missing annotation field: {field}")
            record[field] = row[field]
        record["text_true_urdu"].encode("utf-8").decode("utf-8")
        built.append(record)

    built.sort(key=lambda item: (item["product_id"], ANGLE_ORDER[item["angle"]]))
    products = {item["product_id"] for item in built}
    primary = {item["product_id"] for item in built if item["comparison_role"] == "primary"}
    exploratory = {item["product_id"] for item in built if item["comparison_role"] == "exploratory"}
    if products != {item["product_id"] for item in annotations}:
        raise SystemExit("product set changed while building metadata")
    if len(primary) != 18 or exploratory != EXPLORATORY:
        raise SystemExit(f"comparison roles are {len(primary)} primary and {exploratory}")
    return built


def write_readme() -> None:
    text = r'''---
pretty_name: RegionalPack-VLM Retail Packaging Recognition Evaluation Dataset
license: other
task_categories:
- image-to-text
language:
- en
- ur
size_categories:
- n<1K
annotations_creators:
- expert-generated
language_creators:
- expert-generated
source_datasets:
- original
multilinguality:
- multilingual
tags:
- packaging
- evaluation
---

# RegionalPack-VLM: Retail Packaging Recognition Evaluation Dataset

**Researcher:** Moneeb Hussain  
**Version:** 0.1 | October 2026  
**Location of collection:** Lahore, Pakistan  
**Dataset size:** 60 photographs of 20 retail products

RegionalPack-VLM is a small, manually annotated image evaluation dataset I collected in Lahore, Pakistan to investigate product-line identification, packaging-text transcription, and category recognition in vision-language models.

It contains 60 smartphone photographs of 20 branded retail products, each captured straight-on, tilted, and partially occluded. I manually verified the front-of-pack text annotations for every photograph.

Nine international/Pakistan-market category pairs, representing 18 products, form the primary comparison. One additional apple-drink pair is exploratory. The collection is intended for transparent evaluation and inspection of model behavior, not as a representative benchmark of Pakistani retail products.

## Dataset Composition

Twenty products, three photographs each. The primary categories, one internationally distributed product and one Pakistan-market product each, are cola, biscuit, tea, chocolate, chips, detergent, toothpaste, noodles, and soap. The apple-drink pair, `apple_fruita` and `apple_fruitien`, is exploratory.

Each row of `metadata.csv` points to one JPEG in `images/`. The filename is `<product_id>_<angle>.jpg`. The `file_name` value is the path from this dataset root, for example `images/cola_intl_straight.jpg`.

The Hugging Face image loader names this single collection `train`. That name is the library default for a folder that does not define splits. It is not a training partition. This export does not define validation or test splits.

## Collection Methodology

I photographed the products in Lahore with a phone. The three conditions `straight`, `tilt`, and `occluded` name how I held the phone. They are not a controlled viewpoint or robustness experiment. I am the only photographer and the only annotator.

## Annotation Schema

`brand_true` is the specific product line. It is not the parent company. YOU is the product line on the CandyLand bar. Rite is the product line on the Bisconni biscuit. CandyLand and Bisconni are not stored as `brand_true`.

`comparison_role` is copied from the product manifest, where I recorded it. It is `primary` for 18 products and `exploratory` for the two apple drinks. The photograph annotation file does not contain that column.

| Field | Meaning |
|---|---|
| `file_name` | Path of the JPEG in this dataset |
| `product_id` | Product identifier |
| `angle` | `straight`, `tilt`, or `occluded` |
| `comparison_role` | `primary` or `exploratory`, from the product manifest |
| `origin` | `international`, `local`, or `exploratory` |
| `brand_true` | Specific product line |
| `text_true_latin` | Verified front-of-pack Latin text |
| `text_true_urdu` | Verified Urdu text. Empty when none is in scope |
| `review_status` | `verified` on all 60 rows |
| `category` | Product category, such as cola or biscuit |
| `variant` | Short description of the photographed variant |
| `brand_group` | Collection group: `international`, `pakistan_origin`, or `exploratory` |
| `script_present` | Script recorded for the scored reference |
| `occlusion_note` | Note on what the frame hides, when I wrote one |
| `brand_visible` | Whether the product-line name is visible in that frame |
| `ambiguous_text` | Text I saw but did not put in the scored reference |
| `notes` | Other annotation notes |
| `image_path` | Original path in the research repository, `data/raw/<product_id>/<angle>.jpg` |

## Ground-Truth Verification

I verified the front-of-pack text on all 60 photographs before any model was run. The scored reference keeps the visible brand wordmark, the product or flavor name, and a prominent tagline. Prices, nutrition tables, barcodes, and regulatory fine print are outside the reference.

## Primary and Exploratory Comparison

There is no training split. All 60 images are an evaluation set.

| Comparison | Products | Photographs | How I use it |
|---|---|---|---|
| Primary | 18 | 54 | The paired comparison |
| Exploratory | 2 apple drinks | 6 | Kept in the collection and excluded from the primary comparison |

## Intended Uses

A researcher can inspect the photographs and these annotations to study packaging transcription, category recognition, and product-line identification on this shelf.

The set is small and selected. It is not representative of Pakistani retail products. It is not a basis for a causal claim about geographic bias. There is no training split. A result computed after training on these same images would not be a held-out result.

## Limitations

One annotator. Eighteen primary products. Pairs share a category and are not matched variants. Urdu references exist for 9 photographs. Tilt and occlusion are informal phone conditions.

## Licensing and Intellectual Property

I took the photographs. Brand names and pack designs are trademarks of their owners. This card does not grant rights to those marks. The `license: other` field means this dataset is not released under a standard open license. The photographs, annotations, and accompanying text are not placed under an open license. Model weights are not included.

## Related Resources

| Resource | Where |
|---|---|
| Research repository | [RegionalPack-VLM on GitHub](https://github.com/Moneeb-Hussain/RegionalPack-VLM) |
| Photographs, responses, and technical report | [RegionalPack-VLM Bucket](https://huggingface.co/buckets/moneebhussain/RegionalPack-VLM) |
| Evaluated model | [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |

The technical report is `reports/technical_report.md` in the GitHub repository and in the Bucket. This dataset repository contains the photographs and the verified metadata only.
'''
    (DEST / "README.md").write_text(text, encoding="utf-8")


def export() -> Path:
    records = rows()
    images = DEST / "images"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir(parents=True)
    for record in records:
        source = ROOT / record["image_path"]
        target = DEST / record["file_name"]
        target.write_bytes(source.read_bytes())
        if _sha256(target) != _sha256(source):
            raise SystemExit(f"copy changed bytes: {record['file_name']}")
    copied = sorted(path.relative_to(DEST).as_posix() for path in images.glob("*.jpg"))
    named = [record["file_name"] for record in records]
    if copied != sorted(named):
        raise SystemExit("image files and metadata rows do not match")
    with (DEST / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    write_readme()
    print(f"wrote {DEST} ({len(records)} rows)")
    return DEST


if __name__ == "__main__":
    export()
