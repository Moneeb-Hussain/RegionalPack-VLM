# RegionalPack-VLM: Dataset Card

**Researcher:** Moneeb Hussain  
**Version:** 0.1 | October 2026  
**Location of collection:** Lahore, Pakistan  
**Dataset size:** 60 photographs of 20 retail products

## Dataset Overview

RegionalPack-VLM is a small, manually annotated evaluation dataset I collected to investigate how a vision-language model handles product-line identification, packaging-text transcription, and category recognition on retail packaging.

The dataset contains 60 smartphone photographs of 20 branded products available in Lahore, Pakistan. Each product was photographed under three conditions: straight-on, tilted, and partially occluded.

I selected products from internationally distributed and Pakistan-market brands to explore whether differences in packaging recognition appear across these groups. Nine product-category pairs, representing 18 products and 54 photographs, form the primary comparison. One additional apple-drink pair is included for exploratory analysis and excluded from the primary results.

I manually verified the front-of-pack text annotations for all 60 photographs. These annotations serve as evaluation references, not training labels used to fine-tune the model.

The dataset accompanies my RegionalPack-VLM pilot study evaluating Qwen3-VL-2B-Instruct. It is intended to support transparent examination of the reported results, rather than establish a comprehensive benchmark of retail-product recognition.

## Dataset Composition

Twenty products, three photographs each. The primary categories, one internationally distributed product and one Pakistan-market product each, are cola, biscuit, tea, chocolate, chips, detergent, toothpaste, noodles, and soap. The apple-drink pair, `apple_fruita` and `apple_fruitien`, is exploratory.

Each scored image is one JPEG and one row in [`dataset/annotations.csv`](dataset/annotations.csv). In the Git repository those files are `data/annotations.csv` and `data/raw/<product_id>/{straight,tilt,occluded}.jpg`. The product-level record is [`dataset/product_manifest.csv`](dataset/product_manifest.csv) (`data/product_manifest.csv` in the repository).

## Collection Methodology

I photographed the products in Lahore with a phone. The three filenames `straight`, `tilt`, and `occluded` name how I held the phone. They are not a controlled viewpoint or robustness experiment. I am the only photographer and the only annotator.

## Annotation Schema

`brand_true` is the specific product line I score. It is not the parent company. YOU is the product line on the CandyLand bar. Rite is the product line on the Bisconni biscuit. CandyLand and Bisconni are not accepted as `brand_true`.

The photograph table, [`dataset/annotations.csv`](dataset/annotations.csv), contains:

| Field | Meaning |
|---|---|
| `image_path` | Path of the JPEG |
| `product_id` | Folder name under [`dataset/raw/`](dataset/raw/) |
| `category` | Product category, such as cola or biscuit |
| `variant` | Short description of the photographed variant |
| `brand_true` | Specific product line |
| `brand_group` | Group label used in collection: `international`, `pakistan_origin`, or `exploratory` |
| `origin` | `international`, `local`, or `exploratory` |
| `script_present` | Script recorded for the scored reference |
| `text_true_latin` | Verified front-of-pack Latin text |
| `text_true_urdu` | Verified Urdu text. Empty when none is in scope |
| `angle` | `straight`, `tilt`, or `occluded` |
| `occlusion_note` | Note on what the frame hides, when I wrote one |
| `brand_visible` | Whether the product-line name is visible in that frame |
| `ambiguous_text` | Text I saw but did not put in the scored reference |
| `notes` | Other annotation notes |
| `review_status` | `verified` on all 60 rows |

`comparison_role` is not a column of the annotation file. It is on [`dataset/product_manifest.csv`](dataset/product_manifest.csv): `primary` for the 18-product comparison and `exploratory` for the two apple drinks. The scored result tables copy that value onto each image row.

## Ground-Truth Verification

I verified the front-of-pack text on all 60 photographs before inference. The scored reference keeps the visible brand wordmark, the product or flavor name, and a prominent tagline. Prices, nutrition tables, barcodes, and regulatory fine print are outside the reference. I did not delete them from model output after inference. If the model transcribes them, character error rate counts the extra characters.

## Primary and Exploratory Splits

There is no training split. All 60 images are an evaluation set.

| Split | Products | Photographs | How I use it |
|---|---|---|---|
| Primary | 18 | 54 | Brand and category rates |
| Exploratory | 2 apple drinks | 6 | Kept in the release, excluded from the primary rates |

## Intended Uses

A researcher can inspect the photographs, these annotations, the saved model responses, and the evaluation method to study packaging transcription, category recognition, and product-line identification on this shelf.

The set is small and selected. It is not representative of Pakistani retail products. It is not a basis for a causal claim about geographic bias. There is no training split. A result computed after training on these same images would not be a held-out result.

## Limitations

One annotator. Eighteen primary products. Pairs share a category and are not matched variants. Urdu references exist for 9 photographs. Tilt and occlusion are informal phone conditions.

## Licensing and Intellectual Property

I took the photographs. Brand names and pack designs are trademarks of their owners. This card does not grant rights to those marks. The model weights are not included. Use the public Qwen3-VL-2B-Instruct repository at the revision named in the README. There is no LICENSE file. The photographs, annotations, and accompanying text are not placed under an open license.

## Related Resources

| Resource | Path |
|---|---|
| Technical report | [`reports/technical_report.md`](reports/technical_report.md) |
| Bucket README | [`README.md`](README.md) |
| Benchmark prompt | [`configuration/prompts.yaml`](configuration/prompts.yaml) |
| Source repository | [RegionalPack-VLM on GitHub](https://github.com/Moneeb-Hussain/RegionalPack-VLM) |
| Evaluated model | [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |
