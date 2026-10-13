# Changes fixed before the benchmark

Phase 0 on the Colab T4 is unchanged. Qwen3-VL-2B-Instruct loaded in float16, and all 22 inference calls completed. The Colab files `results/phase0_verification.md`, `results/phase0_predictions.jsonl`, and `results/environment.json` are the Phase 0 record. This document does not rewrite them.

## 1. Bfloat16 reporting

Tesla T4 is Turing, compute capability 7.5. Hardware bfloat16 starts at capability 8.0. The weights are stored as BF16, and the loader still casts them to float16.

PyTorch `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True`. On a T4 that call can return true even though the GPU has no bfloat16 hardware. The old report column used that flag under the label "bfloat16 hardware support."

The probe now stores:

| Field | Meaning on a T4 |
|---|---|
| `bf16_hardware_supported` | false |
| `bf16_pytorch_including_emulation` | whatever PyTorch returns, often true |
| `bf16_supported` | same as hardware support, false |

A downloaded Phase 0 report that shows bfloat16 support as true is the emulation flag. It does not mean the T4 ran bfloat16. The successful load was float16.

## 2. Front-of-pack transcription prompt

Three texts are preserved. None of them was edited for this note.

| File | Role | Wording |
|---|---|---|
| `docs/research_spec.md` | Historical specification. Left unchanged, including the byte copy under `RegionalPack-Cursor-Spec/docs/`. | Prompt 2 still reads: "Transcribe all text visible on the packaging exactly as written, including any text that is not in English. If no text is legible, say so." |
| `prompts/phase0_prompts.yaml` | Prompt actually sent in Phase 0. | The same all-text sentence. Phase 0 responses stay attached to it. |
| `prompts/prompts.yaml` | Prompt actually sent in the 180-call benchmark. | Front-of-pack only: brand wordmark, product or flavor, and prominent tagline. Ingredient lists, nutrition tables, barcodes, and regulatory fine print are excluded. |

The specification's scoring rubric already limited the scored reference to that front-of-pack region. The Prompt 2 string in the specification was not updated to match. The benchmark prompt was fixed before evaluation, in `prompts/prompts.yaml`, and was not changed after the responses were saved. Identification and category match the specification text. The same record is in `reports/PROMPT_PROVENANCE.md`.

## 3. Pinned snapshot

`configs/inference_config.yaml` pins `89644892e4d85e24eaac8bacfd4f463576704203`.

On 2026-10-11 the Hugging Face model API for `Qwen/Qwen3-VL-2B-Instruct` at that revision returned the same sha. The loader now reads `config._commit_hash` after `from_pretrained` and stops if the loaded commit is missing or different. It still does not load the fallback model.

## Phase 1A

The list is approved for a shelf check and is not locked for photography. The store sheet is `data/availability_checklist.md`.

Grouping uses one rule per product line: `pakistan_market_line`, or else `sold_under_same_name_outside_pakistan`. A line cannot be in both. Accepted Prompt 1 spellings are fixed in `configs/brand_aliases.yaml`. Phase 0 filenames are the six names in `RegionalPack-Phase0-Labeled-Photos`.

## Pre-collection category change

The milk pair, Almarai and Olper's, is replaced by a plain milk-chocolate pair before any benchmark photograph:

| Id | Pack |
|---|---|
| `chocolate_intl` | Cadbury Dairy Milk plain bar, preferably 16.5 g |
| `chocolate_local` | CandyLand YOU Creamiest plain bar, preferably 21 g |

The reason is cost and availability. Model performance did not decide the change. The dataset stays 10 categories, 20 products, and 60 photographs.

The collected packs are now photographed. See the photographed-set section.

## Collected-set update

The manifest now names the packs in hand. This update is not based on model performance.

| Pair | Collected products | Recorded limitation |
|---|---|---|
| Toothpaste | Colgate Fresh Gel; Medicam UltraFresh Gel 3-in-1 | Gel claims differ |
| Chips | Lay's Classic; Super Crisp BBQ | Flavors differ |
| Noodles | Knorr Chicken; Shan Shoop Chatpatta | Flavors differ; both are instant noodles |
| Apple drinks | Fruita Vitals Apple; Fruitien Joy Apple | Exploratory. Outside the primary comparison |

Slice, Shezan, Maggi, and Kolson are no longer in the set. The other 16 products stay. The photo count stays 20 products and 60 frames. A category may be photographed in its own place. Both products in a pair share the light and the background.

## Phase 1B

`src/collect.py` prepares `data/raw/<product_id>/` folders, copies camera JPEGs without re-encoding, and writes a collection report. `data/annotations.csv` carries manifest identity. Visible text stays blank until a person fills it. The model is not run.

## Photographed set

`raw.zip` is the image source. 60 JPEGs were copied without re-encoding. `juice_intl` and `juice_local` in that zip are stored as `apple_fruita` and `apple_fruitien`. The apple pair stays exploratory.

Printed variants now on the manifest, from the photographs, not from a model:

| Pair | Photographed packs | Recorded limitation |
|---|---|---|
| Biscuits | Oreo chocolate cream 4-pack; Rite rich chocolate biscuits with vanilla cream | Fillings differ. Bisconni is not an alias. |
| Chocolate | Cadbury Dairy Milk plain bar, wrapper prints 1 serving = 27 g; YOU milk chocolate | YOU is an accepted brand identification. |
| Soap | Lux Vibrant Glow Tangerine; Capri Soothing Glow | Fragrances differ. |
| Apple drinks | Fruita Vitals apple nectar; Fruitien Joy apple fruit drink | Exploratory. Drink types differ. |

The inference loader applies EXIF orientation in memory and does not rewrite the JPEG files.
