# RegionalPack-VLM
**When a Vision-Language Model Can Read a Product Name but Cannot Identify It**

What happens when a vision-language model correctly reads a product's name on its packaging, yet still says it cannot identify the brand?

I encountered this behavior while evaluating Qwen3-VL-2B-Instruct on retail products I collected in Lahore, Pakistan. For example, the model correctly transcribed the visible product-line text on CandyLand YOU across three photographs, but its identification responses still declined to name the specific product line.

That discrepancy led me to a broader question: **How reliably can a small vision-language model distinguish between reading packaging text, recognizing a product category, and identifying a specific product line?**

To investigate, I developed **RegionalPack-VLM**, an independent pilot study involving 60 phone photographs of 20 retail products, including internationally distributed and Pakistan-market brands. I manually verified the front-of-pack text annotations and evaluated three tasks: specific product-line identification, packaging transcription, and general category recognition.

For the primary comparison of 18 products, the frozen automatic scoring protocol (Set A) recognized 13 correctly (72.2%). A separately reported post-hoc human adjudication of five image-level decisions (Set B) increased the count to 15 (83.3%). These findings describe a small, selected sample and do not establish that product origin causes differences in model performance.

This repository contains the evaluation code, experimental configuration, methodology, scoring procedures, and technical documentation. The photographs, verified annotations, and saved evaluation artifacts are available in the accompanying [Hugging Face Bucket](https://huggingface.co/buckets/moneebhussain/RegionalPack-VLM).

**Researcher:** Moneeb Hussain  
**Release:** Version 0.1 | October 2026  
**Evaluated model:** [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)  
**Technical report:** [Read the technical report](reports/technical_report.md)  
**Dataset card:** [Read the dataset card](docs/DATASET_CARD.md)

---

##  Research Motivation: The Blind Spot I Observed

My final-year project was an automated checkout for a moving retail conveyor. The detector was YOLOv4-tiny, and OpenCV handled the camera and the belt. I collected about 3,500 images across 70 products. Inside that fixed list the system could name a pack. Unknown products had no dedicated, validated open-set rejection mechanism. A confidence threshold can yield no detection, so the detector was not forced to emit a known class on every frame. What that project lacked was a tested refusal for an unfamiliar pack.

I then asked a related question of a small vision-language model, using packages from shelves in Pakistan. A model can read the front of a pack, name a general category, and still not commit to the specific product line, or it can name the parent company instead. I am not saying the checkout network and Qwen3-VL fail for the same reason. Eighteen products, unmatched pack designs, and no measurement of how often a name appeared in pretraining are not enough for me to treat product origin as the cause of any difference.

##  Systematic Evaluation: Model and Experimental Design

I photographed 20 products, three phone frames each, and sent three frozen prompts on every frame: identification, front-of-pack transcription, and category. That is 180 calls. All 180 completed. The primary rates use 18 products. The apple-drink pair is exploratory. A product is a brand success when at least 2 of 3 photographs match an accepted product-line alias. Parent-company names are metadata, not aliases. The longer account is [reports/technical_report.md](reports/technical_report.md).

| Item | What I fixed before the run |
|---|---|
| Photographs | 60. 20 products, each with a straight, tilt, and occluded phone photo |
| Primary comparison | 18 products, 9 category pairs. The two apple drinks stay exploratory |
| Prompts | Identification, front-of-pack transcription, and category, in [prompts/prompts.yaml](prompts/prompts.yaml) |
| Decoding | Temperature 0.1, top_p 1.0, top_k unset, repetition penalty 1.0, seed 0, batch size 1 |
| Load | Qwen/Qwen3-VL-2B-Instruct, revision `89644892e4d85e24eaac8bacfd4f463576704203`, float16 on a Tesla T4, SDPA. The loaded snapshot matched the pin |
| OCR | Character error rate on the front-of-pack text I verified. Latin and Urdu are separate |

`straight`, `tilt`, and `occluded` are phone capture conditions. They are not a controlled viewpoint experiment. I did not fine-tune the model. The weights are not in this repository.

## Key Findings

I report two brand numbers. They answer different scoring rules. Set B was decided after I read the saved answers. It does not replace Set A.

| Analysis | Brand, 18 products | What it is |
|---|---|---|
| Set A | 13/18 (72.2%) | Frozen automatic matcher. Wilson 95% interval 49%–88% |
| Set B, post hoc | 15/18 (83.3%) | My post-hoc reading of the same answers. Wilson 95% interval 61%–94% |

Set A by origin: 9/9 international products and 4/9 Pakistan-market products. Set B, post hoc, by origin: 9/9 and 6/9. Frozen category accuracy is 14/18 (77.8%). International and Pakistan-market category majorities are each 7/9. These are counts for this shelf.

On CandyLand YOU, the front-of-pack transcription matches the verified Latin reference `you / MILK / CHOCOLATE` on all three frames, with Latin character error 0.00, and the identification answer still says the specific brand cannot be identified. On Super Crisp, the straight-frame identification quotes “Super Crisp” and then says the brand is not clearly identifiable. The transcription contains the brand line on all three Super Crisp frames.

Bisconni Rite is a different miss. The transcription includes both Bisconni and Rite, with Latin character error 0.00, and the identification assigns the brand role to Bisconni or to the spelling Biscotti. Under the product-line rule, the parent name is not an accepted answer. Rite is a miss in Set A and in Set B.

Latin character error on 60 images has median 0.69 and mean 5.22. The mean includes one repeated Knorr tilt transcription, character error 221. Urdu character error is computed on 9 images. The reference excludes prices, nutrition tables, and tiny regulatory lines, so extra model text raises the error even when the brand words were read. Details are in [reports/OCR_METRIC_INTERPRETATION.md](reports/OCR_METRIC_INTERPRETATION.md).

| What it records | Path |
|---|---|
| 180 raw responses | [results/benchmark/benchmark_predictions.jsonl](results/benchmark/benchmark_predictions.jsonl) |
| Frozen image scores, Set A | [results/scored_predictions.csv](results/scored_predictions.csv) |
| Frozen aggregates | [results/metrics_summary.json](results/metrics_summary.json) |
| Post-hoc image decisions, Set B | [results/adjudication_decisions.csv](results/adjudication_decisions.csv) |
| Technical report | [reports/technical_report.md](reports/technical_report.md) |
| Set A and Set B origin figure | [reports/figures/brand_by_origin.svg](reports/figures/brand_by_origin.svg) |
| Set A capture-condition figure | [reports/figures/brand_by_capture_set_a.svg](reports/figures/brand_by_capture_set_a.svg) |
| Five post-hoc decisions I approved on 12 October 2026 | [reports/ADJUDICATION_APPROVAL.md](reports/ADJUDICATION_APPROVAL.md) |

##  Proposed Path Forward

I did not fine-tune the model, and I have not tested an external abstention gate. The four steps below are the study I would run next.

1. **Matched curation.** Photograph Pakistani product lines and product lines that are also less familiar globally but are not Pakistani, under the same three phone conditions. The present pairs share a category and differ in flavor, filling, fragrance, and pack design, so they cannot separate origin from familiarity or from the package.
2. **Separate the decision.** Record front-of-pack transcription, a candidate product-line name, and abstention as three outputs. CandyLand YOU is the reason: the transcription matches the reference, and the identification still abstains.
3. **An external evidence check.** A calibrated image-text similarity score, or another check that is not the model’s own sentence “I cannot identify the brand,” is a candidate gate for abstention. I have not built that gate.
4. **Held-out evaluation.** Score both correct product-line recognition and safe abstention on products that were not used to choose the rule. These 60 photographs would stay out of any fine-tune and out of the set used to pick the gate.

## Reproducing the Evaluation

These commands score the saved responses. They do not load the model. A clean environment needs its own virtualenv: the scoring scripts import PyYAML, and the tests import Pillow.

```bash
python3 -m venv .venv
.venv/bin/pip install "pyyaml>=6.0.1" "pillow>=11.0" pytest
.venv/bin/python -m pytest tests -q
.venv/bin/python -m src.score_release
.venv/bin/python scripts/analyze_benchmark.py
.venv/bin/python scripts/adjudicate_benchmark.py
.venv/bin/python scripts/prepare_publication_draft.py
```

`src.score_release` rewrites [results/benchmark/v0.1_summary.md](results/benchmark/v0.1_summary.md) from the saved JSONL. The analyzer recomputes Set A. The adjudication script recomputes Set B from the image rows. Set B is 15/18, post hoc, with the five image decisions I approved on 12 October 2026. Figure regeneration leaves [reports/ADJUDICATION_APPROVAL.md](reports/ADJUDICATION_APPROVAL.md) in place once that file exists. The saved responses, annotations, prompts, and scoring configs are inputs.

## Limitations

One model, one annotator, and 18 primary products. Several pairs differ in flavor, filling, gel claim, or fragrance, and the more globally familiar name is also the international one. I do not treat product origin or pretraining frequency as the cause of the counts above. The frozen matcher does not treat a comma or a quotation mark as a word boundary, which is why some explicit Bonus Tristar and Shoop answers are read again in Set B. No significance test is reported. The photographs and pack artwork are not released under an open license. Brand names remain their owners’ trademarks. This is an independent pilot. It has not been peer reviewed.

## Repository Layout

Paths below are relative to this repository. The Bucket page uses a different layout. Its README source is [release/v0.1/README.md](release/v0.1/README.md).

| Path | Role |
|---|---|
| [docs/research_spec.md](docs/research_spec.md) | Study design I locked before the run. The transcription sentence in that file is the older wording. The prompt I sent is in `prompts/prompts.yaml`. |
| [prompts/prompts.yaml](prompts/prompts.yaml) | Prompts used in the 180-call run |
| [reports/technical_report.md](reports/technical_report.md) | Technical report |
| [reports/figures/](reports/figures/) | Figures and captions |
| [docs/DATASET_CARD.md](docs/DATASET_CARD.md) | Dataset card. In the Bucket this file is `DATASET_CARD.md`. |
| [docs/PUBLICATION_INVENTORY.md](docs/PUBLICATION_INVENTORY.md) | File inventory for the Bucket |
