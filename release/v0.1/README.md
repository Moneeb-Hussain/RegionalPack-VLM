# RegionalPack-VLM
**When a Vision-Language Model Can Read a Product Name but Cannot Identify It**

What happens when a vision-language model correctly reads a product's name on its packaging, yet still says it cannot identify the brand?

I encountered this behavior while evaluating Qwen3-VL-2B-Instruct on retail products I collected in Lahore, Pakistan. For example, the model correctly transcribed the visible product-line text on CandyLand YOU across three photographs, but its identification responses still declined to name the specific product line.

That discrepancy led me to a broader question: **How reliably can a small vision-language model distinguish between reading packaging text, recognizing a product category, and identifying a specific product line?**

To investigate, I developed **RegionalPack-VLM**, an independent pilot study involving 60 phone photographs of 20 retail products, including internationally distributed and Pakistan-market brands. I manually verified the front-of-pack text annotations and evaluated three tasks: specific product-line identification, packaging transcription, and general category recognition.

For the primary comparison of 18 products, the frozen automatic scoring protocol (Set A) recognized 13 correctly (72.2%). A separately reported post-hoc human adjudication of five image-level decisions (Set B) increased the count to 15 (83.3%). These findings describe a small, selected sample and do not establish that product origin causes differences in model performance.

This Bucket contains the photographs, verified annotations, saved model responses, evaluation results, and technical report supporting the study.

**Researcher:** Moneeb Hussain  
**Release:** Version 0.1 | October 2026  
**Evaluated model:** [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)  
**Technical report:** [Read the technical report](reports/technical_report.md)  
**Source code:** [RegionalPack-VLM on GitHub](https://github.com/Moneeb-Hussain/RegionalPack-VLM)

---

##  Research Motivation: The Blind Spot I Observed

My final-year project was an automated checkout for a moving retail conveyor. The detector was YOLOv4-tiny, and OpenCV handled the camera and the belt. I collected about 3,500 images across 70 products. Unknown products had no dedicated, validated open-set rejection mechanism. A confidence threshold can yield no detection, so the detector was not forced to emit a known class on every frame.

This pilot asks a related question of a model that can transcribe, name a category, and also abstain in language. I am not saying the checkout network and Qwen3-VL fail for the same reason. The comparison is descriptive. It does not show that origin, or pretraining frequency, caused a difference.

##  Systematic Evaluation: Model and Experimental Design

I sent three frozen prompts on every photograph: specific product-line identification, front-of-pack transcription, and general category. That is 180 calls, and all 180 completed. I did not fine-tune the model.

| Item | Setting |
|---|---|
| Model | Qwen/Qwen3-VL-2B-Instruct |
| Revision | `89644892e4d85e24eaac8bacfd4f463576704203`. The loaded snapshot matched this pin |
| Hardware | Tesla T4, float16, SDPA attention |
| Decoding | Temperature 0.1, top_p 1.0, top_k unset, repetition penalty 1.0, seed 0, batch size 1 |
| Prompts | [`configuration/prompts.yaml`](configuration/prompts.yaml) |
| Decoding record | [`configuration/inference_config.yaml`](configuration/inference_config.yaml) |
| Dataset card | [`DATASET_CARD.md`](DATASET_CARD.md) |

`straight`, `tilt`, and `occluded` are phone capture conditions. They are not a controlled viewpoint experiment. The primary rates use 18 products. `apple_fruita` and `apple_fruitien` are exploratory. A product is a brand success when at least 2 of 3 photographs match an accepted product-line alias. A parent-company name is not an accepted answer.

## Key Findings

I report two brand numbers. Set B was decided after I read the saved answers. It does not replace Set A.

| Analysis | Brand | Label |
|---|---|---|
| Set A | 13/18 (72.2%) | Frozen automatic score |
| Set B, post hoc | 15/18 (83.3%) | Post-hoc reading. Not a replacement for Set A |

Set A origin counts: 9/9 international, 4/9 Pakistan-market. Set B, post hoc, origin counts: 9/9 and 6/9. Frozen category accuracy: 14/18. Each origin group is 7/9 on category.

On CandyLand YOU, the front-of-pack transcription matches the verified Latin reference `you / MILK / CHOCOLATE` on all three frames, with Latin character error 0.00, and the identification responses still abstain. That is a disagreement between transcription and identification. It is not evidence that the model failed to read the text, and it is not evidence about why the identification abstained. On Super Crisp, the straight answer quotes “Super Crisp” and then says the brand is not identifiable. Latin median character error is 0.69. The mean is 5.22 because the Knorr tilt transcription repeats and reaches character error 221. Urdu is 9 images. My reference excludes prices and tiny regulatory text, so extra model text raises the error. The interpretation is in [`reports/OCR_METRIC_INTERPRETATION.md`](reports/OCR_METRIC_INTERPRETATION.md).

##  Proposed Path Forward

I did not fine-tune the model on these photographs, and I have not built a validated external abstention gate. The study I would run next, and have not run, would curate matched photographs of Pakistani product lines and of less globally familiar non-Pakistani product lines, record transcription and identification separately, test an external image-text check as an abstention gate, and score both correct recognition and safe abstention on held-out products.

## Release Contents

| What it records | Path |
|---|---|
| 180 raw responses | [`results/benchmark_predictions.jsonl`](results/benchmark_predictions.jsonl) |
| Frozen image scores, Set A | [`results/scored_predictions.csv`](results/scored_predictions.csv) |
| Frozen aggregates | [`results/metrics_summary.json`](results/metrics_summary.json) |
| Post-hoc image decisions, Set B | [`results/adjudication_decisions.csv`](results/adjudication_decisions.csv) |
| Technical report | [`reports/technical_report.md`](reports/technical_report.md) |
| Prompt provenance | [`reports/PROMPT_PROVENANCE.md`](reports/PROMPT_PROVENANCE.md) |
| Origin figure, Set A and Set B | [`reports/figures/brand_by_origin.svg`](reports/figures/brand_by_origin.svg) |
| Set A capture figure | [`reports/figures/brand_by_capture_set_a.svg`](reports/figures/brand_by_capture_set_a.svg) |
| Five Set B image decisions I approved on 12 October 2026 | [`reports/ADJUDICATION_APPROVAL.md`](reports/ADJUDICATION_APPROVAL.md) |
| Human-verified annotations | [`dataset/annotations.csv`](dataset/annotations.csv) |
| Product manifest | [`dataset/product_manifest.csv`](dataset/product_manifest.csv) |
| 60 photographs | [`dataset/raw/`](dataset/raw/) |
| Frozen scoring rules | [`configuration/scoring_config.yaml`](configuration/scoring_config.yaml) |
| Run environment | [`results/benchmark_environment.json`](results/benchmark_environment.json) |
| Frozen score narrative | [`results/v0.1_summary.md`](results/v0.1_summary.md) |
| Revised analysis, Set A and Set B | [`reports/REVISED_BENCHMARK_ANALYSIS.md`](reports/REVISED_BENCHMARK_ANALYSIS.md) |
| Figure captions | [`reports/figures/CAPTIONS.md`](reports/figures/CAPTIONS.md) |
| SHA-256 manifest | [`SHA256SUMS`](SHA256SUMS) |

## Dataset and Reproducibility

The 60 images are an evaluation set. There is no training split. Scoring a model on these photographs after training on them would not be a held-out result. The commands that rescore the saved responses, without loading the model, are in the GitHub README. Brand names and pack designs are trademarks of their owners. This Bucket does not grant rights to those marks. The photographs, annotations, and text are not placed under an open license.

## Limitations

One model, one annotator, 18 primary products. Pairs are not matched variants, and the international name is also the more familiar one, so the origin counts are descriptive. The frozen matcher misses some explicit Bonus Tristar and Shoop answers because a quotation mark sticks to the word. That is why Set B exists, and why Set B is labeled post hoc. No significance test.
