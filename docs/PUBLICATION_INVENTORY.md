# Publication inventory

Version 0.1 file list for the Hugging Face Bucket `moneebhussain/RegionalPack-VLM`. The five image decisions in `reports/ADJUDICATION_APPROVAL.md` are approved.

## Hugging Face Bucket layout

The Bucket holds a copy of the release files, not the git history. Photographs stay in `dataset/raw/`. Weights are linked, not copied.

```text
README.md                          introduction and results, from release/v0.1/README.md
DATASET_CARD.md                    this dataset card
dataset/product_manifest.csv
dataset/annotations.csv
dataset/raw/<product_id>/{straight,tilt,occluded}.jpg    60 files
configuration/prompts.yaml
configuration/inference_config.yaml
configuration/eval_config.yaml
configuration/brand_aliases.yaml
configuration/scoring_config.yaml
results/benchmark_predictions.jsonl          180 raw responses, unmodified
results/benchmark_environment.json
results/benchmark_run.md
results/v0.1_summary.md                      frozen score narrative
results/scored_predictions.csv               Set A image rows
results/metrics_summary.json                 Set A aggregates
results/adjudication_decisions.csv           Set B image decisions, labeled post hoc
reports/technical_report.md
reports/ADJUDICATION_APPROVAL.md
reports/REVISED_BENCHMARK_ANALYSIS.md
reports/OCR_METRIC_INTERPRETATION.md
reports/figures/brand_by_origin.svg
reports/figures/brand_by_capture_set_a.svg
reports/figures/CAPTIONS.md
reports/PROMPT_PROVENANCE.md
SHA256SUMS
PUBLICATION_READY_CHECKLIST.md
```

The published copy of this list is `release/v0.1/publication/`.

## GitHub README

The repository README is `README.md`. It states the research question, both brand analyses, and the technical report. The study design I locked before the run is `docs/research_spec.md`.

## Frozen inputs

| File | Why |
|---|---|
| `results/benchmark/benchmark_predictions.jsonl` | Raw model output |
| `data/annotations.csv` | Verified references |
| `prompts/prompts.yaml` | Frozen prompts |
| `configs/scoring_config.yaml` | Frozen scoring rules |
| `configs/brand_aliases.yaml` | Frozen aliases |
| `configs/inference_config.yaml` | Frozen decoding and revision |

## Reproducibility

Scoring the saved responses is documented in `README.md`. Those commands do not load the model. The expected brand majorities are frozen 13/18 and post-hoc 15/18.

## Release scope

The Hugging Face Bucket is [moneebhussain/RegionalPack-VLM](https://huggingface.co/buckets/moneebhussain/RegionalPack-VLM). This inventory does not claim that the GitHub repository is public. I did not fine-tune the model, and I did not run a second model.
