# Scoring QA

| Check | Result |
|---|---|
| Image rows scored | 60 |
| Prediction slots represented | 180 unique image/prompt pairs in the scored table |
| Primary brand headline | 13/18 (72%) |
| Abstention override flipped a frozen hit | no |
| Punctuation-blocked images left out of the headline | 6 |

Protected files, SHA-256 recorded before the report files were written:

| File | SHA-256 |
|---|---|
| `data/annotations.csv` | `c6f08bea22883a73deab1d7b0da25d9dd9c3a63b45db1397c2ff69c1833a0ca0` |
| `data/product_manifest.csv` | `d87934a53c08d1c063bf3278325272d82aa8538639774368cf4b38ee03f61be2` |
| `results/benchmark/benchmark_predictions.jsonl` | `e80a439149189eb57e8e08e55de112667562bf5939488ceebbceb261d2434eff` |
| `prompts/prompts.yaml` | `d4ec056c82b0855f68e0073dc98e6b4022da4446a2350192ad580296267f635a` |
| `configs/brand_aliases.yaml` | `32376f6bca68da2f9171e28dad2ad9d1d73e268127c7741a6b6e6425388402a2` |
| `configs/scoring_config.yaml` | `1cdbdb9cfa345a62c1cd0335bcad6e835bb0cac4ef0e645ba51e4754b4fba847` |
| `configs/inference_config.yaml` | `97772b76659249c8a102a49bdd3a021bbf9e73ae4a6179684f05906492c0d186` |
| `configs/eval_config.yaml` | `37cfb9af3434586a030403bfda0cac4db7f1a117e86d8608bfbc4a9f660e1c37` |

The analyzer reads those files and writes only `results/scored_predictions.csv`, `results/metrics_summary.json`, `reports/benchmark_analysis.md`, `reports/failure_cases.md`, `reports/scoring_qa.md`, and `reports/figures/`.

Frozen pieces that were not edited for this score: brand aliases, category accept phrases, OCR normalization, prompts, decoding, and the pinned revision.

Headline brand success is `brand_match` from `src/scoring.py`, except an explicit abstention cannot be a success. On these 180 responses that exception changed no frozen hit.
