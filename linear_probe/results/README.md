# Linear-probe study: results (RegionalPack-VLM, Qwen3-VL-2B-Instruct)

Pre-specified in `../PROTOCOL.md` (committed before the run). One model, 60 photos, 20 products, 3 capture conditions (straight, tilt, occluded). 54 photos belong to the 18 primary products; on 12 of them the model's frozen Set A answer was wrong.

**Question.** When the model's identification answer is wrong, is the product's identity still linearly decodable from its hidden states, on photos from a condition the probe never saw?

## Headline
Yes, but this is not evidence about Qwen. Identity is decodable at ceiling from Qwen's states, from its vision tower, and from an unrelated ImageNet ResNet-18, and a colour histogram alone recovers most of the same photos. In this closed-set design linear decodability is not diagnostic of what the model knows.

## Primary results (leave-one-condition-out, 1000 permutations, alpha = 0.01)

| Site | Accuracy [95% CI over products] | Null mean | p | Model-wrong photos recovered |
|---|---|---|---|---|
| vision_out | 1.00 [1.00, 1.00] | 0.05 | 0.001 | 12 / 12 |
| llm_img, layer 14 | 1.00 [1.00, 1.00] | 0.05 | 0.001 | 12 / 12 |
| llm_last, layer 14 | 0.97 [0.92, 1.00] | 0.05 | 0.001 | 12 / 12 |
| llm_img, layer 28 | 1.00 [1.00, 1.00] | 0.05 | 0.001 | 12 / 12 |
| llm_last, layer 28 | 1.00 [1.00, 1.00] | 0.05 | 0.001 | 12 / 12 |

p = 0.001 is the smallest value 1000 permutations can give.

## Controls
| Control | Accuracy [95% CI] | Model-wrong photos recovered | Note |
|---|---|---|---|
| pixel thumbnail (16x16) | 0.83 [0.73, 0.93] | 9 / 12 | pre-specified |
| colour histogram | 0.87 [0.77, 0.95] | 10 / 12 | pre-specified |
| ResNet-18 (ImageNet), pooled | 0.97 [0.92, 1.00] | 12 / 12 | **exploratory, added after the primary results were seen** |

## Outcome against the protocol
- **H1 (decodable):** met numerically. Both interpretation guards in the protocol triggered: `vision_out` is as decodable as the LLM sites, and the pixel and colour controls are high. H1 is therefore not informative about the language model.
- **H2 (recoverable when the model is wrong):** probe recovers 12 of 12. Colour histogram recovers 10 of 12 and ResNet-18 12 of 12, so the result does not separate Qwen's states from generic image features. The probe also had supervision on these 20 products, while the model had to name them zero-shot.
- **H3 (direction over distance):** across the five primary sites the loss of probe margin tracks alignment (Spearman 0.94 to 0.99) much more than shift size (-0.04 to 0.34), on 40 shift pairs (20 products x 2 shifts, not independent). Margin drop = |g| |d| alignment is an identity, so this describes how large the alignment values are; it is not a discovery. Bootstrap intervals are in `geometry_summary.json`.
- **H4 (exploratory, 7 model-failure pairs of 40):** no consistent predictor. AUROC for alignment ranges 0.39 to 0.81 and for distance 0.26 to 0.74 across sites; the Fisher-style proxy is near chance (0.45 to 0.65) and its ranking still changes with softmax temperature. Inconclusive.

## Exploratory layer sweep (100 permutations per site)
`llm_img` is 1.00 at every layer from 1 to 27. `llm_last` (the decision position) rises from 0.75 at layer 1 and 0.90 at layer 10 to 1.00 from layer 15. This is a descriptive pattern, not a confirmatory result.

## What this supports
- Identity information is present linearly in Qwen's states, including on photos where its answer was wrong.
- The same information is equally available in the vision tower and in an unrelated ImageNet model, so the failures are not explained by a lack of visual identity information in the features. The locus is later (readout or abstention). This is consistent with the earlier transcription-versus-identification gap; it is not shown here.

## What this does not support
- That Qwen "knows" or "uses" the identity. Linear decodability is not use.
- Anything about unseen products: every test product also appears in training under the other conditions (closed-set).
- Any causal statement about geometry (no intervention was run).

## Limits
60 photos, 20 products, one model, one annotator, three conditions that mix viewpoint and occlusion. Failures include abstentions. Probe supervision differs from the model's zero-shot task. The Fisher-style quantity is a proxy inspired by Fisher-IRG, not an implementation of it.

## Implementation notes
- Probes are fitted in the row space of the training data (`fit_probe` in `probe_analysis.py`). With more features than samples this gives the same optimum as the full fit; checked to 5e-14 at tight tolerance on synthetic data, unit tests pass. It was introduced for speed.
- Image preprocessing matches the frozen evaluation (min_pixels 65536, max_pixels 1003520; photo grid 54 x 72, pixel_values_shape [3888, 1536] checked against the saved evaluation record).
- Run with BLAS threads limited to 1 (`OMP_NUM_THREADS=1`) on Colab; otherwise thread contention slowed each fit about 10x.
- Primary sites and controls used 1000 permutations (no deviation from the protocol); exploratory layers used 100.

## Reproduce
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u run_study.py --features features.npz --manifest manifest.csv --out results --n-perm 1000
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u resnet_control.py --manifest manifest.csv --out results_resnet --n-perm 1000
```
