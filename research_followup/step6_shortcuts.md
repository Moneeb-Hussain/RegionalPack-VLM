# Step 6 shortcut baselines

Post-hoc exploratory. This is not a confirmatory result. C stays 1.0. No new permutation test was run. The original 1000-shuffle study was not changed.

Command: `.venv\Scripts\python.exe research_followup\step6_shortcuts.py`

Two tasks, both leave-one-condition-out:

1. Twenty-way: one probe over all 20 products, as in the original study.
2. Matched pairs: nine separate two-class probes. Each probe sees only the international product and the Pakistan-market product in one category. Apple is excluded. Each pair has 6 photos, so the nine pairs have 54 photos. Two classes use the sign rule: a positive score predicts the second class.

The independent-audit claim being checked: Qwen image-token sites 54/54, and the colour histogram 54/54, on these matched pairs.

## Twenty-way

Counts below are a fresh leave-one-condition-out pass on the saved features. Where a step 2 accuracy exists, the fraction matches that saved accuracy.

| site | correct / 60 | correct / 54 primary | matches step 2 |
|---|---:|---:|---|
| llm_img_L1 | 60/60 | 54/54 | True |
| llm_img_L2 | 60/60 | 54/54 | True |
| llm_img_L3 | 60/60 | 54/54 | True |
| llm_img_L4 | 60/60 | 54/54 | True |
| llm_img_L5 | 60/60 | 54/54 | True |
| llm_img_L6 | 60/60 | 54/54 | True |
| llm_img_L7 | 60/60 | 54/54 | True |
| llm_img_L8 | 60/60 | 54/54 | True |
| llm_img_L9 | 60/60 | 54/54 | True |
| llm_img_L10 | 60/60 | 54/54 | True |
| llm_img_L11 | 60/60 | 54/54 | True |
| llm_img_L12 | 60/60 | 54/54 | True |
| llm_img_L13 | 60/60 | 54/54 | True |
| llm_img_L14 | 60/60 | 54/54 | True |
| llm_img_L15 | 60/60 | 54/54 | True |
| llm_img_L16 | 60/60 | 54/54 | True |
| llm_img_L17 | 60/60 | 54/54 | True |
| llm_img_L18 | 60/60 | 54/54 | True |
| llm_img_L19 | 60/60 | 54/54 | True |
| llm_img_L20 | 60/60 | 54/54 | True |
| llm_img_L21 | 60/60 | 54/54 | True |
| llm_img_L22 | 60/60 | 54/54 | True |
| llm_img_L23 | 60/60 | 54/54 | True |
| llm_img_L24 | 60/60 | 54/54 | True |
| llm_img_L25 | 60/60 | 54/54 | True |
| llm_img_L26 | 60/60 | 54/54 | True |
| llm_img_L27 | 60/60 | 54/54 | True |
| llm_img_L28 | 60/60 | 54/54 | True |
| vision_out | 60/60 | 54/54 | True |
| pixel_thumb | 50/60 | 45/54 | True |
| color_hist | 52/60 | 48/54 | True |
| resnet18_pool | 58/60 | 52/54 | published ResNet acc 0.966667, match True |

Image-token sites at 60/60: 28 of 28. Lowest image-token count: 60/60.

## Matched pairs

| site | correct / 54 | photos predicted as the local product |
|---|---:|---:|
| llm_img_L1 | 54/54 | 27 |
| llm_img_L2 | 54/54 | 27 |
| llm_img_L3 | 54/54 | 27 |
| llm_img_L4 | 54/54 | 27 |
| llm_img_L5 | 54/54 | 27 |
| llm_img_L6 | 54/54 | 27 |
| llm_img_L7 | 54/54 | 27 |
| llm_img_L8 | 54/54 | 27 |
| llm_img_L9 | 54/54 | 27 |
| llm_img_L10 | 54/54 | 27 |
| llm_img_L11 | 54/54 | 27 |
| llm_img_L12 | 54/54 | 27 |
| llm_img_L13 | 54/54 | 27 |
| llm_img_L14 | 54/54 | 27 |
| llm_img_L15 | 54/54 | 27 |
| llm_img_L16 | 54/54 | 27 |
| llm_img_L17 | 54/54 | 27 |
| llm_img_L18 | 54/54 | 27 |
| llm_img_L19 | 54/54 | 27 |
| llm_img_L20 | 54/54 | 27 |
| llm_img_L21 | 54/54 | 27 |
| llm_img_L22 | 54/54 | 27 |
| llm_img_L23 | 54/54 | 27 |
| llm_img_L24 | 54/54 | 27 |
| llm_img_L25 | 54/54 | 27 |
| llm_img_L26 | 54/54 | 27 |
| llm_img_L27 | 54/54 | 27 |
| llm_img_L28 | 54/54 | 27 |
| vision_out | 53/54 | 28 |
| pixel_thumb | 47/54 | 26 |
| color_hist | 54/54 | 27 |
| resnet18_pool | 50/54 | 25 |

Image-token sites at 54/54: 28 of 28. Lowest image-token pair count: 54/54.

The 54/54 image-token claim and the 54/54 colour-histogram claim both reproduce under this held-out paired protocol.

Colour histogram ties the Qwen image-token sites at 54/54, so this paired test cannot localise a Qwen-specific effect. It is not at ceiling for every baseline: the 16x16 thumbnail is 47/54, ResNet-18 is 50/54, and the vision tower is 53/54. Misses are listed below.

Misses:

| site | photo | predicted product id |
|---|---|---:|
| vision_out | biscuit_intl/tilt | 3 |
| pixel_thumb | cola_intl/tilt | 1 |
| pixel_thumb | cola_intl/occluded | 1 |
| pixel_thumb | biscuit_intl/straight | 3 |
| pixel_thumb | tea_local/tilt | 4 |
| pixel_thumb | chips_local/tilt | 8 |
| pixel_thumb | toothpaste_local/tilt | 12 |
| pixel_thumb | noodles_local/tilt | 16 |
| resnet18_pool | cola_local/straight | 0 |
| resnet18_pool | cola_local/occluded | 0 |
| resnet18_pool | detergent_intl/tilt | 11 |
| resnet18_pool | detergent_local/occluded | 10 |

## Notes

ResNet-18 is the exploratory control added after the primary results. Its matrix is `linear_probe/results_resnet/features_resnet18.npz`, read only. `resnet_control.py` opens each file with `Image.open` and does not apply the EXIF turn that the Qwen features use. These counts use that saved matrix as-is.

A positive count in the local-product column means the two-class probe predicted the second class. The old one-column argmax rule could not do that.

