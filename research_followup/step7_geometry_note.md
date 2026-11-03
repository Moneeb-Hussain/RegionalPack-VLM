# Step 7 geometry note

Do not treat the geometry results as a new finding. The points below are checked from the saved files. Nothing in `linear_probe/results/` was changed.

Command: `.venv\Scripts\python.exe research_followup\step7_geometry_note.py`

## Margin drop is an identity

For a shift `d` and a margin direction `g`, the drop in probe margin is exactly `|g| |d| alignment`. A high correlation between drop and alignment is that formula, not a separate discovery. The question left by the protocol is only how large the alignment values are.

Recomputed as the largest absolute gap between `drop` and `g_norm * dist * align` in each saved geometry table:

| site | pairs | largest gap |
|---|---:|---:|
| vision_out | 40 | 3.819e-13 |
| llm_img_L14 | 40 | 3.277e-13 |
| llm_last_L14 | 40 | 3.846e-13 |
| llm_img_L28 | 40 | 4.245e-13 |
| llm_last_L28 | 40 | 2.589e-13 |

The largest gap is rounding error, around 1e-13.

## The Fisher-style number is not Fisher-IRG

The code builds a local score from the probe's own softmax. The protocol calls that a proxy inspired by Fisher-IRG, not an implementation of Fisher-IRG. It is not a verified predictor of the model's mistakes. On the same 7 failure pairs, the AUROC for that proxy crosses both sides of 0.5 depending on the site:

| site | AUROC alignment | AUROC distance | AUROC Fisher-style |
|---|---:|---:|---:|
| vision_out | 0.810 | 0.420 | 0.602 |
| llm_img_L14 | 0.571 | 0.351 | 0.589 |
| llm_last_L14 | 0.433 | 0.359 | 0.649 |
| llm_img_L28 | 0.528 | 0.260 | 0.511 |
| llm_last_L28 | 0.385 | 0.745 | 0.446 |

Fisher-style AUROC ranges from 0.446 to 0.649. Chance is 0.5. That spread, on 7 pairs, does not establish a predictor.

## Seven shifted failure pairs, from five products

Primary photos scored wrong: 12. Distinct products: 5. These are not 12 independent cases.

| product | wrong photos |
|---|---:|
| chips_local | 3 |
| chocolate_local | 3 |
| biscuit_local | 2 |
| detergent_local | 2 |
| noodles_local | 2 |

Straight photos among them: 5. Tilt or occluded photos: 7. The geometry tables only score a straight photo against a tilt or occluded photo, so they contain 7 model-failure pairs, the same 7 on every primary site. A result built on 7 pairs is underpowered.

Those shifted pairs are:

- biscuit_local / occluded
- chocolate_local / tilt
- chocolate_local / occluded
- chips_local / tilt
- chips_local / occluded
- detergent_local / occluded
- noodles_local / tilt

