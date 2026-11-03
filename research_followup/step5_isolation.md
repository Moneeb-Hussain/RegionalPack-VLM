# Step 5 isolation and leakage

Post-hoc exploratory. C stays 1.0. The new sanity runs below use 40 label shuffles. The original study used 1000 and is not rerun here.

Command: `.venv\Scripts\python.exe research_followup\step5_isolation.py`

## Held-out condition

Leave-one-condition-out fits the scaler and the probe on photos whose condition is not the held-out one (`loco_predict` in `linear_probe/probe_analysis.py`). The temperature is chosen by more fits inside those remaining conditions only (`calibrate_temperature`). The held-out photos are not in those fits. `test_held_out_condition_never_enters_scaler_probe_or_temperature` checks the exact training rows.

Saved `vision_out` has 40 straight-versus-other pairs. The straight photo is in the training rows for 40 of them. The shifted photo is in the training rows for 0 of them.

## Reference photo inside the geometry training fold

`geometry_pairs` never holds out the straight condition. For a tilt photo, training is straight plus occluded. For an occluded photo, training is straight plus tilt. The straight photo of the same product is one training row, and the rival class is picked from the score at that straight photo. This script leaves that rule in place. It then refits each pair with that one straight photo removed and compares the margin drop, the rival, and the class predicted for the shifted photo.

Removing the straight photo changes the geometry numbers as follows. `published_drop_max_abs_diff` is the gap between this script's with-reference drop and `linear_probe/results/geometry_<site>.csv`, which is left unchanged.

| site | pairs | published drop gap | predictions that agree | rivals that agree | mean abs drop change | max abs drop change | spearman of drops |
|---|---:|---:|---:|---:|---:|---:|---:|
| vision_out | 40 | 1.767e-13 | 39/40 | 33/40 | 1.084994 | 4.324406 | 0.796060 |
| llm_img_L14 | 40 | 2.789e-13 | 38/40 | 39/40 | 1.023515 | 2.616218 | 0.882927 |
| llm_last_L14 | 40 | 1.430e-13 | 33/40 | 37/40 | 2.386097 | 6.027508 | 0.874109 |
| llm_img_L28 | 40 | 4.086e-14 | 39/40 | 38/40 | 0.797108 | 3.229365 | 0.664540 |
| llm_last_L28 | 40 | 3.197e-14 | 40/40 | 37/40 | 0.290357 | 0.650152 | 0.920075 |

## Exact and near-duplicate photos

File SHA-256 matches `data/image_provenance.csv` for 60 of 60 photos. Duplicate file hashes: 0.
Average-hash Hamming distance, 64 bits, after the EXIF turn used at feature extraction. Closest pair of different products: 2 (cola_intl/occluded vs cola_local/occluded). Pairs of different products at distance 10 or less: 79. Closest pair of the same product: 4.

A follow-up read of `step5_image_checks.csv` (`ahash` kept as text) gives the rest of that distribution. Across different products the median distance is 20 of 64 bits (mean 20.68, 1710 pairs). Eight of those pairs are at distance 5 or less. Within one product the median is 14.5 (60 pairs). The hash is coarse, and a few different products still land very close.

## Session and background cues

`photo_location` is filled on 0 of 20 rows in `data/product_manifest.csv`.
GPS EXIF blocks: 0 of 60. DateTimeOriginal EXIF values: 0 of 60. Any of DateTime, DateTimeOriginal, DateTimeDigitized: 0 of 60.
Camera Make values: none. Camera Model values: none.
Whether the three frames of a product were taken in one session cannot be checked from EXIF time or GPS. Those tags are absent. Same for whether a category pair was shot in one place.

The photography guide says a category pair should share a place and a background. That is an instruction, not a measurement. The border color below is a measurement: mean RGB of the outer 8 percent of pixels after the EXIF turn.
Mean distance between the three frames of one product: 13.31 RGB units. Mean distance between the two products of one category: 25.11. Mean distance between products from different categories: 28.52.

| category | product A | product B | border distance |
|---|---|---|---:|
| cola | cola_intl | cola_local | 9.76 |
| biscuit | biscuit_intl | biscuit_local | 6.78 |
| tea | tea_intl | tea_local | 21.32 |
| chocolate | chocolate_intl | chocolate_local | 20.10 |
| chips | chips_intl | chips_local | 32.37 |
| detergent | detergent_intl | detergent_local | 83.04 |
| toothpaste | toothpaste_intl | toothpaste_local | 9.62 |
| apple | apple_fruita | apple_fruitien | 17.87 |
| noodles | noodles_intl | noodles_local | 7.31 |
| soap | soap_intl | soap_local | 42.89 |

## Filenames and paths

`extract_features.py` opens the file with `Image.open` on the manifest path (True) and passes the turned picture, not the path, into the frozen prompt. The message text does not include `photo_id` (True). Saved arrays in `features.npz`: 59. Numeric matrices with one row per photo: 59 of 59.
The feature rows are means of vision or language hidden states, a 16 by 16 pixel thumbnail, or a colour histogram. The path string is not appended to those vectors.

## Chance sanity, post-hoc exploratory

A 20-way probe has chance accuracy 0.05. Each new run uses 40 label shuffles, seed 0, C=1.0. `null_mean` is the accuracy when training labels are shuffled and test labels stay true.

| input | accuracy | shuffled-label mean | p | error |
|---|---:|---:|---:|---|
| vision_out real features | 1.000000 | 0.057500 | 0.024390 |  |
| vision_out all zeros | FAILED | FAILED | FAILED | Found array with 0 feature(s) (shape=(40, 0)) while a minimum of 1 is required by LogisticRegression. |
| vision_out gaussian noise | 0.016667 | 0.052917 | 0.975610 |  |
| pixel_thumb of black images | FAILED | FAILED | FAILED | Found array with 0 feature(s) (shape=(40, 0)) while a minimum of 1 is required by LogisticRegression. |
| color_hist of black images | FAILED | FAILED | FAILED | Found array with 0 feature(s) (shape=(40, 0)) while a minimum of 1 is required by LogisticRegression. |
| pixel_thumb of noise images | 0.016667 | 0.047917 | 0.975610 |  |
| color_hist of noise images | 0.066667 | 0.050000 | 0.414634 |  |

Qwen features of an actual blank or noise photograph were not computed. Phase 1 does not run Qwen. The zero and noise rows above are feature-space stand-ins for `vision_out` only. That image-level Qwen check is NOT VERIFIED.

The step 2 rerun already shuffled labels 1000 times on the saved features. Those null means, read from `research_followup/step2_rerun/decodability_by_site.csv`:

| site | accuracy | 1000-shuffle mean |
|---|---:|---:|
| vision_out | 1.000000 | 0.051100 |
| llm_img_L14 | 1.000000 | 0.050967 |
| llm_last_L14 | 0.966667 | 0.050867 |
| llm_img_L28 | 1.000000 | 0.051267 |
| llm_last_L28 | 1.000000 | 0.051900 |
| pixel_thumb | 0.833333 | 0.050550 |
| color_hist | 0.866667 | 0.048767 |

