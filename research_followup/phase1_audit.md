# Phase 1 audit

Historical Set A stays 13/18. Historical Set B stays 15/18. Neither was overwritten. This file is post-hoc except where it repeats the saved confirmatory run.

Command: `.venv\Scripts\python.exe research_followup\build_phase1_audit.py`

## 1. Plain-language summary

The saved linear probe was rerun from the existing feature file and matched, including 0 disagreements out of 300 twenty-way photo calls after the two-class scoring fix. That fix matters for two-class probes: a colour histogram then scores 54/54 on the nine category pairs, the same as every Qwen image-token layer. The twenty-way colour histogram is 52/60 and the 16x16 thumbnail is 50/60, with ResNet-18 at 58/60. The high probe scores are therefore not evidence specific to Qwen. The geometry correlation with alignment is the identity drop = |g| |d| alignment. The 12 primary misses come from 5 products, and only 7 of them enter the geometry failure pairs.

## 2. Inventory and provenance

| item | result |
|---|---|
| JPEG files under `data/raw` | 60 |
| `manifest.csv` rows | 60 |
| file SHA-256 matches `data/image_provenance.csv` | 60/60 |
| benchmark JSONL rows | 180 |
| identification rows | 60 |
| distinct identification prompts in the JSONL | 1 |
| prompt text equals the frozen sentence | True |
| prompt SHA-256 | `df36d0ab30cd2ea9d0c5737d34807d0d4b0bcf52b96c33d987ce90e92ecc79b2` |
| model revisions on identification rows | 89644892e4d85e24eaac8bacfd4f463576704203 |
| pinned revision in the environment file | `89644892e4d85e24eaac8bacfd4f463576704203` |
| original GPU | Tesla T4, torch 2.11.0+cu130, transformers 5.18.0, float16 |
| original sklearn and numpy | NOT VERIFIED. They are not in `benchmark_environment.json`. |
| this machine | Python 3.14.7, torch 2.15.0+cpu, transformers 5.18.0, sklearn 1.9.1, numpy 2.5.4 |
| `features.npz` | `d9f9d3d20f817e68341595fd2ae6da8a631945b0bb2a4580c337e79d82d170c5`, 21402478 bytes, untracked |

Photos live at `data/raw/<product_id>/{straight,tilt,occluded}.jpg`. The manifest paths are those files relative to `linear_probe/`. The frozen identification prompt is the sentence hashed above. It is in `prompts/prompts.yaml`, `results/benchmark/benchmark_environment.json`, every identification row of `results/benchmark/benchmark_predictions.jsonl`, and `linear_probe/extract_features.py`. Set A is `scripts/analyze_benchmark.py` calling `src.scoring.brand_match`. Headline success is a frozen brand match and no abstention. A product counts as correct when at least 2 of its 3 photos have that headline success.

## 3. The 12 wrong photos

Primary image headline successes recomputed from `results/scored_predictions.csv`: 42/54. Product majority: 13/18. The same frozen majority from `results/adjudication_decisions.csv` is 13/18. Published Set A in `results/benchmark/v0.1_summary.md` is 13/18 products and 42/54 images. These rows were not rescored.

| photo | product | condition | manifest failure | scored outcome | frozen match | abstention | loose match | headline |
|---|---|---|---|---|---|---|---|---|
| biscuit_local/straight | biscuit_local | straight | punctuation_blocked_assertion | punctuation_blocked_assertion | False | False | True | False |
| biscuit_local/occluded | biscuit_local | occluded | punctuation_blocked_assertion | punctuation_blocked_assertion | False | False | True | False |
| chocolate_local/straight | chocolate_local | straight | abstention | abstention | False | True | True | False |
| chocolate_local/tilt | chocolate_local | tilt | abstention | abstention | False | True | False | False |
| chocolate_local/occluded | chocolate_local | occluded | abstention | abstention | False | True | True | False |
| chips_local/straight | chips_local | straight | abstention | abstention | False | True | True | False |
| chips_local/tilt | chips_local | tilt | abstention | abstention | False | True | False | False |
| chips_local/occluded | chips_local | occluded | abstention | abstention | False | True | False | False |
| detergent_local/straight | detergent_local | straight | punctuation_blocked_assertion | punctuation_blocked_assertion | False | False | True | False |
| detergent_local/occluded | detergent_local | occluded | punctuation_blocked_assertion | punctuation_blocked_assertion | False | False | True | False |
| noodles_local/straight | noodles_local | straight | punctuation_blocked_assertion | punctuation_blocked_assertion | False | False | True | False |
| noodles_local/tilt | noodles_local | tilt | punctuation_blocked_assertion | punctuation_blocked_assertion | False | False | True | False |

Distinct products: 5 (biscuit_local, chocolate_local, chips_local, detergent_local, noodles_local). Abstention rows: 6. Punctuation-blocked rows: 6.

Raw identification text from the JSONL. Each one matches the scored-file text.

### biscuit_local/straight

JSONL text equals the scored identification text: True.

```text
The product shown in the image is Bisconni Rite, which is a type of chocolate biscuit with vanilla cream. The specific brand name is Bisconni.
```

### biscuit_local/occluded

JSONL text equals the scored identification text: True.

```text
The product shown in the image is a biscuit bar from the brand Bisconni, specifically the "Rite" flavor. The packaging indicates it is "Rich Chocolate Biscuits with Vanilla Cream."
```

### chocolate_local/straight

JSONL text equals the scored identification text: True.

```text
The product shown in the image is a chocolate bar. The packaging features the text "you MILK CHOCOLATE" in a stylized font. However, the specific brand name cannot be identified with confidence from the image.
```

### chocolate_local/tilt

JSONL text equals the scored identification text: True.

```text
The product shown in the image is a chocolate bar. The specific brand name is not identifiable from the image.
```

### chocolate_local/occluded

JSONL text equals the scored identification text: True.

```text
The product shown in the image is a chocolate bar. The packaging features the text "you MILK CHOCOLATE" in gold lettering on a dark green background, with an image of two chocolate pieces on the right side. The brand name is not clearly visible, so I cannot identify the specific brand with confidence.
```

### chips_local/straight

JSONL text equals the scored identification text: True.

```text
The product shown in the image is a bag of "Super Crisp" flavored potato chips with a "Bar.B.Q" flavor. The specific brand name is not clearly identifiable from the image.
```

### chips_local/tilt

JSONL text equals the scored identification text: True.

```text
The product shown in the image is a bag of potato chips. The specific brand name is not clearly visible in the image.
```

### chips_local/occluded

JSONL text equals the scored identification text: True.

```text
The product shown in the image is a bag of potato chips. The specific brand name is not clearly visible in the image.
```

### detergent_local/straight

JSONL text equals the scored identification text: True.

```text
The product shown in the image is a laundry detergent. The specific brand name is "Bonus Tristar".
```

### detergent_local/occluded

JSONL text equals the scored identification text: True.

```text
The product shown in the image is a cleaning powder, specifically a brand called "Bonus Tristar". The packaging is in a yellow color with a red top and features the brand name "Bonus Tristar" in large white letters. The price is listed as Rs. 50, and the packaging also includes Arabic text.
```

### noodles_local/straight

JSONL text equals the scored identification text: True.

```text
The product shown in the image is a packet of instant noodles. The specific brand name is "Shoop".
```

### noodles_local/tilt

JSONL text equals the scored identification text: True.

```text
The product shown in the image is a packet of instant noodles. The specific brand name is "Shoop".
```

Sensitivity, not a new score. `reports/ADJUDICATION_APPROVAL.md` approves four frozen misses as brand hits (Bonus straight and occluded, Shoop straight and tilt) and one frozen hit as a miss (Rite tilt, which names Bisconni rather than Rite). Applying those five saved decisions in `adjudication_decisions.csv` gives 45/54 images and 15/18 products. Upgrades: 4. Downgrades: 1. The approval note calls the product figure Set B, 15/18. Set A is unchanged.

## 4. Reproduction of the saved results

Earlier command, from `linear_probe/`, one math thread, exit 0:

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u run_study.py --features features.npz --manifest manifest.csv --out ..\research_followup\step2_rerun --n-perm 1000
```

That rerun matched `linear_probe/results/decodability_by_site.csv` on 59 sites and `per_photo_probe.csv` on 300 rows. Output stayed under `research_followup/step2_rerun/`.

| site | accuracy | interval | null mean | p | recovered of 12 |
|---|---:|---|---:|---:|---:|
| vision_out | 1.000000 | 1.000000–1.000000 | 0.051100 | 0.000999 | 12 |
| llm_img_L14 | 1.000000 | 1.000000–1.000000 | 0.050967 | 0.000999 | 12 |
| llm_last_L14 | 0.966667 | 0.916667–1.000000 | 0.050867 | 0.000999 | 12 |
| llm_img_L28 | 1.000000 | 1.000000–1.000000 | 0.051267 | 0.000999 | 12 |
| llm_last_L28 | 1.000000 | 1.000000–1.000000 | 0.051900 | 0.000999 | 12 |
| pixel_thumb | 0.833333 | 0.733333–0.933333 | 0.050550 | 0.000999 | 9 |
| color_hist | 0.866667 | 0.766667–0.950000 | 0.048767 | 0.000999 | 10 |

This report's command refit the five primary sites with the current code, C=1.0, and compared them with the step 2 photo calls: 0 disagreements out of 300.

`PROTOCOL.md` was committed as `d12dcc20c20709ae9e3aef7a97ca4bbbe6e925ab` with message `Pre-register linear-probe protocol` before the results commit. Whether that was before `features.npz` existed on disk is unverified, because the feature file has no git history.

## 5. Bug fix and tests

`_ReducedLR` keeps one coefficient row for two classes. The old argmax over that single column always predicted the first class. The current code returns one score for two classes and treats a positive score as the second class. Geometry and probability helpers raise `ValueError` when a probe has fewer than three classes. Twenty-way prediction is still an argmax over one row per class.

Pytest exit code: 0.

```text
.......                                                                  [100%]
7 passed in 3.95s
```

## 6. Isolation and leakage

Command for the checks below, except the cheap recounts in this report: `.venv\Scripts\python.exe research_followup\step5_isolation.py`

Leave-one-condition-out training rows are the other conditions. On the saved labels, the straight reference photo is inside those training rows for 40 of 40 geometry pairs, and the shifted photo is inside them for 0 of 40. The unit test `test_held_out_condition_never_enters_scaler_probe_or_temperature` checks that the held-out condition is absent from the scaler, the probe, and the temperature fits. The temperature fit is nested inside the training conditions.

Removing that one straight photo and refitting, from `step5_geometry_ablation.csv`:

| site | predictions that agree | rivals that agree | mean absolute drop change |
|---|---:|---:|---:|
| vision_out | 39/40 | 33/40 | 1.085 |
| llm_img_L14 | 38/40 | 39/40 | 1.024 |
| llm_last_L14 | 33/40 | 37/40 | 2.386 |
| llm_img_L28 | 39/40 | 38/40 | 0.797 |
| llm_last_L28 | 40/40 | 37/40 | 0.290 |

The published geometry files were not edited. The straight photo in the training fold changes some rivals and some margin drops. It does not change the twenty-way rule, which never trains on the held-out condition.

Duplicate file hashes: 0. Across different products, the closest 64-bit average hash is 2 bits and the median is 20.0. Pairs at distance 10 or less: 79. The closest pair is `cola_intl/occluded` versus `cola_local/occluded`.

`photo_location` is filled on 0 of 20 product rows. GPS EXIF blocks: 0 of 60. DateTimeOriginal values: 0 of 60. A shared capture session is NOT VERIFIED from the files.

`extract_features.py` passes the turned picture into the model. The path string is not written into `features.npz`. All saved arrays are numeric. Filename leakage into the feature vectors was not found in that code.

Chance checks from the same step 5 command, 40 label shuffles, C=1.0. Chance for 20 products is 0.05. Real `vision_out` accuracy was 1.000000 with shuffle mean 0.057500. Gaussian noise in that shape scored 0.016667. Noise-image colour histograms scored 0.066667. All-zero features, black thumbnails, and black histograms failed before a score existed: the scaler leaves zero columns and logistic regression raises `Found array with 0 feature(s) (shape=(40, 0))`. Qwen features of an actual blank photograph were not computed. That check is NOT VERIFIED.

## 7. Shortcut baselines

Post-hoc exploratory. Command: `.venv\Scripts\python.exe research_followup\step6_shortcuts.py`. Same held-out condition, C=1.0, no new 1000-shuffle test. Pairs are the nine primary category pairs, 54 photos. Two-class scores use the sign rule.

| site | 20-way | matched pairs |
|---|---:|---:|
| llm_img_L14 | 60/60 | 54/54 |
| llm_img_L28 | 60/60 | 54/54 |
| vision_out | 60/60 | 53/54 |
| color_hist | 52/60 | 54/54 |
| pixel_thumb | 50/60 | 47/54 |
| resnet18_pool | 58/60 | 50/54 |

Image-token layers at 54/54: 28 of 28. Colour histogram is also 54/54. The paired test cannot localise a Qwen-specific effect. It is not at ceiling for every baseline: the thumbnail is 47/54, ResNet-18 is 50/54, and the vision tower is 53/54. ResNet features are the saved file. `resnet_control.py` does not apply the EXIF turn used for the Qwen features.

The twenty-way counts match the step 2 accuracies, and the ResNet count matches `linear_probe/results_resnet/resnet_control.csv` (58/60).

## 8. Red flags and what could not be verified

- The probe ceiling is shared with simple image cues. A colour histogram ties Qwen on the paired task.
- Geometry's straight photo is one of the training photos. Margin drop versus alignment is an algebraic identity. The Fisher-style score is not Fisher-IRG.
- Failure-prediction numbers use 7 shifted pairs. Fisher-style AUROC on the saved tables runs from 0.446 to 0.649.
- Four frozen misses look like brand hits to the approved review because quotation marks blocked the matcher. They stay misses in Set A.
- Whether `PROTOCOL.md` was committed before `features.npz` existed on disk: unverified.
- Original sklearn and numpy for the T4 evaluation: NOT VERIFIED.
- Qwen hidden states of a real blank or noise photo: NOT VERIFIED. Phase 1 did not run Qwen.
- Same session or same place for the three frames: NOT VERIFIED. No time, GPS, camera model, or `photo_location` values were in the files.
- The saved ResNet matrix was not rebuilt, and its loader skips the EXIF turn.

## 9. Limits

From `linear_probe/PROTOCOL.md` section 7, plus the geometry note:

- 60 photos, 20 products, one model, one annotator, three phone conditions that mix viewpoint and occlusion.
- Closed set: each test product is seen under the other conditions. This says nothing about unseen products.
- A linear probe can show that a label is readable. It does not show that the model uses that direction.
- The Fisher-style quantity is a proxy inspired by Fisher-IRG, not an implementation of it, and it is not a verified predictor of these mistakes.
- Largest recomputed gap for drop versus `|g| |d| alignment` is 4.245e-13, on the saved geometry tables.
- 12 misses, 5 products, 7 shifted failure pairs. They are not 12 independent cases, and a 7-pair comparison is underpowered.

## 10. Reproducibility record

| item | value |
|---|---|
| branch | `phase1-audit` |
| upstream | no upstream |
| this machine | Python 3.14.7, torch 2.15.0+cpu, transformers 5.18.0, sklearn 1.9.1, numpy 2.5.4 |
| probe seed | 0 |
| C | 1.0 |
| original permutations | 1000 for primary sites and controls |
| prompt SHA-256 | `df36d0ab30cd2ea9d0c5737d34807d0d4b0bcf52b96c33d987ce90e92ecc79b2` |
| features SHA-256 | `d9f9d3d20f817e68341595fd2ae6da8a631945b0bb2a4580c337e79d82d170c5` |
| protocol commit | `d12dcc20c20709ae9e3aef7a97ca4bbbe6e925ab 2026-11-03T02:18:46+05:00 Pre-register linear-probe protocol` |
| pytest | exit 0 |

Commits on `phase1-audit` that are not on `linear-probe`:

- `000ab4dc57481d577fbb1a43b0ec8d3701a847c1 docs: note that the geometry margin drop is an identity`
- `3aa63e2098c4c8803d8c27c0cdc2e4f003736d77 docs: record post-hoc shortcut baseline checks`
- `4dcec95a2b75b0524609184f7c269d2429f05cef docs: record isolation and leakage checks`
- `8ea871cb326de69c422fb25c4ff968f26b776e8d test: check held-out photos stay out of each fit`
- `ed286034ae3c115595ed8b393913b49a22a700e4 test: turn probe checks into pytest tests`
- `32b27027a04465666877531bd35aab082ddf388e fix: score two-class probes by the sign of the single score`
- `14344165540808ba52e3e8255aaaa177d691c4a5 test: reproduce the saved linear-probe accuracies`

Nothing was pushed. `linear_probe/results/` and `linear_probe/results_resnet/` were not modified. `features.npz` is still untracked.

Commands behind the numbers:

- Step 2: `run_study.py --features features.npz --manifest manifest.csv --out ..\research_followup\step2_rerun --n-perm 1000`
- Step 4 and this report: `.venv\Scripts\python.exe -m pytest linear_probe\test_probe_analysis.py -q`
- Step 5: `.venv\Scripts\python.exe research_followup\step5_isolation.py`
- Step 6: `.venv\Scripts\python.exe research_followup\step6_shortcuts.py`
- Step 7: `.venv\Scripts\python.exe research_followup\step7_geometry_note.py`
- This file: `.venv\Scripts\python.exe research_followup\build_phase1_audit.py`

