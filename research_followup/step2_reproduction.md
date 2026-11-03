# Step 2 reproduction record

Command, from `linear_probe/`, with one math thread:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u run_study.py --features features.npz --manifest manifest.csv --out ..\research_followup\step2_rerun --n-perm 1000
```

Exit code 0. Output was not written into `linear_probe/results/`.

## Saved accuracies

Every cell in `decodability_by_site.csv` matched the saved file (59 sites, 0 differences). Per-photo probe calls matched on all 300 saved rows (5 primary sites x 60 photos, 0 disagreements).

| Site | Role | Accuracy | 95% CI | Null mean | p | Recovered of 12 |
|---|---|---|---|---|---|---|
| vision_out | primary | 1.000000 | 1.000000–1.000000 | 0.051100 | 0.000999 | 12 |
| llm_img_L14 | primary | 1.000000 | 1.000000–1.000000 | 0.050967 | 0.000999 | 12 |
| llm_last_L14 | primary | 0.966667 | 0.916667–1.000000 | 0.050867 | 0.000999 | 12 |
| llm_img_L28 | primary | 1.000000 | 1.000000–1.000000 | 0.051267 | 0.000999 | 12 |
| llm_last_L28 | primary | 1.000000 | 1.000000–1.000000 | 0.051900 | 0.000999 | 12 |
| pixel_thumb | control | 0.833333 | 0.733333–0.933333 | 0.050550 | 0.000999 | 9 |
| color_hist | control | 0.866667 | 0.766667–0.950000 | 0.048767 | 0.000999 | 10 |

These are the same figures as `linear_probe/results/README.md` (1.00, 1.00, 0.97, 1.00, 1.00, and controls 0.83 and 0.87). p = 0.000999 is what 1000 permutations print; the README rounds it to 0.001.

## Versions, seeds, revision, commits

| Item | Value |
|---|---|
| Python | 3.14.7 |
| torch | 2.15.0+cpu |
| transformers | 5.18.0 |
| sklearn | 1.9.1 |
| numpy | 2.5.4 |
| Probe seed | `RNG_SEED = 0` in `linear_probe/probe_analysis.py` |
| Permutations | 1000 for primary sites and controls; 100 for exploratory layers |
| C | 1.0 |
| Bootstrap resamples | 2000 |
| Model revision in config and `extract_features.py` | `89644892e4d85e24eaac8bacfd4f463576704203` |
| Prompt SHA-256 | `df36d0ab30cd2ea9d0c5737d34807d0d4b0bcf52b96c33d987ce90e92ecc79b2` |
| `features.npz` SHA-256 | `d9f9d3d20f817e68341595fd2ae6da8a631945b0bb2a4580c337e79d82d170c5` (21402478 bytes) |
| Code commit used for this rerun | `3d0e64abd81e522710a6c2330ae8635704f236c4` |
| First `PROTOCOL.md` commit | `d12dcc20c20709ae9e3aef7a97ca4bbbe6e925ab` `Pre-register linear-probe protocol` (2026-11-03 02:18:46 +0500) |
| Results commit | `535dfec` `Run linear-probe study; row-space fit for speed; add results` |

## Hashes

File SHA-256 of all 60 JPEGs matches `data/image_provenance.csv` (60/60). The pixel SHA-256 used in `results/benchmark/benchmark_predictions.jsonl` also matches all 60, after the same upright-image load. The two hashes are different kinds of hash, so the file hash is not equal to the value stored in the response log. Per-photo file hashes are in `step2_hashes.txt`.

## Protocol before the feature file

unverified

Git shows the protocol commit, with that exact message, before the results commit. `features.npz` has no git history (`git log -- linear_probe/features.npz` is empty; the file is ignored). Git cannot show that the feature file did not already exist on disk when the protocol was committed.
