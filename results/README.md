# Phase 0 results

This directory receives artifacts from `notebooks/phase0_verification.ipynb` after a Colab GPU run.

| File | Written by | Contents |
|---|---|---|
| `phase0_verification.md` | the notebook | Technical record and raw smoke-test responses |
| `phase0_predictions.jsonl` | the notebook | One JSON object per prompt |
| `environment.json` | the notebook | Resolved library versions and GPU details |
| `phase0_artifacts.zip` | the notebook | Those three files, for download |
| `uploads/` | the notebook | Images uploaded inside Colab; not for Git |

Do not fill these files in by hand. A number that did not come from `torch.cuda.max_memory_allocated()` on the Colab GPU is not a measurement.

The completed Colab run is the Phase 0 record. Do not rewrite those files when the benchmark transcription prompt changes. See `docs/prebenchmark_changes.md`. If an older report shows bfloat16 support as true, that value is PyTorch's emulation flag. The T4 load was float16.

Poor Urdu transcription in the smoke test is an observation. It is not, by itself, a reason to switch models.
