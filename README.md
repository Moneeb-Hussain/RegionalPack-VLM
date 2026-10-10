# RegionalPack-VLM

Phase 0 checked that `Qwen/Qwen3-VL-2B-Instruct` can run image inference on a Colab T4 in float16. That run is complete. Scoring has not started.

Phase 1A is the dataset plan in [`data/product_list.md`](data/product_list.md). It is proposed, not locked. The pre-benchmark notes are in [`docs/prebenchmark_changes.md`](docs/prebenchmark_changes.md).

The locked specification is [`docs/research_spec.md`](docs/research_spec.md).

## What this phase produces

| File | When it appears |
|---|---|
| `notebooks/phase0_verification.ipynb` | Run this on a Colab T4 |
| `results/phase0_verification.md` | After the notebook finishes a load attempt |
| `results/phase0_predictions.jsonl` | One JSON object per prompt, appended as each call finishes |
| `results/environment.json` | Versions and GPU counters from that session |
| `results/phase0_artifacts.zip` | Those three files, downloaded from Colab |

A file in `results/` is a measurement only after this notebook has been run on the Colab GPU. Do not fill those files in by hand.

## Run on Colab

1. On your computer, from this directory:

   ```bash
   python3 scripts/build_phase0_bundle.py
   ```

2. Open [Google Colab](https://colab.research.google.com/), then **Runtime → Change runtime type → T4 GPU**.
3. **File → Upload notebook** and choose `notebooks/phase0_verification.ipynb`.
4. Use **Runtime → Disconnect and delete runtime**, then set the runtime type to **T4 GPU** again.
5. Run all cells from the top. When asked, upload `dist/phase0_colab_bundle.zip`.
6. After the public demo image, upload package photos:
   - 2 or 3 images with Urdu-script text, named `urdu_01.jpg`, `urdu_02.jpg`, …
   - 2 or 3 images with Roman Urdu or English text, named `latin_01.jpg`, `latin_02.jpg`, …
7. Download `results/phase0_artifacts.zip` from the last cell. If model loading fails, the load cell downloads that zip before the notebook stops.

The model is public. Do not paste API tokens into the notebook.

The install cell runs before the GPU cell. It keeps Colab's loaded Pillow extension and adds `PIL._typing._Ink` when that build does not have it. It does not upgrade Pillow 11.3.0 to 12.x.

If you already copied the repo to Google Drive, set `PROJECT_ROOT_OVERRIDE` in the first code cell and skip the zip upload.

## Layout

| Path | Role |
|---|---|
| [`docs/research_spec.md`](docs/research_spec.md) | Locked research specification |
| `prompts/prompts.yaml` | Locked identification, transcription, and category prompts |
| `configs/inference_config.yaml` | Locked decoding: temperature 0.1, top_p 1.0, token limits |
| `configs/phase0_config.yaml` | Smoke-test-only settings, including the T4 pixel cap |
| `src/` | Loading, inference, and artifact writing |
| `tests/` | CPU checks that do not download weights |
| `requirements-colab.txt` | Install floor for Colab |

`decoding_overrides` in the Phase 0 config is empty, so smoke-test decoding matches the locked config. The checkpoint's own temperature 0.7 and top_k 20 are overridden in the generate call. `top_k: null` disables that checkpoint default.

`max_pixels: 1003520` limits image tokens during this smoke test so a phone photo is less likely to exhaust a T4. It is not a full-evaluation setting.

## Local checks

These do not load the model and cannot mark the GPU run as passed:

```bash
python3 -m pytest tests -q
```

Needs Python 3.9+, PyYAML, Pillow, and pytest.

## Model-switching rule

Stay on Qwen3-VL-2B-Instruct when it loads and returns text, including when Urdu transcription looks weak. Consider Qwen2-VL-2B-Instruct only after repeated float16 failures on the Colab GPU: the model does not load, it crashes, or weights or outputs are non-finite or empty. This notebook does not load the fallback.

## Known limits

- This repository checkout has not executed the model. Phase 0 acceptance still requires the Colab run.
- Exact `transformers` / `torch` pins are written to `results/environment.json` by that run. `requirements-colab.txt` only sets a floor of `transformers>=4.57.0`.
- The public demo photo is not a retail package. Its answers are a vision check.
- Sampling uses temperature 0.1 with a fixed seed. GPU kernels can still make a later rerun differ by a few tokens.
- Colab may disconnect a long session. Prediction rows are appended after each prompt so a partial JSONL can be downloaded from the file browser.
