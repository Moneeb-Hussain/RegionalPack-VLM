# Phase 0 review summary

Handoff date: 2026-10-10. This file describes the Phase 0 checkout. It does not change the research design, and it does not start Phase 1.

Phase 0 is the Colab compatibility check for `Qwen/Qwen3-VL-2B-Instruct`. The code, notebook, and CPU tests are in the tree. The model has not been executed.

## 1. Files created and their purposes

| Path | Purpose |
|---|---|
| `README.md` | How to run the Phase 0 Colab check |
| `requirements-colab.txt` | Install floor for Colab: `transformers>=4.57.0`, plus accelerate, Pillow, PyYAML, and huggingface_hub. Does not reinstall torch or add FlashAttention |
| `.gitignore` | Excludes secrets, weights, caches, virtualenvs, uploads, and zip bundles |
| `prompts/prompts.yaml` | Locked identification, transcription, and category prompts |
| `configs/inference_config.yaml` | Locked model id, revision, float16, batch size 1, and decoding |
| `configs/phase0_config.yaml` | Smoke-test profile: empty decoding overrides, T4 pixel cap, public-image URLs, output filenames |
| `src/config.py` | Loads both configs and builds prompt jobs |
| `src/versions.py` | Checks the transformers version floor |
| `src/errors.py` | Error types shown in the notebook |
| `src/images.py` | Opens images, downloads the public demo image, reads upload labels |
| `src/messages.py` | Builds the official one-image chat message |
| `src/outputs.py` | Flags empty or repeated-character responses. Does not score OCR |
| `src/environment.py` | Reads GPU name, CUDA, PyTorch, library versions, and CUDA memory counters |
| `src/inference.py` | Loads Qwen3-VL-2B-Instruct in float16 and runs one prompt |
| `src/phase0.py` | Runs the smoke test and writes the report, JSONL, and zip |
| `src/artifacts.py` | Appends JSONL, renders the verification report, packs the download zip |
| `src/bundle.py` | Finds the project root and rejects unsafe zip paths |
| `notebooks/phase0_verification.ipynb` | Colab notebook, top to bottom |
| `scripts/build_phase0_bundle.py` | Builds `dist/phase0_colab_bundle.zip` for upload into Colab |
| `scripts/render_phase0_notebook.py` | Regenerates the notebook from its cell source |
| `tests/` | CPU tests. They do not download weights |
| `results/README.md` | Says what the notebook will write after a GPU run |
| `results/uploads/.gitkeep` | Keeps the upload directory. Uploaded photos are gitignored |

These verification files are not in the tree, because the notebook has not been run:

| Missing file | What it will contain after Colab |
|---|---|
| `results/phase0_verification.md` | Section A, technical record. Section B, raw responses |
| `results/phase0_predictions.jsonl` | One JSON object per prompt |
| `results/environment.json` | Resolved library versions and GPU counters |
| `results/phase0_artifacts.zip` | Those three files |

Locked settings in `configs/inference_config.yaml`:

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-VL-2B-Instruct` |
| Revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| Dtype | `float16` |
| Batch size | 1 |
| temperature | 0.1 |
| top_p | 1.0 |
| top_k | null (checkpoint default of 20 is disabled) |
| max_new_tokens | identification 128, category 128, transcription 256 |

`configs/phase0_config.yaml` sets `max_pixels: 1003520` for this smoke test only. `decoding_overrides` is empty.

## 2. Tests executed

Command, run locally on 2026-10-10:

```bash
.venv/bin/python -m pytest tests -v --tb=no
```

Environment: macOS, Python 3.14.5, pytest 9.1.1. Result: **30 passed, 0 failed**, in 0.34s.

| Test | Result |
|---|---|
| `tests/test_artifacts.py::test_jsonl_appends_and_keeps_earlier_rows` | PASSED |
| `tests/test_artifacts.py::test_report_splits_compatibility_from_raw_responses` | PASSED |
| `tests/test_artifacts.py::test_success_without_a_measured_peak_is_rejected` | PASSED |
| `tests/test_artifacts.py::test_review_status_counts_only_images_with_both_research_prompts` | PASSED |
| `tests/test_artifacts.py::test_zip_contains_the_three_artifacts` | PASSED |
| `tests/test_artifacts.py::test_inspect_response_does_not_score_meaning` | PASSED |
| `tests/test_artifacts.py::test_zip_refuses_a_missing_file` | PASSED |
| `tests/test_bundle_and_notebook.py::test_bundle_round_trip` | PASSED |
| `tests/test_bundle_and_notebook.py::test_extract_rejects_zip_slip` | PASSED |
| `tests/test_bundle_and_notebook.py::test_notebook_cells_parse_and_stay_in_order` | PASSED |
| `tests/test_config.py::test_locked_prompts_match_the_specification` | PASSED |
| `tests/test_config.py::test_locked_decoding_and_token_limits` | PASSED |
| `tests/test_config.py::test_phase0_does_not_override_locked_decoding_by_default` | PASSED |
| `tests/test_config.py::test_prompt_jobs_keep_smoke_prompt_separate` | PASSED |
| `tests/test_config.py::test_generation_kwargs_include_disabled_top_k` | PASSED |
| `tests/test_config.py::test_missing_prompt_file_raises` | PASSED |
| `tests/test_images.py::test_load_rgb_image_accepts_a_real_png` | PASSED |
| `tests/test_images.py::test_load_rgb_image_rejects_corrupt_and_tiny_files` | PASSED |
| `tests/test_images.py::test_fetch_skips_a_failed_url_and_saves_the_next` | PASSED |
| `tests/test_images.py::test_fetch_reports_every_failed_url` | PASSED |
| `tests/test_images.py::test_fetch_rejects_non_http_urls` | PASSED |
| `tests/test_images.py::test_image_from_bytes_rejects_garbage` | PASSED |
| `tests/test_images.py::test_labels_prefer_an_explicit_map_over_the_filename` | PASSED |
| `tests/test_messages_and_inference_contract.py::test_message_contains_the_pil_image_itself` | PASSED |
| `tests/test_messages_and_inference_contract.py::test_message_rejects_a_bare_filename` | PASSED |
| `tests/test_messages_and_inference_contract.py::test_pixel_budget_keeps_existing_size_keys` | PASSED |
| `tests/test_messages_and_inference_contract.py::test_pixel_stats_require_real_pixel_values` | PASSED |
| `tests/test_messages_and_inference_contract.py::test_inference_module_does_not_load_a_fallback_model` | PASSED |
| `tests/test_messages_and_inference_contract.py::test_requirements_do_not_replace_torch_or_add_flash_attention` | PASSED |
| `tests/test_messages_and_inference_contract.py::test_transformers_version_floor` | PASSED |

The notebook test parses every code cell and checks cell order. It does not execute the notebook.

## 3. Model execution

The model was **not** executed on a Colab GPU, and it was **not** executed locally.

Validation so far is local and CPU-only: configuration loading, image checks, serialization, notebook syntax, and static checks that the loader requests float16 and does not call a fallback model. No weight download, no CUDA measurement, and no model response was produced in this checkout.

## 4. Remaining issues and limitations

- The Phase 0 acceptance gate is open. It stays open until a Colab GPU session loads the model, runs the prompts on real images, and writes the three result files.
- `requirements-colab.txt` sets a floor of `transformers>=4.57.0`. The exact transformers, torch, and CUDA versions will be whatever that Colab session resolves, recorded in `results/environment.json`.
- `max_pixels: 1003520` is a smoke-test memory cap in `configs/phase0_config.yaml`. It is separate from the locked evaluation config.
- The public demo image is the Qwen model-card photograph, or a Hugging Face documentation image if that download fails. The notebook records which URL was used. That photo is a vision check, not a retail package.
- Sampling uses seed 0 and temperature 0.1. A second run can still differ by a few tokens.
- If Colab assigns a GPU whose name does not contain `T4`, the notebook still loads float16 and records that the T4 name check was not confirmed.
- A weak Urdu transcription is stored as raw text. The notebook does not load Qwen2-VL because of it.
- Colab can disconnect a long session. JSONL rows are appended after each prompt.

## 5. Run the notebook in Google Colab

1. On your computer, from the project root:

   ```bash
   python scripts/build_phase0_bundle.py
   ```

   This writes `dist/phase0_colab_bundle.zip`.

2. Open https://colab.research.google.com/.
3. Choose **Runtime → Change runtime type → T4 GPU**.
4. Choose **File → Upload notebook** and select `notebooks/phase0_verification.ipynb`.
5. Run the cells from top to bottom.
6. When the first cell asks, upload `dist/phase0_colab_bundle.zip`.
7. Let the public demo image load and generate.
8. Upload package photos:
   - 2 or 3 images with Urdu-script text, named `urdu_01.jpg`, `urdu_02.jpg`, and so on
   - 2 or 3 images with Roman Urdu or English text, named `latin_01.jpg`, `latin_02.jpg`, and so on
   - or set `UPLOAD_LABELS` in the upload cell to `urdu_script` or `latin_script`
9. Run the remaining cells. The last cell downloads `results/phase0_artifacts.zip`.

Do not paste API tokens into the notebook. The model is public.

If the repo is already on Google Drive, mount Drive first and set `PROJECT_ROOT_OVERRIDE` in the first code cell to that folder. Then you can skip the zip upload.

If the runtime disconnects before the last cell, download `results/phase0_predictions.jsonl` from the Colab file browser. Rows are written after every prompt.
