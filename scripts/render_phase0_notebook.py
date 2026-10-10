"""Write notebooks/phase0_verification.ipynb. Run this after editing the cells below."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks" / "phase0_verification.ipynb"


def _source(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    if not lines:
        return []
    source = [line + "\n" for line in lines[:-1]]
    source.append(lines[-1])
    return source


def _cell(cell_type: str, text: str) -> dict:
    cell = {"cell_type": cell_type, "metadata": {}, "source": _source(text)}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


CELLS = [
    _cell(
        "markdown",
        """
# Phase 0 — Qwen3-VL-2B-Instruct compatibility check

This notebook checks that `Qwen/Qwen3-VL-2B-Instruct` can load on a Colab GPU and answer the locked prompts on real images. It does not score accuracy and it does not decide a hypothesis.

## Before you start

1. In Colab: **Runtime → Change runtime type → T4 GPU**.
2. On your computer, from the project root, run `python3 scripts/build_phase0_bundle.py`.
3. Upload this notebook to Colab (**File → Upload notebook**). The GitHub repo is private, so the notebook does not clone it.
4. Use **Runtime → Disconnect and delete runtime**, then set the runtime type to **T4 GPU** again. This drops a Pillow install left by an older run.
5. Run the cells from top to bottom. When asked, upload `dist/phase0_colab_bundle.zip`. If you use `PROJECT_ROOT_OVERRIDE` for a Drive copy, mount Drive before running the first cell.
6. After the public demo image succeeds, upload your own package photos.
7. Download `results/phase0_artifacts.zip` from the last code cell. If model loading fails, the load cell downloads that zip before the notebook stops.

Do not paste API tokens into this notebook. The model is public.

If the runtime disconnects, use the Colab file browser to download `results/phase0_predictions.jsonl` before the session disappears. Rows are appended after every prompt.

Poor Urdu transcription is a research observation. This notebook will not switch to another model because of it.
""",
    ),
    _cell(
        "code",
        """
from pathlib import Path
import zipfile

PROJECT_ROOT_OVERRIDE = ""  # Example: "/content/drive/MyDrive/RegionalPack-VLM"

PHASE0_CONTINUE = True
REQUIRED_FILES = (
    "src/phase0.py",
    "configs/inference_config.yaml",
    "configs/phase0_config.yaml",
    "prompts/prompts.yaml",
    "requirements-colab.txt",
)


def require_previous(*names):
    missing = [name for name in names if name not in globals()]
    if missing:
        raise RuntimeError("Run the notebook from the top. Missing: " + ", ".join(missing))
    if not globals().get("PHASE0_CONTINUE", False):
        raise RuntimeError("An earlier cell failed. Fix that error and rerun from the top.")


def looks_like_root(path):
    return all((path / relative).is_file() for relative in REQUIRED_FILES)


def search_root(path):
    path = Path(path)
    if looks_like_root(path):
        return path
    if path.is_dir():
        for child in sorted(path.iterdir()):
            if child.is_dir() and looks_like_root(child):
                return child
    return None


def root_has_pillow_fix(path):
    environment = path / "src" / "environment.py"
    requirements = path / "requirements-colab.txt"
    if not environment.is_file() or not requirements.is_file():
        return False
    return "def pillow_pin_for_installed_package" in environment.read_text(encoding="utf-8")


try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

candidates = []
if PROJECT_ROOT_OVERRIDE.strip():
    candidates.append(Path(PROJECT_ROOT_OVERRIDE).expanduser())
candidates.extend([Path.cwd(), Path("/content/regionalpack-vlm")])

PROJECT_ROOT = None
for candidate in candidates:
    found = search_root(candidate)
    if found is None:
        continue
    if IN_COLAB and not PROJECT_ROOT_OVERRIDE.strip() and not root_has_pillow_fix(found):
        print(
            f"Ignoring stale project files at {found}. "
            "They do not include the Pillow extension repair. Upload dist/phase0_colab_bundle.zip."
        )
        continue
    PROJECT_ROOT = found
    break

if PROJECT_ROOT is None:
    if not IN_COLAB:
        PHASE0_CONTINUE = False
        raise RuntimeError(
            "Project files were not found. Run this notebook from the repository root, "
            "or open it in Colab and upload dist/phase0_colab_bundle.zip."
        )
    from google.colab import files

    print("Select dist/phase0_colab_bundle.zip from your computer.")
    uploaded = files.upload()
    if len(uploaded) != 1:
        PHASE0_CONTINUE = False
        raise RuntimeError("Upload exactly one zip: dist/phase0_colab_bundle.zip.")
    zip_name, payload = next(iter(uploaded.items()))
    if not zip_name.lower().endswith(".zip"):
        PHASE0_CONTINUE = False
        raise RuntimeError(f"{zip_name} is not a .zip file.")
    zip_path = Path("/content") / Path(zip_name).name
    zip_path.write_bytes(payload)
    dest = Path("/content/regionalpack-vlm")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            parts = Path(info.filename).parts
            if not parts or info.filename.startswith("/") or ".." in parts:
                PHASE0_CONTINUE = False
                raise RuntimeError(f"Refusing to extract unsafe zip path: {info.filename}")
        archive.extractall(dest)
    PROJECT_ROOT = search_root(dest)
    if PROJECT_ROOT is None or not root_has_pillow_fix(PROJECT_ROOT):
        PHASE0_CONTINUE = False
        raise RuntimeError(
            "The zip is missing the Pillow extension repair. On your computer run "
            "python3 scripts/build_phase0_bundle.py, then upload the new "
            "dist/phase0_colab_bundle.zip."
        )

print(f"Project root: {PROJECT_ROOT}")
print(f"Running in Colab: {IN_COLAB}")
""",
    ),
    _cell(
        "markdown",
        """
## 1. Install

This cell runs before the GPU cell. Colab's CUDA build of PyTorch is left in place. Nothing here installs FlashAttention or replaces torch.

Colab loads its Pillow extension before this cell. The cell pins the Python package to that extension. It does not upgrade Pillow 11.3.0 to 12.x. If `_Ink` is missing, the cell adds that alias. No restart is required for this.
""",
    ),
    _cell(
        "code",
        """
import sys

require_previous("PROJECT_ROOT")
sys.path.insert(0, str(PROJECT_ROOT))
for module_name in list(sys.modules):
    if module_name == "src" or module_name.startswith("src."):
        del sys.modules[module_name]

from src.environment import ensure_pillow, probe_libraries

requirements = PROJECT_ROOT / "requirements-colab.txt"
if not requirements.is_file():
    PHASE0_CONTINUE = False
    raise FileNotFoundError(f"Missing {requirements}")

try:
    pillow_version = ensure_pillow(requirements)
    LIBS = probe_libraries()
except Exception:
    PHASE0_CONTINUE = False
    raise

print(f"Pillow extension matches package: {pillow_version}")
print(f"transformers: {LIBS['transformers_version']}")
print(f"torch: {LIBS['pytorch_version']}")
print(f"CUDA: {LIBS['cuda_version']}")
print(f"accelerate: {LIBS['accelerate_version']}")
print(f"pillow: {LIBS['pillow_version']}")
print(f"huggingface_hub: {LIBS['huggingface_hub_version']}")
print(f"transformers meets >=4.57.0: {LIBS['transformers_satisfies_minimum']}")
""",
    ),
    _cell(
        "markdown",
        """
## 2. GPU

The next cell stops if CUDA is missing. Switch the runtime to a T4 and rerun from the top. It runs after the install cell so PyTorch imports the repaired Pillow.

`bfloat16` is not used. A T4 is Turing and has no hardware bfloat16. PyTorch can still report `is_bf16_supported()` as true because that flag includes emulation. The loader casts the model to float16.
""",
    ),
    _cell(
        "code",
        """
import sys

require_previous("PROJECT_ROOT", "LIBS")
sys.path.insert(0, str(PROJECT_ROOT))
for module_name in list(sys.modules):
    if module_name == "src" or module_name.startswith("src."):
        del sys.modules[module_name]

from src.environment import probe_gpu
from src.errors import GpuUnavailableError

try:
    GPU_INFO = probe_gpu()
except GpuUnavailableError:
    PHASE0_CONTINUE = False
    raise

ENVIRONMENT = {**GPU_INFO, **LIBS}
print(f"GPU name: {GPU_INFO['gpu_name']}")
print(f"T4 name confirmed: {GPU_INFO['t4_name_confirmed']}")
print(f"CUDA version: {GPU_INFO['cuda_version']}")
print(f"PyTorch version: {GPU_INFO['pytorch_version']}")
print(f"Compute capability: {GPU_INFO['compute_capability']}")
print(f"bfloat16 hardware support: {GPU_INFO['bf16_hardware_supported']}")
print(
    "PyTorch is_bf16_supported, including emulation: "
    f"{GPU_INFO['bf16_pytorch_including_emulation']}"
)
if GPU_INFO["bf16_emulation_only"]:
    print(
        "This GPU has no hardware bfloat16. The PyTorch flag includes emulation. "
        "The model still loads in float16."
    )
print(f"Device total memory (bytes): {GPU_INFO['total_vram_bytes']}")
print("nvidia-smi:")
print(GPU_INFO["nvidia_smi"])
if not GPU_INFO["t4_name_confirmed"]:
    print(
        "This GPU name does not contain T4. The notebook still loads float16, "
        "and the report will show that the T4 check was not confirmed."
    )
""",
    ),
    _cell(
        "markdown",
        """
## 3. Locked prompts and decoding

The three research prompts are printed below so you can check them before any generation. Phase 0 adds one extra smoke prompt, "Describe this image.", and runs it only on the public demo photo.

`configs/phase0_config.yaml` may override decoding. The shipped file leaves `decoding_overrides` empty, so temperature `0.1`, `top_p` `1.0`, and the locked token limits are used. The checkpoint's own `top_k: 20` and `temperature: 0.7` are overridden on purpose.

`max_pixels` in the Phase 0 config is a T4 memory cap for this smoke test. It is not the full-evaluation setting.
""",
    ),
    _cell(
        "code",
        """
require_previous("PROJECT_ROOT", "ENVIRONMENT")

from src.config import load_inference_config, load_phase0_config, load_phase0_prompts, merged_decoding
from src.phase0 import output_paths

INFERENCE = load_inference_config(PROJECT_ROOT)
PHASE0 = load_phase0_config(PROJECT_ROOT)
PROMPTS = load_phase0_prompts(PROJECT_ROOT)
DECODING, OVERRIDE_KEYS = merged_decoding(INFERENCE, PHASE0)
PATHS = output_paths(PROJECT_ROOT, PHASE0)

print(f"Model: {INFERENCE['model_id']}")
print(f"Pinned revision: {INFERENCE['revision']}")
print(f"Dtype: {INFERENCE['dtype']}")
print(f"Batch size: {INFERENCE['batch_size']}")
print(f"Decoding overrides: {OVERRIDE_KEYS or 'none'}")
print(f"Decoding: {DECODING}")
print(f"Phase 0 pixel budget: min {PHASE0['min_pixels']}, max {PHASE0['max_pixels']}")
print()
for prompt_id, text in PROMPTS.items():
    print(f"[{prompt_id}] max_new_tokens={INFERENCE['max_new_tokens'][prompt_id]}")
    print(text)
    print()
""",
    ),
    _cell(
        "markdown",
        """
## 4. Public demo image

The first URL is the demo image from the Qwen3-VL model card. If that download fails, the notebook tries a Hugging Face documentation image and records which URL was used. The file is opened with PIL and passed to the model as image pixels.
""",
    ),
    _cell(
        "code",
        """
require_previous("PHASE0", "PATHS")

from src.images import fetch_first_available_image, load_rgb_image

demo_path = PATHS["uploads"] / "public_demo.jpg"
try:
    PUBLIC_IMAGE_PATH, PUBLIC_IMAGE_URL = fetch_first_available_image(
        list(PHASE0["public_image_urls"]),
        demo_path,
    )
    PUBLIC_IMAGE = load_rgb_image(PUBLIC_IMAGE_PATH)
except Exception:
    PHASE0_CONTINUE = False
    raise

print(f"Public image URL: {PUBLIC_IMAGE_URL}")
print(f"Saved to: {PUBLIC_IMAGE_PATH}")
print(f"Size: {PUBLIC_IMAGE.size[0]} x {PUBLIC_IMAGE.size[1]} mode={PUBLIC_IMAGE.mode}")

try:
    from IPython.display import Image, display

    display(Image(filename=str(PUBLIC_IMAGE_PATH)))
except Exception as exc:
    print(f"Could not display the image inline: {exc}")
""",
    ),
    _cell(
        "markdown",
        """
## 5. Load the model

This downloads `Qwen/Qwen3-VL-2B-Instruct` at the pinned revision and loads it in float16. The first download is several gigabytes.

The cell records `torch.cuda.max_memory_allocated()` after the load. That number is a measurement. After the weights load, the notebook scans parameters for NaN or infinity. That scan can take about a minute.

A load error is written into `results/phase0_artifacts.zip`. On Colab, this cell downloads that zip before it stops. It does not load a fallback model.
""",
    ),
    _cell(
        "code",
        """
require_previous("INFERENCE", "PHASE0", "ENVIRONMENT", "DECODING", "OVERRIDE_KEYS", "PATHS")

from src.errors import ModelLoadError
from src.phase0 import failure_load_meta, finalize, load_for_phase0


def publish_failure_zip(load_meta):
    # Write the failure artifacts and offer the zip before this cell stops.
    written = finalize(
        paths=PATHS,
        load_meta=load_meta,
        environment=ENVIRONMENT,
        decoding=DECODING,
        decoding_override_keys=OVERRIDE_KEYS,
    )
    print(f"Wrote a failure report to {written['report']}")
    print(f"Failure zip: {written['zip']}")
    if IN_COLAB:
        import time
        from google.colab import files

        files.download(str(written["zip"]))
        # Colab cancels the browser download if the cell raises immediately.
        time.sleep(2)
    else:
        print(f"Not in Colab. Download the failure zip from disk: {written['zip']}")
    return written


try:
    MODEL, PROCESSOR, LOAD_META = load_for_phase0(
        inference_config=INFERENCE,
        phase0_config=PHASE0,
    )
except Exception as exc:
    LOAD_META = failure_load_meta(INFERENCE["model_id"], str(INFERENCE["revision"]), exc)
    try:
        publish_failure_zip(LOAD_META)
    except Exception as report_error:
        print(f"Could not write or download the failure zip: {report_error}")
    PHASE0_CONTINUE = False
    raise

print(f"Load status: {LOAD_META['load_status']}")
print(f"Revision pinned: {LOAD_META['revision_pinned']}")
print(f"Revision resolved: {LOAD_META['revision_resolved']}")
print(f"Parameter dtypes: {LOAD_META['parameter_dtypes']}")
print(f"Attention: {LOAD_META['attn_implementation']}")
print(f"Load notes: {LOAD_META['load_notes'] or 'none'}")
print(f"Non-finite parameter: {LOAD_META['nonfinite_parameter']}")
print(f"Peak allocated bytes at load: {LOAD_META['peak_allocated_bytes']}")
print(f"Allocated bytes after load: {LOAD_META['allocated_bytes_after_load']}")
print(f"Memory source: {LOAD_META['memory_source']}")
if LOAD_META.get("error"):
    try:
        publish_failure_zip(LOAD_META)
    except Exception as report_error:
        print(f"Could not write or download the failure zip: {report_error}")
    PHASE0_CONTINUE = False
    raise ModelLoadError(LOAD_META["error"])
""",
    ),
    _cell(
        "markdown",
        """
## 6. Public-image inference

The demo photo is not a retail package. Its answers only show that the model received pixels and produced text. The locked identification, transcription, and category prompts still run so the same call path is exercised.

Rerunning this cell appends new rows. It does not delete earlier rows.
""",
    ),
    _cell(
        "code",
        """
require_previous("MODEL", "PROCESSOR", "PUBLIC_IMAGE", "LOAD_META", "DECODING", "PATHS")

from src.phase0 import jobs_for_image, run_image

RUNTIME = {**ENVIRONMENT, **LOAD_META}
public_jobs = jobs_for_image(
    prompts=PROMPTS,
    inference_config=INFERENCE,
    phase0_config=PHASE0,
    include_smoke=True,
)
try:
    PUBLIC_RECORDS = run_image(
        model=MODEL,
        processor=PROCESSOR,
        image=PUBLIC_IMAGE,
        image_id="public_demo",
        image_label="public_demo",
        jobs=public_jobs,
        decoding=DECODING,
        runtime=RUNTIME,
        jsonl_path=PATHS["jsonl"],
    )
except Exception:
    PHASE0_CONTINUE = False
    raise

for record in PUBLIC_RECORDS:
    print("=" * 80)
    print(
        f"{record['prompt_id']}  completed={record['inference_completed']}  "
        f"flag={record['output_flag']}  pixels={record.get('pixel_values_shape')}"
    )
    print(record["prompt"])
    print("---")
    print(record["response"] if record["response"] is not None else record["error"])
""",
    ),
    _cell(
        "markdown",
        """
## 7. Your package photos

Upload:

- 2 or 3 photos of packaging with **Urdu-script** text
- 2 or 3 photos of packaging with **Roman Urdu or English** text

Name files `urdu_01.jpg`, `urdu_02.jpg`, `latin_01.jpg`, and so on, **or** fill `UPLOAD_LABELS` before running the upload cell:

```python
UPLOAD_LABELS = {
    "photo_a.jpg": "urdu_script",
    "photo_b.jpg": "latin_script",
}
```

Allowed labels are `urdu_script` and `latin_script`. Unlabeled files are stored as `unlabeled` and do not count toward the sample check.

The model receives the uploaded image, not the filename. You can skip this cell and the next one if you only want the public-image check; the report will then say the script-sample portion is incomplete.
""",
    ),
    _cell(
        "code",
        """
require_previous("PATHS")

UPLOAD_LABELS = {}

from src.errors import ImageValidationError
from src.images import label_for_filename, load_rgb_image

UPLOADED = []
dest_dir = PATHS["uploads"]
dest_dir.mkdir(parents=True, exist_ok=True)

if not IN_COLAB:
    local_dir = PROJECT_ROOT / "samples"
    print(f"Not in Colab. Looking for images in {local_dir}.")
    print("In Colab this cell opens a file picker instead.")
    candidates = []
    if local_dir.is_dir():
        candidates = sorted(
            path for path in local_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        )
    payloads = {path.name: path.read_bytes() for path in candidates}
else:
    from google.colab import files

    print("Select your package photos. You can select more than one file.")
    payloads = files.upload()

for name, payload in payloads.items():
    target = dest_dir / Path(name).name
    if not isinstance(payload, bytes):
        raise TypeError(f"Expected file bytes for {name}, got {type(payload).__name__}.")
    target.write_bytes(payload)
    try:
        label, label_source = label_for_filename(target.name, UPLOAD_LABELS)
        image = load_rgb_image(target)
    except ImageValidationError as exc:
        print(f"Skipping {target.name}: {exc}")
        continue
    UPLOADED.append(
        {
            "image_id": target.name,
            "label": label,
            "label_source": label_source,
            "image": image,
            "path": target,
        }
    )
    print(f"{target.name}: label={label} ({label_source}), size={image.size[0]}x{image.size[1]}")
    try:
        from IPython.display import Image, display

        display(Image(filename=str(target)))
    except Exception as exc:
        print(f"Could not display {target.name}: {exc}")

if not UPLOADED:
    print("No usable uploads. The report will mark the script-sample portion as incomplete.")
""",
    ),
    _cell(
        "markdown",
        """
## 8. Inference on uploaded images

Each image runs the three locked prompts, one prompt at a time. A bad image is recorded and the notebook continues with the next image. An out-of-memory error names `max_pixels` as the setting to lower. The model is not swapped.
""",
    ),
    _cell(
        "code",
        """
require_previous("MODEL", "PROCESSOR", "RUNTIME", "DECODING", "PATHS")

from src.phase0 import jobs_for_image, run_image

if "UPLOADED" not in globals():
    UPLOADED = []
    print("Upload cell was skipped. No package photos will be sent to the model.")

upload_jobs = jobs_for_image(
    prompts=PROMPTS,
    inference_config=INFERENCE,
    phase0_config=PHASE0,
    include_smoke=False,
)
UPLOAD_RECORDS = []
for item in UPLOADED:
    print("=" * 80)
    print(f"{item['image_id']} ({item['label']})")
    rows = run_image(
        model=MODEL,
        processor=PROCESSOR,
        image=item["image"],
        image_id=item["image_id"],
        image_label=item["label"],
        jobs=upload_jobs,
        decoding=DECODING,
        runtime=RUNTIME,
        jsonl_path=PATHS["jsonl"],
    )
    UPLOAD_RECORDS.extend(rows)
    for record in rows:
        print(f"[{record['prompt_id']}] flag={record['output_flag']} completed={record['inference_completed']}")
        print(record["response"] if record["response"] is not None else record["error"])
        print()
""",
    ),
    _cell(
        "markdown",
        """
## 9. Report and download

The report has two parts:

- **A.** Did the model load, which GPU and versions were used, and what CUDA memory counters reported.
- **B.** The raw prompt text and the raw response. These are not scores.

`recorded_for_manual_review` means the technical records are complete enough for you to read. It does not mean the answers are correct, and it does not close Phase 0 until you have checked them.
""",
    ),
    _cell(
        "code",
        """
require_previous("PATHS", "LOAD_META", "ENVIRONMENT", "DECODING", "OVERRIDE_KEYS")

from src.phase0 import finalize

written = finalize(
    paths=PATHS,
    load_meta=LOAD_META,
    environment=ENVIRONMENT,
    decoding=DECODING,
    decoding_override_keys=OVERRIDE_KEYS,
)
print(f"Report: {written['report']}")
print(f"Predictions: {written['jsonl']}")
print(f"Environment: {written['environment']}")
print(f"Zip: {written['zip']}")
print()
report_text = written["report"].read_text(encoding="utf-8")
print("\\n".join(report_text.splitlines()[:12]))
print()
if IN_COLAB:
    from google.colab import files

    files.download(str(written["zip"]))
else:
    print("Not in Colab. The zip is on disk at the path above.")
""",
    ),
    _cell(
        "markdown",
        """
## How to read the result

| Status in the report | Meaning |
|---|---|
| `load_failed` | The model did not finish loading. Use the error text. Do not switch models after one attempt. |
| `inference_failed` | The load did not produce a non-empty public-image response, or an output flag other than `ok` was the only public result. |
| `public_image_ok_script_samples_incomplete` | The public image ran. Fewer than two Urdu-script and two Latin-script photos have both identification and transcription records. |
| `recorded_for_manual_review` | Those records exist. Read section B yourself. |

Phase 0 is ready for your approval only after you have seen a successful Colab run: float16 load, real image responses, CUDA memory counters, and the downloaded zip. This notebook cannot mark that run as passed from a machine that did not execute it.

Copy `results/environment.json` back into the repository when you review the run. Exact package pins belong in that file because they are the versions the GPU session resolved.
""",
    ),
]


def main() -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "colab": {"name": "phase0_verification.ipynb", "provenance": []},
        },
        "cells": CELLS,
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {DEST}")


if __name__ == "__main__":
    main()
