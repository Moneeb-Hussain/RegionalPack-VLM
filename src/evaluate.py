"""Run the locked 60-image evaluation and keep every raw response.

Importing this module does not download weights. Draft OCR labels do not
block inference. They are not used as transcription ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.artifacts import read_jsonl, utc_now, write_json, write_text, zip_artifacts
from src.config import load_inference_config, load_prompts, load_yaml
from src.dataset import ANGLES, validate_dataset
from src.scoring import ANGLES_NOTE

BENCHMARK_PROMPTS = ("identification", "transcription", "category")


def load_eval_config(root: Path) -> dict[str, Any]:
    config = load_yaml(Path(root) / "configs" / "eval_config.yaml")
    for key in (
        "min_pixels",
        "max_pixels",
        "output_dir",
        "predictions_name",
        "environment_name",
        "report_name",
        "zip_name",
    ):
        if key not in config:
            raise ValueError(f"configs/eval_config.yaml is missing {key}.")
    return config


def evaluation_block_reason(root: Path) -> str:
    """Empty when the 60 photographs can be sent to the model.

    Unverified transcription drafts do not block inference. Structural
    dataset errors still do.
    """
    result = validate_dataset(root)
    if result["errors"]:
        return "Dataset validation failed: " + "; ".join(result["errors"][:8])
    return ""


def image_jobs(root: Path) -> list[dict[str, str]]:
    from src.dataset import _read_csv

    _, manifest = _read_csv(Path(root) / "data" / "product_manifest.csv")
    selected = None
    config_path = Path(root) / "configs" / "eval_config.yaml"
    if config_path.is_file():
        ids = (load_eval_config(root).get("inference_subset") or {}).get("image_ids")
        if ids:
            selected = {str(item) for item in ids}
    jobs = []
    for row in manifest:
        product_id = row["product_id"]
        for angle in ANGLES:
            image_id = f"{product_id}/{angle}"
            if selected is not None and image_id not in selected:
                continue
            jobs.append(
                {
                    "image_id": image_id,
                    "image_label": row.get("comparison_role", ""),
                    "image_path": f"data/raw/{product_id}/{angle}.jpg",
                }
            )
    return jobs


def prompt_jobs(root: Path) -> list[dict[str, Any]]:
    prompts = load_prompts(root)
    inference = load_inference_config(root)
    jobs = []
    for prompt_id in BENCHMARK_PROMPTS:
        jobs.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompts[prompt_id],
                "max_new_tokens": int(inference["max_new_tokens"][prompt_id]),
                "locked_research_prompt": True,
            }
        )
    return jobs


def completed_pairs(records: list[dict[str, Any]]) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    for row in records:
        if row.get("record_type") not in (None, "inference"):
            continue
        if row.get("inference_completed") is True and row.get("image_id") and row.get("prompt_id"):
            done.add((str(row["image_id"]), str(row["prompt_id"])))
    return done


def output_paths(root: Path, eval_config: dict[str, Any]) -> dict[str, Path]:
    directory = Path(root) / eval_config["output_dir"]
    return {
        "dir": directory,
        "jsonl": directory / eval_config["predictions_name"],
        "report": directory / eval_config["report_name"],
        "environment": directory / eval_config["environment_name"],
        "zip": directory / eval_config["zip_name"],
    }


def run_benchmark(root: Path) -> dict[str, Path]:
    """Load the pinned model and append one JSONL row per prompt.

    A disconnect keeps the rows already written. The next launch skips them.
    """
    reason = evaluation_block_reason(root)
    if reason:
        raise RuntimeError(reason)
    from src.environment import probe_gpu, probe_libraries
    from src.inference import apply_pixel_budget, load_qwen3_vl
    from src.phase0 import run_image

    root = Path(root)
    inference = load_inference_config(root)
    eval_config = load_eval_config(root)
    paths = output_paths(root, eval_config)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    if not paths["jsonl"].is_file():
        paths["jsonl"].write_text("", encoding="utf-8")
    done = completed_pairs(read_jsonl(paths["jsonl"]))
    jobs = prompt_jobs(root)
    model, processor, meta = load_qwen3_vl(
        model_id=inference["model_id"],
        revision=str(inference["revision"]),
        attn_implementation=str(eval_config.get("attn_implementation", "sdpa")),
    )
    budget = apply_pixel_budget(
        processor,
        min_pixels=int(eval_config["min_pixels"]),
        max_pixels=int(eval_config["max_pixels"]),
    )
    meta.update(budget)
    environment = {"libraries": probe_libraries(), "gpu": probe_gpu()}
    decoding = dict(inference["decoding"])
    gpu = environment["gpu"]
    libraries = environment["libraries"]
    runtime = {
        "model_id": meta.get("model_id"),
        "revision_resolved": meta.get("revision_resolved"),
        "revision_pinned": meta.get("revision_pinned"),
        "gpu_name": gpu.get("gpu_name"),
        "cuda_version": gpu.get("cuda_version"),
        "pytorch_version": libraries.get("pytorch_version"),
        "transformers_version": libraries.get("transformers_version"),
        "nonfinite_parameter": meta.get("nonfinite_parameter"),
    }
    from src.images import load_rgb_image

    for image in image_jobs(root):
        remaining = [job for job in jobs if (image["image_id"], job["prompt_id"]) not in done]
        if not remaining:
            continue
        loaded = load_rgb_image(root / image["image_path"])
        run_image(
            model=model,
            processor=processor,
            image=loaded,
            image_id=image["image_id"],
            image_label=image["image_label"],
            jobs=remaining,
            decoding=decoding,
            runtime=runtime,
            jsonl_path=paths["jsonl"],
        )
    records = read_jsonl(paths["jsonl"])
    write_text(
        paths["report"],
        "\n".join(
            [
                "# Benchmark run",
                "",
                "Raw model responses are in the JSONL file. This file is not a score.",
                ANGLES_NOTE,
                f"Predictions: {len(records)}",
                f"Model: {meta.get('model_id')} @ {meta.get('revision_resolved')}",
                "",
            ]
        ),
    )
    write_json(
        paths["environment"],
        {
            "generated_utc": utc_now(),
            "environment": environment,
            "load": meta,
            "release": eval_config.get("release", "0.1"),
            "prompts": prompt_jobs(root),
            "decoding": decoding,
            "inference_subset": eval_config.get("inference_subset", {}),
            "ocr_subset": eval_config.get("ocr_subset", {}),
            "angles_note": eval_config.get("angles_note", ANGLES_NOTE),
            "record_count": len(records),
        },
    )
    zip_artifacts(paths["zip"], [paths["report"], paths["jsonl"], paths["environment"]])
    return paths
