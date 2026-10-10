"""Run the Phase 0 smoke test and write its artifacts.

Importing this module does not download model weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

from src.artifacts import (
    append_jsonl,
    read_jsonl,
    render_verification_markdown,
    utc_now,
    write_json,
    write_text,
    zip_artifacts,
)
from src.config import generation_kwargs, prompt_jobs
from src.environment import cuda_memory_bytes
from src.errors import InferenceFailedError, ModelLoadError

MEMORY_SOURCE = "torch.cuda.max_memory_allocated"


def output_paths(root: Path, phase0_config: dict[str, Any]) -> dict[str, Path]:
    output_dir = root / phase0_config["output_dir"]
    return {
        "dir": output_dir,
        "jsonl": output_dir / phase0_config["predictions_name"],
        "report": output_dir / phase0_config["report_name"],
        "environment": output_dir / phase0_config["environment_name"],
        "zip": output_dir / "phase0_artifacts.zip",
        "uploads": output_dir / "uploads",
    }


def load_for_phase0(
    *,
    inference_config: dict[str, Any],
    phase0_config: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    from src.inference import apply_pixel_budget, load_qwen3_vl

    model, processor, meta = load_qwen3_vl(
        model_id=inference_config["model_id"],
        revision=str(inference_config["revision"]),
        attn_implementation=str(phase0_config.get("attn_implementation", "sdpa")),
    )
    budget = apply_pixel_budget(
        processor,
        min_pixels=int(phase0_config["min_pixels"]),
        max_pixels=int(phase0_config["max_pixels"]),
    )
    meta.update(budget)
    meta["min_pixels"] = int(phase0_config["min_pixels"])
    meta["max_pixels"] = int(phase0_config["max_pixels"])
    meta["batch_size"] = 1
    meta["inference_peak_allocated_bytes"] = None
    return model, processor, meta


def failure_load_meta(model_id: str, revision: str, exc: Exception) -> dict[str, Any]:
    memory = None
    try:
        memory = cuda_memory_bytes()
    except Exception:
        memory = None
    return {
        "model_id": model_id,
        "revision_pinned": revision,
        "revision_resolved": None,
        "dtype_requested": "float16",
        "parameter_dtypes": None,
        "load_status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "attn_implementation": None,
        "load_notes": [],
        "nonfinite_parameter": None,
        "memory_source": MEMORY_SOURCE,
        "peak_allocated_bytes": None if memory is None else memory.get("peak_allocated_bytes"),
        "allocated_bytes_after_load": None if memory is None else memory.get("allocated_bytes"),
        "batch_size": 1,
        "min_pixels": None,
        "max_pixels": None,
        "inference_peak_allocated_bytes": None,
    }


def run_image(
    *,
    model,
    processor,
    image: Image.Image,
    image_id: str,
    image_label: str,
    jobs: list[dict[str, Any]],
    decoding: dict[str, Any],
    runtime: dict[str, Any],
    jsonl_path: Path,
) -> list[dict[str, Any]]:
    """Run each prompt at batch size 1 and append a JSONL row immediately."""
    from src.images import pixel_sha256
    from src.inference import generate_one

    if runtime.get("nonfinite_parameter"):
        raise ModelLoadError(
            "Stopping inference because a loaded parameter is non-finite: "
            f"{runtime['nonfinite_parameter']}. The fallback model was not loaded."
        )
    written: list[dict[str, Any]] = []
    image_hash = pixel_sha256(image)
    width, height = image.size
    for job in jobs:
        record = {
            "record_type": "inference",
            "image_id": image_id,
            "image_label": image_label,
            "image_sha256": image_hash,
            "image_width": width,
            "image_height": height,
            "prompt_id": job["prompt_id"],
            "prompt": job["prompt"],
            "locked_research_prompt": job["locked_research_prompt"],
            "response": None,
            "output_flag": None,
            "error": None,
            "model_id": runtime["model_id"],
            "model_revision": runtime.get("revision_resolved") or runtime.get("revision_pinned"),
            "decoding": {
                "do_sample": decoding["do_sample"],
                "temperature": decoding["temperature"],
                "top_p": decoding["top_p"],
                "top_k": decoding["top_k"],
                "repetition_penalty": decoding["repetition_penalty"],
                "seed": decoding["seed"],
            },
            "max_new_tokens": job["max_new_tokens"],
            "batch_size": 1,
            "timestamp_utc": utc_now(),
            "gpu_name": runtime.get("gpu_name"),
            "cuda_version": runtime.get("cuda_version"),
            "pytorch_version": runtime.get("pytorch_version"),
            "transformers_version": runtime.get("transformers_version"),
            "dtype": "float16",
            "inference_completed": False,
        }
        try:
            generated = generate_one(
                model=model,
                processor=processor,
                image=image,
                prompt=job["prompt"],
                generate_kwargs=generation_kwargs(decoding, job["max_new_tokens"]),
                seed=int(decoding["seed"]),
            )
        except (InferenceFailedError, ModelLoadError) as exc:
            record["error"] = str(exc)
            record["output_flag"] = "error"
            record["inference_completed"] = False
        else:
            record.update(generated)
        append_jsonl(jsonl_path, record)
        written.append(record)
    return written


def jobs_for_image(
    *,
    prompts: dict[str, str],
    inference_config: dict[str, Any],
    phase0_config: dict[str, Any],
    include_smoke: bool,
) -> list[dict[str, Any]]:
    return prompt_jobs(
        prompts,
        inference_config,
        phase0_config,
        include_smoke=include_smoke,
    )


def finalize(
    *,
    paths: dict[str, Path],
    load_meta: dict[str, Any],
    environment: dict[str, Any],
    decoding: dict[str, Any],
    decoding_override_keys: list[str],
) -> dict[str, Path]:
    """Write the report, environment record, and zip from the JSONL on disk."""
    paths["jsonl"].parent.mkdir(parents=True, exist_ok=True)
    if not paths["jsonl"].is_file():
        paths["jsonl"].write_text("", encoding="utf-8")
    records = read_jsonl(paths["jsonl"])
    peaks = [
        row["peak_allocated_bytes"]
        for row in records
        if isinstance(row.get("peak_allocated_bytes"), int)
    ]
    if peaks:
        load_meta["inference_peak_allocated_bytes"] = max(peaks)
    elif load_meta.get("inference_peak_allocated_bytes") is None:
        load_meta["inference_peak_allocated_bytes"] = None
    report = render_verification_markdown(
        load_meta=load_meta,
        records=records,
        environment=environment,
        decoding=decoding,
        decoding_override_keys=decoding_override_keys,
    )
    write_text(paths["report"], report)
    environment_record = {
        "generated_utc": utc_now(),
        "environment": environment,
        "load": {
            key: load_meta.get(key)
            for key in (
                "model_id",
                "revision_pinned",
                "revision_resolved",
                "dtype_requested",
                "parameter_dtypes",
                "load_status",
                "error",
                "attn_implementation",
                "load_notes",
                "nonfinite_parameter",
                "memory_source",
                "peak_allocated_bytes",
                "allocated_bytes_after_load",
                "inference_peak_allocated_bytes",
                "min_pixels",
                "max_pixels",
                "batch_size",
            )
        },
        "decoding": decoding,
        "decoding_override_keys": decoding_override_keys,
        "record_count": len(records),
    }
    write_json(paths["environment"], environment_record)
    zip_artifacts(
        paths["zip"],
        [paths["report"], paths["jsonl"], paths["environment"]],
    )
    return paths
