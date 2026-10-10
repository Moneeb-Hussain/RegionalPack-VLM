"""Load the locked inference config and the Phase 0 smoke-test config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.errors import Phase0Error

PROMPT_IDS = ("identification", "transcription", "category")
REQUIRED_INFERENCE_KEYS = (
    "model_id",
    "revision",
    "dtype",
    "batch_size",
    "minimum_transformers",
    "decoding",
    "max_new_tokens",
)
REQUIRED_DECODING_KEYS = (
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "seed",
)


def project_root_from(path: Path) -> Path:
    """Return the repository root that contains this config layout."""
    for candidate in (path, *path.parents):
        if (candidate / "configs" / "inference_config.yaml").is_file() and (
            candidate / "prompts" / "prompts.yaml"
        ).is_file():
            return candidate
    raise Phase0Error(
        "Could not find configs/inference_config.yaml and prompts/prompts.yaml. "
        "Upload phase0_colab_bundle.zip or set PROJECT_ROOT_OVERRIDE."
    )


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Phase0Error(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise Phase0Error(f"{path} did not contain a YAML mapping.")
    return loaded


def load_phase0_prompts(root: Path) -> dict[str, str]:
    """Return the prompt text used in the completed Phase 0 run."""
    return _load_prompt_file(root / "prompts" / "phase0_prompts.yaml")


def load_prompts(root: Path) -> dict[str, str]:
    """Return the benchmark prompts, including the front-of-pack transcription prompt."""
    return _load_prompt_file(root / "prompts" / "prompts.yaml")


def _load_prompt_file(path: Path) -> dict[str, str]:
    raw = load_yaml(path)
    prompts: dict[str, str] = {}
    for prompt_id in PROMPT_IDS:
        text = raw.get(prompt_id)
        if not isinstance(text, str) or not text.strip():
            raise Phase0Error(f"{path.name} is missing locked prompt {prompt_id!r}.")
        prompts[prompt_id] = text
    return prompts


def load_inference_config(root: Path) -> dict[str, Any]:
    config = load_yaml(root / "configs" / "inference_config.yaml")
    missing = [key for key in REQUIRED_INFERENCE_KEYS if key not in config]
    if missing:
        raise Phase0Error(f"inference_config.yaml is missing: {', '.join(missing)}")
    decoding = config["decoding"]
    if not isinstance(decoding, dict):
        raise Phase0Error("inference_config.yaml decoding must be a mapping.")
    missing_decoding = [key for key in REQUIRED_DECODING_KEYS if key not in decoding]
    if missing_decoding:
        raise Phase0Error(
            "inference_config.yaml decoding is missing: " + ", ".join(missing_decoding)
        )
    max_new_tokens = config["max_new_tokens"]
    if not isinstance(max_new_tokens, dict):
        raise Phase0Error("max_new_tokens must be a mapping.")
    for prompt_id in PROMPT_IDS:
        value = max_new_tokens.get(prompt_id)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise Phase0Error(f"max_new_tokens.{prompt_id} must be a positive integer.")
    if config["dtype"] != "float16":
        raise Phase0Error("Phase 0 loads with dtype float16. inference_config.yaml does not.")
    if config["batch_size"] != 1:
        raise Phase0Error("Phase 0 uses batch size 1.")
    if config["model_id"] != "Qwen/Qwen3-VL-2B-Instruct":
        raise Phase0Error("inference_config.yaml model_id is not Qwen/Qwen3-VL-2B-Instruct.")
    return config


def load_phase0_config(root: Path) -> dict[str, Any]:
    config = load_yaml(root / "configs" / "phase0_config.yaml")
    overrides = config.get("decoding_overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise Phase0Error("phase0_config.yaml decoding_overrides must be a mapping.")
    config["decoding_overrides"] = overrides
    prompts = config.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise Phase0Error("phase0_config.yaml prompts must be a non-empty list.")
    unknown = [item for item in prompts if item not in PROMPT_IDS]
    if unknown:
        raise Phase0Error(f"Unknown Phase 0 prompt ids: {unknown}")
    smoke = config.get("smoke_prompt")
    if not isinstance(smoke, dict):
        raise Phase0Error("phase0_config.yaml smoke_prompt must be a mapping.")
    for key in ("id", "text", "max_new_tokens"):
        if key not in smoke:
            raise Phase0Error(f"smoke_prompt is missing {key}.")
    if not isinstance(smoke["text"], str) or not smoke["text"].strip():
        raise Phase0Error("smoke_prompt.text must be a non-empty string.")
    if smoke["max_new_tokens"] <= 0:
        raise Phase0Error("smoke_prompt.max_new_tokens must be positive.")
    min_pixels = config.get("min_pixels")
    max_pixels = config.get("max_pixels")
    if not isinstance(min_pixels, int) or not isinstance(max_pixels, int):
        raise Phase0Error("Phase 0 min_pixels and max_pixels must be integers.")
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise Phase0Error("Phase 0 pixel budget is invalid.")
    urls = config.get("public_image_urls")
    if not isinstance(urls, list) or not urls:
        raise Phase0Error("phase0_config.yaml needs at least one public_image_urls entry.")
    return config


def merged_decoding(
    inference_config: dict[str, Any], phase0_config: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Return locked decoding plus any smoke-test overrides, and the override keys."""
    decoding = dict(inference_config["decoding"])
    overrides = dict(phase0_config.get("decoding_overrides") or {})
    decoding.update(overrides)
    return decoding, sorted(overrides)


def prompt_jobs(
    prompts: dict[str, str],
    inference_config: dict[str, Any],
    phase0_config: dict[str, Any],
    *,
    include_smoke: bool,
) -> list[dict[str, Any]]:
    """Build the ordered prompt list for one image."""
    jobs: list[dict[str, Any]] = []
    if include_smoke:
        smoke = phase0_config["smoke_prompt"]
        jobs.append(
            {
                "prompt_id": smoke["id"],
                "prompt": smoke["text"],
                "max_new_tokens": int(smoke["max_new_tokens"]),
                "locked_research_prompt": False,
            }
        )
    for prompt_id in phase0_config["prompts"]:
        jobs.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompts[prompt_id],
                "max_new_tokens": int(inference_config["max_new_tokens"][prompt_id]),
                "locked_research_prompt": True,
            }
        )
    return jobs


def generation_kwargs(decoding: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    return {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(decoding["do_sample"]),
        "temperature": float(decoding["temperature"]),
        "top_p": float(decoding["top_p"]),
        "top_k": decoding["top_k"],
        "repetition_penalty": float(decoding["repetition_penalty"]),
    }
