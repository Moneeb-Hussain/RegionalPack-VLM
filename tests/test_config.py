"""Config loading. These tests do not download a model."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.config import (
    generation_kwargs,
    load_inference_config,
    load_phase0_config,
    load_phase0_prompts,
    load_prompts,
    merged_decoding,
    prompt_jobs,
)
from src.errors import Phase0Error

ROOT = Path(__file__).resolve().parents[1]

IDENTIFICATION = (
    "What product is shown in this image? If you can identify the specific brand, name it. "
    "If you cannot identify the specific brand with confidence, say so explicitly rather than guessing."
)
TRANSCRIPTION = (
    "Transcribe only the text on the front of the package exactly as written, including any text that is not in English. "
    "Include the brand wordmark, the product or flavor name, and any prominent tagline on the main face. "
    "Do not transcribe ingredient lists, nutrition tables, barcodes, or regulatory fine print. "
    "If no front-of-pack text is legible, say so."
)
PHASE0_TRANSCRIPTION = (
    "Transcribe all text visible on the packaging exactly as written, including any text that is not in English. "
    "If no text is legible, say so."
)
CATEGORY = (
    "What general category of product is this, for example beverage, snack, dairy, or cleaning product? "
    "Do not name a specific brand."
)


def test_benchmark_transcription_is_front_of_pack():
    prompts = load_prompts(ROOT)
    assert prompts["identification"] == IDENTIFICATION
    assert prompts["transcription"] == TRANSCRIPTION
    assert prompts["category"] == CATEGORY
    assert "ingredient lists" in prompts["transcription"]
    assert "all text visible on the packaging" not in prompts["transcription"]


def test_phase0_archive_keeps_the_prompt_that_was_run():
    prompts = load_phase0_prompts(ROOT)
    assert prompts["transcription"] == PHASE0_TRANSCRIPTION
    assert prompts["identification"] == IDENTIFICATION
    assert prompts["category"] == CATEGORY


def test_locked_decoding_and_token_limits():
    config = load_inference_config(ROOT)
    decoding = config["decoding"]
    assert config["model_id"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert config["dtype"] == "float16"
    assert config["batch_size"] == 1
    assert config["revision"] == "89644892e4d85e24eaac8bacfd4f463576704203"
    assert decoding["do_sample"] is True
    assert math.isclose(decoding["temperature"], 0.1)
    assert math.isclose(decoding["top_p"], 1.0)
    assert decoding["top_k"] is None
    assert config["max_new_tokens"] == {
        "identification": 128,
        "category": 128,
        "transcription": 256,
    }


def test_phase0_does_not_override_locked_decoding_by_default():
    phase0 = load_phase0_config(ROOT)
    inference = load_inference_config(ROOT)
    decoding, override_keys = merged_decoding(inference, phase0)
    assert override_keys == []
    assert phase0["decoding_overrides"] == {}
    assert math.isclose(decoding["temperature"], 0.1)
    assert phase0["max_pixels"] >= phase0["min_pixels"]
    assert phase0["prompts"] == ["identification", "transcription", "category"]


def test_prompt_jobs_keep_smoke_prompt_separate():
    prompts = load_prompts(ROOT)
    inference = load_inference_config(ROOT)
    phase0 = load_phase0_config(ROOT)
    public_jobs = prompt_jobs(prompts, inference, phase0, include_smoke=True)
    upload_jobs = prompt_jobs(prompts, inference, phase0, include_smoke=False)
    assert public_jobs[0]["prompt_id"] == "smoke_describe"
    assert public_jobs[0]["locked_research_prompt"] is False
    assert [job["prompt_id"] for job in upload_jobs] == [
        "identification",
        "transcription",
        "category",
    ]
    assert all(job["locked_research_prompt"] for job in upload_jobs)
    assert upload_jobs[0]["prompt"] == IDENTIFICATION


def test_generation_kwargs_include_disabled_top_k():
    inference = load_inference_config(ROOT)
    kwargs = generation_kwargs(inference["decoding"], 128)
    assert kwargs["max_new_tokens"] == 128
    assert kwargs["top_k"] is None
    assert kwargs["do_sample"] is True


def test_missing_prompt_file_raises(tmp_path: Path):
    with pytest.raises(Phase0Error):
        load_prompts(tmp_path)
