"""JSONL, report text, and the download zip. No model weights."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.artifacts import (
    append_jsonl,
    read_jsonl,
    render_verification_markdown,
    review_status,
    zip_artifacts,
)
from src.errors import MeasurementError
from src.outputs import inspect_response
from src.phase0 import finalize, output_paths


def _load_meta(**overrides):
    meta = {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "revision_pinned": "abc",
        "revision_resolved": "abc",
        "dtype_requested": "float16",
        "parameter_dtypes": {"torch.float16": 10},
        "load_status": "success",
        "error": None,
        "attn_implementation": "sdpa",
        "nonfinite_parameter": None,
        "memory_source": "torch.cuda.max_memory_allocated",
        "peak_allocated_bytes": 123456789,
        "allocated_bytes_after_load": 100000000,
        "inference_peak_allocated_bytes": 130000000,
        "min_pixels": 65536,
        "max_pixels": 1003520,
        "batch_size": 1,
    }
    meta.update(overrides)
    return meta


def _record(**overrides):
    row = {
        "image_id": "public_demo",
        "image_label": "public_demo",
        "image_sha256": "a" * 64,
        "image_width": 32,
        "image_height": 24,
        "prompt_id": "smoke_describe",
        "prompt": "Describe this image.",
        "locked_research_prompt": False,
        "response": "A photograph of a person and a dog.",
        "output_flag": "ok",
        "error": None,
        "inference_completed": True,
        "timestamp_utc": "2026-10-10T12:00:00+00:00",
        "pixel_values_shape": [16, 32],
        "pixel_values_numel": 512,
    }
    row.update(overrides)
    return row


def test_jsonl_appends_and_keeps_earlier_rows(tmp_path: Path):
    path = tmp_path / "phase0_predictions.jsonl"
    append_jsonl(path, {"image_id": "first", "response": "اردو"})
    append_jsonl(path, {"image_id": "second", "response": "cola"})
    rows = read_jsonl(path)
    assert [row["image_id"] for row in rows] == ["first", "second"]
    assert rows[0]["response"] == "اردو"
    assert path.read_text(encoding="utf-8").count("\n") == 2


def test_report_splits_compatibility_from_raw_responses():
    report = render_verification_markdown(
        load_meta=_load_meta(),
        records=[_record()],
        environment={
            "gpu_name": "Tesla T4",
            "t4_name_confirmed": True,
            "cuda_version": "12.4",
            "pytorch_version": "2.5.0",
            "transformers_version": "4.57.1",
            "bf16_supported": False,
            "total_vram_bytes": 15800000000,
        },
        decoding={"temperature": 0.1, "top_p": 1.0, "top_k": None},
        decoding_override_keys=[],
    )
    assert "## A. Technical compatibility" in report
    assert "## B. Preliminary observations" in report
    assert "123456789 bytes" in report
    assert "torch.cuda.max_memory_allocated" in report
    assert "A photograph of a person and a dog." in report
    assert "not a model-switching criterion" in report
    assert "hypothesis" not in report.lower() or "any hypothesis" in report
    assert "character error rate" in report
    assert "accuracy" in report  # the disclaimer says it does not report accuracy
    assert "Status: `public_image_ok_script_samples_incomplete`" in report
    assert "Hardware bfloat16 requires compute capability 8.0" in report
    assert "| bfloat16 hardware support | False |" in report


def test_success_without_a_measured_peak_is_rejected():
    with pytest.raises(MeasurementError):
        render_verification_markdown(
            load_meta=_load_meta(peak_allocated_bytes="about 4.6 GB"),
            records=[],
            environment={},
            decoding={},
            decoding_override_keys=[],
        )


def test_review_status_counts_only_images_with_both_research_prompts():
    records = [_record()]
    assert review_status(_load_meta(), records) == "public_image_ok_script_samples_incomplete"
    for index, label in enumerate(("urdu_script", "urdu_script", "latin_script", "latin_script")):
        for prompt_id in ("identification", "transcription"):
            records.append(
                _record(
                    image_id=f"{label}_{index}.jpg",
                    image_label=label,
                    prompt_id=prompt_id,
                    locked_research_prompt=True,
                    response="raw text",
                )
            )
    assert review_status(_load_meta(), records) == "recorded_for_manual_review"
    assert review_status(_load_meta(load_status="failed"), records) == "load_failed"


def test_zip_contains_the_three_artifacts(tmp_path: Path):
    root = tmp_path
    (root / "configs").mkdir()
    (root / "prompts").mkdir()
    phase0_config = {
        "output_dir": "results",
        "predictions_name": "phase0_predictions.jsonl",
        "report_name": "phase0_verification.md",
        "environment_name": "environment.json",
    }
    paths = output_paths(root, phase0_config)
    append_jsonl(paths["jsonl"], _record())
    written = finalize(
        paths=paths,
        load_meta=_load_meta(),
        environment={"gpu_name": "Tesla T4", "total_vram_bytes": 1},
        decoding={"temperature": 0.1, "top_k": None},
        decoding_override_keys=[],
    )
    with zipfile.ZipFile(written["zip"]) as archive:
        assert sorted(archive.namelist()) == [
            "environment.json",
            "phase0_predictions.jsonl",
            "phase0_verification.md",
        ]
    assert "Tesla T4" in written["report"].read_text(encoding="utf-8")


def test_inspect_response_does_not_score_meaning():
    assert inspect_response("  ") == "empty"
    assert inspect_response(None) == "empty"
    assert inspect_response("کوکا کولا") == "ok"
    assert inspect_response("a" * 30) == "repeated_character"


def test_zip_refuses_a_missing_file(tmp_path: Path):
    with pytest.raises(MeasurementError):
        zip_artifacts(tmp_path / "out.zip", [tmp_path / "missing.md"])
