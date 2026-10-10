"""Write JSONL predictions, the verification report, and the download zip."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.errors import MeasurementError

SCRIPT_LABELS = ("urdu_script", "latin_script")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise MeasurementError(f"{path} line {line_number} is not valid JSON.") from exc
            if not isinstance(row, dict):
                raise MeasurementError(f"{path} line {line_number} is not a JSON object.")
            rows.append(row)
    return rows


def _mebibytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MiB"


def _require_measured_success(load_meta: dict[str, Any]) -> None:
    if load_meta.get("load_status") != "success":
        return
    peak = load_meta.get("peak_allocated_bytes")
    if not isinstance(peak, int) or peak < 0:
        raise MeasurementError(
            "Refusing to record a successful model load without an integer "
            "peak_allocated_bytes from torch.cuda.max_memory_allocated()."
        )
    if load_meta.get("memory_source") != "torch.cuda.max_memory_allocated":
        raise MeasurementError("Successful load metadata is missing the CUDA memory source.")


def review_status(load_meta: dict[str, Any], records: list[dict[str, Any]]) -> str:
    """Choose a status label. This is not a research conclusion."""
    if load_meta.get("load_status") != "success":
        return "load_failed"
    public_ok = any(
        row.get("image_id") == "public_demo"
        and row.get("inference_completed") is True
        and row.get("output_flag") == "ok"
        for row in records
    )
    if not public_ok:
        return "inference_failed"
    needed = {"identification", "transcription"}
    completed: dict[tuple[str, str], set[str]] = {}
    for row in records:
        label = row.get("image_label")
        prompt_id = row.get("prompt_id")
        if label not in SCRIPT_LABELS or prompt_id not in needed:
            continue
        if row.get("inference_completed") is not True:
            continue
        key = (str(label), str(row.get("image_id")))
        completed.setdefault(key, set()).add(str(prompt_id))
    counts = {label: 0 for label in SCRIPT_LABELS}
    for (label, _image_id), prompt_ids in completed.items():
        if needed <= prompt_ids:
            counts[label] += 1
    if counts["urdu_script"] < 2 or counts["latin_script"] < 2:
        return "public_image_ok_script_samples_incomplete"
    return "recorded_for_manual_review"


def _indent_block(text: str) -> str:
    if text == "":
        return "    "
    return "\n".join(f"    {line}" for line in text.splitlines())


def render_verification_markdown(
    *,
    load_meta: dict[str, Any],
    records: list[dict[str, Any]],
    environment: dict[str, Any],
    decoding: dict[str, Any],
    decoding_override_keys: list[str],
) -> str:
    """Render the Phase 0 report. Section B quotes outputs and does not score them."""
    _require_measured_success(load_meta)
    status = review_status(load_meta, records)
    lines = [
        "# Phase 0 verification",
        "",
        f"Generated (UTC): {utc_now()}",
        f"Status: `{status}`",
        "",
        "This file is a technical smoke test. It does not report category accuracy, brand accuracy,",
        "character error rate, or a result for any hypothesis. OCR quality is not scored here and is",
        "not a model-switching criterion.",
        "",
        "Hardware bfloat16 requires compute capability 8.0 or newer. A Tesla T4 is Turing, capability 7.5.",
        "PyTorch `is_bf16_supported()` can still return true on a T4 because that flag includes emulation.",
        "The loader uses float16. A true emulation flag is not hardware bfloat16 support.",
        "",
        "## A. Technical compatibility",
        "",
        "| Item | Recorded value |",
        "|---|---|",
        f"| GPU name | {environment.get('gpu_name', 'not recorded')} |",
        f"| T4 name confirmed | {environment.get('t4_name_confirmed', 'not recorded')} |",
        f"| CUDA version | {environment.get('cuda_version', 'not recorded')} |",
        f"| PyTorch version | {environment.get('pytorch_version', 'not recorded')} |",
        f"| Transformers version | {environment.get('transformers_version', 'not recorded')} |",
        f"| Compute capability | {environment.get('compute_capability', 'not recorded')} |",
        f"| bfloat16 hardware support | {environment.get('bf16_hardware_supported', environment.get('bf16_supported', 'not recorded'))} |",
        f"| bfloat16 PyTorch flag including emulation | {environment.get('bf16_pytorch_including_emulation', 'not recorded')} |",
        f"| Model id | {load_meta.get('model_id', 'not recorded')} |",
        f"| Revision pinned | {load_meta.get('revision_pinned', 'not recorded')} |",
        f"| Revision resolved | {load_meta.get('revision_resolved', 'not recorded')} |",
        f"| Revision matches pin | {load_meta.get('revision_matches_pin', 'not recorded')} |",
        f"| Dtype requested | {load_meta.get('dtype_requested', 'not recorded')} |",
        f"| Parameter dtypes | {load_meta.get('parameter_dtypes', 'not recorded')} |",
        f"| Load status | {load_meta.get('load_status', 'not recorded')} |",
        f"| Non-finite parameter | {load_meta.get('nonfinite_parameter', 'not recorded')} |",
        f"| Attention implementation | {load_meta.get('attn_implementation', 'not recorded')} |",
        f"| Batch size | {load_meta.get('batch_size', 1)} |",
        f"| Device total memory | {_format_bytes(environment.get('total_vram_bytes'))} |",
        f"| Peak allocated at load | {_format_measured(load_meta.get('peak_allocated_bytes'), load_meta.get('memory_source'))} |",
        f"| Peak allocated during inference | {_format_measured(load_meta.get('inference_peak_allocated_bytes'), load_meta.get('memory_source'))} |",
        f"| Allocated after load | {_format_measured(load_meta.get('allocated_bytes_after_load'), load_meta.get('memory_source'))} |",
        f"| Phase 0 pixel budget | min {load_meta.get('min_pixels')}, max {load_meta.get('max_pixels')} |",
        f"| Decoding override keys | {decoding_override_keys or 'none'} |",
        f"| Decoding | `{json.dumps(decoding, sort_keys=True)}` |",
        f"| Inference records | {len(records)} |",
        f"| Completed inference records | {sum(1 for row in records if row.get('inference_completed') is True)} |",
        "",
        "Device total memory comes from the CUDA device property. Peak and allocated figures come from",
        "PyTorch CUDA counters. Published parameter-count estimates are not used as measurements.",
        "",
        "### Errors and incompatible-runtime notes",
        "",
    ]
    errors = [row for row in records if row.get("error")]
    load_error = load_meta.get("error")
    if load_error:
        lines.append(f"- Load error: {load_error}")
    if not errors and not load_error:
        lines.append("- None recorded.")
    for row in errors:
        lines.append(
            f"- `{row.get('image_id')}` / `{row.get('prompt_id')}`: {row.get('error')}"
        )
    suspicious = [
        row
        for row in records
        if row.get("inference_completed") is True and row.get("output_flag") not in {None, "ok"}
    ]
    if suspicious:
        lines.extend(["", "Output flags other than `ok` (technical check, not an OCR score):", ""])
        for row in suspicious:
            lines.append(
                f"- `{row.get('image_id')}` / `{row.get('prompt_id')}`: `{row.get('output_flag')}`"
            )
    lines.extend(
        [
            "",
            "Fallback model: not loaded. Weak transcription is not a switching trigger.",
            "",
            "## B. Preliminary observations",
            "",
            "The blocks below are raw model responses. They are not scores, and they are not a judgment",
            "of whether the transcription is good enough to keep the model. The public demo image is a",
            "library photograph used to check that pixels reach the model. It is not a retail package",
            "and it is not part of the study dataset.",
            "",
        ]
    )
    if not records:
        lines.append("No inference records were written.")
    current_image = None
    for row in records:
        image_id = row.get("image_id")
        if image_id != current_image:
            current_image = image_id
            lines.extend(
                [
                    f"### {image_id}",
                    "",
                    f"- Label: `{row.get('image_label')}`",
                    f"- SHA-256 of RGB pixels: `{row.get('image_sha256')}`",
                    f"- Size: {row.get('image_width')} x {row.get('image_height')}",
                    "",
                ]
            )
        lines.extend(
            [
                f"#### {row.get('prompt_id')}",
                "",
                f"- Locked research prompt: {row.get('locked_research_prompt')}",
                f"- Timestamp (UTC): {row.get('timestamp_utc')}",
                f"- Output flag: `{row.get('output_flag')}`",
                f"- Inference completed: {row.get('inference_completed')}",
                f"- pixel_values shape: `{row.get('pixel_values_shape')}`",
                f"- pixel_values numel: `{row.get('pixel_values_numel')}`",
                "",
                "Prompt:",
                "",
                _indent_block(str(row.get("prompt", ""))),
                "",
                "Raw response:",
                "",
                _indent_block("" if row.get("response") is None else str(row.get("response"))),
                "",
            ]
        )
    lines.extend(
        [
            "## What this status means",
            "",
            "`recorded_for_manual_review` means the model loaded, the public image produced a non-empty",
            "response, and at least two `urdu_script` images and two `latin_script` images have",
            "identification and transcription records. It does not mean those responses are correct.",
            "",
            "`public_image_ok_script_samples_incomplete` means the technical image path works and the",
            "Urdu / Latin upload portion of the Phase 0 gate still needs samples.",
            "",
            "Switch to Qwen2-VL-2B-Instruct only after Qwen3-VL-2B-Instruct fails to load, crashes, or",
            "produces non-finite or empty output across repeated FP16 attempts on the Colab GPU.",
            "Do not switch because Urdu OCR looks weak.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_bytes(value: object) -> str:
    if not isinstance(value, int):
        return "not recorded"
    return f"{value} bytes ({_mebibytes(value)})"


def _format_measured(value: object, source: object) -> str:
    if not isinstance(value, int):
        return "not measured"
    source_name = source or "torch.cuda measurement missing source label"
    return f"{value} bytes ({_mebibytes(value)}); source: {source_name}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def zip_artifacts(zip_path: Path, files: list[Path]) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            if not file_path.is_file():
                raise MeasurementError(f"Cannot zip missing artifact: {file_path}")
            archive.write(file_path, arcname=file_path.name)
    return zip_path
