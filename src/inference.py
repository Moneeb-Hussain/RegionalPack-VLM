"""Load Qwen3-VL-2B-Instruct and run one image prompt.

This module does not download or import transformers until a load function is called.
A failed load stays a failed load. This module does not load a substitute model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from src.environment import cuda_memory_bytes, reset_peak_memory
from src.errors import InferenceFailedError, ModelLoadError, TransformersVersionError
from src.messages import build_image_message
from src.outputs import inspect_response

MEMORY_SOURCE = "torch.cuda.max_memory_allocated"


def explain_transformers_import_error(exc: BaseException) -> str:
    """Describe an import failure without treating a Pillow bug as a model failure."""
    text = str(exc)
    if "Core version" in text or "_imaging" in text:
        return (
            "Could not import Qwen3-VL because Pillow's Python files and its compiled "
            "_imaging extension are different builds. Colab's extension is 11.3.0. "
            "The install cell pins Pillow to that extension instead of upgrading it. "
            "Re-run from the install cell. This is a library mismatch, not a reason to switch models. "
            f"Original error: {exc}"
        )
    if "_Ink" in text or "PIL._typing" in text:
        return (
            "Could not import Qwen3-VL because Pillow is missing PIL._typing._Ink. "
            "The install cell adds that alias to the Pillow build already loaded. "
            "Re-run from the install cell. This is a library mismatch, not a reason to switch models. "
            f"Original error: {exc}"
        )
    return (
        "Could not import Qwen3VLForConditionalGeneration. Install transformers>=4.57.0 "
        "from PyPI. Building transformers from a git checkout is not the default path. "
        f"Original error: {exc}"
    )


def _import_model_classes():
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise TransformersVersionError(explain_transformers_import_error(exc)) from exc
    return Qwen3VLForConditionalGeneration, AutoProcessor


def _dtype_histogram(model) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in model.parameters():
        name = str(parameter.dtype)
        counts[name] = counts.get(name, 0) + 1
    return counts


def find_nonfinite_parameter(model) -> str | None:
    import torch

    for name, parameter in model.named_parameters():
        if parameter.is_floating_point() and not bool(torch.isfinite(parameter).all()):
            return name
    return None


def apply_pixel_budget(processor, min_pixels: int, max_pixels: int) -> dict[str, Any]:
    """Set the Phase 0 image-size cap on the processor. Returns the previous size."""
    image_processor = processor.image_processor
    previous = getattr(image_processor, "size", None)
    if previous is not None and hasattr(previous, "longest_edge"):
        previous.longest_edge = int(max_pixels)
        if hasattr(previous, "shortest_edge"):
            previous.shortest_edge = int(min_pixels)
    elif isinstance(previous, dict):
        previous["longest_edge"] = int(max_pixels)
        previous["shortest_edge"] = int(min_pixels)
    else:
        image_processor.size = {
            "longest_edge": int(max_pixels),
            "shortest_edge": int(min_pixels),
        }
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = int(max_pixels)
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = int(min_pixels)
    printable = previous if isinstance(previous, dict) else str(previous)
    return {
        "previous_size": printable,
        "applied_min_pixels": int(min_pixels),
        "applied_max_pixels": int(max_pixels),
    }


def assess_loaded_revision(pinned: str, resolved: str | None) -> dict[str, Any]:
    """Compare the snapshot Hugging Face actually loaded with the pinned commit."""
    pinned_text = str(pinned).strip()
    resolved_text = None if resolved is None else str(resolved).strip() or None
    if resolved_text is None:
        status = "missing"
        matches = False
    elif resolved_text == pinned_text:
        status = "match"
        matches = True
    else:
        status = "mismatch"
        matches = False
    return {
        "revision_pinned": pinned_text,
        "revision_resolved": resolved_text,
        "revision_matches_pin": matches,
        "revision_check": status,
    }


def load_qwen3_vl(
    *,
    model_id: str,
    revision: str,
    attn_implementation: str = "sdpa",
) -> tuple[Any, Any, dict[str, Any]]:
    """Load the selected model in float16 on the current CUDA device."""
    import torch

    if model_id != "Qwen/Qwen3-VL-2B-Instruct":
        raise ModelLoadError(
            f"Refusing to load {model_id}. Phase 0 only loads Qwen/Qwen3-VL-2B-Instruct."
        )
    model_cls, processor_cls = _import_model_classes()
    reset_peak_memory()
    load_kwargs = {
        "revision": revision,
        "device_map": "cuda:0",
        "attn_implementation": attn_implementation,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
    }
    notes: list[str] = []
    try:
        try:
            model = model_cls.from_pretrained(model_id, dtype=torch.float16, **load_kwargs)
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            notes.append("from_pretrained rejected dtype; retried with torch_dtype=torch.float16.")
            model = model_cls.from_pretrained(model_id, torch_dtype=torch.float16, **load_kwargs)
    except TypeError as exc:
        if "attn_implementation" not in str(exc):
            raise ModelLoadError(
                "Qwen3-VL-2B-Instruct failed to load. The fallback model was not loaded. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        notes.append(f"attn_implementation={attn_implementation!r} was rejected; retried without it.")
        load_kwargs.pop("attn_implementation")
        try:
            try:
                model = model_cls.from_pretrained(model_id, dtype=torch.float16, **load_kwargs)
            except TypeError as inner:
                if "dtype" not in str(inner):
                    raise
                notes.append(f"Second load used torch_dtype after: {inner}")
                model = model_cls.from_pretrained(
                    model_id, torch_dtype=torch.float16, **load_kwargs
                )
        except Exception as exc:
            raise ModelLoadError(
                "Qwen3-VL-2B-Instruct failed to load after retrying without the attention "
                "implementation argument. The fallback model was not loaded. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
    except Exception as exc:
        raise ModelLoadError(
            "Qwen3-VL-2B-Instruct failed to load or download. The fallback model was not loaded. "
            "Check the Colab GPU runtime, the transformers install, and network access to "
            f"Hugging Face. Original error: {type(exc).__name__}: {exc}"
        ) from exc

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    histogram = _dtype_histogram(model)
    if any(parameter.dtype == torch.bfloat16 for parameter in model.parameters()):
        raise ModelLoadError(
            "The loaded model still has bfloat16 parameters. A T4 cannot run bfloat16 compute. "
            f"Dtype histogram: {histogram}. The fallback model was not loaded."
        )
    if not any(parameter.dtype == torch.float16 for parameter in model.parameters()):
        raise ModelLoadError(
            "The load completed but no float16 parameters were found. "
            f"Dtype histogram: {histogram}. The fallback model was not loaded."
        )

    try:
        processor = processor_cls.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise ModelLoadError(
            "The model weights loaded but AutoProcessor failed. The fallback model was not loaded. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    memory = cuda_memory_bytes() or {}
    print("Scanning loaded parameters for non-finite values.")
    revision_meta = assess_loaded_revision(revision, getattr(model.config, "_commit_hash", None))
    if revision_meta["revision_check"] != "match":
        raise ModelLoadError(
            "The loaded model snapshot does not match the pinned revision "
            f"{revision_meta['revision_pinned']}. "
            f"Resolved commit: {revision_meta['revision_resolved']!r} "
            f"({revision_meta['revision_check']}). "
            "The fallback model was not loaded."
        )
    meta = {
        "model_id": model_id,
        "revision_pinned": revision_meta["revision_pinned"],
        "revision_resolved": revision_meta["revision_resolved"],
        "revision_matches_pin": revision_meta["revision_matches_pin"],
        "revision_check": revision_meta["revision_check"],
        "dtype_requested": "float16",
        "parameter_dtypes": histogram,
        "load_status": "success",
        "error": None,
        "attn_implementation": load_kwargs.get("attn_implementation", "default"),
        "load_notes": notes,
        "nonfinite_parameter": find_nonfinite_parameter(model),
        "memory_source": MEMORY_SOURCE,
        "peak_allocated_bytes": memory.get("peak_allocated_bytes"),
        "allocated_bytes_after_load": memory.get("allocated_bytes"),
        "reserved_bytes_after_load": memory.get("reserved_bytes"),
        "batch_size": 1,
    }
    if meta["nonfinite_parameter"]:
        meta["load_status"] = "nonfinite_weights"
        meta["error"] = (
            "A model parameter is NaN or infinite after the float16 load: "
            f"{meta['nonfinite_parameter']}. This is a technical incompatibility signal. "
            "The fallback model was not loaded."
        )
    return model, processor, meta


def pixel_stats(inputs) -> dict[str, Any]:
    if "pixel_values" not in inputs:
        raise InferenceFailedError(
            "The processor did not return pixel_values, so the model would not see the image. "
            "The message must use the official image content item, not a filename in the prompt."
        )
    values = inputs["pixel_values"]
    numel = int(values.numel())
    if numel <= 0:
        raise InferenceFailedError("pixel_values is empty, so no image pixels were prepared.")
    shape = list(values.shape)
    return {
        "pixel_values_shape": shape,
        "pixel_values_numel": numel,
        "pixel_values_dtype": str(values.dtype),
    }


def _move_to_device(inputs, device):
    try:
        return inputs.to(device)
    except Exception as exc:
        raise InferenceFailedError(
            "Could not move the processor outputs onto the GPU. "
            f"Keys: {list(inputs.keys())}. Original error: {type(exc).__name__}: {exc}"
        ) from exc


def _decode(processor, generated_ids, input_ids) -> str:
    if generated_ids.shape[0] != 1:
        raise InferenceFailedError(
            f"Expected batch size 1, got {generated_ids.shape[0]} sequences."
        )
    trimmed = generated_ids[:, input_ids.shape[-1] :]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def generate_one(
    *,
    model,
    processor,
    image: Image.Image,
    prompt: str,
    generate_kwargs: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Run one prompt on one PIL image. Batch size is one."""
    import torch

    reset_peak_memory()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    passing_mode = "pil"
    notes: list[str] = []
    try:
        result, passing_mode, notes = _generate_with_fallback(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            generate_kwargs=generate_kwargs,
        )
    except InferenceFailedError:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        memory = cuda_memory_bytes() or {}
        torch.cuda.empty_cache()
        raise InferenceFailedError(
            "CUDA out of memory during generation. The fallback model was not loaded. "
            f"Peak allocated bytes at the failure: {memory.get('peak_allocated_bytes')}. "
            "Lower configs/phase0_config.yaml max_pixels and rerun this notebook. "
            f"Original error: {exc}"
        ) from exc
    except Exception as exc:
        raise InferenceFailedError(
            "Generation failed. The original exception is included and the fallback model was not loaded. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    text = result["text"]
    flag = inspect_response(text)
    memory = cuda_memory_bytes() or {}
    return {
        "response": text,
        "output_flag": flag,
        "inference_completed": True,
        "error": None,
        "image_passing_mode": passing_mode,
        "generation_notes": notes,
        "pixel_values_shape": result["pixel_values_shape"],
        "pixel_values_numel": result["pixel_values_numel"],
        "pixel_values_dtype": result["pixel_values_dtype"],
        "allocated_bytes_after": memory.get("allocated_bytes"),
        "peak_allocated_bytes": memory.get("peak_allocated_bytes"),
        "memory_source": MEMORY_SOURCE,
    }


def _generate_with_fallback(*, model, processor, image, prompt, generate_kwargs):
    import torch

    notes: list[str] = []
    try:
        result = _generate_from_source(model, processor, image, prompt, generate_kwargs, image)
        notes.extend(result.pop("generation_notes", []))
        return result, "pil", notes
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as first_error:
        notes.append(
            "Passing the PIL image failed; retrying with a temporary PNG path. "
            f"Original error: {type(first_error).__name__}: {first_error}"
        )
        with tempfile.TemporaryDirectory(prefix="regionalpack-phase0-") as directory:
            path = Path(directory) / "image.png"
            image.save(path, format="PNG")
            try:
                result = _generate_from_source(
                    model, processor, image, prompt, generate_kwargs, str(path)
                )
                notes.extend(result.pop("generation_notes", []))
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as second_error:
                raise InferenceFailedError(
                    "The processor rejected both a PIL image and a PNG path. "
                    f"PIL error: {type(first_error).__name__}: {first_error}. "
                    f"Path error: {type(second_error).__name__}: {second_error}"
                ) from second_error
        return result, "path", notes


def _generate_from_source(model, processor, image, prompt, generate_kwargs, source):
    import torch

    messages = build_image_message(image, prompt, image_source=source)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if hasattr(inputs, "pop"):
        inputs.pop("token_type_ids", None)
    stats = pixel_stats(inputs)
    inputs = _move_to_device(inputs, model.device)
    if inputs["input_ids"].shape[0] != 1:
        raise InferenceFailedError("The processor built a batch larger than 1.")
    call_kwargs = dict(generate_kwargs)
    generation_notes: list[str] = []
    try:
        with torch.inference_mode():
            generated = model.generate(**inputs, **call_kwargs)
    except (TypeError, ValueError) as exc:
        if call_kwargs.get("top_k") is None and "top_k" in str(exc):
            call_kwargs["top_k"] = 0
            generation_notes.append("top_k=None was rejected; retried with top_k=0.")
            with torch.inference_mode():
                generated = model.generate(**inputs, **call_kwargs)
        else:
            raise
    text = _decode(processor, generated, inputs["input_ids"])
    del generated
    del inputs
    return {"text": text, "generation_notes": generation_notes, **stats}
