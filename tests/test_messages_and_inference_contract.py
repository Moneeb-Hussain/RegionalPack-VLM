"""Message shape and static checks that inference does not swap models."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.errors import ImageValidationError, InferenceFailedError
from src.inference import apply_pixel_budget, explain_transformers_import_error, pixel_stats
from src.messages import build_image_message
from src.versions import version_at_least

ROOT = Path(__file__).resolve().parents[1]


def test_message_contains_the_pil_image_itself():
    image = Image.new("RGB", (32, 32), (1, 2, 3))
    prompt = "Transcribe all text visible on the packaging exactly as written."
    message = build_image_message(image, prompt)
    content = message[0]["content"]
    assert content[0] == {"type": "image", "image": image}
    assert content[0]["image"] is image
    assert content[1]["text"] == prompt
    assert "url" not in content[0]
    assert "path" not in content[0]


def test_message_rejects_a_bare_filename():
    image = Image.new("RGB", (32, 32), (1, 2, 3))
    with pytest.raises(ImageValidationError):
        build_image_message(image, "   ")
    with pytest.raises(ImageValidationError):
        build_image_message(image, "hello", image_source=123)  # type: ignore[arg-type]


def test_pixel_budget_keeps_existing_size_keys():
    class ImageProcessor:
        def __init__(self):
            self.size = {"shortest_edge": 4, "longest_edge": 8, "other": 1}
            self.min_pixels = 4
            self.max_pixels = 8

    class Processor:
        def __init__(self):
            self.image_processor = ImageProcessor()

    processor = Processor()
    apply_pixel_budget(processor, 32, 64)
    assert processor.image_processor.size == {
        "shortest_edge": 32,
        "longest_edge": 64,
        "other": 1,
    }
    assert processor.image_processor.max_pixels == 64
    assert processor.image_processor.min_pixels == 32


def test_pixel_stats_require_real_pixel_values():
    class FakeValues:
        shape = (8, 4)
        dtype = "float16"

        def numel(self):
            return 32

    stats = pixel_stats({"pixel_values": FakeValues()})
    assert stats["pixel_values_numel"] == 32
    with pytest.raises(InferenceFailedError, match="pixel_values"):
        pixel_stats({"input_ids": FakeValues()})


def test_inference_module_does_not_load_a_fallback_model():
    source = (ROOT / "src" / "inference.py").read_text(encoding="utf-8")
    phase0 = (ROOT / "src" / "phase0.py").read_text(encoding="utf-8")
    assert "Qwen2-VL" not in source
    assert "Qwen2VL" not in source
    assert "Qwen2-VL" not in phase0
    assert "dtype=torch.float16" in source
    assert "flash_attention" not in source
    assert "dtype=\"auto\"" not in source
    assert "torch.cuda.max_memory_allocated" in (ROOT / "src" / "environment.py").read_text(
        encoding="utf-8"
    )


def test_requirements_do_not_replace_torch_or_add_flash_attention():
    lines = []
    for raw in (ROOT / "requirements-colab.txt").read_text(encoding="utf-8").splitlines():
        code = raw.split("#", 1)[0].strip()
        if code:
            lines.append(code.lower())
    assert any(line.startswith("transformers>=") for line in lines)
    assert not any(line.startswith("pillow") for line in lines)
    assert not any(line.startswith("torch") for line in lines)
    assert not any("flash" in line for line in lines)


def test_ink_import_error_points_at_pillow_12_1():
    message = explain_transformers_import_error(
        ImportError("cannot import name '_Ink' from 'PIL._typing'")
    )
    assert "_Ink" in message
    assert "switch models" in message
    assert "git checkout" not in message


def test_pillow_core_mismatch_tells_user_to_restart_once():
    message = explain_transformers_import_error(
        ImportError(
            "The _imaging extension was built for another version of Pillow or PIL:\n"
            "Core version: 11.3.0\nPillow version: 12.3.0"
        )
    )
    assert "11.3.0" in message
    assert "switch models" in message
    assert "git checkout" not in message


def test_phase0_import_does_not_load_pillow():
    import ast

    tree = ast.parse((ROOT / "src" / "phase0.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert node.module not in {"PIL", "src.inference", "src.images"}
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("PIL") for alias in node.names)


def test_pillow_pin_matches_the_loaded_extension():
    from src.environment import pillow_pin_for_loaded_core

    assert pillow_pin_for_loaded_core("11.3.0") == "pillow==11.3.0"
    assert pillow_pin_for_loaded_core("12.3.0") == "pillow==12.3.0"
    assert pillow_pin_for_loaded_core(None) == "pillow>=12.1.0,<13"


def test_fresh_process_keeps_the_installed_pillow():
    from src.environment import pillow_pin_for_installed_package

    assert pillow_pin_for_installed_package("11.3.0") == "pillow==11.3.0"
    assert pillow_pin_for_installed_package(None) == "pillow>=12.1.0,<13"


def test_pillow_process_status_keeps_a_matching_import():
    from src.environment import pillow_process_status

    status = pillow_process_status()
    assert status["ok"] is True
    assert status["version"]
    assert status["error"] is None


def test_pillow_disk_check_imports_matching_extension():
    from src.environment import assert_pillow_extension_matches, verify_pillow_install

    disk_version = verify_pillow_install()
    loaded_version = assert_pillow_extension_matches()
    assert disk_version
    assert loaded_version == disk_version


def test_other_import_error_still_names_the_transformers_floor():
    message = explain_transformers_import_error(ImportError("No module named transformers"))
    assert "transformers>=4.57.0" in message


def test_transformers_version_floor():
    assert version_at_least("4.57.0", "4.57.0")
    assert version_at_least("4.57.1", "4.57.0")
    assert version_at_least("5.18.0", "4.57.0")
    assert not version_at_least("4.56.2", "4.57.0")
    assert version_at_least("4.57.0+cu124", "4.57.0")
