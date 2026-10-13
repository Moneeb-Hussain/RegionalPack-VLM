"""Bundle zip safety and static notebook checks. No model execution."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import zipfile
from pathlib import Path

import pytest

from src.bundle import safe_extract_zip
from src.errors import Phase0Error

ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    path = ROOT / "scripts" / "build_phase0_bundle.py"
    spec = importlib.util.spec_from_file_location("build_phase0_bundle", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bundle_round_trip(tmp_path: Path):
    builder = _load_builder()
    zip_path = builder.build_bundle(dest=tmp_path / "phase0_colab_bundle.zip")
    extracted = safe_extract_zip(zip_path, tmp_path / "unpacked")
    assert (extracted / "src" / "phase0.py").is_file()
    assert (extracted / "prompts" / "prompts.yaml").is_file()
    assert (extracted / "configs" / "inference_config.yaml").is_file()
    assert (extracted / "configs" / "collection_config.yaml").is_file()
    assert (extracted / "data" / "phase0_seen_products.csv").is_file()


def test_extract_rejects_zip_slip(tmp_path: Path):
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../outside.txt", "nope")
    with pytest.raises(Phase0Error, match="unsafe"):
        safe_extract_zip(zip_path, tmp_path / "out")


def test_notebook_cells_parse_and_stay_in_order():
    notebook_path = ROOT / "notebooks" / "phase0_verification.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] >= 4
    code_cells = []
    for index, cell in enumerate(notebook["cells"]):
        assert cell["cell_type"] in {"code", "markdown"}
        source = "".join(cell["source"])
        assert source.strip()
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            ast.parse(source)
            code_cells.append(source)
            if index > 0:
                assert notebook["cells"][index - 1]["cell_type"] in {"code", "markdown"}
    joined = "\n".join(code_cells)
    order_markers = [
        "phase0_colab_bundle.zip",
        "ensure_pillow(",
        "probe_libraries()",
        "probe_gpu()",
        "fetch_first_available_image",
        "load_for_phase0",
        'image_id="public_demo"',
        "UPLOAD_LABELS",
        'report_text.splitlines()[:12]',
    ]
    positions = []
    for marker in order_markers:
        position = joined.find(marker)
        assert position != -1, marker
        positions.append(position)
    assert positions == sorted(positions)
    assert 'dtype="auto"' not in joined
    assert "flash_attn" not in joined
    assert "Qwen2-VL" not in joined
    assert re.search(r"hf_[A-Za-z0-9]{8,}", joined) is None
    assert "api_key" not in joined
    assert joined.find("finalize") < joined.find("files.download")


def _notebook_code_cells() -> list[str]:
    notebook = json.loads((ROOT / "notebooks" / "phase0_verification.ipynb").read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_colab_setup_rejects_a_stale_extract():
    setup = _notebook_code_cells()[0]
    assert "def root_has_pillow_fix" in setup
    assert "def pillow_pin_for_installed_package" in setup
    assert setup.find("root_has_pillow_fix") < setup.find("files.upload")
    gpu = next(source for source in _notebook_code_cells() if "probe_gpu" in source)
    assert gpu.find('module_name.startswith("src.")') < gpu.find("probe_gpu()")


def test_install_cell_checks_pillow_before_importing_transformers():
    install_cells = [
        source
        for source in _notebook_code_cells()
        if "probe_libraries" in source and "ensure_pillow(" in source
    ]
    assert len(install_cells) == 1
    source = install_cells[0]
    assert "probe_gpu" not in source
    assert "loaded_before" not in source
    assert "Restart session" not in source
    assert source.find("ensure_pillow(") < source.find("probe_libraries()")
    environment = (ROOT / "src" / "environment.py").read_text(encoding="utf-8")
    assert "def ensure_pillow" in environment
    assert 'return f"pillow=={core}"' in environment
    assert "_Ink = float | tuple[int, ...] | str" in environment


def test_load_failure_downloads_zip_before_stopping():
    load_cells = [source for source in _notebook_code_cells() if "def publish_failure_zip" in source]
    assert len(load_cells) == 1
    source = load_cells[0]
    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "publish_failure_zip"
    )
    helper_source = ast.get_source_segment(source, helper)
    assert helper_source is not None
    assert 'files.download(str(written["zip"]))' in helper_source
    assert helper_source.find("finalize(") < helper_source.find("files.download")
    assert not any(isinstance(node, ast.Raise) for node in ast.walk(helper))

    call_sites = source.split("publish_failure_zip(LOAD_META)")
    assert len(call_sites) == 3
    assert "files.download" in call_sites[0]
    for tail in call_sites[1:]:
        raise_at = tail.find("raise")
        assert raise_at != -1
        between = tail[:raise_at]
        assert "PHASE0_CONTINUE = False" in between
        assert "files.download" not in between


def test_readme_links_to_the_repo_copy_of_the_spec():
    spec = ROOT / "docs" / "research_spec.md"
    approved = ROOT / "RegionalPack-Cursor-Spec" / "docs" / "research_spec.md"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert spec.is_file()
    assert spec.read_bytes() == approved.read_bytes()
    assert "docs/research_spec.md" in readme
    assert "RegionalPack-Cursor-Spec/docs/research_spec.md" not in readme
    assert (ROOT / "docs" / "media" / "image1.png").is_file()
