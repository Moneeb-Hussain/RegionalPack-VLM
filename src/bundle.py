"""Find the project root and safely unpack the Colab upload zip."""

from __future__ import annotations

import zipfile
from pathlib import Path

from src.errors import Phase0Error

REQUIRED_RELATIVE_FILES = (
    "src/phase0.py",
    "configs/inference_config.yaml",
    "configs/phase0_config.yaml",
    "prompts/prompts.yaml",
    "requirements-colab.txt",
)


def find_project_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if all((candidate / relative).is_file() for relative in REQUIRED_RELATIVE_FILES):
            return candidate
    if start.is_dir():
        for child in sorted(start.iterdir()):
            if child.is_dir() and all((child / relative).is_file() for relative in REQUIRED_RELATIVE_FILES):
                return child
    return None


def safe_extract_zip(zip_path: Path, dest: Path) -> Path:
    """Extract a bundle and return the directory that contains the project files."""
    if not zip_path.is_file():
        raise Phase0Error(f"Bundle not found: {zip_path}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            parts = Path(info.filename).parts
            if not parts or info.filename.startswith("/") or ".." in parts:
                raise Phase0Error(f"Refusing to extract unsafe zip path: {info.filename}")
        archive.extractall(dest)
    root = find_project_root(dest)
    if root is None:
        raise Phase0Error(
            "The zip did not contain the Phase 0 project files. "
            "Build it with scripts/build_phase0_bundle.py and upload that zip."
        )
    return root
