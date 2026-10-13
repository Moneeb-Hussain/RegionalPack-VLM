"""Build the zip uploaded into a fresh Colab session.

The GitHub repository is private, so the notebook does not clone it.
Run this on your computer, then upload dist/phase0_colab_bundle.zip when the notebook asks.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "dist" / "phase0_colab_bundle.zip"

INCLUDED_FILES = (
    "requirements-colab.txt",
    "prompts/prompts.yaml",
    "prompts/phase0_prompts.yaml",
    "configs/inference_config.yaml",
    "configs/phase0_config.yaml",
    "configs/eval_config.yaml",
    "configs/brand_aliases.yaml",
    "configs/collection_config.yaml",
    "data/product_manifest.csv",
    "data/annotations.csv",
    "data/phase0_seen_products.csv",
    "results/README.md",
)
INCLUDED_DIRECTORIES = ("src",)


def bundle_members(root: Path = ROOT) -> list[Path]:
    files = [root / relative for relative in INCLUDED_FILES]
    for directory_name in INCLUDED_DIRECTORIES:
        directory = root / directory_name
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            files.append(path)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Bundle is missing: " + ", ".join(missing))
    return files


def build_bundle(root: Path = ROOT, dest: Path = DEST) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in bundle_members(root):
            archive.write(path, arcname=path.relative_to(root).as_posix())
    return dest


def main() -> None:
    path = build_bundle()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
