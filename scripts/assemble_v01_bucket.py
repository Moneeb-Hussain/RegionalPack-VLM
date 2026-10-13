"""Copy the Version 0.1 publication files into release/v0.1/publication.

Run this after Colab results are unzipped and src.score_release has been run.
Photographs are copied from data/raw. The script does not invent scores
and does not upload anything.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "release" / "v0.1" / "publication"

FILES = (
    ("release/v0.1/README.md", "README.md"),
    ("docs/DATASET_CARD.md", "DATASET_CARD.md"),
    ("notebooks/evaluate_benchmark.ipynb", "notebook/evaluate_benchmark.ipynb"),
    ("configs/eval_config.yaml", "configuration/eval_config.yaml"),
    ("configs/inference_config.yaml", "configuration/inference_config.yaml"),
    ("configs/brand_aliases.yaml", "configuration/brand_aliases.yaml"),
    ("configs/scoring_config.yaml", "configuration/scoring_config.yaml"),
    ("prompts/prompts.yaml", "configuration/prompts.yaml"),
    ("data/product_manifest.csv", "dataset/product_manifest.csv"),
    ("data/annotations.csv", "dataset/annotations.csv"),
    ("results/benchmark/v0.1_summary.md", "results/v0.1_summary.md"),
    ("results/benchmark/benchmark_predictions.jsonl", "results/benchmark_predictions.jsonl"),
    ("results/benchmark/benchmark_environment.json", "results/benchmark_environment.json"),
    ("results/benchmark/benchmark_run.md", "results/benchmark_run.md"),
    ("results/scored_predictions.csv", "results/scored_predictions.csv"),
    ("results/metrics_summary.json", "results/metrics_summary.json"),
    ("results/adjudication_decisions.csv", "results/adjudication_decisions.csv"),
    ("reports/technical_report.md", "reports/technical_report.md"),
    ("reports/ADJUDICATION_APPROVAL.md", "reports/ADJUDICATION_APPROVAL.md"),
    ("reports/REVISED_BENCHMARK_ANALYSIS.md", "reports/REVISED_BENCHMARK_ANALYSIS.md"),
    ("reports/OCR_METRIC_INTERPRETATION.md", "reports/OCR_METRIC_INTERPRETATION.md"),
    ("reports/figures/brand_by_origin.svg", "reports/figures/brand_by_origin.svg"),
    ("reports/figures/brand_by_capture_set_a.svg", "reports/figures/brand_by_capture_set_a.svg"),
    ("reports/figures/CAPTIONS.md", "reports/figures/CAPTIONS.md"),
    ("reports/PROMPT_PROVENANCE.md", "reports/PROMPT_PROVENANCE.md"),
    ("release/v0.1/PUBLICATION_READY_CHECKLIST.md", "PUBLICATION_READY_CHECKLIST.md"),
)


def required_gaps(root: Path) -> list[str]:
    """Required publication inputs. Checked before any output folder is written."""
    gaps = [name for name, _dest_name in FILES if not (root / name).is_file()]
    photographs = [
        path
        for path in sorted((root / "data" / "raw").glob("*/*.jpg"))
        if not path.name.startswith("._")
    ]
    if len(photographs) != 60:
        gaps.append(f"data/raw photographs: found {len(photographs)}, expected 60")
    return gaps


def assemble(root: Path = ROOT, dest: Path = DEST) -> Path:
    gaps = required_gaps(root)
    if gaps:
        raise FileNotFoundError(
            "Required publication files are missing. No folder or SHA256SUMS was written.\n"
            + "\n".join(f"- {gap}" for gap in gaps)
        )
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    try:
        for source_name, dest_name in FILES:
            source = root / source_name
            target = dest / dest_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        copied = 0
        for path in sorted((root / "data" / "raw").glob("*/*.jpg")):
            if path.name.startswith("._"):
                continue
            relative = Path("dataset") / "raw" / path.parent.name / path.name
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
        if copied != 60:
            raise FileNotFoundError(f"Expected 60 photographs, copied {copied}.")
        write_manifest(dest)
        missing_links = bucket_readme_links_missing(dest)
        if missing_links:
            raise FileNotFoundError(
                "Bucket README links do not match the publication folder:\n"
                + "\n".join(missing_links)
            )
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    print(f"Wrote {dest}")
    return dest


def write_manifest(dest: Path) -> Path:
    """SHA-256 of every file under dest except the manifest itself."""
    lines = []
    for path in sorted(item for item in dest.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(dest).as_posix()}")
    manifest = dest / "SHA256SUMS"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def bucket_readme_links_missing(dest: Path) -> list[str]:
    """Relative links in the Bucket README must exist in the publication folder."""
    readme = dest / "README.md"
    missing = []
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        relative = target.split("#", 1)[0]
        if relative and not (dest / relative).exists():
            missing.append(relative)
    return missing


def main() -> None:
    assemble()


if __name__ == "__main__":
    main()
