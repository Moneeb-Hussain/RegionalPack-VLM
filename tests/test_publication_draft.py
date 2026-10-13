"""Draft publication files stay consistent with the saved scores."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = (
    "results/benchmark/benchmark_predictions.jsonl",
    "data/annotations.csv",
    "prompts/prompts.yaml",
    "configs/scoring_config.yaml",
    "configs/brand_aliases.yaml",
    "configs/inference_config.yaml",
)
CHANGES = (
    "biscuit_local/tilt",
    "detergent_local/straight",
    "detergent_local/occluded",
    "noodles_local/straight",
    "noodles_local/tilt",
)


def _sha(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _load_prepare():
    path = ROOT / "scripts" / "prepare_publication_draft.py"
    spec = importlib.util.spec_from_file_location("prepare_publication_draft", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_assemble_does_not_write_a_folder_when_a_required_file_is_missing(tmp_path):
    module = _load_assemble()
    dest = tmp_path / "publication"
    with pytest.raises(FileNotFoundError, match="No folder or SHA256SUMS was written"):
        module.assemble(root=tmp_path / "empty", dest=dest)
    assert not dest.exists()


def _load_assemble():
    path = ROOT / "scripts" / "assemble_v01_bucket.py"
    spec = importlib.util.spec_from_file_location("assemble_v01_bucket", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_signed_approval_survives_repeated_builds(tmp_path):
    module = _load_prepare()
    signed = "\n".join(
        [
            "# Human adjudication approval checklist",
            "",
            "Status: approved by the reviewer.",
            "",
            "Set A is 13/18. Set B, post hoc, is 15/18.",
            "",
            *[
                f"## {index}. `{image_id}`\n\n- [x] Approve this image decision\n"
                for index, image_id in enumerate(CHANGES, start=1)
            ],
        ]
    )
    path = tmp_path / "ADJUDICATION_APPROVAL.md"
    path.write_text(signed, encoding="utf-8")
    before = path.read_bytes()
    assert module.main(checklist=path) == "kept-signed"
    assert module.main(checklist=path) == "kept-signed"
    assert module.main(checklist=path, write_template=True) == "kept-signed"
    assert path.read_bytes() == before


def test_unsigned_template_is_created_only_when_missing(tmp_path):
    module = _load_prepare()
    path = tmp_path / "ADJUDICATION_APPROVAL.md"
    assert module.main(checklist=path) == "written"
    text = path.read_text(encoding="utf-8")
    for image_id in CHANGES:
        assert image_id in text
    assert text.count("## ") == 5
    assert "13/18" in text and "15/18" in text
    assert "- [x] Approve this image decision" not in text
    assert "Status: pending" in text
    kept = path.read_bytes()
    assert module.main(checklist=path) == "kept"
    assert path.read_bytes() == kept


def test_draft_materials_match_the_two_scores():
    before = {relative: _sha(relative) for relative in PROTECTED}
    approval = ROOT / "reports" / "ADJUDICATION_APPROVAL.md"
    approval_before = approval.read_bytes()
    assert _load_prepare().main() == "kept-signed"
    assert {relative: _sha(relative) for relative in PROTECTED} == before
    assert approval.read_bytes() == approval_before

    checklist = approval.read_text(encoding="utf-8")
    for image_id in CHANGES:
        assert image_id in checklist
    assert checklist.count("## ") == 5
    assert "13/18" in checklist and "15/18" in checklist

    figure = (ROOT / "reports" / "figures" / "brand_by_origin.svg").read_text(encoding="utf-8")
    for label in ("13/18", "15/18", "9/9", "4/9", "6/9"):
        assert label in figure
    capture = (ROOT / "reports" / "figures" / "brand_by_capture_set_a.svg").read_text(encoding="utf-8")
    for label in ("13/18", "15/18", "14/18"):
        assert label in capture

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    report = (ROOT / "reports" / "technical_report.md").read_text(encoding="utf-8")
    bucket = (ROOT / "release" / "v0.1" / "README.md").read_text(encoding="utf-8")
    for text in (readme, report, bucket):
        assert "caused by origin" not in text.casefold()
        assert "caused by pretraining" not in text.casefold()
        assert "proves a geographic" not in text.casefold()
    assert "##  Research Motivation: The Blind Spot I Observed" in readme
    assert "##  Proposed Path Forward" in readme
    for hypothesis in ("H1", "H2", "H3", "H4"):
        assert hypothesis in report
    for text in (readme, report, bucket):
        for line in text.splitlines():
            if "15/18" in line:
                lowered = line.casefold()
                assert "set b" in lowered or "post-hoc" in lowered or "post hoc" in lowered
