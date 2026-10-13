"""Reporting checks that do not load a model."""

from __future__ import annotations

from src.environment import classify_bf16_support
from src.inference import assess_loaded_revision, commit_from_snapshot_path

PIN = "89644892e4d85e24eaac8bacfd4f463576704203"


def test_t4_capability_is_not_hardware_bf16():
    report = classify_bf16_support(
        compute_capability=(7, 5),
        pytorch_including_emulation=True,
    )
    assert report["bf16_hardware_supported"] is False
    assert report["bf16_supported"] is False
    assert report["bf16_pytorch_including_emulation"] is True
    assert report["bf16_emulation_only"] is True


def test_ampere_capability_is_hardware_bf16():
    report = classify_bf16_support(
        compute_capability=(8, 0),
        pytorch_including_emulation=True,
    )
    assert report["bf16_hardware_supported"] is True
    assert report["bf16_emulation_only"] is False


def test_loaded_revision_must_equal_the_pin():
    match = assess_loaded_revision(PIN, PIN)
    assert match["revision_matches_pin"] is True
    assert match["revision_check"] == "match"
    mismatch = assess_loaded_revision(PIN, "abc")
    assert mismatch["revision_matches_pin"] is False
    assert mismatch["revision_check"] == "mismatch"
    missing = assess_loaded_revision(PIN, None)
    assert missing["revision_check"] == "missing"
    assert missing["revision_matches_pin"] is False


def test_snapshot_path_yields_the_commit_directory():
    path = (
        "/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/"
        f"snapshots/{PIN}/config.json"
    )
    assert commit_from_snapshot_path(path) == PIN
    assert commit_from_snapshot_path(None) is None
    assert commit_from_snapshot_path("/tmp/config.json") is None
