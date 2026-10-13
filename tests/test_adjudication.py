"""The post-hoc brand reading is counted from image decisions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "adjudicate_benchmark.py"
    spec = importlib.util.spec_from_file_location("adjudicate_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_written_rules_and_recounted_majority():
    module = _module()
    decisions = module.decide(module.load_scored(ROOT / "results" / "scored_predictions.csv"))
    by_id = {row["image_id"]: row for row in decisions}
    assert by_id["biscuit_local/tilt"]["frozen_brand_success"] is True
    assert by_id["biscuit_local/tilt"]["adjudicated_brand_success"] is False
    assert by_id["biscuit_local/occluded"]["adjudicated_brand_success"] is False
    assert by_id["biscuit_local/straight"]["adjudicated_brand_success"] is False
    assert by_id["detergent_local/straight"]["frozen_brand_success"] is False
    assert by_id["detergent_local/straight"]["adjudicated_brand_success"] is True
    assert by_id["detergent_local/tilt"]["adjudicated_brand_success"] is True
    assert "gum" in by_id["detergent_local/tilt"]["rationale"]
    assert by_id["noodles_local/straight"]["adjudicated_brand_success"] is True
    assert by_id["chocolate_local/straight"]["adjudicated_brand_success"] is False
    assert by_id["chips_local/straight"]["adjudicated_brand_success"] is False
    frozen = module.product_rates(decisions, "frozen_brand_success")
    adjudicated = module.product_rates(decisions, "adjudicated_brand_success")
    assert frozen["label"] == "13/18"
    counted = 0
    products = {row["product_id"] for row in decisions if row["comparison_role"] == "primary"}
    for product_id in products:
        flags = [
            row["adjudicated_brand_success"]
            for row in decisions
            if row["product_id"] == product_id
        ]
        counted += int(module._majority(flags))
    assert adjudicated["hits"] == counted
    assert adjudicated["n"] == 18
    assert all(
        row["category_interpretation"] == "broad_cleaning_product"
        for row in decisions
        if row["frozen_category_match"] is False
    )
