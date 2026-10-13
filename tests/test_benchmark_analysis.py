"""Checks for the post-run analysis. It does not load a model."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _analyzer():
    path = ROOT / "scripts" / "analyze_benchmark.py"
    spec = importlib.util.spec_from_file_location("analyze_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_abstention_is_not_a_brand_success_and_quotes_are_not_silently_accepted():
    analyzer = _analyzer()
    rows = {row["image_id"]: row for row in analyzer.score_rows(analyzer.load_inputs(ROOT))}
    you = rows["chocolate_local/straight"]
    assert you["explicit_abstention"] is True
    assert you["frozen_brand_match"] is False
    assert you["brand_headline_success"] is False
    assert you["brand_outcome"] == "abstention"
    crisp = rows["chips_local/straight"]
    assert crisp["brand_outcome"] == "abstention"
    assert crisp["brand_headline_success"] is False
    shoop = rows["noodles_local/straight"]
    assert shoop["brand_outcome"] == "punctuation_blocked_assertion"
    assert shoop["adjudication"] == "pending_human"
    assert shoop["brand_headline_success"] is False
    bonus = rows["detergent_local/tilt"]
    assert bonus["brand_headline_success"] is True
    assert "identification_calls_it_gum" in bonus["flags"]
    assert not any(row["frozen_brand_match"] and row["explicit_abstention"] for row in rows.values())


def test_saved_run_covers_every_verified_image():
    analyzer = _analyzer()
    inputs = analyzer.load_inputs(ROOT)
    assert len(inputs["predictions"]) == 180
    rows = analyzer.score_rows(inputs)
    assert len(rows) == 60
    assert all(row["comparison_role"] in {"primary", "exploratory"} for row in rows)
    summary = analyzer.summarize(rows)
    assert summary["product_brand"]["hits"] == 13
    assert summary["product_brand"]["n"] == 18
    assert summary["product_category"]["n"] == 18
    assert summary["exploratory_products"] == ["apple_fruita", "apple_fruitien"]
