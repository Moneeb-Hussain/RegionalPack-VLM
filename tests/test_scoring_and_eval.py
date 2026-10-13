"""Scoring and the evaluation gate. No model weights."""

from __future__ import annotations

from pathlib import Path

from src.annotate import apply_annotation, pending
from src.dataset import ANNOTATION_COLUMNS
from src.evaluate import completed_pairs, evaluation_block_reason, image_jobs
from src.scoring import brand_match, score_predictions
from test_dataset import ROOT, _annotation, _scaffold, _write_csv


def test_you_is_a_brand_hit_and_a_substring_is_not():
    assert brand_match("The brand is YOU.", ["YOU"])
    assert brand_match("CandyLand YOU milk chocolate", ["YOU", "CandyLand YOU"])
    assert not brand_match("Youthful Glow", ["YOU"])
    assert not brand_match("CandyLand", ["YOU", "CandyLand YOU"])
    assert not brand_match("Bisconni", ["Rite"])


def test_parent_name_alone_fails_and_product_brand_variants_pass():
    import yaml

    products = yaml.safe_load((ROOT / "configs/brand_aliases.yaml").read_text(encoding="utf-8"))["products"]
    cases = (
        ("noodles_local", "Shan", False),
        ("noodles_local", "Shan Foods", False),
        ("noodles_local", "The brand is Shoop.", True),
        ("noodles_local", "shoop Chattpata", True),
        ("noodles_local", "Shan Shoop", True),
        ("chocolate_local", "CandyLand", False),
        ("chocolate_local", "Candyland", False),
        ("chocolate_local", "This is YOU chocolate.", True),
        ("chocolate_local", "CandyLand YOU", True),
        ("apple_fruita", "Nestlé", False),
        ("apple_fruita", "Nestle", False),
        ("apple_fruita", "Fruita Vitals apple nectar", True),
        ("apple_fruita", "Nestlé Fruita Vitals", True),
        ("biscuit_local", "Bisconni", False),
        ("biscuit_local", "Rite biscuits", True),
        ("cola_intl", "Coke", True),
        ("cola_local", "Next", False),
        ("cola_local", "Mezan", False),
        ("tea_local", "Danedar", False),
        ("detergent_intl", "Excel", False),
        ("detergent_intl", "Surf excel", True),
        ("detergent_local", "Tristar", False),
        ("detergent_local", "Bonus Tristar", True),
        ("noodles_intl", "Chicken", False),
        ("noodles_intl", "Unilever", False),
        ("apple_fruitien", "Joy", False),
        ("apple_fruitien", "Citropak", False),
        ("apple_fruitien", "Fruitien Joy", True),
        ("chips_intl", "Lays", True),
        ("chips_local", "SuperCrisp", True),
    )
    for product_id, response, expected in cases:
        aliases = products[product_id]["aliases"]
        assert brand_match(response, aliases) is expected, (product_id, response)
    for product_id, entry in products.items():
        aliases = entry["aliases"]
        for parent_name in entry.get("parent_names") or []:
            assert brand_match(parent_name, aliases) is False, (product_id, parent_name)


def test_hidden_text_is_not_a_missed_transcription():
    annotations = [
        _annotation("cola_intl", "straight", text_true_latin="Coca-Cola"),
        _annotation("cola_intl", "tilt", text_true_latin="Coca-Cola"),
        _annotation("cola_intl", "occluded", text_true_latin="", occlusion_note="Wordmark covered."),
    ]
    predictions = []
    for angle, response in (
        ("straight", "Coca-Cola"),
        ("tilt", "Coca-Cola"),
        ("occluded", "unreadable"),
    ):
        predictions.append(
            {
                "image_id": f"cola_intl/{angle}",
                "prompt_id": "transcription",
                "response": response,
                "inference_completed": True,
            }
        )
        predictions.append(
            {
                "image_id": f"cola_intl/{angle}",
                "prompt_id": "identification",
                "response": "Coca-Cola",
                "inference_completed": True,
            }
        )
        predictions.append(
            {
                "image_id": f"cola_intl/{angle}",
                "prompt_id": "category",
                "response": "beverage",
                "inference_completed": True,
            }
        )
    scored = score_predictions(
        annotations=annotations,
        predictions=predictions,
        aliases_by_product={"cola_intl": ["Coca-Cola", "Coke"]},
        manifest=[{"product_id": "cola_intl", "comparison_role": "primary"}],
    )
    occluded = [row for row in scored["images"] if row["angle"] == "occluded"][0]
    assert occluded["latin_scored"] is False
    assert occluded["latin_cer"] is None
    assert scored["products"][0]["brand"] == "correct"
    assert scored["primary_brand_accuracy"] == 1.0
    assert "not a controlled 3D viewpoint experiment" in scored["angles_note"]


def test_exploratory_products_stay_out_of_the_primary_rate():
    annotations = [
        _annotation("cola_intl", angle, text_true_latin="Coca-Cola") for angle in ("straight", "tilt", "occluded")
    ] + [
        _annotation(
            "apple_fruita",
            angle,
            category="apple",
            brand_true="Fruita Vitals",
            brand_group="exploratory",
            origin="exploratory",
            text_true_latin="Fruita Vitals",
        )
        for angle in ("straight", "tilt", "occluded")
    ]
    predictions = []
    for image_id, brand, category in (
        ("cola_intl/straight", "Coca-Cola", "beverage"),
        ("cola_intl/tilt", "Coca-Cola", "beverage"),
        ("cola_intl/occluded", "unknown", "beverage"),
        ("apple_fruita/straight", "guess", "snack"),
        ("apple_fruita/tilt", "guess", "snack"),
        ("apple_fruita/occluded", "guess", "snack"),
    ):
        predictions.append({"image_id": image_id, "prompt_id": "identification", "response": brand, "inference_completed": True})
        predictions.append({"image_id": image_id, "prompt_id": "category", "response": category, "inference_completed": True})
        predictions.append({"image_id": image_id, "prompt_id": "transcription", "response": brand, "inference_completed": True})
    scored = score_predictions(
        annotations=annotations,
        predictions=predictions,
        aliases_by_product={"cola_intl": ["Coca-Cola"], "apple_fruita": ["Fruita Vitals"]},
        manifest=[
            {"product_id": "cola_intl", "comparison_role": "primary"},
            {"product_id": "apple_fruita", "comparison_role": "exploratory"},
        ],
    )
    assert scored["primary_brand_accuracy"] == 1.0
    assert scored["exploratory_product_ids"] == ["apple_fruita"]


def test_resume_skips_completed_prompt_rows():
    done = completed_pairs(
        [
            {"image_id": "cola_intl/straight", "prompt_id": "identification", "inference_completed": True},
            {"image_id": "cola_intl/straight", "prompt_id": "category", "inference_completed": False},
        ]
    )
    assert done == {("cola_intl/straight", "identification")}


def test_inference_runs_while_ocr_drafts_remain_unverified():
    reason = evaluation_block_reason(ROOT)
    assert reason == ""
    jobs = image_jobs(ROOT)
    assert len(jobs) == 60
    assert {job["image_id"].rsplit("/", 1)[1] for job in jobs} == {"straight", "tilt", "occluded"}


def test_unverified_drafts_are_not_ocr_scores():
    from src.evaluate import load_eval_config

    config = load_eval_config(ROOT)
    assert config["ocr_subset"]["scope"] == "all_verified"
    subset = {
        "cola_intl/straight",
        "cola_intl/tilt",
        "cola_intl/occluded",
        "cola_local/straight",
        "cola_local/tilt",
        "cola_local/occluded",
    }
    annotations = [
        _annotation("cola_intl", "straight", review_status="needs_review", text_true_latin="Coca-Cola"),
        _annotation("cola_local", "straight", review_status="verified", text_true_latin="Cola NEXT"),
        _annotation("tea_intl", "straight", review_status="verified", text_true_latin="Lipton"),
    ]
    predictions = []
    for image_id, text in (
        ("cola_intl/straight", "Coca-Cola"),
        ("cola_local/straight", "Cola NEXT"),
        ("tea_intl/straight", "Lipton"),
    ):
        predictions.append({"image_id": image_id, "prompt_id": "transcription", "response": text, "inference_completed": True})
        predictions.append({"image_id": image_id, "prompt_id": "identification", "response": text, "inference_completed": True})
        predictions.append({"image_id": image_id, "prompt_id": "category", "response": "beverage", "inference_completed": True})
    scored = score_predictions(
        annotations=annotations,
        predictions=predictions,
        aliases_by_product={"cola_intl": ["Coca-Cola"], "cola_local": ["Cola Next"], "tea_intl": ["Lipton"]},
        manifest=[
            {"product_id": "cola_intl", "comparison_role": "primary"},
            {"product_id": "cola_local", "comparison_role": "primary"},
            {"product_id": "tea_intl", "comparison_role": "primary"},
        ],
        ocr_image_ids=subset,
        require_verified_ocr=True,
    )
    by_id = {row["image_id"]: row for row in scored["images"]}
    assert by_id["cola_intl/straight"]["ocr_evaluated"] is False
    assert by_id["tea_intl/straight"]["ocr_evaluated"] is False
    assert by_id["cola_local/straight"]["ocr_evaluated"] is True
    assert scored["ocr_evaluated_count"] == 1


def test_annotate_writes_only_the_named_photograph(tmp_path: Path):
    root = _scaffold(tmp_path)
    rows = [_annotation("cola_intl", angle, script_present="", text_true_latin="") for angle in ("straight", "tilt", "occluded")]
    _write_csv(root / "data" / "annotations.csv", ANNOTATION_COLUMNS, rows)
    loaded = list(__import__("csv").DictReader((root / "data" / "annotations.csv").open(encoding="utf-8")))
    assert len(pending(loaded)) == 3
    apply_annotation(
        loaded,
        image_path="data/raw/cola_intl/straight.jpg",
        latin="Coca-Cola",
        urdu="",
        script="latin",
        brand_visible="yes",
        occlusion="",
        ambiguous="",
    )
    assert pending(loaded)[0]["image_path"] == "data/raw/cola_intl/tilt.jpg"
    assert loaded[0]["text_true_latin"] == "Coca-Cola"
    assert loaded[0].get("review_status") != "verified"
    assert loaded[1]["text_true_latin"] == ""


def test_latin_casefold_does_not_change_urdu_or_mark_a_blank_as_absent():
    from src.qa import CASE_NOT_REVIEWED, classify_row
    from src.scoring import _cer

    assert _cer("Coca-Cola\nORIGINAL TASTE", "coca-cola original taste", script="latin") == 0.0
    assert _cer("", "anything", script="urdu") is None
    draft = _annotation(
        "detergent_intl",
        "straight",
        review_status="needs_review",
        script_present="mixed",
        text_true_urdu="",
        ambiguous_text="Urdu is printed. The spelling is not clear enough to transcribe.",
    )
    classified = classify_row(draft)
    assert classified["latin_case"] == CASE_NOT_REVIEWED
    assert classified["urdu_case"] == CASE_NOT_REVIEWED
    assert "urdu_blank_is_not_recorded_as_absent" in classified["signals"]
    assert "draft_note_says_urdu_is_unreadable" in classified["signals"]


def test_scoring_config_matches_the_category_table():
    import yaml

    from src.scoring import CATEGORY_ACCEPT

    loaded = yaml.safe_load((ROOT / "configs" / "scoring_config.yaml").read_text(encoding="utf-8"))
    assert loaded["frozen_before_responses"] is True
    assert loaded["category_accept"] == {key: list(value) for key, value in CATEGORY_ACCEPT.items()}
    assert loaded["primary_comparison"]["exploratory_product_ids"] == ["apple_fruita", "apple_fruitien"]
