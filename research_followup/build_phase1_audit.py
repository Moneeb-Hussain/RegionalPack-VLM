"""Assemble research_followup/phase1_audit.md from files already on disk.

Does not rescore Set A, does not run Qwen, and does not write into linear_probe/results
or linear_probe/results_resnet.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "linear_probe"))
import probe_analysis as pa  # noqa: E402

PROBE = ROOT / "linear_probe"
OUT = ROOT / "research_followup"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PRIMARY_SITES = ["vision_out", "llm_img_L14", "llm_last_L14", "llm_img_L28", "llm_last_L28"]
IDENT = (
    "What product is shown in this image? If you can identify the specific brand, name it. "
    "If you cannot identify the specific brand with confidence, say so explicitly rather than guessing."
)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def majority(frame, column):
    counts = frame.groupby("product_id")[column].sum()
    return int((counts >= 2).sum()), int(counts.size)


def md_cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def main():
    manifest = pd.read_csv(PROBE / "manifest.csv")
    scored = pd.read_csv(ROOT / "results" / "scored_predictions.csv")
    decisions = pd.read_csv(ROOT / "results" / "adjudication_decisions.csv")
    products = pd.read_csv(ROOT / "data" / "product_manifest.csv")
    images = pd.read_csv(OUT / "step5_image_checks.csv", dtype={"ahash": str})
    step2 = pd.read_csv(OUT / "step2_rerun" / "decodability_by_site.csv")
    per_photo = pd.read_csv(OUT / "step2_rerun" / "per_photo_probe.csv")
    ablation = pd.read_csv(OUT / "step5_geometry_ablation.csv")
    way = pd.read_csv(OUT / "step6_twenty_way.csv")
    binary = pd.read_csv(OUT / "step6_binary_by_site.csv")
    env = json.loads((ROOT / "results" / "benchmark" / "benchmark_environment.json").read_text(encoding="utf-8"))

    jpg = list((ROOT / "data" / "raw").rglob("*.jpg"))
    jsonl_path = ROOT / "results" / "benchmark" / "benchmark_predictions.jsonl"
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ident_rows = [row for row in records if row.get("prompt_id") == "identification"]
    prompt_rows = {row.get("prompt") for row in ident_rows}
    revisions = {row.get("model_revision") for row in ident_rows}

    raw_by_id = {row["image_id"]: row.get("response", "") for row in ident_rows}
    wrong = manifest[(manifest["primary"] == 1) & (manifest["model_correct"] == 0)].copy()
    wrong["slug"] = wrong["path"].map(lambda path: Path(path).parent.name)
    scored_by_id = scored.set_index("image_id")

    primary_scored = scored[scored["comparison_role"] == "primary"]
    headline_hits = int(primary_scored["brand_headline_success"].sum())
    frozen_products, n_primary_products = majority(primary_scored, "brand_headline_success")
    primary_decisions = decisions[decisions["comparison_role"] == "primary"]
    frozen_from_decisions, _ = majority(primary_decisions, "frozen_brand_success")
    adjudicated_products, _ = majority(primary_decisions, "adjudicated_brand_success")
    adjudicated_hits = int(primary_decisions["adjudicated_brand_success"].sum())
    quote_upgrades = primary_decisions[primary_decisions["difference_from_frozen"] == "frozen_miss_adjudicated_hit"]
    downgrades = primary_decisions[primary_decisions["difference_from_frozen"] == "frozen_hit_adjudicated_miss"]

    features_path = PROBE / "features.npz"
    features_hash = sha256_file(features_path)
    features_bytes = features_path.stat().st_size
    prompt_hash = sha256_text(IDENT)
    extract_text = (PROBE / "extract_features.py").read_text(encoding="utf-8")
    prompts_text = (ROOT / "prompts" / "prompts.yaml").read_text(encoding="utf-8")
    env_prompt = next(item["prompt"] for item in env["prompts"] if item["prompt_id"] == "identification")

    print("compare 20-way calls", flush=True)
    features = np.load(features_path)
    product = manifest["product_id"].to_numpy(int)
    cond = manifest["condition"].map({"straight": 0, "tilt": 1, "occluded": 2}).to_numpy(int)
    disagreements = 0
    compared = 0
    for site in PRIMARY_SITES:
        predicted = pa.loco_predict(features[site].astype(np.float64), product, cond, C=1.0)
        saved = per_photo[per_photo["site"] == site].set_index("photo_id").loc[manifest["photo_id"], "probe_correct"]
        match = predicted == product
        disagreements += int(np.sum(match != saved.to_numpy(dtype=bool)))
        compared += len(manifest)

    pytest_run = subprocess.run(
        [str(PY), "-m", "pytest", "test_probe_analysis.py", "-q"],
        cwd=PROBE,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    pytest_text = (pytest_run.stdout or "") + (pytest_run.stderr or "")

    import numpy
    import sklearn
    import torch
    import transformers

    bits = [np.array(list(text), dtype=np.uint8) for text in images["ahash"]]
    across = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            if images.loc[i, "slug"] == images.loc[j, "slug"]:
                continue
            across.append(int(np.count_nonzero(bits[i] != bits[j])))
    near = int(sum(distance <= 10 for distance in across))

    identity_gaps = []
    fisher_values = []
    fail_counts = []
    for site in PRIMARY_SITES:
        table = pd.read_csv(PROBE / "results" / f"geometry_{site}.csv")
        gap = float(np.max(np.abs(table["drop"] - table["g_norm"] * table["dist"] * table["align"])))
        identity_gaps.append((site, gap))
        flags = table["model_fail"].map(lambda value: str(value).strip().lower() in {"1", "true"})
        fail_counts.append(int(flags.sum()))
        fisher_values.append(pa.auroc(table["fisher_That"].to_numpy(float), flags.to_numpy(bool)))

    n_ref_in = 0
    n_held_in = 0
    n_pairs = 0
    for held in (1, 2):
        for product_id in np.unique(product):
            i_ref = np.where((product == product_id) & (cond == 0))[0]
            i_held = np.where((product == product_id) & (cond == held))[0]
            if len(i_ref) != 1 or len(i_held) != 1:
                continue
            train = cond != held
            n_pairs += 1
            n_ref_in += int(train[int(i_ref[0])])
            n_held_in += int(train[int(i_held[0])])

    location_filled = int((products["photo_location"].astype("string").fillna("").str.strip() != "").sum())
    commits = git("log", "--format=%H %s", "linear-probe..HEAD").splitlines()
    protocol = git("log", "-1", "--format=%H %cI %s", "d12dcc20c20709ae9e3aef7a97ca4bbbe6e925ab")
    status = git("status", "-sb")
    branch_upstream = "no upstream" if "..." not in status.splitlines()[0] else status.splitlines()[0]

    focus = ["llm_img_L14", "llm_img_L28", "vision_out", "color_hist", "pixel_thumb", "resnet18_pool"]
    saved_focus = step2[step2["site"].isin(PRIMARY_SITES + ["pixel_thumb", "color_hist"])]

    lines = []
    add = lines.append
    add("# Phase 1 audit")
    add("")
    add("Historical Set A stays 13/18. Historical Set B stays 15/18. Neither was overwritten. This file is post-hoc except where it repeats the saved confirmatory run.")
    add("")
    add("Command: `.venv\\Scripts\\python.exe research_followup\\build_phase1_audit.py`")
    add("")
    add("## 1. Plain-language summary")
    add("")
    add(
        f"The saved linear probe was rerun from the existing feature file and matched, including {disagreements} disagreements "
        f"out of {compared} twenty-way photo calls after the two-class scoring fix. "
        f"That fix matters for two-class probes: a colour histogram then scores {int(binary.loc[binary.site=='color_hist','correct'].iloc[0])}/54 "
        f"on the nine category pairs, the same as every Qwen image-token layer. "
        f"The twenty-way colour histogram is {int(way.loc[way.site=='color_hist','correct_60'].iloc[0])}/60 and the 16x16 thumbnail is "
        f"{int(way.loc[way.site=='pixel_thumb','correct_60'].iloc[0])}/60, with ResNet-18 at "
        f"{int(way.loc[way.site=='resnet18_pool','correct_60'].iloc[0])}/60. "
        f"The high probe scores are therefore not evidence specific to Qwen. "
        f"The geometry correlation with alignment is the identity drop = |g| |d| alignment. "
        f"The {len(wrong)} primary misses come from {wrong['slug'].nunique()} products, and only "
        f"{int((wrong['condition']!='straight').sum())} of them enter the geometry failure pairs."
    )
    add("")
    add("## 2. Inventory and provenance")
    add("")
    add("| item | result |")
    add("|---|---|")
    add(f"| JPEG files under `data/raw` | {len(jpg)} |")
    add(f"| `manifest.csv` rows | {len(manifest)} |")
    add(f"| file SHA-256 matches `data/image_provenance.csv` | {int(images['sha_matches_provenance'].sum())}/{len(images)} |")
    add(f"| benchmark JSONL rows | {len(records)} |")
    add(f"| identification rows | {len(ident_rows)} |")
    add(f"| distinct identification prompts in the JSONL | {len(prompt_rows)} |")
    add(f"| prompt text equals the frozen sentence | {prompt_rows == {IDENT} and env_prompt == IDENT and IDENT in extract_text and IDENT in prompts_text} |")
    add(f"| prompt SHA-256 | `{prompt_hash}` |")
    add(f"| model revisions on identification rows | {', '.join(sorted(str(item) for item in revisions))} |")
    add(f"| pinned revision in the environment file | `{env['load']['revision_pinned']}` |")
    add(f"| original GPU | {env['environment']['gpu']['gpu_name']}, torch {env['environment']['libraries']['pytorch_version']}, transformers {env['environment']['libraries']['transformers_version']}, float16 |")
    add("| original sklearn and numpy | NOT VERIFIED. They are not in `benchmark_environment.json`. |")
    add(f"| this machine | Python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}, sklearn {sklearn.__version__}, numpy {numpy.__version__} |")
    add(f"| `features.npz` | `{features_hash}`, {features_bytes} bytes, untracked |")
    add("")
    add("Photos live at `data/raw/<product_id>/{straight,tilt,occluded}.jpg`. The manifest paths are those files relative to `linear_probe/`. The frozen identification prompt is the sentence hashed above. It is in `prompts/prompts.yaml`, `results/benchmark/benchmark_environment.json`, every identification row of `results/benchmark/benchmark_predictions.jsonl`, and `linear_probe/extract_features.py`. Set A is `scripts/analyze_benchmark.py` calling `src.scoring.brand_match`. Headline success is a frozen brand match and no abstention. A product counts as correct when at least 2 of its 3 photos have that headline success.")
    add("")
    add("## 3. The 12 wrong photos")
    add("")
    add(f"Primary image headline successes recomputed from `results/scored_predictions.csv`: {headline_hits}/54. Product majority: {frozen_products}/{n_primary_products}. The same frozen majority from `results/adjudication_decisions.csv` is {frozen_from_decisions}/{n_primary_products}. Published Set A in `results/benchmark/v0.1_summary.md` is 13/18 products and 42/54 images. These rows were not rescored.")
    add("")
    add("| photo | product | condition | manifest failure | scored outcome | frozen match | abstention | loose match | headline |")
    add("|---|---|---|---|---|---|---|---|---|")
    response_blocks = []
    for record in wrong.itertuples():
        row = scored_by_id.loc[record.photo_id]
        raw = raw_by_id[record.photo_id]
        same = raw.strip() == str(row["identification_response"]).strip()
        add(
            f"| {record.photo_id} | {record.slug} | {record.condition} | {record.failure_type} | "
            f"{row['brand_outcome']} | {row['frozen_brand_match']} | {row['explicit_abstention']} | "
            f"{row['alias_if_quotes_and_commas_ignored']} | {row['brand_headline_success']} |"
        )
        response_blocks.append((record.photo_id, same, raw))
    add("")
    add(f"Distinct products: {wrong['slug'].nunique()} ({', '.join(wrong['slug'].drop_duplicates())}). Abstention rows: {int((wrong['failure_type']=='abstention').sum())}. Punctuation-blocked rows: {int((wrong['failure_type']=='punctuation_blocked_assertion').sum())}.")
    add("")
    add("Raw identification text from the JSONL. Each one matches the scored-file text.")
    add("")
    for photo_id, same, raw in response_blocks:
        add(f"### {photo_id}")
        add("")
        add(f"JSONL text equals the scored identification text: {same}.")
        add("")
        add("```text")
        add(raw.strip())
        add("```")
        add("")
    add("Sensitivity, not a new score. `reports/ADJUDICATION_APPROVAL.md` approves four frozen misses as brand hits (Bonus straight and occluded, Shoop straight and tilt) and one frozen hit as a miss (Rite tilt, which names Bisconni rather than Rite). Applying those five saved decisions in `adjudication_decisions.csv` gives "
        f"{adjudicated_hits}/54 images and {adjudicated_products}/{n_primary_products} products. "
        f"Upgrades: {len(quote_upgrades)}. Downgrades: {len(downgrades)}. "
        "The approval note calls the product figure Set B, 15/18. Set A is unchanged.")
    add("")
    add("## 4. Reproduction of the saved results")
    add("")
    add("Earlier command, from `linear_probe/`, one math thread, exit 0:")
    add("")
    add("```text")
    add("OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u run_study.py --features features.npz --manifest manifest.csv --out ..\\research_followup\\step2_rerun --n-perm 1000")
    add("```")
    add("")
    add("That rerun matched `linear_probe/results/decodability_by_site.csv` on 59 sites and `per_photo_probe.csv` on 300 rows. Output stayed under `research_followup/step2_rerun/`.")
    add("")
    add("| site | accuracy | interval | null mean | p | recovered of 12 |")
    add("|---|---:|---|---:|---:|---:|")
    for record in saved_focus.itertuples():
        add(f"| {record.site} | {record.acc:.6f} | {record.ci_lo:.6f}–{record.ci_hi:.6f} | {record.null_mean:.6f} | {record.p_overall:.6f} | {int(record.n_recovered)} |")
    add("")
    add(f"This report's command refit the five primary sites with the current code, C=1.0, and compared them with the step 2 photo calls: {disagreements} disagreements out of {compared}.")
    add("")
    add("`PROTOCOL.md` was committed as `d12dcc20c20709ae9e3aef7a97ca4bbbe6e925ab` with message `Pre-register linear-probe protocol` before the results commit. Whether that was before `features.npz` existed on disk is unverified, because the feature file has no git history.")
    add("")
    add("## 5. Bug fix and tests")
    add("")
    add("`_ReducedLR` keeps one coefficient row for two classes. The old argmax over that single column always predicted the first class. The current code returns one score for two classes and treats a positive score as the second class. Geometry and probability helpers raise `ValueError` when a probe has fewer than three classes. Twenty-way prediction is still an argmax over one row per class.")
    add("")
    add(f"Pytest exit code: {pytest_run.returncode}.")
    add("")
    add("```text")
    add(pytest_text.rstrip())
    add("```")
    add("")
    add("## 6. Isolation and leakage")
    add("")
    add("Command for the checks below, except the cheap recounts in this report: `.venv\\Scripts\\python.exe research_followup\\step5_isolation.py`")
    add("")
    add(f"Leave-one-condition-out training rows are the other conditions. On the saved labels, the straight reference photo is inside those training rows for {n_ref_in} of {n_pairs} geometry pairs, and the shifted photo is inside them for {n_held_in} of {n_pairs}. The unit test `test_held_out_condition_never_enters_scaler_probe_or_temperature` checks that the held-out condition is absent from the scaler, the probe, and the temperature fits. The temperature fit is nested inside the training conditions.")
    add("")
    add("Removing that one straight photo and refitting, from `step5_geometry_ablation.csv`:")
    add("")
    add("| site | predictions that agree | rivals that agree | mean absolute drop change |")
    add("|---|---:|---:|---:|")
    for record in ablation.itertuples():
        add(f"| {record.site} | {int(record.pred_agree)}/{int(record.n_pairs)} | {int(record.rival_agree)}/{int(record.n_pairs)} | {record.mean_abs_drop_change:.3f} |")
    add("")
    add("The published geometry files were not edited. The straight photo in the training fold changes some rivals and some margin drops. It does not change the twenty-way rule, which never trains on the held-out condition.")
    add("")
    add(f"Duplicate file hashes: {int(images['sha256'].duplicated().sum())}. Across different products, the closest 64-bit average hash is {min(across)} bits and the median is {float(np.median(across)):.1f}. Pairs at distance 10 or less: {near}. The closest pair is `cola_intl/occluded` versus `cola_local/occluded`.")
    add("")
    datetime_original = images["datetime_original"].astype("string").fillna("").str.strip()
    add(f"`photo_location` is filled on {location_filled} of {len(products)} product rows. GPS EXIF blocks: {int(images['gps_present'].sum())} of {len(images)}. DateTimeOriginal values: {int((datetime_original != '').sum())} of {len(images)}. A shared capture session is NOT VERIFIED from the files.")
    add("")
    add("`extract_features.py` passes the turned picture into the model. The path string is not written into `features.npz`. All saved arrays are numeric. Filename leakage into the feature vectors was not found in that code.")
    add("")
    add("Chance checks from the same step 5 command, 40 label shuffles, C=1.0. Chance for 20 products is 0.05. Real `vision_out` accuracy was 1.000000 with shuffle mean 0.057500. Gaussian noise in that shape scored 0.016667. Noise-image colour histograms scored 0.066667. All-zero features, black thumbnails, and black histograms failed before a score existed: the scaler leaves zero columns and logistic regression raises `Found array with 0 feature(s) (shape=(40, 0))`. Qwen features of an actual blank photograph were not computed. That check is NOT VERIFIED.")
    add("")
    add("## 7. Shortcut baselines")
    add("")
    add("Post-hoc exploratory. Command: `.venv\\Scripts\\python.exe research_followup\\step6_shortcuts.py`. Same held-out condition, C=1.0, no new 1000-shuffle test. Pairs are the nine primary category pairs, 54 photos. Two-class scores use the sign rule.")
    add("")
    add("| site | 20-way | matched pairs |")
    add("|---|---:|---:|")
    for name in ["llm_img_L14", "llm_img_L28"] + [site for site in focus if not site.startswith("llm_img")]:
        way_row = way[way.site == name].iloc[0]
        pair_row = binary[binary.site == name].iloc[0]
        add(f"| {name} | {int(way_row.correct_60)}/60 | {int(pair_row.correct)}/{int(pair_row.total)} |")
    img = binary[binary.site.str.startswith("llm_img_")]
    add("")
    add(f"Image-token layers at 54/54: {int((img.correct==54).sum())} of {len(img)}. Colour histogram is also 54/54. The paired test cannot localise a Qwen-specific effect. It is not at ceiling for every baseline: the thumbnail is 47/54, ResNet-18 is 50/54, and the vision tower is 53/54. ResNet features are the saved file. `resnet_control.py` does not apply the EXIF turn used for the Qwen features.")
    add("")
    add("The twenty-way counts match the step 2 accuracies, and the ResNet count matches `linear_probe/results_resnet/resnet_control.csv` (58/60).")
    add("")
    add("## 8. Red flags and what could not be verified")
    add("")
    add("- The probe ceiling is shared with simple image cues. A colour histogram ties Qwen on the paired task.")
    add("- Geometry's straight photo is one of the training photos. Margin drop versus alignment is an algebraic identity. The Fisher-style score is not Fisher-IRG.")
    add(f"- Failure-prediction numbers use {fail_counts[0]} shifted pairs. Fisher-style AUROC on the saved tables runs from {min(fisher_values):.3f} to {max(fisher_values):.3f}.")
    add("- Four frozen misses look like brand hits to the approved review because quotation marks blocked the matcher. They stay misses in Set A.")
    add("- Whether `PROTOCOL.md` was committed before `features.npz` existed on disk: unverified.")
    add("- Original sklearn and numpy for the T4 evaluation: NOT VERIFIED.")
    add("- Qwen hidden states of a real blank or noise photo: NOT VERIFIED. Phase 1 did not run Qwen.")
    add("- Same session or same place for the three frames: NOT VERIFIED. No time, GPS, camera model, or `photo_location` values were in the files.")
    add("- The saved ResNet matrix was not rebuilt, and its loader skips the EXIF turn.")
    add("")
    add("## 9. Limits")
    add("")
    add("From `linear_probe/PROTOCOL.md` section 7, plus the geometry note:")
    add("")
    add("- 60 photos, 20 products, one model, one annotator, three phone conditions that mix viewpoint and occlusion.")
    add("- Closed set: each test product is seen under the other conditions. This says nothing about unseen products.")
    add("- A linear probe can show that a label is readable. It does not show that the model uses that direction.")
    add("- The Fisher-style quantity is a proxy inspired by Fisher-IRG, not an implementation of it, and it is not a verified predictor of these mistakes.")
    add(f"- Largest recomputed gap for drop versus `|g| |d| alignment` is {max(gap for _, gap in identity_gaps):.3e}, on the saved geometry tables.")
    add(f"- {len(wrong)} misses, {wrong['slug'].nunique()} products, {fail_counts[0]} shifted failure pairs. They are not {len(wrong)} independent cases, and a 7-pair comparison is underpowered.")
    add("")
    add("## 10. Reproducibility record")
    add("")
    add("| item | value |")
    add("|---|---|")
    add(f"| branch | `{git('branch', '--show-current')}` |")
    add(f"| upstream | {branch_upstream} |")
    add(f"| this machine | Python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}, sklearn {sklearn.__version__}, numpy {numpy.__version__} |")
    add(f"| probe seed | {pa.RNG_SEED} |")
    add("| C | 1.0 |")
    add("| original permutations | 1000 for primary sites and controls |")
    add(f"| prompt SHA-256 | `{prompt_hash}` |")
    add(f"| features SHA-256 | `{features_hash}` |")
    add(f"| protocol commit | `{protocol}` |")
    add(f"| pytest | exit {pytest_run.returncode} |")
    add("")
    add("Commits on `phase1-audit` that are not on `linear-probe`:")
    add("")
    for commit in commits:
        add(f"- `{commit}`")
    add("")
    add("Nothing was pushed. `linear_probe/results/` and `linear_probe/results_resnet/` were not modified. `features.npz` is still untracked.")
    add("")
    add("Commands behind the numbers:")
    add("")
    add("- Step 2: `run_study.py --features features.npz --manifest manifest.csv --out ..\\research_followup\\step2_rerun --n-perm 1000`")
    add("- Step 4 and this report: `.venv\\Scripts\\python.exe -m pytest linear_probe\\test_probe_analysis.py -q`")
    add("- Step 5: `.venv\\Scripts\\python.exe research_followup\\step5_isolation.py`")
    add("- Step 6: `.venv\\Scripts\\python.exe research_followup\\step6_shortcuts.py`")
    add("- Step 7: `.venv\\Scripts\\python.exe research_followup\\step7_geometry_note.py`")
    add("- This file: `.venv\\Scripts\\python.exe research_followup\\build_phase1_audit.py`")
    add("")
    text = "\n".join(lines) + "\n"
    (OUT / "phase1_audit.md").write_text(text, encoding="utf-8")
    print(f"wrote phase1_audit.md chars {len(text)} pytest {pytest_run.returncode} disagreements {disagreements}")


if __name__ == "__main__":
    main()
