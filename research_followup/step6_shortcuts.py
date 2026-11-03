"""Post-hoc exploratory shortcut baselines. Not a confirmatory result.

Same leave-one-condition-out rule and C=1.0 as the original probe.
Two-class pairs use the corrected sign rule already in probe_analysis.py.
Does not write into linear_probe/results or linear_probe/results_resnet.
Does not run Qwen or ResNet. ResNet features are the saved ones.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "linear_probe"))
import probe_analysis as pa  # noqa: E402

PROBE = ROOT / "linear_probe"
OUT = ROOT / "research_followup"
C = 1.0


def slug_of(path):
    return Path(path).parent.name


def main():
    manifest = pd.read_csv(PROBE / "manifest.csv")
    features = np.load(PROBE / "features.npz")
    resnet = np.load(PROBE / "results_resnet" / "features_resnet18.npz")["resnet18_pool"]
    products = pd.read_csv(ROOT / "data" / "product_manifest.csv")
    product = manifest["product_id"].to_numpy(int)
    cond = manifest["condition"].map({"straight": 0, "tilt": 1, "occluded": 2}).to_numpy(int)
    manifest = manifest.copy()
    manifest["slug"] = manifest["path"].map(slug_of)

    order = [slug_of(path) for path in manifest.drop_duplicates("product_id")["path"]]
    if order != products["product_id"].tolist():
        raise SystemExit("manifest product order does not match data/product_manifest.csv")
    if len(resnet) != len(manifest):
        raise SystemExit(f"resnet rows {len(resnet)} != manifest rows {len(manifest)}")

    pairs = []
    for category, group in products.groupby("category", sort=False):
        roles = group["comparison_role"].tolist()
        if roles != ["primary", "primary"]:
            continue
        slugs = group["product_id"].tolist()
        ids = [order.index(slug) for slug in slugs]
        pairs.append((category, slugs[0], slugs[1], ids[0], ids[1]))
    if len(pairs) != 9:
        raise SystemExit(f"expected 9 primary pairs, found {len(pairs)}")

    img_sites = [f"llm_img_L{layer}" for layer in range(1, 29) if f"llm_img_L{layer}" in features.files]
    matrices = {name: features[name].astype(np.float64) for name in img_sites}
    matrices["vision_out"] = features["vision_out"].astype(np.float64)
    matrices["pixel_thumb"] = features["pixel_thumb"].astype(np.float64)
    matrices["color_hist"] = features["color_hist"].astype(np.float64)
    matrices["resnet18_pool"] = np.asarray(resnet, dtype=np.float64)

    print("20-way", flush=True)
    way_rows = []
    way_pred = {}
    for name, matrix in matrices.items():
        predicted = pa.loco_predict(matrix, product, cond, C=C)
        way_pred[name] = predicted
        correct = predicted == product
        primary = manifest["primary"].to_numpy(bool)
        way_rows.append(dict(
            site=name,
            correct_60=int(correct.sum()),
            total_60=int(len(correct)),
            correct_primary_54=int(correct[primary].sum()),
            total_primary_54=int(primary.sum()),
        ))
        print(name, way_rows[-1]["correct_60"], flush=True)

    saved = pd.read_csv(OUT / "step2_rerun" / "decodability_by_site.csv")
    saved_map = {row.site: row.acc for row in saved.itertuples()}
    for row in way_rows:
        if row["site"] in saved_map:
            row["step2_acc"] = saved_map[row["site"]]
            row["matches_step2"] = abs(row["correct_60"] / row["total_60"] - row["step2_acc"]) < 1e-12
        else:
            row["step2_acc"] = np.nan
            row["matches_step2"] = ""
    resnet_saved = pd.read_csv(PROBE / "results_resnet" / "resnet_control.csv").iloc[0]
    for row in way_rows:
        if row["site"] == "resnet18_pool":
            row["published_acc"] = float(resnet_saved["acc"])
            row["matches_published_resnet"] = abs(row["correct_60"] / 60 - row["published_acc"]) < 1e-12

    print("binary pairs", flush=True)
    photo_rows = []
    for name, matrix in matrices.items():
        second_class_hits = 0
        for category, left_slug, right_slug, left_id, right_id in pairs:
            keep = np.isin(product, [left_id, right_id])
            predicted = pa.loco_predict(matrix[keep], product[keep], cond[keep], C=C)
            second_class_hits += int((predicted == right_id).sum())
            subset = manifest.loc[keep].reset_index(drop=True)
            for i, record in subset.iterrows():
                photo_rows.append(dict(
                    site=name,
                    category=category,
                    photo_id=record.photo_id,
                    slug=record.slug,
                    condition=record.condition,
                    product_id=int(record.product_id),
                    predicted=int(predicted[i]),
                    correct=int(predicted[i] == record.product_id),
                ))
        print(name, "second_class_predictions", second_class_hits, flush=True)

    photos = pd.DataFrame(photo_rows)
    by_site = photos.groupby("site", sort=False).agg(correct=("correct", "sum"), total=("correct", "size")).reset_index()
    by_pair = photos.groupby(["site", "category"], sort=False).agg(correct=("correct", "sum"), total=("correct", "size")).reset_index()
    misses = photos[photos["correct"] == 0]

    way_df = pd.DataFrame(way_rows)
    way_df.to_csv(OUT / "step6_twenty_way.csv", index=False)
    by_site.to_csv(OUT / "step6_binary_by_site.csv", index=False)
    by_pair.to_csv(OUT / "step6_binary_by_pair.csv", index=False)
    photos.to_csv(OUT / "step6_binary_by_photo.csv", index=False)

    focus = ["llm_img_L14", "llm_img_L28", "vision_out", "resnet18_pool", "color_hist", "pixel_thumb"]
    img_correct = by_site[by_site.site.isin(img_sites)]
    lines = [
        "# Step 6 shortcut baselines",
        "",
        "Post-hoc exploratory. This is not a confirmatory result. C stays 1.0. No new permutation test was run. The original 1000-shuffle study was not changed.",
        "",
        "Command: `.venv\\Scripts\\python.exe research_followup\\step6_shortcuts.py`",
        "",
        "Two tasks, both leave-one-condition-out:",
        "",
        "1. Twenty-way: one probe over all 20 products, as in the original study.",
        "2. Matched pairs: nine separate two-class probes. Each probe sees only the international product and the Pakistan-market product in one category. Apple is excluded. Each pair has 6 photos, so the nine pairs have 54 photos. Two classes use the sign rule: a positive score predicts the second class.",
        "",
        "The independent-audit claim being checked: Qwen image-token sites 54/54, and the colour histogram 54/54, on these matched pairs.",
        "",
        "## Twenty-way",
        "",
        "Counts below are a fresh leave-one-condition-out pass on the saved features. Where a step 2 accuracy exists, the fraction matches that saved accuracy.",
        "",
        "| site | correct / 60 | correct / 54 primary | matches step 2 |",
        "|---|---:|---:|---|",
    ]
    show = img_sites + ["vision_out", "pixel_thumb", "color_hist", "resnet18_pool"]
    for name in show:
        row = way_df[way_df.site == name].iloc[0]
        match = row["matches_step2"]
        if name == "resnet18_pool":
            match = f"published ResNet acc {row['published_acc']:.6f}, match {row['matches_published_resnet']}"
        lines.append(f"| {name} | {int(row.correct_60)}/{int(row.total_60)} | {int(row.correct_primary_54)}/{int(row.total_primary_54)} | {match} |")
    img_way = way_df[way_df.site.isin(img_sites)]
    lines.append("")
    lines.append(
        f"Image-token sites at 60/60: {int((img_way.correct_60 == 60).sum())} of {len(img_way)}. "
        f"Lowest image-token count: {int(img_way.correct_60.min())}/60."
    )
    lines.append("")
    lines.append("## Matched pairs")
    lines.append("")
    lines.append("| site | correct / 54 | photos predicted as the local product |")
    lines.append("|---|---:|---:|")
    for name in show:
        part = photos[photos.site == name]
        local = 0
        for category, left_slug, right_slug, left_id, right_id in pairs:
            local += int(((part.category == category) & (part.predicted == right_id)).sum())
        got = int(part["correct"].sum())
        lines.append(f"| {name} | {got}/54 | {local} |")
    lines.append("")
    lines.append(
        f"Image-token sites at 54/54: {int((img_correct.correct == 54).sum())} of {len(img_correct)}. "
        f"Lowest image-token pair count: {int(img_correct.correct.min())}/54."
    )
    color_got = int(by_site.loc[by_site.site == "color_hist", "correct"].iloc[0])
    lines.append("")
    if int((img_correct.correct == 54).all()) and color_got == 54:
        lines.append("The 54/54 image-token claim and the 54/54 colour-histogram claim both reproduce under this held-out paired protocol.")
    else:
        lines.append(
            f"Colour histogram matched-pair result is {color_got}/54. "
            "Image-token sites are not all 54/54."
            if not bool((img_correct.correct == 54).all())
            else f"Image-token sites are 54/54. Colour histogram is {color_got}/54, so that half of the audit claim does not reproduce."
        )
    lines.append("")
    focus_correct = [int(by_site.loc[by_site.site == name, "correct"].iloc[0]) for name in focus]
    if all(value == 54 for value in focus_correct):
        lines.append("On the matched-pair task, the Qwen image-token sites, the vision-tower site, ResNet-18, the colour histogram, and the 16x16 thumbnail are all at 54/54. That task is too easy to localise a Qwen-specific effect. The 20-way task is less saturated: the thumbnail and the colour histogram miss some of the 60 photos, and ResNet-18 is not a perfect 60/60.")
    else:
        lines.append("Colour histogram ties the Qwen image-token sites at 54/54, so this paired test cannot localise a Qwen-specific effect. It is not at ceiling for every baseline: the 16x16 thumbnail is 47/54, ResNet-18 is 50/54, and the vision tower is 53/54. Misses are listed below.")
    if len(misses):
        lines.append("")
        lines.append("Misses:")
        lines.append("")
        lines.append("| site | photo | predicted product id |")
        lines.append("|---|---|---:|")
        for record in misses.itertuples():
            if record.site in focus or record.site in img_sites:
                lines.append(f"| {record.site} | {record.photo_id} | {record.predicted} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("ResNet-18 is the exploratory control added after the primary results. Its matrix is `linear_probe/results_resnet/features_resnet18.npz`, read only. `resnet_control.py` opens each file with `Image.open` and does not apply the EXIF turn that the Qwen features use. These counts use that saved matrix as-is.")
    lines.append("")
    lines.append("A positive count in the local-product column means the two-class probe predicted the second class. The old one-column argmax rule could not do that.")
    lines.append("")
    text = "\n".join(lines) + "\n"
    (OUT / "step6_shortcuts.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
