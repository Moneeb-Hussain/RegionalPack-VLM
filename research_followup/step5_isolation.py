"""Post-hoc exploratory checks for train/test isolation and easy image cues.

Does not change C, the permutation count of the original run, or any saved result.
Qwen is not run. Blank-image features of Qwen are therefore not produced here.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "linear_probe"))
import probe_analysis as pa  # noqa: E402

PROBE = ROOT / "linear_probe"
OUT = ROOT / "research_followup"
C = 1.0
N_PERM = 40
CHANCE = 1.0 / 20.0
PRIMARY = [
    "vision_out",
    "llm_img_L14",
    "llm_last_L14",
    "llm_img_L28",
    "llm_last_L28",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ahash_bits(img):
    small = img.convert("L").resize((8, 8), Image.Resampling.BOX)
    values = np.asarray(small, dtype=np.float64)
    return (values > values.mean()).astype(np.uint8).reshape(-1)


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def border_rgb(img, frac=0.08):
    pixels = np.asarray(img)
    height, width = pixels.shape[:2]
    band = max(1, int(round(frac * min(height, width))))
    mask = np.ones((height, width), dtype=bool)
    if height > 2 * band and width > 2 * band:
        mask[band:-band, band:-band] = False
    return pixels[mask].reshape(-1, 3).mean(axis=0)


def exif_fields(img):
    exif = img.getexif()
    gps = exif.get_ifd(0x8825)
    return {
        "make": str(exif.get(271, "")),
        "model": str(exif.get(272, "")),
        "software": str(exif.get(305, "")),
        "datetime": str(exif.get(306, "")),
        "datetime_original": str(exif.get(36867, "")),
        "datetime_digitized": str(exif.get(36868, "")),
        "orientation": str(exif.get(274, "")),
        "gps_present": bool(gps),
    }


def color_hist(img, bins=8):
    pixels = np.asarray(img.convert("RGB").resize((128, 128))).reshape(-1, 3)
    counts = np.concatenate([
        np.histogram(pixels[:, channel], bins=bins, range=(0, 255))[0] for channel in range(3)
    ]).astype(np.float32)
    return counts / counts.sum()


def pixel_thumb(img):
    return np.asarray(img.resize((16, 16))).astype(np.float32).ravel() / 255.0


def load_rgb(path):
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def pair_rows(X, product, cond, drop_ref):
    rows = []
    for held in np.unique(cond):
        if held == 0:
            continue
        for p_id in np.unique(product):
            i_ref = np.where((product == p_id) & (cond == 0))[0]
            i_held = np.where((product == p_id) & (cond == held))[0]
            if len(i_ref) != 1 or len(i_held) != 1:
                continue
            i_ref, i_held = int(i_ref[0]), int(i_held[0])
            train = np.array(cond != held)
            ref_in = bool(train[i_ref])
            held_in = bool(train[i_held])
            if drop_ref:
                train[i_ref] = False
            scaler, clf = pa.fit_probe(X[train], product[train], C)
            z_ref = scaler.transform(X[[i_ref]])[0]
            z_held = scaler.transform(X[[i_held]])[0]
            true_row = int(np.where(clf.classes_ == p_id)[0][0])
            scores = clf.coef_ @ z_ref + clf.intercept_
            rival_scores = scores.copy()
            rival_scores[true_row] = -np.inf
            rival_row = int(np.argmax(rival_scores))
            direction = clf.coef_[true_row] - clf.coef_[rival_row]
            intercept_gap = clf.intercept_[true_row] - clf.intercept_[rival_row]
            margin_ref = float(direction @ z_ref + intercept_gap)
            margin_held = float(direction @ z_held + intercept_gap)
            shift = z_held - z_ref
            align = float((shift @ -direction) / (np.linalg.norm(shift) * np.linalg.norm(direction) + 1e-12))
            predicted = int(clf.predict(z_held.reshape(1, -1))[0])
            rows.append(dict(
                product=int(p_id), cond=int(held), ref_in_train=ref_in, held_in_train=held_in,
                drop=margin_ref - margin_held, align=align, margin_ref=margin_ref,
                rival=int(clf.classes_[rival_row]), pred=predicted,
            ))
    return rows


def sanity(X, product, cond, model_correct, primary):
    try:
        report = pa.decodability_report(X, product, cond, model_correct, primary, n_perm=N_PERM, C=C)
    except ValueError as exc:
        return dict(acc="FAILED", null_mean="FAILED", p_overall="FAILED", error=str(exc))
    return dict(
        acc=f"{float(report['acc']):.6f}",
        null_mean=f"{float(report['null_mean']):.6f}",
        p_overall=f"{float(report['p_overall']):.6f}",
        error="",
    )


def main():
    manifest = pd.read_csv(PROBE / "manifest.csv")
    features = np.load(PROBE / "features.npz")
    product = manifest["product_id"].to_numpy(int)
    cond = manifest["condition"].map({"straight": 0, "tilt": 1, "occluded": 2}).to_numpy(int)
    primary = manifest["primary"].to_numpy(bool)
    model_correct = manifest["model_correct"].to_numpy(bool)
    lines = [
        "# Step 5 isolation and leakage",
        "",
        "Post-hoc exploratory. C stays 1.0. The new sanity runs below use 40 label shuffles. The original study used 1000 and is not rerun here.",
        "",
        "Command: `.venv\\Scripts\\python.exe research_followup\\step5_isolation.py`",
        "",
        "## Held-out condition",
        "",
        "Leave-one-condition-out fits the scaler and the probe on photos whose condition is not the held-out one (`loco_predict` in `linear_probe/probe_analysis.py`). The temperature is chosen by more fits inside those remaining conditions only (`calibrate_temperature`). The held-out photos are not in those fits. `test_held_out_condition_never_enters_scaler_probe_or_temperature` checks the exact training rows.",
        "",
    ]

    print("membership", flush=True)
    base_vision = pair_rows(features["vision_out"].astype(np.float64), product, cond, drop_ref=False)
    n_pairs = len(base_vision)
    n_ref_in = sum(row["ref_in_train"] for row in base_vision)
    n_held_in = sum(row["held_in_train"] for row in base_vision)
    lines.append(
        f"Saved `vision_out` has {n_pairs} straight-versus-other pairs. "
        f"The straight photo is in the training rows for {n_ref_in} of them. "
        f"The shifted photo is in the training rows for {n_held_in} of them."
    )
    lines.append("")
    lines.append("## Reference photo inside the geometry training fold")
    lines.append("")
    lines.append(
        "`geometry_pairs` never holds out the straight condition. "
        "For a tilt photo, training is straight plus occluded. "
        "For an occluded photo, training is straight plus tilt. "
        "The straight photo of the same product is one training row, and the rival class is picked from the score at that straight photo. "
        "This script leaves that rule in place. "
        "It then refits each pair with that one straight photo removed and compares the margin drop, the rival, and the class predicted for the shifted photo."
    )
    lines.append("")

    summaries = []
    details = []
    for site in PRIMARY:
        print("ablation", site, flush=True)
        matrix = features[site].astype(np.float64)
        base = pair_rows(matrix, product, cond, drop_ref=False)
        dropped = pair_rows(matrix, product, cond, drop_ref=True)
        published = pd.read_csv(PROBE / "results" / f"geometry_{site}.csv")
        base_drop = np.array([row["drop"] for row in base])
        published_drop = published["drop"].to_numpy(float)
        if len(published_drop) == len(base_drop):
            published_gap = float(np.max(np.abs(base_drop - published_drop)))
        else:
            published_gap = float("nan")
        dropped_drop = np.array([row["drop"] for row in dropped])
        same_pred = sum(int(a["pred"] == b["pred"]) for a, b in zip(base, dropped))
        same_rival = sum(int(a["rival"] == b["rival"]) for a, b in zip(base, dropped))
        mean_abs = float(np.mean(np.abs(base_drop - dropped_drop)))
        max_abs = float(np.max(np.abs(base_drop - dropped_drop)))
        spearman = float(pa.spearmanr(base_drop, dropped_drop)[0])
        summaries.append(dict(
            site=site, n_pairs=len(base), published_drop_max_abs_diff=published_gap,
            pred_agree=same_pred, rival_agree=same_rival,
            mean_abs_drop_change=mean_abs, max_abs_drop_change=max_abs, spearman_drop=spearman,
        ))
        for kind, rows in (("with_reference", base), ("without_reference", dropped)):
            for row in rows:
                details.append(dict(site=site, fit=kind, **row))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "step5_geometry_ablation.csv", index=False)
    pd.DataFrame(details).to_csv(OUT / "step5_geometry_pairs.csv", index=False)
    lines.append("Removing the straight photo changes the geometry numbers as follows. `published_drop_max_abs_diff` is the gap between this script's with-reference drop and `linear_probe/results/geometry_<site>.csv`, which is left unchanged.")
    lines.append("")
    lines.append("| site | pairs | published drop gap | predictions that agree | rivals that agree | mean abs drop change | max abs drop change | spearman of drops |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        lines.append(
            f"| {row['site']} | {row['n_pairs']} | {row['published_drop_max_abs_diff']:.3e} | "
            f"{row['pred_agree']}/{row['n_pairs']} | {row['rival_agree']}/{row['n_pairs']} | "
            f"{row['mean_abs_drop_change']:.6f} | {row['max_abs_drop_change']:.6f} | {row['spearman_drop']:.6f} |"
        )
    lines.append("")

    print("images", flush=True)
    products = pd.read_csv(ROOT / "data" / "product_manifest.csv")
    if "photo_location" in products.columns:
        location_text = products["photo_location"].astype("string").fillna("").str.strip()
        photo_location_filled = int((location_text != "").sum())
    else:
        photo_location_filled = -1
    provenance = pd.read_csv(ROOT / "data" / "image_provenance.csv")
    provenance_by_dest = {row.dest_path.replace("\\", "/"): row.sha256 for row in provenance.itertuples()}
    image_rows = []
    for record in manifest.itertuples():
        path = (PROBE / record.path).resolve()
        slug = path.parent.name
        with Image.open(path) as raw:
            fields = exif_fields(raw)
        rgb = load_rgb(path)
        bits = ahash_bits(rgb)
        image_rows.append(dict(
            photo_id=record.photo_id,
            slug=slug,
            condition=record.condition,
            sha256=sha256_file(path),
            provenance_sha256=provenance_by_dest.get(f"data/raw/{slug}/{record.condition}.jpg", ""),
            ahash="".join(str(int(bit)) for bit in bits),
            border_r=float(border_rgb(rgb)[0]),
            border_g=float(border_rgb(rgb)[1]),
            border_b=float(border_rgb(rgb)[2]),
            **fields,
        ))
    images = pd.DataFrame(image_rows)
    images["sha_matches_provenance"] = images["sha256"] == images["provenance_sha256"]
    images.to_csv(OUT / "step5_image_checks.csv", index=False)

    sha_dupes = images["sha256"].duplicated().sum()
    hashes = [np.array(list(text), dtype=np.uint8) for text in images["ahash"]]
    pair_distances = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            distance = hamming(hashes[i], hashes[j])
            same_product = images.loc[i, "slug"] == images.loc[j, "slug"]
            pair_distances.append((distance, same_product, images.loc[i, "photo_id"], images.loc[j, "photo_id"]))
    across = [item for item in pair_distances if not item[1]]
    within = [item for item in pair_distances if item[1]]
    near_across = [item for item in across if item[0] <= 10]
    lines.append("## Exact and near-duplicate photos")
    lines.append("")
    lines.append(f"File SHA-256 matches `data/image_provenance.csv` for {int(images['sha_matches_provenance'].sum())} of {len(images)} photos. Duplicate file hashes: {int(sha_dupes)}.")
    lines.append(
        f"Average-hash Hamming distance, 64 bits, after the EXIF turn used at feature extraction. "
        f"Closest pair of different products: {min(item[0] for item in across)} "
        f"({min(across, key=lambda item: item[0])[2]} vs {min(across, key=lambda item: item[0])[3]}). "
        f"Pairs of different products at distance 10 or less: {len(near_across)}. "
        f"Closest pair of the same product: {min(item[0] for item in within)}."
    )
    lines.append("")
    lines.append("## Session and background cues")
    lines.append("")
    lines.append(f"`photo_location` is filled on {photo_location_filled} of {len(products)} rows in `data/product_manifest.csv`.")
    gps_n = int(images["gps_present"].sum())
    datetime_n = int((images["datetime_original"].str.strip() != "").sum())
    datetime_any = int(((images["datetime"].str.strip() != "") | (images["datetime_original"].str.strip() != "") | (images["datetime_digitized"].str.strip() != "")).sum())
    makes = sorted({value for value in images["make"] if value.strip()})
    models = sorted({value for value in images["model"] if value.strip()})
    lines.append(f"GPS EXIF blocks: {gps_n} of {len(images)}. DateTimeOriginal EXIF values: {datetime_n} of {len(images)}. Any of DateTime, DateTimeOriginal, DateTimeDigitized: {datetime_any} of {len(images)}.")
    lines.append(f"Camera Make values: {makes if makes else 'none'}. Camera Model values: {models if models else 'none'}.")
    if datetime_any == 0 and gps_n == 0:
        lines.append("Whether the three frames of a product were taken in one session cannot be checked from EXIF time or GPS. Those tags are absent. Same for whether a category pair was shot in one place.")
    lines.append("")
    lines.append("The photography guide says a category pair should share a place and a background. That is an instruction, not a measurement. The border color below is a measurement: mean RGB of the outer 8 percent of pixels after the EXIF turn.")
    within_distances = []
    for slug, group in images.groupby("slug"):
        colors = group[["border_r", "border_g", "border_b"]].to_numpy(float)
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                within_distances.append(float(np.linalg.norm(colors[i] - colors[j])))
    pair_distances_rgb = []
    other_distances = []
    centers = images.groupby("slug")[["border_r", "border_g", "border_b"]].mean()
    slugs = products["product_id"].tolist()
    categories = products["category"].tolist()
    by_cat = {}
    for slug, category in zip(slugs, categories):
        by_cat.setdefault(category, []).append(slug)
    paired = []
    for category, members in by_cat.items():
        if len(members) == 2 and all(slug in centers.index for slug in members):
            distance = float(np.linalg.norm(centers.loc[members[0]] - centers.loc[members[1]]))
            pair_distances_rgb.append(distance)
            paired.append((category, members[0], members[1], distance))
    center_items = list(centers.itertuples())
    for i in range(len(center_items)):
        for j in range(i + 1, len(center_items)):
            left, right = center_items[i], center_items[j]
            same_cat = False
            for members in by_cat.values():
                if left.Index in members and right.Index in members:
                    same_cat = True
            if same_cat:
                continue
            other_distances.append(float(np.linalg.norm(
                np.array([left.border_r, left.border_g, left.border_b])
                - np.array([right.border_r, right.border_g, right.border_b])
            )))
    lines.append(
        f"Mean distance between the three frames of one product: {np.mean(within_distances):.2f} RGB units. "
        f"Mean distance between the two products of one category: {np.mean(pair_distances_rgb):.2f}. "
        f"Mean distance between products from different categories: {np.mean(other_distances):.2f}."
    )
    lines.append("")
    lines.append("| category | product A | product B | border distance |")
    lines.append("|---|---|---|---:|")
    for category, left, right, distance in paired:
        lines.append(f"| {category} | {left} | {right} | {distance:.2f} |")
    lines.append("")

    extract_source = (PROBE / "extract_features.py").read_text(encoding="utf-8")
    numeric = []
    for key in features.files:
        array = features[key]
        numeric.append(array.dtype.kind in "fiu" and array.ndim == 2 and array.shape[0] == len(manifest))
    path_used_for_open = 'Image.open(r["path"])' in extract_source
    prompt_is_constant = "IDENT_PROMPT" in extract_source and "photo_id" not in extract_source.split("msgs =", 1)[1].split("inputs =", 1)[0]
    lines.append("## Filenames and paths")
    lines.append("")
    lines.append(
        f"`extract_features.py` opens the file with `Image.open` on the manifest path ({path_used_for_open}) and passes the turned picture, not the path, into the frozen prompt. "
        f"The message text does not include `photo_id` ({prompt_is_constant}). "
        f"Saved arrays in `features.npz`: {len(features.files)}. Numeric matrices with one row per photo: {sum(numeric)} of {len(features.files)}."
    )
    lines.append("The feature rows are means of vision or language hidden states, a 16 by 16 pixel thumbnail, or a colour histogram. The path string is not appended to those vectors.")
    lines.append("")

    print("sanity", flush=True)
    rng = np.random.default_rng(0)
    sanity_rows = []
    jobs = [
        ("vision_out real features", features["vision_out"].astype(np.float64)),
        ("vision_out all zeros", np.zeros_like(features["vision_out"], dtype=np.float64)),
        ("vision_out gaussian noise", rng.normal(size=features["vision_out"].shape)),
    ]
    blank = Image.new("RGB", (64, 64), (0, 0, 0))
    noise_thumbs, noise_hists = [], []
    for _ in range(len(manifest)):
        noise = Image.fromarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8), mode="RGB")
        noise_thumbs.append(pixel_thumb(noise))
        noise_hists.append(color_hist(noise))
    jobs.extend([
        ("pixel_thumb of black images", np.stack([pixel_thumb(blank) for _ in range(len(manifest))]).astype(np.float64)),
        ("color_hist of black images", np.stack([color_hist(blank) for _ in range(len(manifest))]).astype(np.float64)),
        ("pixel_thumb of noise images", np.stack(noise_thumbs).astype(np.float64)),
        ("color_hist of noise images", np.stack(noise_hists).astype(np.float64)),
    ])
    for name, matrix in jobs:
        print("sanity", name, flush=True)
        result = sanity(matrix, product, cond, model_correct, primary)
        result["name"] = name
        sanity_rows.append(result)
    step2 = pd.read_csv(OUT / "step2_rerun" / "decodability_by_site.csv")
    lines.append("## Chance sanity, post-hoc exploratory")
    lines.append("")
    lines.append(f"A 20-way probe has chance accuracy {CHANCE:.2f}. Each new run uses {N_PERM} label shuffles, seed 0, C={C}. `null_mean` is the accuracy when training labels are shuffled and test labels stay true.")
    lines.append("")
    lines.append("| input | accuracy | shuffled-label mean | p | error |")
    lines.append("|---|---:|---:|---:|---|")
    for row in sanity_rows:
        lines.append(
            f"| {row['name']} | {row['acc']} | {row['null_mean']} | {row['p_overall']} | {row['error']} |"
        )
    lines.append("")
    lines.append("Qwen features of an actual blank or noise photograph were not computed. Phase 1 does not run Qwen. The zero and noise rows above are feature-space stand-ins for `vision_out` only. That image-level Qwen check is NOT VERIFIED.")
    lines.append("")
    lines.append("The step 2 rerun already shuffled labels 1000 times on the saved features. Those null means, read from `research_followup/step2_rerun/decodability_by_site.csv`:")
    lines.append("")
    lines.append("| site | accuracy | 1000-shuffle mean |")
    lines.append("|---|---:|---:|")
    keep = step2[step2["role"].isin(["primary", "control"])]
    for row in keep.itertuples():
        lines.append(f"| {row.site} | {row.acc:.6f} | {row.null_mean:.6f} |")
    lines.append("")
    text = "\n".join(lines) + "\n"
    (OUT / "step5_isolation.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
