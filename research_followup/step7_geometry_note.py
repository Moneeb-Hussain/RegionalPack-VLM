"""Check the geometry claims and write a short note. Does not edit saved results."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "linear_probe"))
import probe_analysis as pa  # noqa: E402

PROBE = ROOT / "linear_probe"
OUT = ROOT / "research_followup"
SITES = ["vision_out", "llm_img_L14", "llm_last_L14", "llm_img_L28", "llm_last_L28"]


def main():
    manifest = pd.read_csv(PROBE / "manifest.csv")
    manifest["slug"] = manifest["path"].map(lambda path: Path(path).parent.name)
    wrong = manifest[(manifest["primary"] == 1) & (manifest["model_correct"] == 0)].copy()
    shifted = wrong[wrong["condition"] != "straight"]
    by_product = wrong.groupby("slug").size().sort_values(ascending=False)

    lines = [
        "# Step 7 geometry note",
        "",
        "Do not treat the geometry results as a new finding. The points below are checked from the saved files. Nothing in `linear_probe/results/` was changed.",
        "",
        "Command: `.venv\\Scripts\\python.exe research_followup\\step7_geometry_note.py`",
        "",
        "## Margin drop is an identity",
        "",
        "For a shift `d` and a margin direction `g`, the drop in probe margin is exactly `|g| |d| alignment`. A high correlation between drop and alignment is that formula, not a separate discovery. The question left by the protocol is only how large the alignment values are.",
        "",
        "Recomputed as the largest absolute gap between `drop` and `g_norm * dist * align` in each saved geometry table:",
        "",
        "| site | pairs | largest gap |",
        "|---|---:|---:|",
    ]
    fail_ids = None
    auroc_rows = []
    for site in SITES:
        table = pd.read_csv(PROBE / "results" / f"geometry_{site}.csv")
        gap = np.max(np.abs(table["drop"] - table["g_norm"] * table["dist"] * table["align"]))
        lines.append(f"| {site} | {len(table)} | {gap:.3e} |")
        flags = table["model_fail"].map(lambda value: str(value).strip().lower() in {"1", "true"})
        ids = set(zip(table.loc[flags, "product"].astype(int), table.loc[flags, "cond"].astype(int)))
        if fail_ids is None:
            fail_ids = ids
        elif ids != fail_ids:
            raise SystemExit(f"{site} failure pairs differ from the first site")
        auroc_rows.append(dict(
            site=site,
            align=pa.auroc(table["align"].to_numpy(float), flags.to_numpy(bool)),
            dist=pa.auroc(table["dist"].to_numpy(float), flags.to_numpy(bool)),
            fisher=pa.auroc(table["fisher_That"].to_numpy(float), flags.to_numpy(bool)),
        ))
    lines.append("")
    lines.append("The largest gap is rounding error, around 1e-13.")
    lines.append("")
    lines.append("## The Fisher-style number is not Fisher-IRG")
    lines.append("")
    lines.append("The code builds a local score from the probe's own softmax. The protocol calls that a proxy inspired by Fisher-IRG, not an implementation of Fisher-IRG. It is not a verified predictor of the model's mistakes. On the same 7 failure pairs, the AUROC for that proxy crosses both sides of 0.5 depending on the site:")
    lines.append("")
    lines.append("| site | AUROC alignment | AUROC distance | AUROC Fisher-style |")
    lines.append("|---|---:|---:|---:|")
    for row in auroc_rows:
        lines.append(f"| {row['site']} | {row['align']:.3f} | {row['dist']:.3f} | {row['fisher']:.3f} |")
    fishers = [row["fisher"] for row in auroc_rows]
    lines.append("")
    lines.append(f"Fisher-style AUROC ranges from {min(fishers):.3f} to {max(fishers):.3f}. Chance is 0.5. That spread, on 7 pairs, does not establish a predictor.")
    lines.append("")
    lines.append("## Seven shifted failure pairs, from five products")
    lines.append("")
    lines.append(f"Primary photos scored wrong: {len(wrong)}. Distinct products: {wrong['slug'].nunique()}. These are not {len(wrong)} independent cases.")
    lines.append("")
    lines.append("| product | wrong photos |")
    lines.append("|---|---:|")
    for slug, count in by_product.items():
        lines.append(f"| {slug} | {int(count)} |")
    lines.append("")
    lines.append(f"Straight photos among them: {int((wrong['condition'] == 'straight').sum())}. Tilt or occluded photos: {len(shifted)}. The geometry tables only score a straight photo against a tilt or occluded photo, so they contain {len(fail_ids)} model-failure pairs, the same {len(fail_ids)} on every primary site. A result built on 7 pairs is underpowered.")
    lines.append("")
    lines.append("Those shifted pairs are:")
    lines.append("")
    id_to_slug = manifest.drop_duplicates("product_id").set_index("product_id")["slug"].to_dict()
    cond_name = {0: "straight", 1: "tilt", 2: "occluded"}
    for product_id, cond in sorted(fail_ids):
        lines.append(f"- {id_to_slug[product_id]} / {cond_name[cond]}")
    lines.append("")
    text = "\n".join(lines) + "\n"
    (OUT / "step7_geometry_note.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
