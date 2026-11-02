"""
Run the full pre-specified analysis on features.npz + manifest.csv (see PROTOCOL.md).
    python run_study.py --features features.npz --manifest manifest.csv --out results --n-perm 1000
"""
import argparse, json, os, re, numpy as np, pandas as pd
import probe_analysis as pa

def primary_sites(keys):
    Ls = sorted({int(m.group(1)) for k in keys for m in [re.match(r"llm_img_L(\d+)$", k)] if m})
    last = Ls[-1]; mid = min(Ls, key=lambda l: abs(l - last // 2))
    return ["vision_out", f"llm_img_L{mid}", f"llm_img_L{last}", f"llm_last_L{mid}", f"llm_last_L{last}"], Ls

def main(features, manifest, out, n_perm):
    os.makedirs(out, exist_ok=True)
    F = np.load(features); df = pd.read_csv(manifest)
    cmap = {"straight": 0, "tilt": 1, "occluded": 2}
    product = df["product_id"].to_numpy(int); cond = df["condition"].map(cmap).to_numpy(int)
    primary = df["primary"].to_numpy(bool); model_correct = df["model_correct"].to_numpy(bool)
    prim, Ls = primary_sites(F.files)
    controls = [k for k in ("pixel_thumb", "color_hist") if k in F.files]
    sites = [k for k in F.files if k in prim or k in controls]                 # confirmatory + controls first
    sites += [f"llm_img_L{l}" for l in Ls if f"llm_img_L{l}" not in sites] + [f"llm_last_L{l}" for l in Ls if f"llm_last_L{l}" not in sites]
    rows, geo_all, photo_rows = [], {}, []
    for s in sites:
        X = F[s].astype(np.float64)
        n = n_perm if (s in prim or s in controls) else max(50, n_perm // 10)  # exploratory layers get fewer permutations
        rep = pa.decodability_report(X, product, cond, model_correct, primary, n_perm=n)
        mc = rep["mcnemar_probe_only_vs_model_only_p"]
        rows.append(dict(site=s, role=("primary" if s in prim else "control" if s in controls else "exploratory"),
                         acc=rep["acc"], ci_lo=rep["ci"][0], ci_hi=rep["ci"][1], null_mean=rep["null_mean"], p_overall=rep["p_overall"],
                         n_model_wrong=rep["n_model_wrong"], n_recovered=rep.get("n_recovered"),
                         probe_acc_when_model_wrong=rep.get("probe_acc_when_model_wrong"), p_when_model_wrong=rep.get("p_when_model_wrong"),
                         mcnemar_probe_only=mc[0], mcnemar_model_only=mc[1], mcnemar_p=mc[2]))
        if s in prim:
            g = pa.geometry_pairs(X, product, cond, model_correct)
            pd.DataFrame(g).to_csv(os.path.join(out, f"geometry_{s}.csv"), index=False)
            geo_all[s] = pa.geometry_report(g)
            for i, ok in enumerate(rep["per_photo_probe_correct"]):
                photo_rows.append(dict(site=s, photo_id=df.iloc[i]["photo_id"], product_id=int(product[i]),
                                       condition=df.iloc[i]["condition"], primary=bool(primary[i]),
                                       model_correct=bool(model_correct[i]), probe_correct=bool(ok)))
        print(f"{s:16s} acc={rep['acc']:.2f} [{rep['ci'][0]:.2f},{rep['ci'][1]:.2f}] null={rep['null_mean']:.2f} p={rep['p_overall']:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(out, "decodability_by_site.csv"), index=False)
    if photo_rows:
        pd.DataFrame(photo_rows).to_csv(os.path.join(out, "per_photo_probe.csv"), index=False)
    json.dump(geo_all, open(os.path.join(out, "geometry_summary.json"), "w"), indent=1, default=float)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        R = pd.DataFrame(rows)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for name, pre in (("evidence (image tokens)", "llm_img_L"), ("decision (last token)", "llm_last_L")):
            sub = R[R.site.str.startswith(pre)].copy(); sub["L"] = sub.site.str.extract(r"L(\d+)").astype(int); sub = sub.sort_values("L")
            ax[0].plot(sub.L, sub.acc, marker="o", label=name); ax[0].fill_between(sub.L, sub.ci_lo, sub.ci_hi, alpha=0.15)
            ax[1].plot(sub.L, sub.probe_acc_when_model_wrong, marker="o", label=name)
        ax[0].axhline(1 / 20, ls="--", c="gray", label="chance"); ax[0].set_title("Probe accuracy by LLM layer (leave-one-condition-out)")
        ax[1].axhline(1 / 20, ls="--", c="gray"); ax[1].set_title("Probe accuracy on photos the model got WRONG")
        for a in ax: a.set_xlabel("layer"); a.set_ylabel("accuracy"); a.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out, "layerwise_decodability.png"), dpi=150)
    except Exception as e:
        print("plot skipped:", e)
    print("\nPrimary sites use alpha = 0.01 (Bonferroni over 5). See PROTOCOL.md for how to read each outcome.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--features", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="results"); ap.add_argument("--n-perm", type=int, default=1000)
    a = ap.parse_args(); main(a.features, a.manifest, a.out, a.n_perm)
