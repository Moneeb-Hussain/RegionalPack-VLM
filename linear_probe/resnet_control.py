"""
Exploratory control (added AFTER the primary results were seen; label it that way in the write-up).
Question: is product identity just as decodable from an ordinary ImageNet ResNet-18, which has nothing to do with Qwen?
Same manifest, same LOCO probe, same permutation null, same stats as run_study.py.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u resnet_control.py --manifest manifest.csv --out results_resnet --n-perm 1000
"""
import argparse, os, numpy as np, pandas as pd, torch
from PIL import Image
import probe_analysis as pa

def main(manifest, out, n_perm, weights):
    os.makedirs(out, exist_ok=True)
    import torchvision
    from torchvision import transforms as T
    torch.manual_seed(0)
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if weights == "imagenet" else None
    net = torchvision.models.resnet18(weights=w); net.fc = torch.nn.Identity(); net.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; net.to(dev)
    # whole photo resized to 224x224 (no centre crop, so packaging edges are kept)
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    df = pd.read_csv(manifest)
    feats = []
    with torch.no_grad():
        for p in df["path"]:
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
            feats.append(net(x).squeeze(0).cpu().numpy())
    X = np.stack(feats).astype(np.float64); print("features", X.shape, "weights:", weights)
    np.savez(os.path.join(out, "features_resnet18.npz"), resnet18_pool=X)

    cmap = {"straight": 0, "tilt": 1, "occluded": 2}
    product = df["product_id"].to_numpy(int); cond = df["condition"].map(cmap).to_numpy(int)
    primary = df["primary"].to_numpy(bool); model_correct = df["model_correct"].to_numpy(bool)
    rep = pa.decodability_report(X, product, cond, model_correct, primary, n_perm=n_perm)
    mc = rep["mcnemar_probe_only_vs_model_only_p"]
    row = dict(site="resnet18_pool", role="exploratory_control_added_after_primary", acc=rep["acc"], ci_lo=rep["ci"][0], ci_hi=rep["ci"][1],
               null_mean=rep["null_mean"], p_overall=rep["p_overall"], n_model_wrong=rep["n_model_wrong"], n_recovered=rep.get("n_recovered"),
               probe_acc_when_model_wrong=rep.get("probe_acc_when_model_wrong"), p_when_model_wrong=rep.get("p_when_model_wrong"),
               mcnemar_probe_only=mc[0], mcnemar_model_only=mc[1], mcnemar_p=mc[2])
    pd.DataFrame([row]).to_csv(os.path.join(out, "resnet_control.csv"), index=False)
    print(f"resnet18_pool  acc={rep['acc']:.2f} [{rep['ci'][0]:.2f},{rep['ci'][1]:.2f}] null={rep['null_mean']:.2f} p={rep['p_overall']:.3f}")
    print(f"model-wrong photos recovered: {rep.get('n_recovered')} of {rep['n_model_wrong']}  (Qwen sites: 12 of 12, colour histogram: 10 of 12)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--out", default="results_resnet")
    ap.add_argument("--n-perm", type=int, default=1000); ap.add_argument("--weights", default="imagenet", choices=["imagenet", "none"])
    a = ap.parse_args(); main(a.manifest, a.out, a.n_perm, a.weights)
