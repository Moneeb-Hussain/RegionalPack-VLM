import numpy as np, probe_analysis as pa
rng = np.random.default_rng(1)
P, C, D = 20, 3, 128
ident = rng.normal(size=(P, D))
cond_shift = rng.normal(size=(C, D)) * 0.8
X, prod, cond = [], [], []
for p in range(P):
    for c in range(C):
        X.append(ident[p] + cond_shift[c] + rng.normal(size=D) * 1.5); prod.append(p); cond.append(c)
X, prod, cond = np.array(X), np.array(prod), np.array(cond)
model_correct = rng.random(len(prod)) > 0.35
primary = prod < 18
rep = pa.decodability_report(X, prod, cond, model_correct, primary, n_perm=40)
print({k: v for k, v in rep.items() if k != "per_photo_probe_correct"})
assert rep["acc"] > 0.8 and rep["null_mean"] < 0.2 and rep["p_overall"] < 0.05
rows = pa.geometry_pairs(X, prod, cond, model_correct)
g = pa.geometry_report(rows, n_boot=200)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items() if k != "fisher_rank_corr_across_temps"})
print("fisher temp rank corr:", {k: round(v, 2) for k, v in g["fisher_rank_corr_across_temps"].items()})
assert g["identity_max_err"] < 1e-6 and g["n_pairs"] == 40
# null case: pure-noise features must NOT look decodable
Xn = rng.normal(size=X.shape)
rn = pa.decodability_report(Xn, prod, cond, model_correct, primary, n_perm=40)
print("noise acc", round(rn["acc"], 3), "p", round(rn["p_overall"], 3)); assert rn["p_overall"] > 0.05
print("ALL TESTS PASSED")
