"""
Analysis library for the RegionalPack-VLM linear-probe study.

Input per feature "site" (e.g. vision_out, llm_img_L14, llm_last_L28): an array (N, D).
Metadata arrays of length N: product (int 0..P-1), cond (int 0..C-1, 0 = straight reference),
model_correct (bool, from the FROZEN Set A scorer), primary (bool, True for the 18 primary products).

Everything here is a pure function of numpy arrays so it can be tested without a GPU.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import binomtest, spearmanr, rankdata

RNG_SEED = 0

# ---------------------------------------------------------------- probes
class _ReducedLR:
    """L2-regularised multinomial logistic regression fitted in the row space of the training data.
    With more features than samples the optimum lies in the span of the training rows (the penalty only grows
    for components outside it), so fitting on rank-r coordinates gives the same solution as the full fit.
    coef_ is mapped back to the standardised feature space so callers see an ordinary fitted model."""
    def __init__(self, lr, V):
        self.classes_ = lr.classes_
        self.coef_ = lr.coef_ @ V.T
        self.intercept_ = lr.intercept_
    def decision_function(self, Z):
        scores = Z @ self.coef_.T + self.intercept_
        # Two classes: sklearn keeps one row. The score is for classes_[1].
        if len(self.classes_) == 2:
            return np.asarray(scores).reshape(-1)
        return scores
    def predict(self, Z):
        scores = self.decision_function(Z)
        if len(self.classes_) == 2:
            # score > 0 is the second class. A zero score stays the first class.
            return np.where(scores > 0, self.classes_[1], self.classes_[0])
        return self.classes_[np.argmax(scores, axis=1)]

def _require_multiclass(clf, caller):
    """Geometry and class probabilities index one coefficient row per class.
    A two-class probe does not have that layout."""
    n = len(clf.classes_)
    if n < 3:
        raise ValueError(
            f"{caller} needs one coefficient row per class. "
            f"This probe has {n} classes; a two-class model stores a single score, "
            "and score > 0 means the second class."
        )

def fit_probe(Xtr, ytr, C=1.0):
    sc = StandardScaler().fit(Xtr)
    Z = sc.transform(Xtr)
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    r = int((S > 1e-10 * S[0]).sum())
    V = Vt[:r].T
    lr = LogisticRegression(C=C, max_iter=5000).fit(Z @ V, ytr)
    return sc, _ReducedLR(lr, V)

def loco_predict(X, y, cond, C=1.0, y_train_override=None):
    """Leave-one-condition-out: train on the other conditions, predict the held-out one.
    y_train_override lets the permutation test shuffle TRAIN labels while testing on TRUE labels."""
    pred = np.empty(len(y), dtype=int)
    for c in np.unique(cond):
        tr, te = cond != c, cond == c
        ytr = (y if y_train_override is None else y_train_override)[tr]
        sc, clf = fit_probe(X[tr], ytr, C)
        pred[te] = clf.predict(sc.transform(X[te]))
    return pred

def loco_accuracy_with_null(X, y, cond, n_perm=200, C=1.0, seed=RNG_SEED):
    """Observed per-photo correctness + a permutation null (shuffled train labels, true test labels)."""
    rng = np.random.default_rng(seed)
    obs = loco_predict(X, y, cond, C) == y
    null = np.zeros((n_perm, len(y)), dtype=bool)
    for i in range(n_perm):
        y_shuf = y.copy()
        for c in np.unique(cond):                      # shuffle labels inside each training condition
            idx = np.where(cond == c)[0]
            y_shuf[idx] = rng.permutation(y[idx])
        null[i] = loco_predict(X, y, cond, C, y_train_override=y_shuf) == y
    return obs, null

# ---------------------------------------------------------------- statistics
def cluster_bootstrap_mean(values, groups, n_boot=2000, seed=RNG_SEED):
    """Mean of `values` with a 95% CI from resampling whole groups (products), not single photos."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    per = {g: values[groups == g] for g in ug}
    means = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        v = np.concatenate([per[g] for g in pick])
        means.append(v.mean())
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def perm_pvalue(observed_stat, null_stats):
    return float((1 + np.sum(null_stats >= observed_stat)) / (1 + len(null_stats)))

def mcnemar_exact(a_correct, b_correct):
    """Exact McNemar test: are the two classifiers' errors different? returns (a_only, b_only, p)."""
    a_only = int(np.sum(a_correct & ~b_correct)); b_only = int(np.sum(~a_correct & b_correct))
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(a_only, n, 0.5).pvalue)
    return a_only, b_only, p

def auroc(score, label):
    """P(score of a positive > score of a negative). label: bool, True = positive (model failure)."""
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

# ---------------------------------------------------------------- decodability + dissociation (H1, H2)
def decodability_report(X, product, cond, model_correct, primary, n_perm=200, C=1.0):
    obs, null = loco_accuracy_with_null(X, product, cond, n_perm, C)
    acc, lo, hi = cluster_bootstrap_mean(obs.astype(float), product)
    null_acc = null.mean(axis=1)
    out = dict(acc=acc, ci=(lo, hi), null_mean=float(null_acc.mean()), p_overall=perm_pvalue(acc, null_acc))
    # H2: among photos where the model's own answer was wrong (primary products only)
    wrong = primary & ~model_correct
    out["n_model_wrong"] = int(wrong.sum())
    if wrong.sum() > 0:
        w_acc = float(obs[wrong].mean())
        w_null = null[:, wrong].mean(axis=1)
        out["probe_acc_when_model_wrong"] = w_acc
        out["probe_acc_when_model_wrong_null_mean"] = float(w_null.mean())
        out["p_when_model_wrong"] = perm_pvalue(w_acc, w_null)
        out["n_recovered"] = int(obs[wrong].sum())
    a_only, b_only, p = mcnemar_exact(obs[primary], model_correct[primary])
    out["mcnemar_probe_only_vs_model_only_p"] = (a_only, b_only, p)
    out["per_photo_probe_correct"] = obs
    return out

# ---------------------------------------------------------------- geometry (H3, H4)
def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True); e = np.exp(z); return e / e.sum(axis=-1, keepdims=True)

def calibrate_temperature(X, y, cond, train_conds, C=1.0, grid=None):
    """Temperature chosen by held-out NLL INSIDE the training conditions (nested), so it is not fitted on test data."""
    grid = np.logspace(-1, 1, 41) if grid is None else grid
    logits_all, y_all = [], []
    for c in train_conds:
        rest = [k for k in train_conds if k != c]
        tr = np.isin(cond, rest); te = cond == c
        sc, clf = fit_probe(X[tr], y[tr], C)
        _require_multiclass(clf, "calibrate_temperature")
        full = np.full((te.sum(), len(np.unique(y))), -30.0)
        full[:, clf.classes_] = clf.decision_function(sc.transform(X[te]))
        logits_all.append(full); y_all.append(y[te])
    L, Y = np.concatenate(logits_all), np.concatenate(y_all)
    nll = [-np.mean(np.log(_softmax(L / T)[np.arange(len(Y)), Y] + 1e-12)) for T in grid]
    return float(grid[int(np.argmin(nll))])

def fisher_sensitivity(clf, z_std, d_std, T):
    """d^T F d / |d|^2, F = W^T (diag(p) - p p^T) W / T^2 for a softmax probe (a local, probe-defined Fisher form)."""
    _require_multiclass(clf, "fisher_sensitivity")
    W = clf.coef_; p = _softmax((W @ z_std + clf.intercept_) / T)
    M = W.T @ ((np.diag(p) - np.outer(p, p)) @ W) / (T ** 2)
    return float(d_std @ M @ d_std / (d_std @ d_std + 1e-12))

def geometry_pairs(X, product, cond, model_correct, ref_cond=0, C=1.0, temps=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """For each (product, held-out condition c != ref): compare the straight (reference) photo with the photo
    in condition c, under a probe trained on the conditions other than c (which include the reference).
    Returns one dict per pair with distance, alignment, margin drop, and Fisher-style sensitivity."""
    rows = []
    for c in np.unique(cond):
        if c == ref_cond: continue
        tr_conds = [k for k in np.unique(cond) if k != c]
        tr = np.isin(cond, tr_conds)
        sc, clf = fit_probe(X[tr], product[tr], C)
        _require_multiclass(clf, "geometry_pairs")
        T_hat = calibrate_temperature(X, product, cond, tr_conds, C)
        for p_id in np.unique(product):
            i_ref = np.where((product == p_id) & (cond == ref_cond))[0]
            i_c = np.where((product == p_id) & (cond == c))[0]
            if len(i_ref) != 1 or len(i_c) != 1: continue
            i_ref, i_c = i_ref[0], i_c[0]
            z0, z1 = sc.transform(X[[i_ref]])[0], sc.transform(X[[i_c]])[0]
            scores0 = clf.coef_ @ z0 + clf.intercept_
            y_idx = int(np.where(clf.classes_ == p_id)[0][0])
            s = scores0.copy(); s[y_idx] = -np.inf; r = int(np.argmax(s))        # rival fixed at the reference photo
            g = clf.coef_[y_idx] - clf.coef_[r]
            m0 = float(g @ z0 + clf.intercept_[y_idx] - clf.intercept_[r])
            m1 = float(g @ z1 + clf.intercept_[y_idx] - clf.intercept_[r])
            d = z1 - z0
            row = dict(product=int(p_id), cond=int(c), dist=float(np.linalg.norm(d)),
                       align=float((d @ -g) / (np.linalg.norm(d) * np.linalg.norm(g) + 1e-12)),
                       g_norm=float(np.linalg.norm(g)), drop=m0 - m1, margin_ref=m0,
                       model_fail=bool(not model_correct[i_c]), T_hat=T_hat,
                       fisher_That=fisher_sensitivity(clf, z0, d, T_hat))
            for T in temps: row[f"fisher_T{T}"] = fisher_sensitivity(clf, z0, d, T)
            rows.append(row)
    return rows

def geometry_report(rows, n_boot=2000, seed=RNG_SEED):
    A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    prod = A["product"]
    out = {"n_pairs": len(rows), "identity_max_err": float(np.max(np.abs(A["drop"] - A["g_norm"] * A["dist"] * A["align"])))}
    out["spearman_drop_vs_dist"] = float(spearmanr(A["dist"], A["drop"])[0])
    out["spearman_drop_vs_align"] = float(spearmanr(A["align"], A["drop"])[0])
    out["spearman_drop_vs_fisher_That"] = float(spearmanr(A["fisher_That"], A["drop"])[0])
    # cluster bootstrap for: corr(align, drop) - corr(dist, drop)
    rng = np.random.default_rng(seed); ug = np.unique(prod); diffs = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([np.where(prod == g)[0] for g in pick])
        a = spearmanr(A["align"][idx], A["drop"][idx])[0]; d = spearmanr(A["dist"][idx], A["drop"][idx])[0]
        if np.isfinite(a) and np.isfinite(d): diffs.append(a - d)
    out["align_minus_dist_corr_ci"] = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    fail = A["model_fail"]; out["n_model_fail_pairs"] = int(fail.sum())
    if 0 < fail.sum() < len(fail):
        out["auroc_fail_by_dist"] = auroc(A["dist"], fail)
        out["auroc_fail_by_align"] = auroc(A["align"], fail)
        out["auroc_fail_by_fisher_That"] = auroc(A["fisher_That"], fail)
        out["auroc_fail_by_low_margin_ref"] = auroc(-A["margin_ref"], fail)
    # temperature sensitivity of the Fisher-style quantity (the instability seen earlier, now reported explicitly)
    out["fisher_rank_corr_across_temps"] = {k: float(spearmanr(A[k], A["fisher_That"])[0]) for k in A if k.startswith("fisher_T") and k != "fisher_That"}
    return out
