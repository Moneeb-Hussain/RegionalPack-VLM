"""Pytest checks for the linear probe. These are functions, so pytest collects them."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import probe_analysis as pa


def _toy(n_classes, n_cond=3, d=32, seed=1, noise=0.4):
    rng = np.random.default_rng(seed)
    rows, labels, conds = [], [], []
    centers = rng.normal(size=(n_classes, d)) * 3
    for c in range(n_cond):
        shift = rng.normal(size=d) * 0.2
        for k in range(n_classes):
            rows.append(centers[k] + shift + rng.normal(size=d) * noise)
            labels.append(k)
            conds.append(c)
    return np.array(rows), np.array(labels), np.array(conds)


def test_two_class_toy_predicts_class_1():
    X, y, _cond = _toy(2, d=8, noise=0.05)
    sc, clf = pa.fit_probe(X, y)
    pred = clf.predict(sc.transform(X))
    assert set(pred.tolist()) == {0, 1}
    assert int((pred == 1).sum()) == int((y == 1).sum())
    assert np.array_equal(pred, y)


def test_twenty_class_toy_is_decodable():
    rng = np.random.default_rng(1)
    n_classes, n_cond, d = 20, 3, 128
    ident = rng.normal(size=(n_classes, d))
    cond_shift = rng.normal(size=(n_cond, d)) * 0.8
    rows, labels, conds = [], [], []
    for k in range(n_classes):
        for c in range(n_cond):
            rows.append(ident[k] + cond_shift[c] + rng.normal(size=d) * 1.5)
            labels.append(k)
            conds.append(c)
    X = np.array(rows)
    y = np.array(labels)
    cond = np.array(conds)
    model_correct = rng.random(len(y)) > 0.35
    primary = y < 18
    rep = pa.decodability_report(X, y, cond, model_correct, primary, n_perm=40)
    assert rep["acc"] > 0.8
    assert rep["null_mean"] < 0.2
    assert rep["p_overall"] < 0.05
    rows_g = pa.geometry_pairs(X, y, cond, model_correct)
    report = pa.geometry_report(rows_g, n_boot=200)
    assert report["n_pairs"] == 40
    assert report["identity_max_err"] < 1e-6
    noise = pa.decodability_report(rng.normal(size=X.shape), y, cond, model_correct, primary, n_perm=40)
    assert noise["p_overall"] > 0.05


def test_reduced_fit_matches_direct_sklearn(monkeypatch):
    # Default lbfgs tol stops early, so coefficients can differ by ~1e-3
    # even though both fits are the same problem. Tighten both solvers.
    real = pa.LogisticRegression

    def tight_lr(*args, **kwargs):
        kwargs["tol"] = 1e-14
        kwargs["max_iter"] = 20000
        return real(*args, **kwargs)

    monkeypatch.setattr(pa, "LogisticRegression", tight_lr)
    # 20 classes in 200 dimensions is the wide case the reduced fit is for.
    for n_classes, d in ((2, 40), (20, 200)):
        X, y, _cond = _toy(n_classes, d=d, noise=0.3, seed=2)
        sc, clf = pa.fit_probe(X, y, C=1.0)
        Z = sc.transform(X)
        direct = real(C=1.0, max_iter=20000, tol=1e-14).fit(Z, y)
        extra = Z + np.random.default_rng(0).normal(scale=0.1, size=Z.shape)
        assert np.array_equal(clf.predict(Z), direct.predict(Z))
        assert np.array_equal(clf.predict(extra), direct.predict(extra))
        np.testing.assert_allclose(clf.coef_, direct.coef_, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(clf.intercept_, direct.intercept_, rtol=1e-8, atol=1e-8)


def test_permutation_null_shuffles_train_labels_only(monkeypatch):
    X, y, cond = _toy(4, d=6, noise=0.05, seed=3)
    y_over = y.copy()
    for c in np.unique(cond):
        idx = np.where(cond == c)[0]
        y_over[idx] = y[idx][::-1]
    assert not np.array_equal(y_over, y)
    seen = []
    real_fit = pa.fit_probe

    def spy(Xtr, ytr, C=1.0):
        seen.append(np.array(ytr, copy=True))
        return real_fit(Xtr, ytr, C)

    monkeypatch.setattr(pa, "fit_probe", spy)
    pred = pa.loco_predict(X, y, cond, y_train_override=y_over)
    assert len(seen) == 3
    for c, ytr in zip(np.unique(cond), seen):
        train = cond != c
        assert np.array_equal(ytr, y_over[train])
        assert not np.array_equal(ytr, y[train])
    # Held-out photos are scored against the true labels, not the shuffled ones.
    obs, null = pa.loco_accuracy_with_null(X, y, cond, n_perm=1, seed=0)
    assert obs.shape == y.shape
    assert np.array_equal(obs, pa.loco_predict(X, y, cond) == y)
    assert null.shape == (1, len(y))


def test_scaler_is_fit_on_training_rows_only(monkeypatch):
    n_per = 5
    cond = np.repeat([0, 1, 2], n_per)
    X = np.zeros((len(cond), 2))
    X[:, 0] = cond.astype(float) * 100
    y = np.tile(np.arange(n_per), 3)
    seen = []
    real = pa.StandardScaler

    class Capture(real):
        def fit(self, data, y=None):
            seen.append(np.array(data, copy=True))
            return super().fit(data, y)

    monkeypatch.setattr(pa, "StandardScaler", Capture)
    pa.loco_predict(X, y, cond)
    assert len(seen) == 3
    for c, block in zip(np.unique(cond), seen):
        train = cond != c
        assert block.shape[0] == int(train.sum())
        assert not np.any(np.isclose(block[:, 0], float(c) * 100))
        assert np.allclose(block.mean(axis=0), X[train].mean(axis=0))


def test_two_class_geometry_raises():
    X, y, cond = _toy(2, d=4, noise=0.1)
    with pytest.raises(ValueError, match="two-class"):
        pa.geometry_pairs(X, y, cond, np.ones(len(y), dtype=bool))


def test_held_out_condition_never_enters_scaler_probe_or_temperature(monkeypatch):
    """Each fit may use only the rows of the conditions it is allowed to train on.

    The temperature is chosen by fits inside those training conditions.
    The straight reference photo is allowed in that training set; the held-out
    condition is not.
    """
    X, y, cond = _toy(4, n_cond=3, d=6, noise=0.2, seed=5)
    X = np.column_stack([np.arange(len(y), dtype=float), X])
    seen = []
    real_fit = pa.fit_probe
    real_scaler = pa.StandardScaler

    class Capture(real_scaler):
        def fit(self, data, y=None):
            seen.append(("scaler", data[:, 0].astype(int).copy()))
            return super().fit(data, y)

    def spy(Xtr, ytr, C=1.0):
        seen.append(("probe", Xtr[:, 0].astype(int).copy()))
        return real_fit(Xtr, ytr, C)

    monkeypatch.setattr(pa, "StandardScaler", Capture)
    monkeypatch.setattr(pa, "fit_probe", spy)

    def cond_sets(kind):
        out = []
        for name, rows in seen:
            if name != kind:
                continue
            present = sorted({int(cond[i]) for i in rows})
            expected = set(np.where(np.isin(cond, present))[0].tolist())
            assert set(rows.tolist()) == expected
            out.append(tuple(present))
        return out

    pa.loco_predict(X, y, cond)
    assert cond_sets("probe") == [(1, 2), (0, 2), (0, 1)]
    assert cond_sets("scaler") == cond_sets("probe")

    seen.clear()
    pa.geometry_pairs(X, y, cond, np.ones(len(y), dtype=bool))
    # Held out tilt, then occluded. Temperature fits use one training condition at a time.
    assert cond_sets("probe") == [(0, 2), (2,), (0,), (0, 1), (1,), (0,)]
    assert cond_sets("scaler") == cond_sets("probe")
    probe_rows = [rows for name, rows in seen if name == "probe"]
    for rows in probe_rows[:3]:
        assert not np.any(cond[rows] == 1)
    for rows in probe_rows[3:]:
        assert not np.any(cond[rows] == 2)
