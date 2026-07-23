"""Deployable soft blend: isotonic-calibrated ExtraTrees + LogReg."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_et():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("clf", ExtraTreesClassifier(
            n_estimators=500, max_depth=7, min_samples_leaf=4,
            class_weight="balanced", max_features="sqrt",
            random_state=0, n_jobs=-1)),
    ])


def make_lr():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", C=1.0,
                                   max_iter=4000, random_state=0)),
    ])


class IsoETBlend(BaseEstimator, ClassifierMixin):
    """0.7 * mean(isotonic-calibrated ExtraTrees) + 0.3 * LogReg."""

    def __init__(self, w_et=0.7):
        self.w_et = w_et

    def fit(self, X, y, groups=None):
        self.lr_ = make_lr()
        self.lr_.fit(X, y)
        self.cal_models_ = []
        if groups is None:
            n = len(y)
            idx = np.arange(n)
            rng = np.random.RandomState(0)
            rng.shuffle(idx)
            folds = np.array_split(idx, 3)
            splits = []
            for i in range(3):
                cal_idx = folds[i]
                fit_idx = np.concatenate([folds[j] for j in range(3) if j != i])
                splits.append((fit_idx, cal_idx))
        else:
            splits = list(GroupKFold(n_splits=3).split(X, y, groups))
        for fi, ci in splits:
            base = make_et()
            base.fit(X[fi], y[fi])
            raw = base.predict_proba(X[ci])[:, 1]
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(raw, y[ci])
            self.cal_models_.append((base, cal))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        ps = []
        for base, cal in self.cal_models_:
            raw = base.predict_proba(X)[:, 1]
            ps.append(np.clip(cal.predict(raw), 0, 1))
        p_et = np.mean(ps, axis=0)
        p_lr = self.lr_.predict_proba(X)[:, 1]
        p = np.clip(self.w_et * p_et + (1.0 - self.w_et) * p_lr, 0, 1)
        return np.column_stack([1 - p, p])
