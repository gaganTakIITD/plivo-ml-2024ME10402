"""Deployable blends for EOT: 2-way iso-ET+LR and 3-way + iso-HGB."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
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


def make_hgb():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=3, min_samples_leaf=15,
        l2_regularization=1.0, random_state=0)


def _fit_iso_ensemble(factory, X, y, groups=None, n_splits=3):
    models = []
    if groups is None:
        n = len(y)
        idx = np.arange(n)
        rng = np.random.RandomState(0)
        rng.shuffle(idx)
        folds = np.array_split(idx, n_splits)
        splits = []
        for i in range(n_splits):
            cal_idx = folds[i]
            fit_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
            splits.append((fit_idx, cal_idx))
    else:
        splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    for fi, ci in splits:
        base = factory()
        base.fit(X[fi], y[fi])
        raw = base.predict_proba(X[ci])[:, 1]
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(raw, y[ci])
        models.append((base, cal))
    return models


def _predict_iso(models, X):
    ps = []
    for base, cal in models:
        raw = base.predict_proba(X)[:, 1]
        ps.append(np.clip(cal.predict(raw), 0, 1))
    return np.mean(ps, axis=0)


class IsoETBlend(BaseEstimator, ClassifierMixin):
    """Legacy 2-way: w_et * iso-ET + (1-w_et) * LR."""

    def __init__(self, w_et=0.7):
        self.w_et = w_et

    def fit(self, X, y, groups=None):
        self.lr_ = make_lr()
        self.lr_.fit(X, y)
        self.cal_models_ = _fit_iso_ensemble(make_et, X, y, groups)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p_et = _predict_iso(self.cal_models_, X)
        p_lr = self.lr_.predict_proba(X)[:, 1]
        p = np.clip(self.w_et * p_et + (1.0 - self.w_et) * p_lr, 0, 1)
        return np.column_stack([1 - p, p])


class ThreeWayBlend(BaseEstimator, ClassifierMixin):
    """a * iso-ET + b * LR + c * iso-HGB (a+b+c should be 1)."""

    def __init__(self, w_et=0.5, w_lr=0.2, w_hgb=0.3):
        self.w_et = w_et
        self.w_lr = w_lr
        self.w_hgb = w_hgb

    def fit(self, X, y, groups=None):
        self.lr_ = make_lr()
        self.lr_.fit(X, y)
        self.et_models_ = _fit_iso_ensemble(make_et, X, y, groups)
        self.hgb_models_ = _fit_iso_ensemble(make_hgb, X, y, groups)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p_et = _predict_iso(self.et_models_, X)
        p_lr = self.lr_.predict_proba(X)[:, 1]
        p_hgb = _predict_iso(self.hgb_models_, X)
        p = (self.w_et * p_et + self.w_lr * p_lr + self.w_hgb * p_hgb)
        p = np.clip(p, 0, 1)
        return np.column_stack([1 - p, p])
