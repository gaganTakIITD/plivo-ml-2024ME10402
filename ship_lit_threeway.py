"""Ship lit-features v4 ThreeWayBlend (0.3 iso-ET + 0.2 LR + 0.5 iso-HGB).

Recomputes honest OOF; ships only if mean < 964.
"""
from __future__ import annotations

import csv
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from blend_model import ThreeWayBlend
from eot_dataset import build_matrix
from features_ext import FEATURE_NAMES, FEATURE_VERSION

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import score as official

DEF_EN = "/mnt/d/pilvo test/eot_handout/eot_data/eot_data/english"
DEF_HI = "/mnt/d/pilvo test/eot_handout/eot_data/eot_data/hindi"
CACHE = os.path.join(ROOT, ".cache")
GATE = 964.0
W_ET, W_LR, W_HGB = 0.30, 0.20, 0.50


def write_pred_csv(path, keys, probs):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), p in zip(keys, probs):
            w.writerow([tid, pi, f"{p:.6f}"])


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


def oof_raw(factory, X, y, groups):
    p = np.zeros(len(y), dtype=np.float64)
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        m = factory()
        m.fit(X[tr], y[tr])
        p[va] = m.predict_proba(X[va])[:, 1]
    return p


def oof_iso(factory, X, y, groups):
    p = np.zeros(len(y), dtype=np.float64)
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        Xt, yt, gt = X[tr], y[tr], groups[tr]
        parts = []
        for fi, ci in GroupKFold(n_splits=3).split(Xt, yt, gt):
            base = factory()
            base.fit(Xt[fi], yt[fi])
            raw = base.predict_proba(Xt[ci])[:, 1]
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(raw, yt[ci])
            parts.append((base, cal))
        ps = [np.clip(cal.predict(base.predict_proba(X[va])[:, 1]), 0, 1)
              for base, cal in parts]
        p[va] = np.mean(ps, axis=0)
    return p


def main():
    assert FEATURE_VERSION == 4, FEATURE_VERSION
    Xe, ye, ke, ge = build_matrix(DEF_EN, cache_dir=CACHE, use_cache=True)
    Xh, yh, kh, gh = build_matrix(DEF_HI, cache_dir=CACHE, use_cache=True)
    assert Xe.shape[1] == len(FEATURE_NAMES) == 60
    X = np.vstack([Xe, Xh])
    y = np.concatenate([ye, yh]).astype(int)
    groups = np.concatenate([ge, gh])
    keys = ke + kh
    n_en = len(ke)
    print(f"v4 {X.shape}")

    print("OOF iso-ET / LR / iso-HGB...")
    p_et = oof_iso(make_et, X, y, groups)
    p_lr = oof_raw(make_lr, X, y, groups)
    p_hgb = oof_iso(make_hgb, X, y, groups)
    p = W_ET * p_et + W_LR * p_lr + W_HGB * p_hgb

    labels_en = os.path.join(DEF_EN, "labels.csv")
    labels_hi = os.path.join(DEF_HI, "labels.csv")
    oof_en = os.path.join(ROOT, "oof_english.csv")
    oof_hi = os.path.join(ROOT, "oof_hindi.csv")
    write_pred_csv(oof_en, keys[:n_en], p[:n_en])
    write_pred_csv(oof_hi, keys[n_en:], p[n_en:])
    r_en = official.score(labels_en, oof_en)
    r_hi = official.score(labels_hi, oof_hi)
    mean = (r_en["latency"] + r_hi["latency"]) / 2 * 1000
    print(f"OOF EN {r_en['latency']*1000:.1f}/{r_en['auc']:.3f} "
          f"HI {r_hi['latency']*1000:.1f}/{r_hi['auc']:.3f} mean {mean:.1f}")
    if mean >= GATE:
        print("NO SHIP")
        sys.exit(2)

    model = ThreeWayBlend(w_et=W_ET, w_lr=W_LR, w_hgb=W_HGB)
    model.fit(X, y, groups=groups)
    joblib.dump({
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "feature_version": FEATURE_VERSION,
        "model_name": "threeway_et30_lr20_hgb50_v4lit",
        "oof_en_ms": r_en["latency"] * 1000,
        "oof_hi_ms": r_hi["latency"] * 1000,
        "oof_en_auc": r_en["auc"],
        "oof_hi_auc": r_hi["auc"],
        "oof_mean_ms": mean,
        "blend_weights": {"et": W_ET, "lr": W_LR, "hgb": W_HGB},
    }, os.path.join(ROOT, "model.joblib"))
    print(f"SHIPPED v4 ThreeWayBlend mean {mean:.1f}")


if __name__ == "__main__":
    main()
