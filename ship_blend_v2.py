"""Ship calibrated ExtraTrees + LogReg blend on features v2 if OOF < gate.

Recipe: OOF p = 0.7 * isotonic-calibrated-ET + 0.3 * LogReg
(nested GroupKFold: outer 5 for OOF, inner 3 for calibration on train).
Final artifact: IsoETBlend from blend_model.py.
"""
from __future__ import annotations

import csv
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from blend_model import IsoETBlend
from eot_dataset import build_matrix
from features_ext import FEATURE_NAMES, FEATURE_VERSION

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import score as official

DEF_EN = "/mnt/d/pilvo test/eot_handout/eot_data/eot_data/english"
DEF_HI = "/mnt/d/pilvo test/eot_handout/eot_data/eot_data/hindi"
CACHE = os.path.join(ROOT, ".cache")
GATE_MS = 987.5
W_ET = 0.7


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


def oof_lr(X, y, groups):
    proba = np.zeros(len(y), dtype=np.float64)
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        m = make_lr()
        m.fit(X[tr], y[tr])
        proba[va] = m.predict_proba(X[va])[:, 1]
    return proba


def oof_et_iso(X, y, groups):
    proba = np.zeros(len(y), dtype=np.float64)
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        Xt, yt, gt = X[tr], y[tr], groups[tr]
        parts = []
        for fi, ci in GroupKFold(n_splits=3).split(Xt, yt, gt):
            base = make_et()
            base.fit(Xt[fi], yt[fi])
            raw = base.predict_proba(Xt[ci])[:, 1]
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(raw, yt[ci])
            parts.append((base, cal))
        ps = []
        for base, cal in parts:
            raw = base.predict_proba(X[va])[:, 1]
            ps.append(np.clip(cal.predict(raw), 0, 1))
        proba[va] = np.mean(ps, axis=0)
    return proba


def main():
    assert FEATURE_VERSION == 2, f"expected v2 features, got v{FEATURE_VERSION}"
    os.makedirs(CACHE, exist_ok=True)
    Xe, ye, ke, ge = build_matrix(DEF_EN, cache_dir=CACHE, use_cache=True)
    Xh, yh, kh, gh = build_matrix(DEF_HI, cache_dir=CACHE, use_cache=True)
    n_en = len(ke)
    X = np.vstack([Xe, Xh])
    y = np.concatenate([ye, yh]).astype(int)
    groups = np.concatenate([ge, gh])
    keys = ke + kh
    print(f"features v{FEATURE_VERSION}: {X.shape}")

    labels_en = os.path.join(DEF_EN, "labels.csv")
    labels_hi = os.path.join(DEF_HI, "labels.csv")

    print("Computing OOF isotonic-ET + LR blend...")
    p_iso = oof_et_iso(X, y, groups)
    p_lr = oof_lr(X, y, groups)
    p = W_ET * p_iso + (1.0 - W_ET) * p_lr

    oof_en = os.path.join(ROOT, "oof_english.csv")
    oof_hi = os.path.join(ROOT, "oof_hindi.csv")
    write_pred_csv(oof_en, keys[:n_en], p[:n_en])
    write_pred_csv(oof_hi, keys[n_en:], p[n_en:])
    r_en = official.score(labels_en, oof_en)
    r_hi = official.score(labels_hi, oof_hi)
    mean_ms = (r_en["latency"] + r_hi["latency"]) / 2 * 1000
    print(f"OOF EN {r_en['latency']*1000:.1f} (AUC {r_en['auc']:.3f}) | "
          f"HI {r_hi['latency']*1000:.1f} (AUC {r_hi['auc']:.3f}) | "
          f"mean {mean_ms:.1f}")

    if mean_ms >= GATE_MS:
        print(f"NO SHIP: {mean_ms:.1f} >= {GATE_MS}")
        sys.exit(2)

    model = IsoETBlend(w_et=W_ET)
    model.fit(X, y, groups=groups)
    joblib.dump({
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "feature_version": FEATURE_VERSION,
        "model_name": "iso_et_lr_blend70_v2",
        "oof_en_ms": r_en["latency"] * 1000,
        "oof_hi_ms": r_hi["latency"] * 1000,
        "oof_en_auc": r_en["auc"],
        "oof_hi_auc": r_hi["auc"],
        "oof_mean_ms": mean_ms,
    }, os.path.join(ROOT, "model.joblib"))
    print(f"SHIPPED model.joblib mean OOF {mean_ms:.1f} ms")


if __name__ == "__main__":
    main()
