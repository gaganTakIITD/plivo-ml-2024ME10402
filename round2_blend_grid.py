"""Round 2 cheap experiments: blend weight grid + 3-way HGB blend.

Uses features v2 cache. Writes scored results to .cache/round2_results.txt
Does NOT overwrite model.joblib.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eot_dataset import build_matrix
from features_ext import FEATURE_VERSION

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import score as official

DEF_EN = "/mnt/d/pilvo test/eot_handout/eot_data/eot_data/english"
DEF_HI = "/mnt/d/pilvo test/eot_handout/eot_data/eot_data/hindi"
CACHE = os.path.join(ROOT, ".cache")
GATE = 964.0


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


def oof_predict(factory, X, y, groups):
    proba = np.zeros(len(y), dtype=np.float64)
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        m = factory()
        m.fit(X[tr], y[tr])
        proba[va] = m.predict_proba(X[va])[:, 1]
    return proba


def oof_isotonic(factory, X, y, groups):
    proba = np.zeros(len(y), dtype=np.float64)
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
        ps = []
        for base, cal in parts:
            raw = base.predict_proba(X[va])[:, 1]
            ps.append(np.clip(cal.predict(raw), 0, 1))
        proba[va] = np.mean(ps, axis=0)
    return proba


def score_pair(name, keys, n_en, proba, labels_en, labels_hi):
    en = os.path.join(CACHE, f"r2_{name}_en.csv")
    hi = os.path.join(CACHE, f"r2_{name}_hi.csv")
    write_pred_csv(en, keys[:n_en], proba[:n_en])
    write_pred_csv(hi, keys[n_en:], proba[n_en:])
    r_en = official.score(labels_en, en)
    r_hi = official.score(labels_hi, hi)
    mean = (r_en["latency"] + r_hi["latency"]) / 2 * 1000
    line = (f"[{name}] EN {r_en['latency']*1000:.0f} (AUC {r_en['auc']:.3f}) | "
            f"HI {r_hi['latency']*1000:.0f} (AUC {r_hi['auc']:.3f}) | "
            f"mean {mean:.1f}" + (" *** BEATS GATE" if mean < GATE else ""))
    print(line)
    return mean, r_en, r_hi, line


def main():
    os.makedirs(CACHE, exist_ok=True)
    Xe, ye, ke, ge = build_matrix(DEF_EN, cache_dir=CACHE, use_cache=True)
    Xh, yh, kh, gh = build_matrix(DEF_HI, cache_dir=CACHE, use_cache=True)
    n_en = len(ke)
    X = np.vstack([Xe, Xh])
    y = np.concatenate([ye, yh]).astype(int)
    groups = np.concatenate([ge, gh])
    keys = ke + kh
    print(f"v{FEATURE_VERSION} {X.shape}")

    labels_en = os.path.join(DEF_EN, "labels.csv")
    labels_hi = os.path.join(DEF_HI, "labels.csv")
    lines = []

    print("\n=== base OOF probs ===")
    p_et = oof_predict(make_et, X, y, groups)
    score_pair("et_raw", keys, n_en, p_et, labels_en, labels_hi)
    p_lr = oof_predict(make_lr, X, y, groups)
    score_pair("lr_raw", keys, n_en, p_lr, labels_en, labels_hi)
    p_et_iso = oof_isotonic(make_et, X, y, groups)
    score_pair("et_iso", keys, n_en, p_et_iso, labels_en, labels_hi)
    p_hgb = oof_predict(make_hgb, X, y, groups)
    score_pair("hgb_raw", keys, n_en, p_hgb, labels_en, labels_hi)
    p_hgb_iso = oof_isotonic(make_hgb, X, y, groups)
    score_pair("hgb_iso", keys, n_en, p_hgb_iso, labels_en, labels_hi)

    print("\n=== ET-iso / LR weight grid ===")
    best = (1e9, None, None)
    for w10 in range(0, 11):
        w = w10 / 10.0
        p = w * p_et_iso + (1 - w) * p_lr
        mean, r_en, r_hi, line = score_pair(
            f"blend_w{w10}", keys, n_en, p, labels_en, labels_hi)
        lines.append(line)
        if mean < best[0]:
            best = (mean, f"blend_w{w10}", (r_en, r_hi, p, w))

    print("\n=== 3-way blends (et_iso, lr, hgb_iso) ===")
    # simplex grid with step 0.2
    for a10 in range(0, 11, 2):
        for b10 in range(0, 11 - a10, 2):
            c10 = 10 - a10 - b10
            a, b, c = a10 / 10.0, b10 / 10.0, c10 / 10.0
            p = a * p_et_iso + b * p_lr + c * p_hgb_iso
            tag = f"3way_e{a10}_l{b10}_h{c10}"
            mean, r_en, r_hi, line = score_pair(
                tag, keys, n_en, p, labels_en, labels_hi)
            lines.append(line)
            if mean < best[0]:
                best = (mean, tag, (r_en, r_hi, p, (a, b, c)))

    # also try raw et + lr grid (no iso)
    print("\n=== ET-raw / LR weight grid ===")
    for w10 in range(0, 11):
        w = w10 / 10.0
        p = w * p_et + (1 - w) * p_lr
        mean, r_en, r_hi, line = score_pair(
            f"rawblend_w{w10}", keys, n_en, p, labels_en, labels_hi)
        lines.append(line)
        if mean < best[0]:
            best = (mean, f"rawblend_w{w10}", (r_en, r_hi, p, w))

    print(f"\nBEST so far: {best[1]} mean {best[0]:.1f} (gate {GATE})")
    np.savez(os.path.join(CACHE, "round2_probs.npz"),
             p_et=p_et, p_lr=p_lr, p_et_iso=p_et_iso,
             p_hgb=p_hgb, p_hgb_iso=p_hgb_iso, y=y, groups=groups)
    with open(os.path.join(CACHE, "round2_blend_lines.txt"), "w") as fh:
        fh.write("\n".join(lines) + f"\nBEST={best[1]} {best[0]:.1f}\n")


if __name__ == "__main__":
    main()
