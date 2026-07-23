"""Hindi-weighted ExtraTrees (+ iso/LR blend) OOF experiments."""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eot_dataset import build_matrix

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import score as official

DEF_EN = "/mnt/d/pilvo test/eot_handout/eot_data/eot_data/english"
DEF_HI = "/mnt/d/pilvo test/eot_handout/eot_data/eot_data/hindi"
CACHE = os.path.join(ROOT, ".cache")
GATE = 964.0
BASE_HI = 850.0  # current shipped HI; do not regress


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


def oof_et_weighted(X, y, groups, sw):
    """ExtraTrees with sample_weight; isotonic calib on unweighted holdout."""
    proba = np.zeros(len(y), dtype=np.float64)
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        Xt, yt, gt, st = X[tr], y[tr], groups[tr], sw[tr]
        parts = []
        for fi, ci in GroupKFold(n_splits=3).split(Xt, yt, gt):
            # fit ET with weights via underlying classifier after impute
            imp = SimpleImputer(strategy="median", keep_empty_features=True)
            Xf = imp.fit_transform(Xt[fi])
            clf = ExtraTreesClassifier(
                n_estimators=500, max_depth=7, min_samples_leaf=4,
                class_weight="balanced", max_features="sqrt",
                random_state=0, n_jobs=-1)
            clf.fit(Xf, yt[fi], sample_weight=st[fi])
            Xc = imp.transform(Xt[ci])
            raw = clf.predict_proba(Xc)[:, 1]
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(raw, yt[ci])
            parts.append((imp, clf, cal))
        ps = []
        for imp, clf, cal in parts:
            Xv = imp.transform(X[va])
            raw = clf.predict_proba(Xv)[:, 1]
            ps.append(np.clip(cal.predict(raw), 0, 1))
        proba[va] = np.mean(ps, axis=0)
    return proba


def oof_lr_weighted(X, y, groups, sw):
    proba = np.zeros(len(y), dtype=np.float64)
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        sc = StandardScaler()
        Xtr = sc.fit_transform(imp.fit_transform(X[tr]))
        clf = LogisticRegression(class_weight="balanced", C=1.0,
                                 max_iter=4000, random_state=0)
        clf.fit(Xtr, y[tr], sample_weight=sw[tr])
        Xva = sc.transform(imp.transform(X[va]))
        proba[va] = clf.predict_proba(Xva)[:, 1]
    return proba


def score(name, keys, n_en, proba, labels_en, labels_hi):
    en = os.path.join(CACHE, f"r2_{name}_en.csv")
    hi = os.path.join(CACHE, f"r2_{name}_hi.csv")
    write_pred_csv(en, keys[:n_en], proba[:n_en])
    write_pred_csv(hi, keys[n_en:], proba[n_en:])
    r_en = official.score(labels_en, en)
    r_hi = official.score(labels_hi, hi)
    mean = (r_en["latency"] + r_hi["latency"]) / 2 * 1000
    hi_ms = r_hi["latency"] * 1000
    ok_hi = hi_ms <= BASE_HI + 1e-6
    flag = ""
    if mean < GATE and ok_hi:
        flag = " *** CANDIDATE"
    elif mean < GATE and not ok_hi:
        flag = " (mean ok but HI regress)"
    print(f"[{name}] EN {r_en['latency']*1000:.0f} HI {hi_ms:.0f} "
          f"mean {mean:.1f}{flag}")
    return mean, r_en, r_hi


def main():
    Xe, ye, ke, ge = build_matrix(DEF_EN, cache_dir=CACHE, use_cache=True)
    Xh, yh, kh, gh = build_matrix(DEF_HI, cache_dir=CACHE, use_cache=True)
    n_en = len(ke)
    X = np.vstack([Xe, Xh])
    y = np.concatenate([ye, yh]).astype(int)
    groups = np.concatenate([ge, gh])
    keys = ke + kh
    is_hi = np.concatenate([np.zeros(n_en), np.ones(len(kh))])
    labels_en = os.path.join(DEF_EN, "labels.csv")
    labels_hi = os.path.join(DEF_HI, "labels.csv")

    # load unweighted lr for blending option
    z = np.load(os.path.join(CACHE, "round2_probs.npz"))
    p_lr0 = z["p_lr"]

    for hw in (1.5, 2.0, 3.0):
        sw = np.where(is_hi == 1, hw, 1.0)
        print(f"\n=== hindi weight {hw}x ===")
        p_iso = oof_et_weighted(X, y, groups, sw)
        score(f"hiw{hw}_etiso", keys, n_en, p_iso, labels_en, labels_hi)
        p_lr = oof_lr_weighted(X, y, groups, sw)
        score(f"hiw{hw}_lr", keys, n_en, p_lr, labels_en, labels_hi)
        for w in (0.5, 0.7, 0.4):
            p = w * p_iso + (1 - w) * p_lr
            score(f"hiw{hw}_blend{int(w*10)}", keys, n_en, p,
                  labels_en, labels_hi)
            # blend with unweighted lr too
            p2 = w * p_iso + (1 - w) * p_lr0
            score(f"hiw{hw}_etiso_lr0_w{int(w*10)}", keys, n_en, p2,
                  labels_en, labels_hi)


if __name__ == "__main__":
    main()
