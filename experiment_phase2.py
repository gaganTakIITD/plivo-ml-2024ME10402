"""Phase 2 experiments: ExtraTrees OOF, calibration, LogReg blend.

Writes OOF CSVs under .cache/ and prints official score.py results.
Does NOT overwrite model.joblib.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
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


def write_pred_csv(path, keys, probs):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), p in zip(keys, probs):
            w.writerow([tid, pi, f"{p:.6f}"])


def make_et():
    # Match shipped model.joblib: bare ExtraTrees after median impute.
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("clf", ExtraTreesClassifier(
            n_estimators=500, max_depth=7, min_samples_leaf=4,
            class_weight="balanced", max_features="sqrt",
            random_state=0, n_jobs=-1)),
    ])


def make_et_bare():
    return ExtraTreesClassifier(
        n_estimators=500, max_depth=7, min_samples_leaf=4,
        class_weight="balanced", max_features="sqrt",
        random_state=0, n_jobs=-1)


def make_lr():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", C=1.0,
                                   max_iter=4000, random_state=0)),
    ])


def oof_predict(est, X, y, groups):
    """Manual GroupKFold OOF probabilities."""
    cv = GroupKFold(n_splits=5)
    proba = np.zeros(len(y), dtype=np.float64)
    for tr, va in cv.split(X, y, groups):
        m = clone(est)
        m.fit(X[tr], y[tr])
        proba[va] = m.predict_proba(X[va])[:, 1]
    return proba


def oof_et_preimputed(X, y, groups):
    """Global-median fill (matches likely ship path) then bare ExtraTrees OOF."""
    col_med = np.nanmedian(X, axis=0)
    Xf = np.where(np.isnan(X), col_med, X)
    return oof_predict(make_et_bare(), Xf, y, groups)


def oof_calibrated(base_factory, X, y, groups, method="isotonic"):
    """For each outer fold: fit base on ~2/3 of train groups, calibrate on
    remaining train groups (isotonic or Platt). Never touches outer val.
    """
    cv_outer = GroupKFold(n_splits=5)
    proba = np.zeros(len(y), dtype=np.float64)
    for tr, va in cv_outer.split(X, y, groups):
        cal = _manual_calibrate(base_factory, X[tr], y[tr], groups[tr],
                                method=method)
        proba[va] = cal.predict_proba(X[va])[:, 1]
    return proba


def _manual_calibrate(base_factory, X, y, groups, method="isotonic"):
    """Average 3 calibrated models (GroupKFold on train), like CalibratedCV."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression as LR

    gkf = GroupKFold(n_splits=3)
    models = []
    for fit_idx, cal_idx in gkf.split(X, y, groups):
        base = base_factory()
        base.fit(X[fit_idx], y[fit_idx])
        raw = base.predict_proba(X[cal_idx])[:, 1]
        if method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(raw, y[cal_idx])
        else:
            cal = LR(max_iter=1000)
            cal.fit(raw.reshape(-1, 1), y[cal_idx])
        models.append((base, cal, method))

    class _Wrap:
        def predict_proba(self, Xq):
            ps = []
            for base, cal, meth in models:
                raw = base.predict_proba(Xq)[:, 1]
                if meth == "isotonic":
                    c = np.clip(cal.predict(raw), 0, 1)
                else:
                    c = cal.predict_proba(raw.reshape(-1, 1))[:, 1]
                ps.append(c)
            c = np.mean(ps, axis=0)
            return np.column_stack([1 - c, c])
    return _Wrap()


def evaluate(name, keys, n_en, proba, labels_en, labels_hi):
    oof_en = os.path.join(CACHE, f"oof_{name}_en.csv")
    oof_hi = os.path.join(CACHE, f"oof_{name}_hi.csv")
    write_pred_csv(oof_en, keys[:n_en], proba[:n_en])
    write_pred_csv(oof_hi, keys[n_en:], proba[n_en:])
    r_en = official.score(labels_en, oof_en)
    r_hi = official.score(labels_hi, oof_hi)
    mean_ms = (r_en["latency"] + r_hi["latency"]) / 2 * 1000
    print(f"[{name}] OOF  EN {r_en['latency']*1000:.0f} ms "
          f"(AUC {r_en['auc']:.3f}, cut {r_en['cutoff']*100:.1f}%, "
          f"t={r_en['threshold']}, d={r_en['delay']*1000:.0f} ms) | "
          f"HI {r_hi['latency']*1000:.0f} ms (AUC {r_hi['auc']:.3f}, "
          f"cut {r_hi['cutoff']*100:.1f}%, t={r_hi['threshold']}, "
          f"d={r_hi['delay']*1000:.0f} ms) | mean {mean_ms:.1f} ms")
    return mean_ms, r_en, r_hi


def main():
    os.makedirs(CACHE, exist_ok=True)
    Xe, ye, ke, ge = build_matrix(DEF_EN, cache_dir=CACHE, use_cache=True)
    Xh, yh, kh, gh = build_matrix(DEF_HI, cache_dir=CACHE, use_cache=True)
    n_en = len(ke)
    X = np.vstack([Xe, Xh])
    y = np.concatenate([ye, yh]).astype(int)
    groups = np.concatenate([ge, gh])
    keys = ke + kh
    print(f"features v{FEATURE_VERSION}: {X.shape[0]} pauses x {X.shape[1]} "
          f"(en {n_en}, hi {len(kh)})")

    labels_en = os.path.join(DEF_EN, "labels.csv")
    labels_hi = os.path.join(DEF_HI, "labels.csv")

    # 1) ExtraTrees baselines (pipeline impute vs global-median like ship)
    print("\n--- ExtraTrees (pipeline impute) ---")
    p_et = oof_predict(make_et(), X, y, groups)
    evaluate("et", keys, n_en, p_et, labels_en, labels_hi)

    print("\n--- ExtraTrees (global median + bare) ---")
    p_et_g = oof_et_preimputed(X, y, groups)
    evaluate("et_globalimp", keys, n_en, p_et_g, labels_en, labels_hi)

    # Prefer the better ET parent for blends
    if official.score(labels_en, os.path.join(CACHE, "oof_et_globalimp_en.csv"))["latency"] <= \
       official.score(labels_en, os.path.join(CACHE, "oof_et_en.csv"))["latency"]:
        p_et_best = p_et_g
        print("Using et_globalimp as ET parent for blends")
    else:
        p_et_best = p_et
        print("Using et (pipeline) as ET parent for blends")

    # 2) LogReg
    print("\n--- LogReg ---")
    p_lr = oof_predict(make_lr(), X, y, groups)
    evaluate("lr", keys, n_en, p_lr, labels_en, labels_hi)

    # 3) Calibrated ExtraTrees (isotonic)
    print("\n--- ExtraTrees + isotonic calib ---")
    p_et_iso = oof_calibrated(make_et, X, y, groups, method="isotonic")
    evaluate("et_iso", keys, n_en, p_et_iso, labels_en, labels_hi)

    # 4) Calibrated ExtraTrees (sigmoid/Platt)
    print("\n--- ExtraTrees + sigmoid calib ---")
    p_et_sig = oof_calibrated(make_et, X, y, groups, method="sigmoid")
    evaluate("et_sig", keys, n_en, p_et_sig, labels_en, labels_hi)

    # 5) Soft blends of best ET + LR
    print("\n--- Blends ---")
    for w, tag in ((0.5, "blend50"), (0.7, "blend70et"), (0.3, "blend30et"),
                   (0.6, "blend60et"), (0.4, "blend40et")):
        p = w * p_et_best + (1 - w) * p_lr
        evaluate(tag, keys, n_en, p, labels_en, labels_hi)

    # 6) Blend calibrated ET iso + LR
    for w, tag in ((0.5, "blend_iso50"), (0.7, "blend_iso70et")):
        p = w * p_et_iso + (1 - w) * p_lr
        evaluate(tag, keys, n_en, p, labels_en, labels_hi)

    # save best raw arrays for inspection
    np.savez(os.path.join(CACHE, "oof_phase2.npz"),
             p_et=p_et, p_et_g=p_et_g, p_lr=p_lr,
             p_et_iso=p_et_iso, p_et_sig=p_et_sig,
             y=y, groups=groups)
    print("\nSaved .cache/oof_phase2.npz")


if __name__ == "__main__":
    main()
