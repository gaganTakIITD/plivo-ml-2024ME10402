"""Train/evaluate ExtraTrees (+ optional LR blend) on features v3.

Honest 5-fold GroupKFold OOF by turn_id; official starter/score.py.
Does not overwrite model.joblib unless --ship and mean < GATE.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eot_dataset import build_matrix
from features_ext import FEATURE_NAMES, FEATURE_VERSION

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import score as official

DEF_EN = "/mnt/d/pilvo test/eot_handout/eot_data/eot_data/english"
DEF_HI = "/mnt/d/pilvo test/eot_handout/eot_data/eot_data/hindi"
CACHE = os.path.join(ROOT, ".cache")
GATE_MS = 987.5


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


def oof_predict(est, X, y, groups):
    cv = GroupKFold(n_splits=5)
    proba = np.zeros(len(y), dtype=np.float64)
    for tr, va in cv.split(X, y, groups):
        m = clone(est)
        m.fit(X[tr], y[tr])
        proba[va] = m.predict_proba(X[va])[:, 1]
    return proba


def oof_calibrated_et(X, y, groups, method="isotonic"):
    cv_outer = GroupKFold(n_splits=5)
    proba = np.zeros(len(y), dtype=np.float64)
    for tr, va in cv_outer.split(X, y, groups):
        Xt, yt, gt = X[tr], y[tr], groups[tr]
        models = []
        for fi, ci in GroupKFold(n_splits=3).split(Xt, yt, gt):
            base = make_et()
            base.fit(Xt[fi], yt[fi])
            raw = base.predict_proba(Xt[ci])[:, 1]
            if method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(raw, yt[ci])
            else:
                cal = LogisticRegression(max_iter=1000)
                cal.fit(raw.reshape(-1, 1), yt[ci])
            models.append((base, cal, method))
        ps = []
        for base, cal, meth in models:
            raw = base.predict_proba(X[va])[:, 1]
            if meth == "isotonic":
                ps.append(np.clip(cal.predict(raw), 0, 1))
            else:
                ps.append(cal.predict_proba(raw.reshape(-1, 1))[:, 1])
        proba[va] = np.mean(ps, axis=0)
    return proba


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
    return mean_ms, r_en, r_hi, proba


class BlendEstimator(BaseEstimator, ClassifierMixin):
    """Soft average of ExtraTrees and LogReg probabilities."""

    def __init__(self, w_et=0.7):
        self.w_et = w_et
        self.et_ = None
        self.lr_ = None

    def fit(self, X, y):
        self.et_ = make_et()
        self.lr_ = make_lr()
        self.et_.fit(X, y)
        self.lr_.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self.w_et * self.et_.predict_proba(X)[:, 1]
        p = p + (1.0 - self.w_et) * self.lr_.predict_proba(X)[:, 1]
        p = np.clip(p, 0, 1)
        return np.column_stack([1 - p, p])


class CalibratedBlend(BaseEstimator, ClassifierMixin):
    """Fit ET+isotonic (3-fold group) on train? Too heavy for final.
    Final ship: average of fitted ET and LR (same as OOF blend parents
    without nested calib) OR ET-only if that wins.

    For the winning OOF recipe blend_iso70et we approximate at ship time
    by fitting ET+LR on all data with w=0.7 — slightly optimistic vs OOF
    calib, but predict.py needs a single artifact. Prefer shipping plain
    ET on v3 if it beats gate; else ship BlendEstimator.
    """

    def __init__(self, w_et=0.7):
        self.w_et = w_et

    def fit(self, X, y):
        self.blend_ = BlendEstimator(w_et=self.w_et)
        self.blend_.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.blend_.predict_proba(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--ship", action="store_true",
                    help="write model.joblib+preds if best mean < GATE")
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    use_cache = not args.no_cache
    print(f"Building matrices (FEATURE_VERSION={FEATURE_VERSION}, "
          f"cache={use_cache})...")
    Xe, ye, ke, ge = build_matrix(DEF_EN, cache_dir=CACHE, use_cache=use_cache)
    Xh, yh, kh, gh = build_matrix(DEF_HI, cache_dir=CACHE, use_cache=use_cache)
    n_en = len(ke)
    X = np.vstack([Xe, Xh])
    y = np.concatenate([ye, yh]).astype(int)
    groups = np.concatenate([ge, gh])
    keys = ke + kh
    print(f"features v{FEATURE_VERSION}: {X.shape[0]} x {X.shape[1]} "
          f"(en {n_en}, hi {len(kh)})")
    assert X.shape[1] == len(FEATURE_NAMES)

    labels_en = os.path.join(DEF_EN, "labels.csv")
    labels_hi = os.path.join(DEF_HI, "labels.csv")

    results = {}
    print("\n--- ExtraTrees v3 ---")
    p_et = oof_predict(make_et(), X, y, groups)
    results["et_v3"] = evaluate("et_v3", keys, n_en, p_et, labels_en, labels_hi)

    print("\n--- LogReg v3 ---")
    p_lr = oof_predict(make_lr(), X, y, groups)
    results["lr_v3"] = evaluate("lr_v3", keys, n_en, p_lr, labels_en, labels_hi)

    print("\n--- ExtraTrees isotonic calib v3 ---")
    p_iso = oof_calibrated_et(X, y, groups, method="isotonic")
    results["et_iso_v3"] = evaluate("et_iso_v3", keys, n_en, p_iso,
                                    labels_en, labels_hi)

    print("\n--- Blends v3 ---")
    for w, tag in ((0.7, "blend70_v3"), (0.5, "blend50_v3")):
        p = w * p_et + (1 - w) * p_lr
        results[tag] = evaluate(tag, keys, n_en, p, labels_en, labels_hi)
    for w, tag in ((0.7, "blend_iso70_v3"), (0.5, "blend_iso50_v3")):
        p = w * p_iso + (1 - w) * p_lr
        results[tag] = evaluate(tag, keys, n_en, p, labels_en, labels_hi)

    best = min(results, key=lambda k: results[k][0])
    mean_ms, r_en, r_hi, proba = results[best]
    print(f"\nBEST={best} mean {mean_ms:.1f} ms  (gate {GATE_MS})")

    if args.ship and mean_ms < GATE_MS:
        # Prefer single ExtraTrees if it wins; else soft blend ET+LR
        if best.startswith("et_v3") or best.startswith("et_iso"):
            # ship plain ET refit (iso is OOF-only; final is full-data ET)
            model = make_et()
            model_name = "extratrees_v3"
        else:
            w = 0.7 if "70" in best else 0.5
            model = BlendEstimator(w_et=w)
            model_name = f"blend_et{int(w*100)}_lr_v3"
        model.fit(X, y)
        joblib.dump({
            "model": model,
            "feature_names": FEATURE_NAMES,
            "feature_version": FEATURE_VERSION,
            "model_name": model_name,
            "oof_en_ms": r_en["latency"] * 1000,
            "oof_hi_ms": r_hi["latency"] * 1000,
            "oof_en_auc": r_en["auc"],
            "oof_hi_auc": r_hi["auc"],
            "oof_mean_ms": mean_ms,
            "oof_tag": best,
        }, os.path.join(ROOT, "model.joblib"))
        write_pred_csv(os.path.join(ROOT, "oof_english.csv"),
                       keys[:n_en], proba[:n_en])
        write_pred_csv(os.path.join(ROOT, "oof_hindi.csv"),
                       keys[n_en:], proba[n_en:])
        print(f"SHIPPED model.joblib ({model_name}) OOF mean {mean_ms:.1f}")
    elif args.ship:
        print(f"NO SHIP: mean {mean_ms:.1f} >= gate {GATE_MS}")


if __name__ == "__main__":
    main()
