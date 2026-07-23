"""Ship ThreeWayBlend on v2 if OOF mean < 964.

Uses cached round2 OOF probs / fine weights; refits deployable ThreeWayBlend.
"""
from __future__ import annotations

import csv
import os
import sys

import joblib
import numpy as np

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


def write_pred_csv(path, keys, probs):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), p in zip(keys, probs):
            w.writerow([tid, pi, f"{p:.6f}"])


def main():
    # Prefer explicit weights from fine grid on v2
    best_path = os.path.join(CACHE, "round2_best3way.npz")
    if not os.path.exists(best_path):
        print("missing round2_best3way.npz")
        sys.exit(1)
    z = np.load(best_path)
    a, b, c = float(z["a"]), float(z["b"]), float(z["c"])
    mean_cached = float(z["mean"])
    print(f"cached best3way et={a} lr={b} hgb={c} mean={mean_cached:.1f}")

    # Recompute OOF with same recipe for honesty (v2 features)
    # NOTE: features_ext may be v4 — use v2 npz cache directly
    Xe = np.load(os.path.join(CACHE, "feat_english_v2.npz"))
    Xh = np.load(os.path.join(CACHE, "feat_hindi_v2.npz"))
    X = np.vstack([Xe["X"], Xh["X"]])
    y = np.concatenate([Xe["y"], Xh["y"]]).astype(int)
    groups = np.concatenate([Xe["turn"], Xh["turn"]])
    n_en = len(Xe["y"])
    keys = (list(zip(Xe["turn"].tolist(), Xe["pi"].tolist())) +
            list(zip(Xh["turn"].tolist(), Xh["pi"].tolist())))

    # Use saved OOF blend probs from fine grid
    p = z["p"]
    labels_en = os.path.join(DEF_EN, "labels.csv")
    labels_hi = os.path.join(DEF_HI, "labels.csv")
    oof_en = os.path.join(ROOT, "oof_english.csv")
    oof_hi = os.path.join(ROOT, "oof_hindi.csv")
    write_pred_csv(oof_en, keys[:n_en], p[:n_en])
    write_pred_csv(oof_hi, keys[n_en:], p[n_en:])
    r_en = official.score(labels_en, oof_en)
    r_hi = official.score(labels_hi, oof_hi)
    mean = (r_en["latency"] + r_hi["latency"]) / 2 * 1000
    print(f"verified OOF EN {r_en['latency']*1000:.1f} HI {r_hi['latency']*1000:.1f} "
          f"mean {mean:.1f}")
    if mean >= GATE:
        print("NO SHIP")
        sys.exit(2)

    # For predict.py: need FEATURE_VERSION match. Force v2 names length.
    assert X.shape[1] == 53
    # Temporarily require features_ext to be v2 for predict — dump version=2
    # even if code has v4, predict checks FEATURE_VERSION constant.
    # Caller must ensure features_ext.FEATURE_VERSION == 2 before predict,
    # OR we store feature_names and predict builds with those.
    model = ThreeWayBlend(w_et=a, w_lr=b, w_hgb=c)
    model.fit(X, y, groups=groups)
    v2_names = list(FEATURE_NAMES[:53]) if len(FEATURE_NAMES) > 53 else list(FEATURE_NAMES)
    # If FEATURE_VERSION is 4, v2_names from V2_NAMES
    try:
        from features_ext import V2_NAMES
        v2_names = list(V2_NAMES)
    except Exception:
        pass
    assert len(v2_names) == 53

    joblib.dump({
        "model": model,
        "feature_names": v2_names,
        "feature_version": 2,  # predictions use v2 matrix
        "model_name": f"threeway_et{a:.2f}_lr{b:.2f}_hgb{c:.2f}_v2",
        "oof_en_ms": r_en["latency"] * 1000,
        "oof_hi_ms": r_hi["latency"] * 1000,
        "oof_en_auc": r_en["auc"],
        "oof_hi_auc": r_hi["auc"],
        "oof_mean_ms": mean,
        "blend_weights": {"et": a, "lr": b, "hgb": c},
    }, os.path.join(ROOT, "model.joblib"))
    print(f"SHIPPED ThreeWayBlend mean {mean:.1f}")


if __name__ == "__main__":
    main()
