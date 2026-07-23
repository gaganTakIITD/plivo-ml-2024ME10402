"""Train and honestly evaluate the end-of-turn model.

Protocol: pooled english+hindi pauses, 5-fold GroupKFold by turn_id
(a turn never straddles train/validation), out-of-fold p_eot for every
pause, scored per language with the OFFICIAL starter/score.py. The final
model is refit on all data and saved to model.joblib for predict.py.

    python train_model.py [--model auto|logreg|hgb] [--no-cache]
"""
import argparse
import csv
import os
import sys

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eot_dataset import build_matrix
from features_ext import FEATURE_NAMES, FEATURE_VERSION

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import score as official  # the untouched official scorer

DEF_EN = "/mnt/d/pilvo test/eot_handout/eot_data/eot_data/english"
DEF_HI = "/mnt/d/pilvo test/eot_handout/eot_data/eot_data/hindi"
CACHE = os.path.join(ROOT, ".cache")


def make_models():
    logreg = Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", C=1.0,
                                   max_iter=4000, random_state=0)),
    ])
    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=3, min_samples_leaf=15,
        l2_regularization=1.0, random_state=0)
    return {"logreg": logreg, "hgb": hgb}


def write_pred_csv(path, keys, probs):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), p in zip(keys, probs):
            w.writerow([tid, pi, f"{p:.4f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", default=DEF_EN)
    ap.add_argument("--hi", default=DEF_HI)
    ap.add_argument("--model", default="auto",
                    choices=["auto", "logreg", "hgb"])
    ap.add_argument("--tune", action="store_true",
                    help="small HGB grid search by OOF mean latency")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    use_cache = not args.no_cache
    Xe, ye, ke, ge = build_matrix(args.en, cache_dir=CACHE, use_cache=use_cache)
    Xh, yh, kh, gh = build_matrix(args.hi, cache_dir=CACHE, use_cache=use_cache)
    n_en = len(ke)
    X = np.vstack([Xe, Xh])
    y = np.concatenate([ye, yh]).astype(int)
    groups = np.concatenate([ge, gh])
    keys = ke + kh
    print(f"features v{FEATURE_VERSION}: {X.shape[0]} pauses x "
          f"{X.shape[1]} features (en {n_en}, hi {len(kh)})")

    labels_en = os.path.join(args.en, "labels.csv")
    labels_hi = os.path.join(args.hi, "labels.csv")
    cv = GroupKFold(n_splits=5)
    models = make_models()
    if args.model != "auto":
        models = {args.model: models[args.model]}

    def evaluate(name, est):
        proba = cross_val_predict(est, X, y, groups=groups, cv=cv,
                                  method="predict_proba")[:, 1]
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
              f"d={r_hi['delay']*1000:.0f} ms) | mean {mean_ms:.0f} ms")
        return mean_ms, r_en, r_hi, proba

    candidates = dict(models)
    if args.tune:
        candidates = {}
        for lr in (0.03, 0.06, 0.1):
            for mi in (300, 600):
                for md in (2, 3):
                    for msl in (10, 20):
                        name = f"hgb_lr{lr}_it{mi}_d{md}_l{msl}"
                        candidates[name] = HistGradientBoostingClassifier(
                            learning_rate=lr, max_iter=mi, max_depth=md,
                            min_samples_leaf=msl, l2_regularization=1.0,
                            random_state=0)

    results = {}
    for name, est in candidates.items():
        results[name] = evaluate(name, est)

    best = min(results, key=lambda k: results[k][0])
    mean_ms, r_en, r_hi, proba = results[best]
    print(f"BEST={best}  mean {mean_ms:.0f} ms")

    # promote best OOF files for error analysis
    for lang, sl in (("english", slice(0, n_en)), ("hindi", slice(n_en, None))):
        write_pred_csv(os.path.join(ROOT, f"oof_{lang}.csv"),
                       keys[sl], proba[sl])

    est = clone(candidates[best])
    est.fit(X, y)
    joblib.dump({"model": est, "feature_names": FEATURE_NAMES,
                 "feature_version": FEATURE_VERSION, "model_name": best},
                os.path.join(ROOT, "model.joblib"))
    print(f"saved model.joblib ({best}, refit on all {len(y)} pauses)")


if __name__ == "__main__":
    main()
