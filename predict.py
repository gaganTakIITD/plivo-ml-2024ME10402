"""Official prediction interface.

    python predict.py --data_dir <folder> --out predictions.csv

<folder> must contain labels.csv and the audio/ files it references (same
schema as the handout). Writes turn_id,pause_index,p_eot for EVERY row.

Causality: features use only audio strictly before each pause_start plus
boundaries of EARLIER pauses in the same turn (past events). The current
pause's `pause_end`/`label` are never used as inputs (see features_ext.py
and eot_dataset.py). The model is loaded from model.joblib next to this
file -- no refitting on the evaluation data.
"""
import argparse
import csv
import os

import joblib
import numpy as np

from eot_dataset import build_matrix
from features_ext import FEATURE_VERSION

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    bundle = joblib.load(os.path.join(ROOT, "model.joblib"))
    if bundle["feature_version"] != FEATURE_VERSION:
        raise SystemExit(
            f"model.joblib was built with feature v{bundle['feature_version']}"
            f" but code is v{FEATURE_VERSION}; retrain with train_model.py")

    X, _, keys, _ = build_matrix(args.data_dir, cache_dir=None,
                                 use_cache=False)
    p = bundle["model"].predict_proba(X)[:, 1]
    p = np.clip(np.nan_to_num(p, nan=0.4), 0.0, 1.0)

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pp in zip(keys, p):
            w.writerow([tid, pi, f"{pp:.4f}"])
    print(f"wrote {len(p)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
