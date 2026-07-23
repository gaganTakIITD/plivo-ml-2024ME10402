"""Fine 3-way grid around best (0.4 et_iso, 0.2 lr, 0.4 hgb_iso)."""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import score as official

CACHE = os.path.join(ROOT, ".cache")
DEF_EN = "/mnt/d/pilvo test/eot_handout/eot_data/eot_data/english"
DEF_HI = "/mnt/d/pilvo test/eot_handout/eot_data/eot_data/hindi"
GATE = 964.0


def write_pred_csv(path, keys, probs):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), p in zip(keys, probs):
            w.writerow([tid, pi, f"{p:.6f}"])


def main():
    z = np.load(os.path.join(CACHE, "round2_probs.npz"), allow_pickle=True)
    p_et_iso = z["p_et_iso"]
    p_lr = z["p_lr"]
    p_hgb_iso = z["p_hgb_iso"]

    from eot_dataset import build_matrix
    Xe, ye, ke, ge = build_matrix(DEF_EN, cache_dir=CACHE, use_cache=True)
    Xh, yh, kh, gh = build_matrix(DEF_HI, cache_dir=CACHE, use_cache=True)
    n_en = len(ke)
    keys = ke + kh
    labels_en = os.path.join(DEF_EN, "labels.csv")
    labels_hi = os.path.join(DEF_HI, "labels.csv")

    best = (1e9, None)
    # step 0.05 simplex
    for a20 in range(0, 21):
        for b20 in range(0, 21 - a20):
            c20 = 20 - a20 - b20
            a, b, c = a20 / 20.0, b20 / 20.0, c20 / 20.0
            # focus near previous best, but full fine grid is cheap
            p = a * p_et_iso + b * p_lr + c * p_hgb_iso
            en = os.path.join(CACHE, "_tmp_en.csv")
            hi = os.path.join(CACHE, "_tmp_hi.csv")
            write_pred_csv(en, keys[:n_en], p[:n_en])
            write_pred_csv(hi, keys[n_en:], p[n_en:])
            r_en = official.score(labels_en, en)
            r_hi = official.score(labels_hi, hi)
            mean = (r_en["latency"] + r_hi["latency"]) / 2 * 1000
            if mean < best[0]:
                best = (mean, (a, b, c, r_en, r_hi, p.copy()))
                mark = " ***" if mean < GATE else ""
                print(f"NEW BEST a={a:.2f} b={b:.2f} c={c:.2f} "
                      f"EN {r_en['latency']*1000:.0f} HI {r_hi['latency']*1000:.0f} "
                      f"mean {mean:.1f}{mark}")

    mean, (a, b, c, r_en, r_hi, p) = best[0], best[1]
    print(f"\nFINE BEST: et={a:.2f} lr={b:.2f} hgb={c:.2f} mean {mean:.1f}")
    print(f"  EN {r_en['latency']*1000:.1f} AUC {r_en['auc']:.3f}")
    print(f"  HI {r_hi['latency']*1000:.1f} AUC {r_hi['auc']:.3f}")
    np.savez(os.path.join(CACHE, "round2_best3way.npz"),
             p=p, a=a, b=b, c=c, mean=mean,
             en_ms=r_en["latency"] * 1000, hi_ms=r_hi["latency"] * 1000,
             en_auc=r_en["auc"], hi_auc=r_hi["auc"])


if __name__ == "__main__":
    main()
