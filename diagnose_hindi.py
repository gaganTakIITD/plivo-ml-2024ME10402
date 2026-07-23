"""Diagnose why Hindi OOF delay is stuck at 850 ms despite AUC ~0.718."""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import score as official

DEF_HI = "/mnt/d/pilvo test/eot_handout/eot_data/eot_data/hindi"
OOF_HI = os.path.join(ROOT, "oof_hindi.csv")
LABELS = os.path.join(DEF_HI, "labels.csv")


def load_joined(labels_csv, pred_csv):
    preds = {}
    with open(pred_csv) as f:
        for r in csv.DictReader(f):
            preds[(r["turn_id"], int(r["pause_index"]))] = float(r["p_eot"])
    rows = []
    with open(labels_csv) as f:
        for r in csv.DictReader(f):
            key = (r["turn_id"], int(r["pause_index"]))
            dur = float(r["pause_end"]) - float(r["pause_start"])
            rows.append({
                "turn_id": r["turn_id"],
                "pause_index": int(r["pause_index"]),
                "dur": dur,
                "label": r["label"],
                "p": preds[key],
                "pause_start": float(r["pause_start"]),
            })
    return rows


def main():
    rows = load_joined(LABELS, OOF_HI)
    r = official.score(LABELS, OOF_HI)
    t, d = r["threshold"], r["delay"]
    print("=== Official best op-point ===")
    print(f"latency={r['latency']*1000:.1f} ms  cut={r['cutoff']*100:.2f}%  "
          f"t={t}  delay={d*1000:.0f} ms  AUC={r['auc']:.3f}")

    holds = [x for x in rows if x["label"] == "hold"]
    eots = [x for x in rows if x["label"] == "eot"]
    print(f"\npauses: {len(rows)}  holds={len(holds)}  eots={len(eots)}  "
          f"turns={len({x['turn_id'] for x in rows})}")

    # fire / timeout at operating point
    n_fire_eot = sum(1 for x in eots if x["p"] >= t)
    n_timeout = len(eots) - n_fire_eot
    n_hold_fire_risky = sum(1 for x in holds if x["p"] >= t and d < x["dur"])
    n_hold_fire_forgiven = sum(1 for x in holds if x["p"] >= t and d >= x["dur"])
    print("\n=== At operating point ===")
    print(f"EOT fires: {n_fire_eot}/{len(eots)} ({100*n_fire_eot/len(eots):.1f}%)")
    print(f"EOT timeouts: {n_timeout}/{len(eots)} ({100*n_timeout/len(eots):.1f}%)")
    print(f"HOLD risky fires (false cutoff): {n_hold_fire_risky}")
    print(f"HOLD forgiven fires (delay >= dur): {n_hold_fire_forgiven}")

    # latency composition
    lat_fire = d
    lat_to = official.TIMEOUT_S
    mean_from_fire = (n_fire_eot * lat_fire + n_timeout * lat_to) / len(eots)
    print(f"\nMean latency breakdown: fire@{d*1000:.0f} + timeout@1600 "
          f"-> {mean_from_fire*1000:.1f} ms")
    # What if all EOTs fired at this delay?
    print(f"If 100% EOT fire at delay: {d*1000:.0f} ms")
    # What delay would give 850 if we fired all?
    # 850 comes from delay=0.85 with all fires, OR mix

    # Check if 850 == exactly some delay with full fire rate
    print("\n=== Sweep: best latency per delay (budget 5%) ===")
    best_by_delay = {}
    for dd in official.DELAYS:
        best = None
        for tt in official.THRESHOLDS:
            cut, lat = official.evaluate(rows, tt, dd)
            if cut <= 0.05 and (best is None or lat < best[0]):
                best = (lat, tt, cut)
        if best:
            best_by_delay[dd] = best
            mark = " <--" if abs(dd - d) < 1e-9 else ""
            print(f"  delay={dd*1000:4.0f} ms -> lat={best[0]*1000:6.1f} "
                  f"t={best[1]:.2f} cut={best[2]*100:.1f}%{mark}")

    # p_eot distribution overlap
    ph = np.array([x["p"] for x in holds])
    pe = np.array([x["p"] for x in eots])
    print("\n=== p_eot distribution ===")
    for name, arr in (("HOLD", ph), ("EOT", pe)):
        qs = np.percentile(arr, [10, 25, 50, 75, 90])
        print(f"  {name}: mean={arr.mean():.3f} std={arr.std():.3f} "
              f"p10/25/50/75/90={qs.round(3).tolist()}")

    # overlap: fraction of HOLDs above various EOT percentiles
    for pct in (25, 50, 75):
        thr = np.percentile(pe, pct)
        frac = (ph >= thr).mean()
        print(f"  HOLD >= EOT p{pct} ({thr:.3f}): {100*frac:.1f}%")

    # Long holds with high p — binding constraint?
    print("\n=== High-p long HOLDs (p>=0.5, dur>0.5s) ===")
    long_hi = [x for x in holds if x["p"] >= 0.5 and x["dur"] > 0.5]
    long_hi.sort(key=lambda x: -x["p"])
    print(f"count={len(long_hi)}")
    for x in long_hi[:15]:
        print(f"  {x['turn_id']} idx={x['pause_index']} "
              f"p={x['p']:.3f} dur={x['dur']:.2f}s start={x['pause_start']:.2f}")

    # Binding turns: which false cutoffs appear at aggressive op-points?
    print("\n=== Binding false-cutoff turns at delay=0.5, best t under budget ===")
    for dd in (0.3, 0.5, 0.7, 0.85):
        best = None
        for tt in official.THRESHOLDS:
            cut, lat = official.evaluate(rows, tt, dd)
            if cut <= 0.05 and (best is None or lat < best[0]):
                best = (lat, tt, cut)
        # also show cut at t that would fire most EOTs
        if best:
            tt = best[1]
            cut_turns = set()
            for x in holds:
                if x["p"] >= tt and dd < x["dur"]:
                    cut_turns.add(x["turn_id"])
            n_fire = sum(1 for x in eots if x["p"] >= tt)
            print(f"  d={dd*1000:.0f}: best t={tt} lat={best[0]*1000:.0f} "
                  f"cut={best[2]*100:.1f}% eot_fire={n_fire}/{len(eots)} "
                  f"cut_turns={sorted(cut_turns)[:8]}")

    # First-pause EOTs
    print("\n=== First-pause (idx=0) EOTs ===")
    e0 = [x for x in eots if x["pause_index"] == 0]
    h0 = [x for x in holds if x["pause_index"] == 0]
    print(f"EOT idx0: n={len(e0)} mean_p={np.mean([x['p'] for x in e0]):.3f}")
    print(f"HOLD idx0: n={len(h0)} mean_p={np.mean([x['p'] for x in h0]):.3f}")
    n_miss0 = sum(1 for x in e0 if x["p"] < t)
    print(f"At op-point, idx0 EOT misses (timeout): {n_miss0}/{len(e0)}")

    # Is 850ms literally the delay with near-full fire?
    print("\n=== Hypothesis check ===")
    # If latency == delay, nearly all EOTs fire
    if abs(r["latency"] - d) < 1e-6:
        print("Latency == delay => virtually ALL EOTs fire; floor is the "
              "minimum safe delay under 5% cutoff budget.")
    else:
        mix = (r["latency"] - lat_to) / (d - lat_to) if d != lat_to else float("nan")
        print(f"Latency != delay; implied fire fraction ~ {mix:.3f}")
    print("Done.")


if __name__ == "__main__":
    main()
