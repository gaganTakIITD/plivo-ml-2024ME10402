"""Show the worst pauses so a human can LISTEN to them.

    python error_analysis.py --data_dir <folder> --pred oof_english.csv --top 12

Prints holds ranked by highest p_eot (false-cutoff risk) and eots ranked by
lowest p_eot (each costs the 1.6 s timeout), with times for quick listening.
"""
import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    preds = {}
    with open(args.pred, newline="") as fh:
        for r in csv.DictReader(fh):
            preds[(r["turn_id"], int(r["pause_index"]))] = float(r["p_eot"])

    rows = []
    with open(os.path.join(args.data_dir, "labels.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["turn_id"], int(r["pause_index"]))
            rows.append({
                "turn": r["turn_id"], "pi": int(r["pause_index"]),
                "start": float(r["pause_start"]),
                "end": float(r["pause_end"]),
                "label": r["label"], "p": preds.get(key, float("nan")),
                "file": r["audio_file"],
            })

    holds = sorted([r for r in rows if r["label"] == "hold"],
                   key=lambda r: -r["p"])[: args.top]
    eots = sorted([r for r in rows if r["label"] == "eot"],
                  key=lambda r: r["p"])[: args.top]

    print(f"== HOLD pauses with HIGHEST p_eot (false-cutoff risk, "
          f"dur matters: cut only if delay < dur) ==")
    for r in holds:
        print(f"  p={r['p']:.3f}  {r['turn']} pi={r['pi']} "
              f"@{r['start']:.2f}-{r['end']:.2f}s dur={r['end']-r['start']:.2f}s"
              f"  {r['file']}")
    print(f"== EOT pauses with LOWEST p_eot (each miss ~1.6 s timeout) ==")
    for r in eots:
        print(f"  p={r['p']:.3f}  {r['turn']} pi={r['pi']} "
              f"@{r['start']:.2f}s  {r['file']}")


if __name__ == "__main__":
    main()
