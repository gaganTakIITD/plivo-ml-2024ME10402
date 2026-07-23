"""Round-2 remaining + literature eval on one clock.

1) v2 permutation prune + retrain ET/blend
2) tiny MLP OOF
3) rebuild v4 lit features; ablate flux / jitter / semitone / all
4) tiny GRU on frame sequences; blend if AUC>0.65
5) re-grid blends; print table; write .cache/round2_final_summary.txt

Does NOT overwrite model.joblib.
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import score as official

from eot_dataset import build_matrix
from features_ext import (FEATURE_NAMES, FEATURE_VERSION, LIT_ALL, LIT_FLUX,
                          LIT_JITTER, LIT_SEMITONE, V2_NAMES)

DEF_EN = "/mnt/d/pilvo test/eot_handout/eot_data/eot_data/english"
DEF_HI = "/mnt/d/pilvo test/eot_handout/eot_data/eot_data/hindi"
CACHE = os.path.join(ROOT, ".cache")
GATE = 964.0
RESULTS = []


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


def oof_raw(factory, X, y, groups):
    p = np.zeros(len(y), dtype=np.float64)
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        m = factory()
        m.fit(X[tr], y[tr])
        p[va] = m.predict_proba(X[va])[:, 1]
    return p


def oof_iso(factory, X, y, groups):
    p = np.zeros(len(y), dtype=np.float64)
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
        p[va] = np.mean(ps, axis=0)
    return p


def evaluate(name, keys, n_en, proba, labels_en, labels_hi):
    en = os.path.join(CACHE, f"r2f_{name}_en.csv")
    hi = os.path.join(CACHE, f"r2f_{name}_hi.csv")
    write_pred_csv(en, keys[:n_en], proba[:n_en])
    write_pred_csv(hi, keys[n_en:], proba[n_en:])
    r_en = official.score(labels_en, en)
    r_hi = official.score(labels_hi, hi)
    mean = (r_en["latency"] + r_hi["latency"]) / 2 * 1000
    row = {
        "name": name,
        "en_ms": r_en["latency"] * 1000, "en_auc": r_en["auc"],
        "hi_ms": r_hi["latency"] * 1000, "hi_auc": r_hi["auc"],
        "mean": mean,
    }
    RESULTS.append(row)
    flag = " ***" if mean < GATE else ""
    print(f"[{name}] EN {row['en_ms']:.0f}/{row['en_auc']:.3f} "
          f"HI {row['hi_ms']:.0f}/{row['hi_auc']:.3f} mean {mean:.1f}{flag}")
    return row


def cols_for(names):
    idx = [FEATURE_NAMES.index(n) for n in names]
    return np.array(idx, dtype=int)


def run_prune_v2(keys, n_en, labels_en, labels_hi):
    print("\n==== V2 permutation prune ====")
    # force v2 cache
    Xe = np.load(os.path.join(CACHE, "feat_english_v2.npz"))
    Xh = np.load(os.path.join(CACHE, "feat_hindi_v2.npz"))
    X = np.vstack([Xe["X"], Xh["X"]])
    y = np.concatenate([Xe["y"], Xh["y"]]).astype(int)
    groups = np.concatenate([Xe["turn"], Xh["turn"]])
    # OOF importance: average permutation importance on val folds
    imp = np.zeros(X.shape[1], dtype=np.float64)
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        m = make_et()
        m.fit(X[tr], y[tr])
        r = permutation_importance(m, X[va], y[va], n_repeats=5,
                                   random_state=0, scoring="roc_auc",
                                   n_jobs=-1)
        imp += r.importances_mean
    imp /= 5.0
    order = np.argsort(imp)
    drop_n = 18
    keep = np.sort(order[drop_n:])
    print(f"dropping {drop_n} lowest-importance features; keep {len(keep)}")
    Xp = X[:, keep]
    p_et = oof_iso(make_et, Xp, y, groups)
    p_lr = oof_raw(make_lr, Xp, y, groups)
    p_hgb = oof_iso(make_hgb, Xp, y, groups)
    evaluate("prune18_etiso", keys, n_en, p_et, labels_en, labels_hi)
    evaluate("prune18_blend70", keys, n_en, 0.7 * p_et + 0.3 * p_lr,
             labels_en, labels_hi)
    evaluate("prune18_3way_502030", keys, n_en,
             0.5 * p_et + 0.2 * p_lr + 0.3 * p_hgb, labels_en, labels_hi)
    return keep, p_et, p_lr, p_hgb


def run_mlp(X, y, groups, keys, n_en, labels_en, labels_hi, tag="mlp"):
    print(f"\n==== Tiny MLP ({tag}) ====")
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("torch missing; skip MLP")
        return None

    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    # median impute + standardize per fold
    proba = np.zeros(len(y), dtype=np.float64)
    torch.manual_seed(0)
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        med = np.nanmedian(X[tr], axis=0)
        Xtr = np.where(np.isnan(X[tr]), med, X[tr])
        Xva = np.where(np.isnan(X[va]), med, X[va])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xva = (Xva - mu) / sd
        xt = torch.tensor(Xtr, dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.float32)
        xv = torch.tensor(Xva, dtype=torch.float32)
        model = MLP(X.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        model.train()
        for epoch in range(80):
            opt.zero_grad()
            loss = loss_fn(model(xt), yt)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            proba[va] = torch.sigmoid(model(xv)).numpy()
    evaluate(tag, keys, n_en, proba, labels_en, labels_hi)
    return proba


def run_gru(keys, n_en, labels_en, labels_hi, y, groups):
    print("\n==== Tiny GRU on frame sequences ====")
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("torch missing; skip GRU")
        return None

    seq_path = os.path.join(CACHE, "frame_seq_v4.npz")
    if not os.path.exists(seq_path):
        print("building frame sequences (slow-ish)...")
        from eot_dataset import read_turns
        from features_ext import frame_sequence, load_wav
        seqs = []
        for data_dir in (DEF_EN, DEF_HI):
            turns = read_turns(data_dir)
            for turn_id in sorted(turns):
                rows = turns[turn_id]
                wav = os.path.join(data_dir, rows[0]["audio_file"])
                try:
                    x, sr = load_wav(wav)
                except Exception:
                    x, sr = np.zeros(1, dtype=np.float32), 16000
                for r in rows:
                    seqs.append(frame_sequence(x, sr, float(r["pause_start"])))
        S = np.stack(seqs).astype(np.float32)
        np.savez(seq_path, S=S)
    else:
        S = np.load(seq_path)["S"]
    print(f"sequences {S.shape}")

    class TinyGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_size=4, hidden_size=32, num_layers=1,
                              batch_first=True, dropout=0.0)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(32, 1)

        def forward(self, x):
            # x: [B,T,4]
            out, h = self.gru(x)
            h = self.drop(h[-1])
            return self.fc(h).squeeze(-1)

    proba = np.zeros(len(y), dtype=np.float64)
    torch.manual_seed(0)
    for tr, va in GroupKFold(n_splits=5).split(S, y, groups):
        # standardize using train
        mu = S[tr].mean(axis=(0, 1), keepdims=True)
        sd = S[tr].std(axis=(0, 1), keepdims=True) + 1e-6
        Str = (S[tr] - mu) / sd
        Sva = (S[va] - mu) / sd
        xt = torch.tensor(Str, dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.float32)
        xv = torch.tensor(Sva, dtype=torch.float32)
        model = TinyGRU()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        model.train()
        bs = 64
        for epoch in range(40):
            perm = np.random.RandomState(epoch).permutation(len(tr))
            for i in range(0, len(tr), bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = loss_fn(model(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            proba[va] = torch.sigmoid(model(xv)).numpy()
    row = evaluate("gru", keys, n_en, proba, labels_en, labels_hi)
    return proba if row["en_auc"] > 0.65 or row["hi_auc"] > 0.65 else None


def fine_3way(p_et, p_lr, p_hgb, keys, n_en, labels_en, labels_hi, tag_prefix):
    print(f"\n==== fine 3-way grid ({tag_prefix}) ====")
    best = (1e9, None, None)
    for a20 in range(0, 21):
        for b20 in range(0, 21 - a20):
            c20 = 20 - a20 - b20
            a, b, c = a20 / 20.0, b20 / 20.0, c20 / 20.0
            p = a * p_et + b * p_lr + c * p_hgb
            en = os.path.join(CACHE, "_t_en.csv")
            hi = os.path.join(CACHE, "_t_hi.csv")
            write_pred_csv(en, keys[:n_en], p[:n_en])
            write_pred_csv(hi, keys[n_en:], p[n_en:])
            r_en = official.score(labels_en, en)
            r_hi = official.score(labels_hi, hi)
            mean = (r_en["latency"] + r_hi["latency"]) / 2 * 1000
            if mean < best[0]:
                best = (mean, (a, b, c), p.copy())
                if mean < GATE:
                    print(f"  best {tag_prefix} a={a:.2f} b={b:.2f} c={c:.2f} "
                          f"EN {r_en['latency']*1000:.0f} HI {r_hi['latency']*1000:.0f} "
                          f"mean {mean:.1f}")
    a, b, c = best[1]
    evaluate(f"{tag_prefix}_3way_{int(a*100)}_{int(b*100)}_{int(c*100)}",
             keys, n_en, best[2], labels_en, labels_hi)
    return best


def main():
    t0 = time.time()
    os.makedirs(CACHE, exist_ok=True)
    labels_en = os.path.join(DEF_EN, "labels.csv")
    labels_hi = os.path.join(DEF_HI, "labels.csv")

    # keys from v2 (same ordering)
    Xe2 = np.load(os.path.join(CACHE, "feat_english_v2.npz"))
    Xh2 = np.load(os.path.join(CACHE, "feat_hindi_v2.npz"))
    n_en = len(Xe2["y"])
    keys = (list(zip(Xe2["turn"].tolist(), Xe2["pi"].tolist())) +
            list(zip(Xh2["turn"].tolist(), Xh2["pi"].tolist())))
    y = np.concatenate([Xe2["y"], Xh2["y"]]).astype(int)
    groups = np.concatenate([Xe2["turn"], Xh2["turn"]])

    # --- prune ---
    run_prune_v2(keys, n_en, labels_en, labels_hi)

    # --- MLP on v2 ---
    X2 = np.vstack([Xe2["X"], Xh2["X"]])
    p_mlp = run_mlp(X2, y, groups, keys, n_en, labels_en, labels_hi, "mlp_v2")

    # load prior iso probs if present for blend with mlp
    zpath = os.path.join(CACHE, "round2_probs.npz")
    if os.path.exists(zpath) and p_mlp is not None:
        z = np.load(zpath)
        for w in (0.2, 0.3):
            p = (0.5 - w / 2) * z["p_et_iso"] + 0.2 * z["p_lr"] + \
                (0.3 - w / 2) * z["p_hgb_iso"] + w * p_mlp
            # renormalize conceptually: just try additive mix that sums ~1
            evaluate(f"4way_mlp_w{int(w*10)}", keys, n_en, p,
                     labels_en, labels_hi)
        # cleaner: grid mlp into 3way best
        p = 0.4 * z["p_et_iso"] + 0.15 * z["p_lr"] + 0.25 * z["p_hgb_iso"] + 0.2 * p_mlp
        evaluate("4way_mlp_40251520", keys, n_en, p, labels_en, labels_hi)

    print(f"\n==== rebuild features v{FEATURE_VERSION} ====")
    assert FEATURE_VERSION == 4
    Xe, ye, ke, ge = build_matrix(DEF_EN, cache_dir=CACHE, use_cache=False)
    Xh, yh, kh, gh = build_matrix(DEF_HI, cache_dir=CACHE, use_cache=False)
    assert ke == keys[:n_en]
    X = np.vstack([Xe, Xh])
    print(f"v4 matrix {X.shape}")

    # ablations
    def subset(extra_names):
        names = V2_NAMES + list(extra_names)
        idx = cols_for(names)
        return X[:, idx]

    for tag, extra in [
        ("v2only_from_v4", []),
        ("flux_only", LIT_FLUX),
        ("jitter_only", LIT_JITTER),
        ("semitone_only", LIT_SEMITONE),
        ("lit_all", LIT_ALL),
    ]:
        Xs = subset(extra)
        print(f"\n-- ET-iso on {tag} d={Xs.shape[1]} --")
        p_et = oof_iso(make_et, Xs, y, groups)
        evaluate(f"{tag}_etiso", keys, n_en, p_et, labels_en, labels_hi)
        if tag == "lit_all":
            p_lr = oof_raw(make_lr, Xs, y, groups)
            p_hgb = oof_iso(make_hgb, Xs, y, groups)
            evaluate("lit_all_lr", keys, n_en, p_lr, labels_en, labels_hi)
            fine_3way(p_et, p_lr, p_hgb, keys, n_en, labels_en, labels_hi, "lit")
            p_mlp4 = run_mlp(Xs, y, groups, keys, n_en, labels_en, labels_hi,
                             "mlp_v4")
            # save for gru blend
            np.savez(os.path.join(CACHE, "round2_lit_probs.npz"),
                     p_et=p_et, p_lr=p_lr, p_hgb=p_hgb,
                     p_mlp=p_mlp4 if p_mlp4 is not None else np.zeros(len(y)))

    # GRU
    p_gru = run_gru(keys, n_en, labels_en, labels_hi, y, groups)
    if p_gru is not None and os.path.exists(os.path.join(CACHE, "round2_lit_probs.npz")):
        z = np.load(os.path.join(CACHE, "round2_lit_probs.npz"))
        for w in (0.15, 0.25):
            p = (0.5 - w / 2) * z["p_et"] + 0.2 * z["p_lr"] + \
                (0.3 - w / 2) * z["p_hgb"] + w * p_gru
            evaluate(f"lit_3way_plus_gru_w{int(w*100)}", keys, n_en, p,
                     labels_en, labels_hi)
    elif p_gru is None:
        print("GRU AUC too low or failed; not added to blend")

    # summary
    RESULTS.sort(key=lambda r: r["mean"])
    print("\n======== SORTED RESULTS ========")
    for r in RESULTS:
        print(f"{r['name']:40s} mean {r['mean']:7.1f}  "
              f"EN {r['en_ms']:.0f}/{r['en_auc']:.3f}  "
              f"HI {r['hi_ms']:.0f}/{r['hi_auc']:.3f}")
    best = RESULTS[0]
    print(f"\nBEST={best['name']} mean {best['mean']:.1f} (gate {GATE})")
    print(f"elapsed {time.time()-t0:.0f}s")
    with open(os.path.join(CACHE, "round2_final_summary.txt"), "w") as fh:
        for r in RESULTS:
            fh.write(f"{r['name']}\t{r['mean']:.2f}\t{r['en_ms']:.1f}\t"
                     f"{r['en_auc']:.3f}\t{r['hi_ms']:.1f}\t{r['hi_auc']:.3f}\n")
        fh.write(f"BEST\t{best['name']}\t{best['mean']:.2f}\n")


if __name__ == "__main__":
    main()
