"""labels.csv + wavs -> feature matrix, with on-disk caching.

Causality is enforced structurally here: for each turn we walk pauses in
pause_index order and append a pause's (start, end) to `prior` only AFTER
its own features are extracted. So pause j never sees its own end (future),
while pause j+1 may see pause j's boundaries (past). The current row's
`label` is used only as the training target, never as a feature.
"""
import csv
import os

import numpy as np

from features_ext import (FEATURE_NAMES, FEATURE_VERSION, features_vec,
                          load_wav, pause_features)


def read_turns(data_dir):
    """dict turn_id -> rows sorted by pause_index."""
    turns = {}
    with open(os.path.join(data_dir, "labels.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            turns.setdefault(r["turn_id"], []).append(r)
    for rows in turns.values():
        rows.sort(key=lambda r: int(r["pause_index"]))
    return turns


def build_matrix(data_dir, cache_dir=None, use_cache=True):
    """Returns X (n,f) float32, y (n,) int8, keys [(turn_id, pause_index)],
    groups (n,) array of turn ids."""
    tag = os.path.basename(os.path.normpath(data_dir))
    cpath = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cpath = os.path.join(cache_dir, f"feat_{tag}_v{FEATURE_VERSION}.npz")
        if use_cache and os.path.exists(cpath):
            z = np.load(cpath, allow_pickle=False)
            keys = list(zip(z["turn"].tolist(), z["pi"].tolist()))
            return z["X"], z["y"], keys, z["turn"]

    turns = read_turns(data_dir)
    X, y, tids, pis = [], [], [], []
    for turn_id in sorted(turns):
        rows = turns[turn_id]
        wav_path = os.path.join(data_dir, rows[0]["audio_file"])
        try:
            x, sr = load_wav(wav_path)
        except Exception as exc:  # unreadable audio: emit NaN rows, never crash
            print(f"WARN could not read {wav_path}: {exc}")
            x, sr = np.zeros(1, dtype=np.float32), 16000
        prior = []
        for r in rows:
            ps = float(r["pause_start"])
            fd = pause_features(x, sr, ps, prior)
            X.append(features_vec(fd))
            y.append(1 if r.get("label") == "eot" else 0)
            tids.append(r["turn_id"])
            pis.append(int(r["pause_index"]))
            # append AFTER extraction: own end stays invisible to this pause
            pe = r.get("pause_end")
            prior.append((ps, float(pe) if pe not in (None, "") else ps))

    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.int8)
    turn = np.array(tids)
    pi = np.array(pis, dtype=np.int32)
    if cpath:
        np.savez(cpath, X=X, y=y, turn=turn, pi=pi)
    keys = list(zip(turn.tolist(), pi.tolist()))
    return X, y, keys, turn
