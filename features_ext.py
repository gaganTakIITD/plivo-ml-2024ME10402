"""Causal features for end-of-turn detection.

CAUSALITY CONTRACT
------------------
For a pause with boundary time t = pause_start, every feature uses ONLY:
  * audio samples x[0 : int(t * sr)]  (strictly before the pause), and
  * metadata already known at time t: this pause's index (= how many pauses
    came before it) and the boundaries of PREVIOUS pauses in the same turn.
    Every previous pause ended at pause_end_prev <= t, so it is past data.
The current row's `pause_end` and `label` are NEVER read here.

The pitch tracker reproduces starter/features.py (autocorrelation, 40 ms
frames / 10 ms hop, voicing threshold 0.30) but runs the autocorrelation of
all frames in one batched FFT, so a full-dataset feature rebuild takes
seconds instead of ~10 minutes.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
import librosa

FRAME_MS = 25
HOP_MS = 10
F0_FRAME_MS = 40
F0_MIN, F0_MAX = 60.0, 400.0
VOICING_THRESH = 0.30

SEG_S = 1.5   # analysis window immediately before the pause
CTX_S = 4.0   # longer context for speaker-relative normalisation
TAIL_S = 0.6  # spectral/MFCC window at the very end of speech

FEATURE_VERSION = 2  # bump to invalidate cached feature matrices

FEATURE_NAMES = [
    # timing / turn structure (past metadata only)
    "elapsed_s", "pause_index", "prev_pause_dur", "since_prev_end",
    "mean_prior_dur", "speech_ratio", "seg_dur_s",
    "is_first_pause", "speech_so_far", "pause_rate", "log1p_elapsed",
    # energy trajectory into the pause
    "e_last5", "e_last15", "e_slope_300", "e_slope_150", "e_drop", "e_std",
    "e_last_rel_ctx", "e_peak_count", "e_tail_vs_head", "e_max",
    # pitch behaviour at the end of speech
    "voiced_frac", "trail_unvoiced_ms", "last_voiced_run_ms",
    "final_lengthening", "f0_last", "f0_slope_lastrun", "f0_slope_500",
    "f0_med_seg", "f0_last_rel", "f0_range_seg", "n_voiced_runs",
    # speaker-relative pitch (longer context)
    "f0_med_ctx", "f0_last_rel_ctx",
    # spectral shape at the end
    "cent_mean", "cent_slope", "rolloff_mean", "rolloff_slope",
    "zcr_mean", "zcr_slope", "rms_tail",
] + [f"mfcc{i}" for i in range(8)] + [f"dmfcc{i}" for i in range(4)]


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def _frames(x, sr, frame_ms, hop_ms):
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]


def frame_energy_db(x, sr):
    fr = _frames(x, sr, FRAME_MS, HOP_MS)
    if len(fr) == 0:
        return np.zeros(0, dtype=np.float32)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    return (20 * np.log10(rms + 1e-12)).astype(np.float32)


def f0_contour_fast(x, sr, frame_ms=F0_FRAME_MS, hop_ms=HOP_MS):
    """Batched-FFT version of starter's autocorr_f0 over all frames."""
    fr = _frames(x, sr, frame_ms, hop_ms)
    if len(fr) == 0:
        return np.zeros(0, dtype=np.float32)
    fr = fr - fr.mean(axis=1, keepdims=True)
    amp = np.max(np.abs(fr), axis=1)
    nfft = 1 << int(np.ceil(np.log2(2 * fr.shape[1])))
    spec = np.fft.rfft(fr, n=nfft, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), n=nfft, axis=1)[:, : fr.shape[1]]
    ac0 = ac[:, 0].copy()
    ok = (ac0 > 0) & (amp >= 1e-4)
    ac = np.where(ok[:, None], ac / np.maximum(ac0[:, None], 1e-12), 0.0)
    lo = int(sr / F0_MAX)
    hi = min(int(sr / F0_MIN), ac.shape[1] - 1)
    f0 = np.zeros(len(fr), dtype=np.float32)
    if hi > lo:
        band = ac[:, lo:hi]
        lag = lo + np.argmax(band, axis=1)
        peak = band[np.arange(len(fr)), lag - lo]
        voiced = ok & (peak >= VOICING_THRESH)
        f0[voiced] = (sr / lag[voiced]).astype(np.float32)
    return f0


def _runs(mask):
    """(start, end) index pairs of contiguous True runs."""
    if len(mask) == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]
    return list(zip(starts, ends))


def _slope(y, x_step=1.0):
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 2:
        return np.nan
    x = np.arange(len(y), dtype=np.float64) * x_step
    x = x - x.mean()
    denom = float((x ** 2).sum())
    if denom == 0.0:
        return np.nan
    return float((x * (y - y.mean())).sum() / denom)


def pause_features(x, sr, pause_start, prior_pauses):
    """Feature dict for one pause.

    prior_pauses: [(start, end), ...] for pauses with a SMALLER pause_index
    in the same turn -- strictly past events at time pause_start.
    """
    f = {k: np.nan for k in FEATURE_NAMES}
    hop_s = HOP_MS / 1000.0

    # ---- timing / structure ----
    f["elapsed_s"] = float(pause_start)
    f["log1p_elapsed"] = float(np.log1p(pause_start))
    f["pause_index"] = float(len(prior_pauses))
    f["is_first_pause"] = 1.0 if not prior_pauses else 0.0
    if prior_pauses:
        prev_s, prev_e = prior_pauses[-1]
        durs = [e - s for s, e in prior_pauses]
        f["prev_pause_dur"] = float(prev_e - prev_s)
        f["since_prev_end"] = float(pause_start - prev_e)
        f["mean_prior_dur"] = float(np.mean(durs))
        pause_time = float(sum(durs))
    else:
        pause_time = 0.0
    speech_so_far = max(pause_start - pause_time, 0.0)
    f["speech_so_far"] = float(speech_so_far)
    f["speech_ratio"] = float(speech_so_far / max(pause_start, 1e-6))
    f["pause_rate"] = float(len(prior_pauses) / max(pause_start, 1e-6))

    t_end = min(int(pause_start * sr), len(x))
    seg = x[max(0, t_end - int(SEG_S * sr)): t_end]
    f["seg_dur_s"] = float(len(seg) / sr)
    if len(seg) < sr // 10:  # under 100 ms of usable context
        return f

    # ---- energy trajectory ----
    e = frame_energy_db(seg, sr)
    if len(e) >= 5:
        f["e_last5"] = float(np.mean(e[-5:]))
        f["e_last15"] = float(np.mean(e[-15:]))
        f["e_slope_300"] = _slope(e[-30:], hop_s)
        f["e_slope_150"] = _slope(e[-15:], hop_s)
        f["e_drop"] = float(np.mean(e[-5:]) - np.mean(e))
        f["e_std"] = float(np.std(e))
        f["e_max"] = float(np.max(e))
        # local peaks: phrase-ish energy bumps before the pause
        if len(e) >= 7:
            peaks = (e[1:-1] > e[:-2]) & (e[1:-1] > e[2:]) & (e[1:-1] > np.median(e))
            f["e_peak_count"] = float(peaks.sum())
        mid = len(e) // 2
        if mid > 0:
            f["e_tail_vs_head"] = float(np.mean(e[mid:]) - np.mean(e[:mid]))

    # ---- pitch ----
    f0 = f0_contour_fast(seg, sr)
    v = f0 > 0
    if len(v):
        f["voiced_frac"] = float(v.mean())
    runs = _runs(v)
    f["n_voiced_runs"] = float(len(runs))
    if runs:
        ls, le = runs[-1]
        f["trail_unvoiced_ms"] = float((len(v) - le) * HOP_MS)
        f["last_voiced_run_ms"] = float((le - ls) * HOP_MS)
        mean_run = float(np.mean([(b - a) for a, b in runs]))
        f["final_lengthening"] = float((le - ls) / max(mean_run, 1e-6))
        last_run = f0[ls:le]
        f["f0_last"] = float(last_run[-1])
        f["f0_slope_lastrun"] = _slope(last_run, hop_s)
        k = min(len(last_run), 50)
        f["f0_slope_500"] = _slope(last_run[-k:], hop_s)
        vseg = f0[v]
        med = float(np.median(vseg))
        f["f0_med_seg"] = med
        f["f0_last_rel"] = float(np.log(max(f["f0_last"], 1e-3) / max(med, 1e-3)))
        f["f0_range_seg"] = float(np.percentile(vseg, 90) - np.percentile(vseg, 10))

    # ---- speaker-relative context ----
    ctx = x[max(0, t_end - int(CTX_S * sr)): t_end]
    f0c = f0_contour_fast(ctx, sr)
    vc = f0c[f0c > 0]
    if len(vc):
        f["f0_med_ctx"] = float(np.median(vc))
        if runs:
            f["f0_last_rel_ctx"] = float(
                np.log(max(f["f0_last"], 1e-3) / max(f["f0_med_ctx"], 1e-3)))
    ec = frame_energy_db(ctx, sr)
    if len(ec) and not np.isnan(f["e_last5"]):
        f["e_last_rel_ctx"] = float(f["e_last5"] - np.median(ec))

    # ---- spectral shape of the final moments ----
    tail = seg[-int(TAIL_S * sr):]
    if len(tail) >= 400:
        f["rms_tail"] = float(np.sqrt(np.mean(tail ** 2) + 1e-12))
        cent = librosa.feature.spectral_centroid(
            y=tail, sr=sr, n_fft=512, hop_length=160)[0]
        f["cent_mean"] = float(np.mean(cent))
        f["cent_slope"] = _slope(cent, hop_s)
        roll = librosa.feature.spectral_rolloff(
            y=tail, sr=sr, n_fft=512, hop_length=160)[0]
        f["rolloff_mean"] = float(np.mean(roll))
        f["rolloff_slope"] = _slope(roll, hop_s)
        zcr = librosa.feature.zero_crossing_rate(
            tail, frame_length=400, hop_length=160)[0]
        f["zcr_mean"] = float(np.mean(zcr))
        f["zcr_slope"] = _slope(zcr, hop_s)
        m = librosa.feature.mfcc(y=tail, sr=sr, n_mfcc=8, n_fft=512,
                                 hop_length=160)
        for i in range(8):
            f[f"mfcc{i}"] = float(np.mean(m[i]))
        if m.shape[1] >= 2:
            dm = np.diff(m, axis=1)
            for i in range(4):
                f[f"dmfcc{i}"] = float(np.mean(dm[i]))

    return f


def features_vec(fdict):
    return np.array([fdict[k] for k in FEATURE_NAMES], dtype=np.float32)
