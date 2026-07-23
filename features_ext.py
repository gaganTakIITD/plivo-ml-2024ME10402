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

FEATURE_VERSION = 4  # literature cues: flux, jitter/shimmer, semitone-z

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
] + [f"mfcc{i}" for i in range(8)] + [f"dmfcc{i}" for i in range(4)] + [
    # v4 literature-derived (causal)
    "cepstral_flux_tail",   # mean ||mfcc_t - mfcc_{t-1}|| in last ~400 ms
    "cepstral_flux_ratio",  # tail flux / earlier-seg flux
    "f0_jitter",            # mean |Δperiod| / mean period on last voiced run
    "amp_shimmer",          # mean |Δrms| / mean rms on last voiced run
    "f0_st_last",           # last F0 in semitones re: context median
    "f0_st_z_ctx",          # (f0_st_last - mean_ctx) / std_ctx
    "f0_st_slope",          # semitone slope on last voiced run
]

# column index groups for ablation scoring (relative to FEATURE_NAMES)
LIT_FLUX = ["cepstral_flux_tail", "cepstral_flux_ratio"]
LIT_JITTER = ["f0_jitter", "amp_shimmer"]
LIT_SEMITONE = ["f0_st_last", "f0_st_z_ctx", "f0_st_slope"]
LIT_ALL = LIT_FLUX + LIT_JITTER + LIT_SEMITONE
V2_NAMES = [n for n in FEATURE_NAMES if n not in LIT_ALL]


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
    mfcc_frames = None
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
        mfcc_frames = m
        for i in range(8):
            f[f"mfcc{i}"] = float(np.mean(m[i]))
        if m.shape[1] >= 2:
            dm = np.diff(m, axis=1)
            for i in range(4):
                f[f"dmfcc{i}"] = float(np.mean(dm[i]))

    # ---- v4: cepstral flux (phoneme lengthening cue) ----
    # Use MFCC frames over the full seg for ratio; hop 160 ~10 ms at 16 kHz
    if len(seg) >= int(0.5 * sr):
        m_seg = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=8, n_fft=512,
                                     hop_length=160)
        if m_seg.shape[1] >= 3:
            d = np.linalg.norm(np.diff(m_seg, axis=1), axis=0)  # per-step flux
            # last ~400 ms ≈ 40 frames at 10 ms; earlier = preceding same length
            n_tail = min(40, len(d))
            flux_tail = float(np.mean(d[-n_tail:]))
            f["cepstral_flux_tail"] = flux_tail
            if len(d) >= 2 * n_tail:
                flux_prev = float(np.mean(d[-2 * n_tail: -n_tail]))
                f["cepstral_flux_ratio"] = float(flux_tail / max(flux_prev, 1e-6))

    # ---- v4: jitter + shimmer on last voiced run ----
    if runs:
        ls, le = runs[-1]
        last_f0 = f0[ls:le]
        # frame RMS aligned to hop of energy frames — use seg energy near end
        if len(e) >= le and (le - ls) >= 3:
            # map voiced-run frame indices (F0 hop=10ms) onto energy frames
            e_run = e[max(0, ls): min(len(e), le)]
            if len(e_run) >= 3:
                # shimmer on linear-ish amplitude proxy: 10**(e/20)
                amp = np.power(10.0, e_run / 20.0)
                f["amp_shimmer"] = float(
                    np.mean(np.abs(np.diff(amp))) / max(np.mean(amp), 1e-9))
        voiced_f0 = last_f0[last_f0 > 0]
        if len(voiced_f0) >= 3:
            periods = 1.0 / voiced_f0
            f["f0_jitter"] = float(
                np.mean(np.abs(np.diff(periods))) / max(np.mean(periods), 1e-9))

    # ---- v4: semitone pitch z-normalized by context ----
    # ref = context median F0; convert last F0 and slope to semitones
    if len(vc) >= 3 and runs and not np.isnan(f["f0_last"]):
        ref = float(np.median(vc))
        st_ctx = 12.0 * np.log2(np.maximum(vc, 1e-3) / max(ref, 1e-3))
        st_last = 12.0 * np.log2(max(f["f0_last"], 1e-3) / max(ref, 1e-3))
        f["f0_st_last"] = float(st_last)
        mu, sd = float(np.mean(st_ctx)), float(np.std(st_ctx) + 1e-6)
        f["f0_st_z_ctx"] = float((st_last - mu) / sd)
        ls, le = runs[-1]
        last_f0 = f0[ls:le]
        last_f0 = last_f0[last_f0 > 0]
        if len(last_f0) >= 3:
            st_run = 12.0 * np.log2(np.maximum(last_f0, 1e-3) / max(ref, 1e-3))
            f["f0_st_slope"] = _slope(st_run, hop_s)

    return f


def frame_sequence(x, sr, pause_start, max_steps=60, step_ms=50):
    """Causal frame sequence for tiny GRU: [T, 4] = energy_db, voiced,
    semitone (re: context med), local cepstral flux. T<=max_steps (~3 s).
    """
    t_end = min(int(pause_start * sr), len(x))
    win = x[max(0, t_end - int(3.0 * sr)): t_end]
    T = max_steps
    feats = np.zeros((T, 4), dtype=np.float32)
    if len(win) < sr // 10:
        return feats
    hop = int(sr * step_ms / 1000)
    fl = hop  # non-overlapping-ish blocks
    # context median F0 for semitone ref
    f0_all = f0_contour_fast(win, sr)
    vc = f0_all[f0_all > 0]
    ref = float(np.median(vc)) if len(vc) else 150.0
    # mfcc over win for flux
    try:
        m = librosa.feature.mfcc(y=win, sr=sr, n_mfcc=8, n_fft=512,
                                 hop_length=max(hop // 5, 1))
        flux_full = np.zeros(m.shape[1], dtype=np.float32)
        if m.shape[1] >= 2:
            flux_full[1:] = np.linalg.norm(np.diff(m, axis=1), axis=0).astype(np.float32)
    except Exception:
        flux_full = np.zeros(1, dtype=np.float32)

    n_frames = max(1, (len(win) - fl) // hop + 1)
    rows = []
    for i in range(n_frames):
        s = i * hop
        chunk = win[s: s + fl]
        if len(chunk) < fl // 2:
            break
        edb = float(20 * np.log10(np.sqrt(np.mean(chunk ** 2) + 1e-12) + 1e-12))
        # f0 near this block
        f0i = f0_contour_fast(chunk, sr)
        voiced = 1.0 if np.any(f0i > 0) else 0.0
        f0m = float(np.median(f0i[f0i > 0])) if voiced else ref
        st = float(12.0 * np.log2(max(f0m, 1e-3) / max(ref, 1e-3)))
        # map to mfcc flux index
        fi = min(int(i * (len(flux_full) / max(n_frames, 1))), len(flux_full) - 1)
        rows.append([edb, voiced, st, float(flux_full[fi])])
    if not rows:
        return feats
    arr = np.asarray(rows, dtype=np.float32)
    if len(arr) >= T:
        feats[:] = arr[-T:]
    else:
        feats[-len(arr):] = arr
    return feats


def features_vec(fdict):
    return np.array([fdict[k] for k in FEATURE_NAMES], dtype=np.float32)

