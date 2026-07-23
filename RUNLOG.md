# RUNLOG — End-of-Turn Detection (STT track)

Metric: **mean response delay (ms) at <= 5% interrupted turns** via the
untouched `starter/score.py`. Lower is better. AUC is the ranking
diagnostic. All model rows are **out-of-fold** (5-fold GroupKFold by
turn_id over pooled english+hindi) — no in-sample scoring anywhere.

| # | run | EN ms | EN AUC | HI ms | HI AUC | change / why |
|---|-----|-------|--------|-------|--------|--------------|
| 0a | silence baseline (p=1) | ~1600 | ~0.5 | ~1600 | ~0.5 | status quo: every pause looks like EOT; agent waits on a silence timer |
| 1a | features v1 + LogisticRegression | 1213 | 0.647 | 850 | 0.669 | first real model: 42 causal prosodic features (energy decay, F0 slope/level, final lengthening, spectral tail, turn structure) |
| 1b | features v1 + HistGradientBoosting | 1225 | 0.679 | 767 | 0.708 | same features, tree model; better AUC + better Hindi (mean 996 ms) |
| 2 | HGB hyperparam grid (24 configs) | — | — | — | — | best grid point mean ~1000 ms; no gain over 1b → abandoned |
| 3a | features v2 + LogReg C=1 | 1135 | 0.671 | 850 | 0.687 | added early-turn markers (is_first_pause, speech_so_far, pause_rate), energy peaks, rolloff, n_voiced_runs — error analysis showed many missed EOTs are pause_index=0 |
| 3b | features v2 + ExtraTrees | 1125 | 0.734 | 850 | 0.718 | previous ship: mean **987.5 ms** |
| 4a | diagnose Hindi 850 floor | — | — | 850 | 0.718 | at best op-point latency==delay, **100% EOT fire**, cut=5%; floor = min safe delay under interrupt budget (binding long high-p HOLDs), not missed EOTs |
| 4b | v2 ExtraTrees + isotonic calib (nested GroupKFold) | 1134 | 0.719 | 850 | 0.703 | calib alone: mean 992; better EN delay ordering, HI still glued at 850 |
| 4c | v2 soft blend 0.7 iso-ET + 0.3 LogReg | **1078** | 0.716 | **850** | 0.707 | **selected**: mean **964.0 ms** (beats 987.5 gate) |
| 5a | features v3 (+trail_rms_ratio, hnr_drop_200, speech_burst) + ET | 1134 | 0.730 | 850 | 0.723 | tight EN-oriented cues; mean 992 — no gate beat |
| 5b | v3 ET isotonic / blends | ≥1126 | — | 850 | — | best v3 mean **988.0** (fails gate by 0.5 ms) → not shipped |

## Error-analysis notes

- High-p HOLD pauses are often short (0.1–0.5 s); the scorer already forgives fires when delay ≥ pause duration.
- Low-p EOT misses cluster on early / first pauses — short complete turns look like mid-sentence holds. v2 features target that.
- Hindi delay stuck at 850 ms even with AUC ~0.72: with t≈0.05 every EOT fires, so latency equals the delay needed to keep ≤5 long high-p HOLDs from counting as interrupts. Moving Hindi requires pushing those HOLDs down in rank, not recovering timeouts.
- Hindi scores better than English on AUC throughout; hidden test is mostly Hindi.

## Final selection

`IsoETBlend` on features v2: nested GroupKFold isotonic-calibrated ExtraTrees soft-averaged with LogReg (w=0.7 / 0.3), trained on pooled en+hi, shipped as `model.joblib` (+ `blend_model.py`).
Honest OOF: EN **1078 ms** / AUC 0.716, HI **850 ms** / AUC 0.707, mean **964.0 ms**.
