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
| 3b | features v2 + ExtraTrees | 1125 | 0.734 | 850 | 0.718 | **selected**: mean **987.5 ms**; highest English AUC; RF was 1015 ms, LogReg C grid did not beat C=1 |

## Error-analysis notes (after run 1b)

- High-p HOLD pauses are often short (0.1–0.5 s); the scorer already forgives fires when delay ≥ pause duration.
- Low-p EOT misses cluster on early / first pauses — short complete turns look like mid-sentence holds. v2 features target that.
- Hindi scores better than English throughout; hidden test is mostly Hindi, which is favorable but we still train pooled.

## Final selection

`ExtraTreesClassifier` on features v2, trained on pooled en+hi, shipped as `model.joblib`.
Honest OOF: EN 1125 ms / AUC 0.734, HI 850 ms / AUC 0.718.
