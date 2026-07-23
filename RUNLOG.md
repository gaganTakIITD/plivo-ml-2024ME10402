# RUNLOG — End-of-Turn Detection (STT track)

Metric: **mean response delay (ms) at <= 5% interrupted turns** via the
untouched `starter/score.py`. Lower is better. AUC is the ranking
diagnostic. All model rows are **out-of-fold** (5-fold GroupKFold by
turn_id over pooled english+hindi) — no in-sample scoring anywhere.

| # | run | EN ms | EN AUC | HI ms | HI AUC | change / why |
|---|-----|-------|--------|-------|--------|--------------|
| 0a | silence baseline (p=1) | ~1600 | ~0.5 | ~1600 | ~0.5 | status quo: every pause looks like EOT; agent waits on a silence timer |
| 1a | features v1 + LogisticRegression | 1213 | 0.647 | 850 | 0.669 | first real model: 42 causal prosodic features |
| 1b | features v1 + HistGradientBoosting | 1225 | 0.679 | 767 | 0.708 | same features, tree model; mean 996 ms |
| 2 | HGB hyperparam grid (24 configs) | — | — | — | — | no gain over 1b → abandoned |
| 3a | features v2 + LogReg C=1 | 1135 | 0.671 | 850 | 0.687 | early-turn markers + energy peaks + rolloff |
| 3b | features v2 + ExtraTrees | 1125 | 0.734 | 850 | 0.718 | prior ExtraTrees-only ship: mean 987.5 |
| 4a | diagnose Hindi 850 floor | — | — | 850 | 0.718 | latency==delay, 100% EOT fire, cut=5%; delay-floor from long high-p HOLDs |
| 4b | v2 ExtraTrees + isotonic calib | 1134 | 0.719 | 850 | 0.703 | mean 992 |
| 4c | v2 blend 0.7 iso-ET + 0.3 LogReg | 1078 | 0.716 | 850 | 0.707 | prior blend ship: mean **964.0** |
| 5a | features v3 (RMS ratio/HNR/burst) + ET | 1134 | 0.730 | 850 | 0.723 | mean 992 — no ship |
| 5b | v3 ET iso / blends | ≥1126 | — | 850 | — | best 988 — fails prior gate |
| 6a | ET-iso/LR weight grid (0–1 step 0.1) | best@0.7 | — | 850 | — | only w=0.7 hits 964; neighbours worse |
| 6b | 3-way iso-ET + LR + iso-HGB fine grid | **1123** | 0.719 | **784** | 0.713 | v2 best 0.50/0.20/0.30 → mean **953.5** (beats 964) |
| 6c | Hindi sample_weight 1.5–3× | ≥977 | — | ≥801 | — | no mean <964 with HI≤850; rejected |
| 6d | prune 18 lowest tree-importance feats + 3-way | 1058 | 0.722 | 826 | 0.721 | mean **942.5** (0/0.2/0.8) — strong but not best |
| 6e | tiny MLP (2×64/32, dropout) on v2 | 1068 | 0.691 | 857 | 0.688 | mean 962.7; 4-way mixes ≥965 |
| 7a | +cepstral flux only + ET-iso | 1112 | 0.733 | 850 | 0.689 | mean 981 — flux helps EN AUC, not delay floor |
| 7b | +jitter/shimmer only + ET-iso | 1163 | 0.706 | 850 | 0.694 | mean 1006 — hurts |
| 7c | +semitone-z F0 only + ET-iso | 1087 | 0.718 | 850 | 0.702 | mean 968.5 — mild EN help |
| 7d | lit pack (flux+jitter+semitone) + ET-iso | 1204 | 0.712 | 850 | 0.713 | mean 1027 alone; LogReg 958.7 |
| 7e | **lit pack + 3-way 0.30/0.20/0.50** | **1032** | **0.720** | **792** | **0.715** | **selected**: mean **912.2** |
| 7f | tiny GRU (h=32) on 50 ms frames | 1185 | 0.640 | 850 | 0.696 | EN AUC <0.65; HI ok; +GRU blends 955–962.5 — no beat over 7e |
| 7g | ambitious 1D CNN on log-mel (planned) | — | — | — | — | **not run**: remaining clock reserved for shipping 7e; with n=496 a from-scratch CNN on 1.5 s mel would need careful regularization and risk missing the push window. Would try next with GroupKFold + heavy dropout / SpecAugment-lite, expecting high variance before it beats a calibrated tree blend. |

## Error-analysis notes

- Hindi delay-floor: once every EOT fires, latency = min delay that keeps ≤5% long high-p HOLDs from interrupting. 3-way blends that include HGB push those HOLDs down → HI 850→784–792.
- Literature cues: semitone-z and flux help ranking; jitter/shimmer alone overfit; the pack shines inside a calibrated 3-way blend, not as ET-only.
- GRU got usable HI AUC but weak EN AUC; not worth the predict.py complexity vs 7e.

## Final selection

`ThreeWayBlend` on **features v4** (v2 + cepstral flux, jitter/shimmer, semitone-z):
0.30 isotonic-ExtraTrees + 0.20 LogReg + 0.50 isotonic-HGB.
Honest OOF: EN **1032 ms** / AUC 0.720, HI **792 ms** / AUC 0.715, mean **912.2 ms**.
