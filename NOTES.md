# NOTES

1. The model scores each pause with causal prosody: energy decay into the pause, F0 slope/level and final lengthening, spectral shape of the last ~0.6 s, plus turn structure (pause index, speech so far, prior pause timing).
2. Silence-only endpointing fails because long mid-turn holds look identical to true ends; ranking by pre-pause speech shape is what unlocks lower delay at a fixed interrupt budget.
3. We train ExtraTrees on pooled English+Hindi pauses and evaluate with 5-fold GroupKFold by turn so no turn leaks across folds.
4. Features never use audio after pause_start, never use the current pause_end/duration, and only use earlier pauses' boundaries (past events).
5. Error analysis showed many missed EOTs are first pauses of short turns; v2 added is_first_pause, speech_so_far, pause_rate, energy peak count.
6. Hyperparameter grids on HistGradientBoosting did not beat a well-regularized ExtraTrees on this metric.
7. Hindi is easier than English here (higher OOF AUC); the hidden set is mostly Hindi, so we kept pooled training rather than English-only features.
8. The model still fails on short complete answers that end with rising or flat pitch, and on long holds after list-like intonation.
9. With one more day: add lightweight phone/duration proxies without ASR, calibrate probabilities for the 5% cutoff operating point, and try a tiny causal CNN on the last 1.5 s spectrogram under the same CPU/no-pretrained rules.
10. Human contribution was hypothesis choice and which scores to trust; the coding agent wrote the feature/training plumbing and SUMMARY.html.
