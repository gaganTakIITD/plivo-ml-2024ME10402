# NOTES

1. The model scores each pause with causal prosody: energy decay into the pause, F0 slope/level and final lengthening, spectral shape of the last ~0.6 s, plus turn structure (pause index, speech so far, prior pause timing).
2. Silence-only endpointing fails because long mid-turn holds look identical to true ends; ranking by pre-pause speech shape is what unlocks lower delay at a fixed interrupt budget.
3. We train on pooled English+Hindi pauses and evaluate with 5-fold GroupKFold by turn so no turn leaks across folds. Final ship is a soft blend of isotonic-calibrated ExtraTrees (w=0.7) and LogisticRegression (w=0.3).
4. Features never use audio after pause_start, never use the current pause_end/duration, and only use earlier pauses' boundaries (past events).
5. Error analysis showed many missed EOTs are first pauses of short turns; v2 added is_first_pause, speech_so_far, pause_rate, energy peak count.
6. Hindi’s 850 ms OOF delay is a **delay-floor under the 5% cutoff**: at the best operating point every EOT already fires, so latency equals the minimum safe action delay. A handful of long high-p HOLDs bind the budget.
7. Probability calibration (isotonic inside each GroupKFold) plus a LogReg blend cut English delay (1125 → 1078 ms) without moving Hindi; mean 964 ms.
8. A tight features-v3 bump (trailing RMS ratio, HNR drop, speech-burst length) was tried; best OOF mean 988 ms — did not beat the 987.5 gate, so not shipped.
9. With one more day: attack the binding Hindi long-holds with list-intonation / relative-peak energy, and keep calibration in the training loop from the start.
10. Human contribution was hypothesis choice and which OOF scores to trust; the coding agent wrote the feature/training plumbing and SUMMARY.html.
