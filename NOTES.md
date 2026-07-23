# NOTES

1. The model scores each pause with causal prosody: energy decay, F0 (Hz + semitone-z vs context), final lengthening, cepstral flux in the tail, jitter/shimmer on the last voiced run, spectral/MFCC shape, plus turn structure.
2. Silence-only endpointing fails because long mid-turn holds look identical to true ends; ranking by pre-pause speech shape unlocks lower delay at a fixed interrupt budget.
3. Final ship is a soft **3-way blend**: isotonic-calibrated ExtraTrees (0.30) + LogisticRegression (0.20) + isotonic-calibrated HistGradientBoosting (0.50), GroupKFold by turn, pooled en+hi.
4. Features never use audio after pause_start, never use the current pause_end/duration, and only use earlier pauses' boundaries (past events).
5. Hindi’s old 850 ms OOF delay was a **delay-floor** under the 5% cutoff (all EOTs already firing). Adding calibrated HGB into the blend was what finally moved Hindi (→ ~792 ms) by demoting long high-p HOLDs.
6. Literature v4 cues: cepstral flux (lengthening), jitter/shimmer (creak), semitone-z F0. Ablations: flux/semitone help; jitter alone hurts; the pack wins inside the 3-way blend (mean 912.2).
7. Tiny MLP (~963 ms) and GRU (EN AUC 0.64) did not beat the tree blend; CNN on log-mel was deferred to protect the ship window (documented in RUNLOG 7g).
8. Feature pruning (drop 18) + 3-way reached 942.5 ms — useful but worse than lit 3-way.
9. Hindi upweighting (1.5–3×) did not clear the gate without HI regression.
10. Human contribution: hypothesis priority and which OOF scores to trust; coding agent ran the feature/training/predict plumbing and SUMMARY.
