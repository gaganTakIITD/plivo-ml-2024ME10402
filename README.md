# plivo-ml-2024ME10402 — End-of-Turn Detection (STT track)

Predicts `p_eot` (probability that a silence pause is the true end of a
user's turn) for every annotated pause, from audio strictly before the
pause. Scored by `starter/score.py`: mean response delay at <= 5%
interrupted turns.

## Run on any data folder (handout schema)

```
python predict.py --data_dir <folder> --out predictions.csv
```

`<folder>` must contain `labels.csv` and the `audio/` it references.
Loads the committed `model.joblib`; no refitting.

## Reproduce training

```
python ship_lit_threeway.py
```

Rebuilds v4 features (if needed), runs nested GroupKFold OOF for the
0.30/0.20/0.50 iso-ET + LR + iso-HGB blend, and rewrites `model.joblib`
only if mean OOF delay beats the gate. Requires `blend_model.py` next to
`predict.py`.

Older recipes: `ship_blend_v2.py` (2-way), `ship_threeway_v2.py` (v2 3-way).

## Causality

For a pause starting at `t`: features use only `audio[0 : t]`, the pause's
index, and boundaries of EARLIER pauses (they ended before `t`). The
current pause's `pause_end`/`label` are never inputs. Enforced in
`eot_dataset.build_matrix` (a pause's boundaries are appended to the
`prior` list only AFTER its own features are extracted) and documented in
`features_ext.py`.

## Layout

- `features_ext.py` — causal prosodic features (energy decay, pitch slope,
  final lengthening, spectral tail, turn structure); vectorized FFT
  autocorrelation pitch tracker
- `eot_dataset.py` — labels.csv + wavs -> feature matrix (with caching)
- `train_model.py` — grouped OOF evaluation + final model -> `model.joblib`
- `predict.py` — official CLI (above)
- `error_analysis.py` — worst holds/eots for listening
- `starter/` — unmodified handout starter kit incl. official `score.py`
- `RUNLOG.md`, `NOTES.md`, `SUMMARY.html` — required deliverables
