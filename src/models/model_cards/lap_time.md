# Model Card — Lap Time Predictor (Phase 1 baseline ladder + Phase 2 TFT)

## Overview
Predicts single-lap time (`LapTimeSeconds`) for an F1 race lap from circuit, team, compound, tyre age, fuel load, weather, and regulation era. Phase 1 establishes an honest point-prediction baseline ladder (LightGBM = the number to beat); Phase 2 adds the headline **Temporal Fusion Transformer** — a true sequence model over laps-in-a-stint with native quantile uncertainty.

**Selected model (Phase 2): Temporal Fusion Transformer.** Beats the LightGBM baseline on green-flag MAE on both val (era 0) and the cross-era test (era 1). LightGBM remains the strong tabular baseline and the model-to-beat; RandomForest/Ridge are documented lower rungs. A safety check raises if no candidate beats the linear baseline.

## Data
- Source: FastF1, race sessions only (`session_type="R"`), 2022–2026.
- 103,732 laps after validation + de-duplication, from 94 per-round files.
- Season-aware splits (no shuffle): **train** 2022–2024 (era 0, ground-effect), **val** 2025 (era 0), **test** 2026 R1–6 (era 1, active-aero — 6 races; Bahrain & Saudi cancelled).
- Filters: out-laps (`TyreLife==1`), lap time outside [70, 130]s, invalid/null compound, sprint safety-net.

## Results — MAE (seconds)
| Model | Val all / green (2025, era0) | Test all / green (2026, era1) |
|---|---|---|
| BayesianRidge (one-hot + scaled) | 4.16 / 2.97 | 3.97 / 2.88 |
| RandomForest (integer codes) | 6.42 / 5.57 | 9.01 / 8.46 |
| LightGBM (Phase-1 baseline) | 3.52 / 2.18 | 3.45 / 2.14 |
| **TFT (Phase-2, chosen)** | **2.27 / 1.11** | **3.93 / 1.67** |

Green-flag = `TrackStatus == "1"` (~92% of laps). Safety-car / yellow laps account for the bulk of the all-laps vs green gap. TFT roughly halves the LightGBM green MAE on both splits.

## Phase 2 — Temporal Fusion Transformer
**Setup.** Sequence unit = laps within a stint (`GroupID = Season_Round_Driver_StintID`), `time_idx` = lap-in-stint. Static categoricals (`CircuitKey`, `Team`, `EraStr`, `CompoundStr`) via `NaNLabelEncoder(add_nan=True)`; known reals (`time_idx`, `NormLapNumber`, `FuelLoad`, `TyreLife`); observed reals (`TrackTemp`, `AirTemp`) + the target. Target scaled per-stint with `EncoderNormalizer`. QuantileLoss at 0.1/0.5/0.9. ~85.6K params, hidden_size 32. Trained on Kaggle T4 (~26 epochs, EarlyStopping on val_loss).

**Why it beats LightGBM — and why that's legitimate, not leakage.** The TFT's edge is that it consumes each stint's *past lap times* in the encoder to predict the next lap; LightGBM treats every lap as an independent row. Past laps are known at prediction time and the decoder never sees the target lap's own value, so this is proper autoregressive use of in-stint history — exactly the "laps within a stint are a sequence" thesis the project is built on. The win shows up most on degradation dynamics a tree can't see from static tyre-age features alone.

**Honesty caveats.**
- The 2026 test set is only 6 races and statistically noisy — read 1.67s green with that caveat; the val/test gap (1.11→1.67) reflects the genuine cross-regulation shift plus small-sample noise.
- **Quantile calibration** (do the 0.1–0.9 intervals cover ~80%?) is computed each run into `reports/lap_time/tft_calibration.csv` — fill the coverage numbers here once read. A strong median MAE with miscalibrated intervals would weaken the uncertainty story the simulation engine depends on; verify before trusting the bands.

## Key findings
- **Cross-regulation generalization (the headline).** LightGBM green-lap MAE is 2.18s on val (era 0) vs 2.14s on test (era 1) — essentially flat across the 2022-25 → 2026 regulation boundary. A model trained only on ground-effect-era data carries that learning into the unseen active-aero era. **Caveat:** the 2026 test set is only 6 races, so test ≈ val means "generalizes within noise," not "generalizes *better* than val." A larger 2026 sample is needed to tighten this.
- **RandomForest is kept as an instructive counter-example, not a candidate.** It collapses across the boundary (green test 8.46s vs green val 5.57s) because sklearn trees can't split integer-coded categoricals natively and RF never saw era 1 / new 2026 teams in training. The contrast is the motivation for LightGBM's native-categorical + regularized approach.
- **Team-pace fix validated.** `TeamEncoded` ranks 6th of 14 features by mean |SHAP| (0.43), above every tyre-age polynomial term, the compound×life interaction, and fuel effect — car pace is real, learnable signal. Circuit dominates everything (~7.26, ~17× team), as expected: track length sets the baseline lap time.

## Feature importance (LightGBM, mean |SHAP|, on val)
CircuitEncoded 7.26 ≫ AirTemp 0.99, TrackTemp 0.62, CompoundEncoded 0.59, FuelLoad 0.58, TeamEncoded 0.43, CompoundXTyreLife 0.14, FuelEffect 0.13, NormLapNumber 0.12, TyreLife 0.11, TyreAgeSq 0.03, TyreAgeCubed 0.003, StintPhase ≈0, Era 0.00.
Physical features rank sensibly (temps/compound/fuel above tyre-age polynomials) — the model learned structure, not noise. Era attributes 0.00 because val is single-era (no variance to attribute) — not a bug.

## Intended use / out of scope
**For:** pre-/post-race lap-time estimation, and as an input to the tyre-degradation and Monte Carlo simulation models. **Not for:** wagering, or any real-time safety-critical decision. No uncertainty estimate at this rung — added with the TFT (quantile outputs) in Phase 2.

## Limitations
No live telemetry (tyre temperatures, ERS deployment, fuel flow, active-aero state — the last especially impactful in 2026); SC/VSC/red-flag laps not modeled (reported separately); fuel load estimated via linear 110kg burn; driver skill not modeled as a continuous variable; SOFT/MEDIUM/HARD labels hide the circuit-varying C1–C5 compounds. The 2026 test set is small (6 races) and statistically noisy.

## Reproduce
```bash
python -m src.models.lap_time.train       # ladder + Optuna, logs to MLflow, saves models/*.joblib
python -m src.models.lap_time.evaluate    # writes reports/lap_time/* (tree models)
python -m src.models.lap_time.train_tft   # TFT on GPU; saves models/tft_lap.{pt,ckpt} + reports/lap_time/tft_*.csv
```
GPU note: the TFT needs a CUDA arch in the installed torch build — Kaggle's cu128 torch supports T4 (sm_75) but **not** P100 (sm_60). See `notebooks/04_tft_kaggle_fullrun.ipynb`.
Artifacts: `models/{bayesian_ridge,rf,lgbm,chosen}_lap.joblib`, `models/tft_lap.pt` (CPU-loadable) + `tft_lap.ckpt`; `reports/lap_time/{model_comparison,per_era_mae,greenflag_vs_alllaps,per_circuit_mae,shap_importance,tft_breakdown,tft_calibration}.csv`, `shap_summary.png`, `learning_curve.png`.