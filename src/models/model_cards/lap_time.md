# Model Card — Lap Time Predictor (Phase 1: baseline ladder)

## Overview
Predicts single-lap time (`LapTimeSeconds`) for an F1 race lap from circuit, team, compound, tyre age, fuel load, weather, and regulation era. Phase 1 establishes an honest point-prediction baseline ladder; the headline sequence model (Temporal Fusion Transformer, with calibrated uncertainty) follows in Phase 2.

**Selected model:** LightGBM (Optuna-tuned, native categorical handling). Chosen as the lowest-val-MAE candidate; beats the linear baseline on every split. A safety check raises if no candidate beats the baseline.

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
| **LightGBM (chosen)** | **3.52 / 2.18** | **3.45 / 2.14** |

Green-flag = `TrackStatus == "1"` (91.8% of val laps, 88.5% of test laps). Safety-car / yellow laps account for ~1.3s of the all-laps error.

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
python -m src.models.lap_time.train      # ladder + Optuna, logs to MLflow, saves models/*.joblib
python -m src.models.lap_time.evaluate   # writes reports/lap_time/*
```
Artifacts: `models/{bayesian_ridge,rf,lgbm,chosen}_lap.joblib`; `reports/lap_time/{model_comparison,per_era_mae,greenflag_vs_alllaps,per_circuit_mae,shap_importance}.csv`, `shap_summary.png`, `learning_curve.png`.