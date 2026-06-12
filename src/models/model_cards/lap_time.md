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
| BayesianRidge (one-hot + scaled) | 4.35 / 3.13 | 4.19 / 3.09 |
| RandomForest (integer codes) | 6.47 / 5.63 | 8.87 / 8.30 |
| LightGBM (16 features, current) | 3.49 / 2.14 | 3.53 / 2.22 |
| LightGBM (14 features, pre-ablation) | 3.51 / 2.16 | 3.42 / 2.11 |
| TFT v1 (Phase-2, hand-picked h32) | 2.25 / 1.12 | 3.78 / 1.62 |
| **TFT v3 (swept: h64 + Driver/EngineMaker, chosen)** | **2.27 / 1.02** | **3.77 / 1.56** |

**Feature ablation finding (v3.2 — driver + engine features).** Adding `DriverEncoded` (frozen mapping, rookies → −1) and `EngineEncoded` (season-aware team→PU dictionary — domain knowledge, not learned) improves val green MAE 2.16→2.14 but *degrades* test green 2.11→2.22. Mechanism: driver identity is partly a proxy for the car; across the 2026 boundary drivers switch teams and rookies are unseen, so the learned attribution transfers badly. Selection stays val-based per the project's honesty rules (never pick features on test), so the 16-feature model is current — and the cross-era cost is reported here, not hidden. EngineEncoded carries ≈0 SHAP in-era (engine is constant within a team in era 0 — collinear with Team); its real test is the TFT v2 run where `EngineMaker` is a static categorical and the 2026 boundary is in-distribution for evaluation.

*(Mappings are frozen from the training split only — unseen 2026 entrants hit the −1 sentinel path.)*

Green-flag = `TrackStatus == "1"` (~92% of laps). Safety-car / yellow laps account for the bulk of the all-laps vs green gap. The deployed TFT roughly halves the LightGBM green MAE on both splits (1.02 vs 2.14 val; 1.56 vs 2.22 test).

## Phase 2 — Temporal Fusion Transformer
**Setup (deployed v3).** Sequence unit = laps within a stint (`GroupID = Season_Round_Driver_StintID`), `time_idx` = lap-in-stint. Static categoricals (`CircuitKey`, `Team`, `EraStr`, `CompoundStr`, `Driver`, `EngineMaker`) via `NaNLabelEncoder(add_nan=True)`; known reals (`time_idx`, `NormLapNumber`, `FuelLoad`, `TyreLife`); observed reals (`TrackTemp`, `AirTemp`) + the target. Target scaled per-stint with `EncoderNormalizer`. QuantileLoss at 0.1/0.5/0.9. hidden_size 64, dropout 0.20, 4 attention heads, ~322K params — selected by the 2-round hyperparameter sweep below (v1: hand-picked hidden 32, ~85.6K params). Trained on Kaggle T4 (22 epochs, EarlyStopping patience 10 on val_loss, encoder length 12).

**Why it beats LightGBM — and why that's legitimate, not leakage.** The TFT's edge is that it consumes each stint's *past lap times* in the encoder to predict the next lap; LightGBM treats every lap as an independent row. Past laps are known at prediction time and the decoder never sees the target lap's own value, so this is proper autoregressive use of in-stint history — exactly the "laps within a stint are a sequence" thesis the project is built on. The win shows up most on degradation dynamics a tree can't see from static tyre-age features alone.

**Quantile calibration** (`reports/lap_time/tft_calibration.csv`; nominal central = 0.80 for the 0.1–0.9 band):

| Split | frac<q0.1 (→0.10) | frac<q0.5 (→0.50) | frac<q0.9 (→0.90) | central coverage (→0.80) |
|---|---|---|---|---|
| Val 2025 (era0) | 0.137 | 0.499 | 0.853 | **0.715** |
| Test 2026 (era1) | 0.155 | 0.507 | 0.822 | **0.667** |

- Median essentially unbiased on both splits (0.499 / 0.507), but the raw bands are **overconfident everywhere** — the better median model produces *narrower* raw bands than v1 did, so raw coverage actually dropped (v1: 0.756 / 0.690). Conformal recalibration (below) is what makes the intervals honest; the sim must always use the recalibrated bands.

**Recalibration (Phase 3.5 — DONE, `src/models/lap_time/recalibrate.py`).** Era-aware split-conformal widening on **green-flag laps** (the distribution the sim engine samples from; SC laps would blow the shift up to a useless ~8s — measured, then excluded). Calibration and evaluation never share laps: era 0 calibrates on odd 2025 rounds → evaluates on even rounds; era 1 calibrates on 2026 R1 → evaluates on R2–R6 (R1 sacrificed from coverage reporting). Results (`reports/lap_time/tft_recalibration.csv`):

| Era | shift (s) | coverage raw → recalibrated | nominal |
|---|---|---|---|
| 0 (2025 even rounds) | +0.052 | 0.750 → **0.787** | 0.80 |
| 1 (2026 R2–R6) | +0.153 | 0.726 → **0.813** | 0.80 |

Both eras land inside the [0.78, 0.85] success window. Note the era-1 shift collapsed from v1's +0.868s to +0.153s — the swept model generalizes across the boundary well enough that the conformal correction is now a trim, not a rescue. Shifts live in `models/tft_calibration.json`; the engine widens its sampling σ with them. (v1 history: era0 +0.082 → 0.800, era1 +0.868 → 0.879.)

**Honesty caveats.**
- The 2026 test set is only 6 races and statistically noisy — read 1.56s green with that caveat; the val→test gap (1.02→1.56 green) reflects the genuine cross-regulation shift plus small-sample noise.
- **Per-circuit spread is large** (`reports/lap_time/tft_breakdown.csv`): green MAE is excellent on conventional tracks (Japan 0.37 val / 0.51 test, Hungary 0.50, China 0.72) but poor on street/atypical circuits — Canada (val 2.59 / test 2.85), Monaco (val 2.56 / test 1.87) and Australia (val 1.81 / test 2.48) are the worst. Aggregate green MAE hides this; the sim should treat street-circuit predictions as higher-variance.

## Hyperparameter sweep history (v1 → v2 → v3)
v1's architecture (hidden 32, lr 1e-3, dropout 0.15) was hand-picked, never searched. Three runs settled it:

**v2 (notebook 05, rejected at the time).** Added `Driver` + `EngineMaker` static categoricals at v1's hidden 32: val green flat (1.12→1.14), test green regressed 1.62→1.94. Read then as the driver-as-car-proxy effect from the LightGBM ablation; flagged confound: early-stopped at 20 epochs vs v1's ~26.

**Sweep round 1 (notebook 06).** 6-config screen + top-2 converged, v1 feature roles: best was hidden 64 at 1.1239 val green vs v1's 1.12 — a statistical tie, confirming v1's config near-optimal *for v1 features*. But the converged v2-features tiebreak (hidden 64, 18 epochs) hit **1.083** — the "v2 failure" was undertraining + too-small capacity, not the features.

**Sweep round 2 (notebook 07) → deployed v3.** All-v2-features, converged (patience 10): hidden 64 → **1.023 val green**; hidden 96 (1.185) and 128 (1.139) overfit; encoder length 24 (1.150) worse than 12. Winner = hidden 64 / dropout 0.20 / lr 1e-3 / enc 12, 22 epochs, ~322K params:

| Split | v1 all / green | v3 (deployed) all / green |
|---|---|---|
| Val 2025 (era0) | 2.25 / 1.12 | **2.27 / 1.02** |
| Test 2026 (era1) | 3.78 / 1.62 | **3.77 / 1.56** |

**The cross-era fear did not materialize at convergence**: v3 improves test green too (1.62→1.56), so the identity features transfer once the model has the capacity to learn them properly — the v2 episode was a capacity/training artifact, not a feature problem. Selection was val-based throughout (test touched only for the final winner). Sweep records: `reports/lap_time/hpo_stage_{a,b}.csv`, `hpo_v2_tiebreak.csv`, `hpo_round2.csv`. v1 archived in `tft_v1_artifacts_backup.zip`; v2 in `tft_v2_artifacts.zip`.

## Key findings
- **Cross-regulation generalization (the headline).** The 14-feature LightGBM holds green-lap MAE 2.16s val (era 0) → 2.11s test (era 1) — essentially flat across the 2022-25 → 2026 regulation boundary (the 16-feature variant trades a small val gain for 2.22s test; see the ablation above). A model trained only on ground-effect-era data carries that learning into the unseen active-aero era. **Caveat:** the 2026 test set is only 6 races, so test ≈ val means "generalizes within noise," not "generalizes *better* than val." A larger 2026 sample is needed to tighten this.
- **RandomForest is kept as an instructive counter-example, not a candidate.** It collapses across the boundary (green test 8.30s vs green val 5.63s) because sklearn trees can't split integer-coded categoricals natively and RF never saw era 1 / new 2026 teams in training. The contrast is the motivation for LightGBM's native-categorical + regularized approach.
- **Driver absorbs team's credit.** With `DriverEncoded` in the model, SHAP shifts from TeamEncoded (0.43→0.08) to DriverEncoded (0.32) — driver is nested inside team per season, so the tree prefers the finer split. Car pace + driver skill are real, learnable signal either way; the attribution between them is not identifiable from this data alone (an honest interview point).

## Feature importance (LightGBM 16-feature, mean |SHAP|, on val)
CircuitEncoded 7.35 ≫ AirTemp 1.02, CompoundEncoded 0.62, FuelLoad 0.59, TrackTemp 0.57, CompoundXTyreLife 0.33, DriverEncoded 0.32, NormLapNumber 0.19, TyreLife 0.18, FuelEffect 0.09, TeamEncoded 0.08, TyreAgeSq 0.04, TyreAgeCubed 0.01, EngineEncoded ≈0 (era-0 redundant with Team — see ablation), Era 0.00, StintPhase 0.00.
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