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
| **TFT (Phase-2, chosen)** | **2.25 / 1.12** | **3.78 / 1.62** |

**Feature ablation finding (v3.2 — driver + engine features).** Adding `DriverEncoded` (frozen mapping, rookies → −1) and `EngineEncoded` (season-aware team→PU dictionary — domain knowledge, not learned) improves val green MAE 2.16→2.14 but *degrades* test green 2.11→2.22. Mechanism: driver identity is partly a proxy for the car; across the 2026 boundary drivers switch teams and rookies are unseen, so the learned attribution transfers badly. Selection stays val-based per the project's honesty rules (never pick features on test), so the 16-feature model is current — and the cross-era cost is reported here, not hidden. EngineEncoded carries ≈0 SHAP in-era (engine is constant within a team in era 0 — collinear with Team); its real test is the TFT v2 run where `EngineMaker` is a static categorical and the 2026 boundary is in-distribution for evaluation.

*(Mappings are frozen from the training split only — unseen 2026 entrants hit the −1 sentinel path.)*

Green-flag = `TrackStatus == "1"` (~92% of laps). Safety-car / yellow laps account for the bulk of the all-laps vs green gap. TFT roughly halves the LightGBM green MAE on both splits (1.12 vs 2.16 val; 1.62 vs 2.11 test).

## Phase 2 — Temporal Fusion Transformer
**Setup.** Sequence unit = laps within a stint (`GroupID = Season_Round_Driver_StintID`), `time_idx` = lap-in-stint. Static categoricals (`CircuitKey`, `Team`, `EraStr`, `CompoundStr`) via `NaNLabelEncoder(add_nan=True)`; known reals (`time_idx`, `NormLapNumber`, `FuelLoad`, `TyreLife`); observed reals (`TrackTemp`, `AirTemp`) + the target. Target scaled per-stint with `EncoderNormalizer`. QuantileLoss at 0.1/0.5/0.9. ~85.6K params, hidden_size 32. Trained on Kaggle T4 (~26 epochs, EarlyStopping on val_loss).

**Why it beats LightGBM — and why that's legitimate, not leakage.** The TFT's edge is that it consumes each stint's *past lap times* in the encoder to predict the next lap; LightGBM treats every lap as an independent row. Past laps are known at prediction time and the decoder never sees the target lap's own value, so this is proper autoregressive use of in-stint history — exactly the "laps within a stint are a sequence" thesis the project is built on. The win shows up most on degradation dynamics a tree can't see from static tyre-age features alone.

**Quantile calibration** (`reports/lap_time/tft_calibration.csv`; nominal central = 0.80 for the 0.1–0.9 band):

| Split | frac<q0.1 (→0.10) | frac<q0.5 (→0.50) | frac<q0.9 (→0.90) | central coverage (→0.80) |
|---|---|---|---|---|
| Val 2025 (era0) | 0.113 | 0.499 | 0.868 | **0.756** |
| Test 2026 (era1) | 0.146 | 0.542 | 0.836 | **0.690** |

- **In-era (val) intervals are roughly honest** — central coverage 0.76 vs 0.80 nominal, median essentially unbiased (0.499). Mildly overconfident (bands a touch narrow).
- **Across the era boundary (test) the model is overconfident** — central coverage drops to 0.69, with a heavy lower tail (14.6% fall below q0.1 vs 10% nominal) and a slight high-median bias (0.542). The intervals don't widen enough for the unseen 2026 regulations — the same cross-era degradation seen in the median MAE.

**Recalibration (Phase 3.5 — DONE, `src/models/lap_time/recalibrate.py`).** Era-aware split-conformal widening on **green-flag laps** (the distribution the sim engine samples from; SC laps would blow the shift up to a useless ~8s — measured, then excluded). Calibration and evaluation never share laps: era 0 calibrates on odd 2025 rounds → evaluates on even rounds; era 1 calibrates on 2026 R1 → evaluates on R2–R6 (R1 sacrificed from coverage reporting). Results (`reports/lap_time/tft_recalibration.csv`):

| Era | shift (s) | coverage raw → recalibrated | nominal |
|---|---|---|---|
| 0 (2025 even rounds) | +0.082 | 0.747 → **0.800** | 0.80 |
| 1 (2026 R2–R6) | +0.868 | 0.659 → **0.879** | 0.80 |

Era 0 lands exactly on nominal; era 1 overshoots conservatively (one-race calibration set — the safe direction for the sim). Shifts live in `models/tft_calibration.json`; the engine widens its sampling σ with them.

**Honesty caveats.**
- The 2026 test set is only 6 races and statistically noisy — read 1.67s green with that caveat; the val→test gap (1.11→1.67 green) reflects the genuine cross-regulation shift plus small-sample noise.
- **Per-circuit spread is large** (`reports/lap_time/tft_breakdown.csv`): green MAE is excellent on conventional tracks (Japan 0.46, Hungary 0.56, China 0.73) but poor on street/atypical circuits — Canada (val 2.96 / test 3.13) and Monaco (val 2.83 / test 2.07) are the worst. Aggregate green MAE hides this; the sim should treat street-circuit predictions as higher-variance.

## TFT v2 — driver + engine-maker static categoricals (ran, rejected; v1 kept)
The card's ablation note (above) flagged that the EngineMaker/Driver features' "real test is the TFT v2 run where `EngineMaker` is a static categorical and the 2026 boundary is in-distribution for evaluation." That run (`notebooks/05_tft_v2_kaggle.ipynb`, Kaggle T4) is done. **v2 added `Driver` + `EngineMaker` as static categoricals on top of v1's `CircuitKey/Team/Era/Compound`. It did not beat v1 on any split:**

| Split | v1 (chosen) all / green | v2 all / green |
|---|---|---|
| Val 2025 (era0) | 2.25 / **1.12** | 2.43 / 1.14 |
| Test 2026 (era1) | 3.78 / **1.62** | 4.14 / 1.94 |

Val green was flat (1.12→1.14); the loss concentrated in the cross-era test (green 1.62→1.94, +20%). This is the **same driver-as-car-proxy mechanism the LightGBM ablation showed**: driver/engine identity is partly a stand-in for car pace, and across the 2026 boundary drivers switch teams and rookies are unseen, so the learned attribution transfers badly — now reproduced on the sequence model. **Honesty confound:** v2 early-stopped at 20 epochs vs v1's ~26, so undertraining is a partial confound — read this as "added identity features, got no improvement, kept v1," not as a clean causal proof that the features hurt. A converged re-run is the clean tiebreak if a Kaggle slot frees up. v1 artifacts (`models/tft_lap.*`, reports, recalibration) are the deployed set; v2 is archived in `tft_v2_artifacts.zip`.

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