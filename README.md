# 🏎️ F1 AI Race Strategist

**An AI system that predicts F1 lap times across regulation eras with a Temporal Fusion Transformer, models tyre degradation with physics-informed curves, and composes both — *with their uncertainties* — into a Monte Carlo race-simulation engine that recommends pit strategy as win-probability distributions.**

Most F1 ML projects stop at "I trained a model on a table." This project composes uncertainty-aware models into a simulation engine that outputs **probability distributions over strategies, not point estimates** — which is what real motorsport strategy teams actually do.

> **Status:** lap-time models (baseline ladder + TFT) and tyre degradation curves complete and honestly evaluated. Monte Carlo simulation engine + strategy MDP in progress. See [roadmap](#roadmap).

---

## Headline results

**Lap-time MAE (seconds), season-aware splits — train 2022–24, val 2025, test 2026:**

| Model | Val all / green (2025, era 0) | Test all / green (2026, era 1) |
|---|---|---|
| BayesianRidge (one-hot + scaled) | 4.16 / 2.97 | 3.97 / 2.88 |
| RandomForest (integer codes) | 6.45 / 5.61 | 9.05 / 8.52 |
| LightGBM (Optuna-tuned baseline) | 3.51 / 2.16 | 3.42 / 2.11 |
| **Temporal Fusion Transformer** | **2.25 / 1.12** | **3.78 / 1.62** |

Three results worth reading closely:

1. **The TFT roughly halves the LightGBM green-flag MAE on both splits.** Its edge is legitimate sequence modeling, not leakage: the encoder consumes each stint's *past lap times* (known at prediction time) to predict the next lap — degradation dynamics a tree treating laps as independent rows cannot see.
2. **Cross-regulation generalization.** 2026 introduced new aero/power-unit regulations and three new teams (test set = unseen era, 6 races). LightGBM holds 2.16 → 2.11 green MAE across the boundary; the TFT holds 1.12 → 1.62. Era is an explicit feature, encodings are frozen from train, and the degradation across the boundary is reported, not hidden.
3. **RandomForest is kept as a documented failure case.** It collapses across the era boundary (5.61 → 8.52) because it can't handle integer-coded categoricals or unseen teams — the motivating contrast for LightGBM's native categorical handling. Honest baselines beat quiet deletions.

**Uncertainty is a first-class output.** The TFT predicts 0.1/0.5/0.9 quantiles; calibration is *measured*, not assumed: central coverage is 0.76 in-era (vs 0.80 nominal — mildly overconfident) and 0.69 across the 2026 boundary (overconfident; flagged for recalibration before the simulation engine trusts the bands).

**Tyre degradation (Phase 3):** quadratic curves `b·age + c·age²` fit per (compound × circuit × era) on fuel-corrected, green-flag-only stint deltas, with 95% CIs from the covariance and hierarchical pooling for thin cells (2026 cells pool to era level — wide CIs in low-data regimes, never false precision). Era-0 degradation rates land in physically sensible territory: HARD 0.024 / MEDIUM 0.033 s/lap linear terms, SOFT carrying the largest cliff coefficient.

<p align="center">
  <img src="reports/lap_time/shap_summary.png" width="46%" alt="SHAP feature importance"/>
  <img src="reports/tyre/degradation_curves_era0.png" width="46%" alt="Tyre degradation curves"/>
</p>

---

## Architecture

```
FastF1 (2022–2026 race laps, cached)
        │
   ingest → validate (Pandera) → features → season splits (no shuffle, frozen encodings)
        │
        ├── Baseline ladder: Ridge → RF → LightGBM (Optuna)     [models to beat]
        ├── TFT (pytorch-forecasting): stint sequences,          [headline model]
        │        quantile uncertainty, cross-era evaluation
        └── Tyre curves: curve_fit per compound×circuit×era,     [physics-informed]
                 hierarchical pooling, 95% CIs
        │
        ▼
   Monte Carlo simulation engine  ◄── THE SHOWCASE (in progress)
   composes lap-time + tyre + pit-loss models WITH uncertainty
   N ≥ 1000 rollouts → strategy win-probability distributions
        │
        ▼
   Pit-strategy MDP (+ RL stretch) → FastAPI → Streamlit dashboard
```

## Evaluation discipline

- **Season-aware temporal splits only** — shuffled splits would leak across regulation changes and overestimate accuracy massively.
- **Per-circuit and per-era MAE always** — aggregates hide failures (the TFT is excellent at Suzuka, 0.46s green; poor at street circuits, ~3s at Canada — reported, and the sim will treat street circuits as higher-variance).
- **Green-flag vs all-laps split** — safety cars are not predictable from pre-lap features; the gap is quantified instead of polluting the headline.
- **Calibration measured** for every uncertainty estimate.
- **Frozen mappings** — circuit/team encodings built from train only; unseen teams (Audi, Cadillac…) hit a −1 sentinel, the same path a brand-new team takes at serving time.
- Every model card lives in [`src/models/model_cards/`](src/models/model_cards/).

## Repo map

```
src/pipeline/        ingest, validate, features, splits  (column contracts = law)
src/models/lap_time/ train.py (ladder), train_tft.py, evaluate.py
src/models/tyre/     fit.py (degradation curves + pooling)
src/models/model_cards/  honest model cards with real numbers
reports/             per-circuit / per-era / calibration CSVs + plots
tests/               50 unit tests (pipeline, features, tyre, TFT structure)
notebooks/           EDA + Kaggle GPU training notebooks
```

## Reproduce

```bash
python -m venv venv && venv/Scripts/activate     # Python 3.13
pip install fastf1 pandas pandera scikit-learn scipy mlflow shap matplotlib \
            joblib pytest lightgbm optuna                       # core CPU deps
python -m src.pipeline.ingest                    # fetch FastF1 data (cached)
python -m src.models.lap_time.train              # ladder + Optuna → MLflow
python -m src.models.lap_time.evaluate           # all report CSVs/plots
python -m src.models.tyre.fit                    # degradation curves
pytest -q                                        # 50 tests
```

The TFT trains on GPU (`notebooks/04_tft_kaggle_fullrun.ipynb`, Kaggle T4 — pinned `pytorch-forecasting==1.7.0` / `lightning==2.6.5` / torch cu128; P100 unsupported by the cu128 wheel).

## Known limitations (read before trusting any number)

No live telemetry (tyre temps, fuel flow, ERS, active-aero state — the last especially impactful in 2026). Fuel load is a linear estimate. SOFT/MEDIUM/HARD labels hide circuit-varying C1–C5 compounds. The 2026 test set is six races and statistically noisy. TFT intervals are overconfident across the era boundary (measured, documented, recalibration planned). The simulation's outputs will inherit every one of these.

## Roadmap

- [x] Data pipeline + validation + leakage-proof splits (2022–2026)
- [x] Baseline ladder: Ridge → RF → LightGBM (Optuna), SHAP, model card
- [x] TFT with quantile uncertainty, cross-era eval, calibration measurement
- [x] Tyre degradation curves with CIs + hierarchical pooling
- [ ] TFT quantile recalibration (conformal, era-aware)
- [ ] Pit-strategy MDP (value iteration, explicit undercut model)
- [ ] **Monte Carlo race-simulation engine** — validated by replaying historical races
- [ ] RL pit agent vs MDP (stretch)
- [ ] FastAPI + Streamlit dashboard + deployment
