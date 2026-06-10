# F1 AI Race Strategist — Master Context v3.2
### Paste this at the start of every new Claude conversation
### v3 supersedes v2. Scope = "depth, then one ambitious showcase."
### v3.1 (2026-06-10): Phase 3 (tyre curves) DELIVERED; mapping-freeze contract fixed; README;
### Section 10 rewritten as step-by-step PHASE PLAYBOOKS.
### v3.2 (2026-06-10, same day): Phases 3.5 + 4 DELIVERED AND VALIDATED —
###   - EngineMaker (season-aware team→PU dictionary) + DriverEncoded features, ablation run
###   - TFT conformal recalibration run locally on CPU (era-0 coverage 0.800 on the nose)
###   - pit_loss.py (2,153 measured events), mdp.py (exact backward induction),
###     simulation/engine.py (vectorized MC showcase), validate_sim.py (replay coverage 0.79)
###   - notebooks/05_tft_v2_kaggle.ipynb ready: TFT v2 (Driver+EngineMaker static cats)
###   - TUTOR.md added: full teaching walkthrough of the project for interview prep
### GPU note: Phase 2 ran on Kaggle T4. ~28 Kaggle GPU hours remain; next GPU job =
### notebook 05 (TFT v2 + recalibration, ~2-4h). Then RL (Phase 5, ~10h).

---

## CLAUDE INSTRUCTIONS — READ THIS FIRST

When the user says **"update master context"**, do the following without asking questions:
1. Read this entire file
2. Update Section 8 (progress tracker) based on what was discussed
3. Update Section 9 (next actions) to reflect the immediate next step
4. Add any new bugs to Section 16 (errors log)
5. Update Section 5 (file specs) with any fixes made
6. Output the complete updated file — no truncation

When the user says **"start phase N"**:
1. Read Section 8 to confirm what's done
2. Read Section 10's playbook for phase N — it contains the design decisions; do not re-derive them
3. Read Section 5 for all file specs before writing any code
4. Read Section 3 before suggesting any pip install
5. Read Section 16 (errors log) before writing any code — do not reintroduce fixed bugs
6. State the plan and data contracts before writing any code

**Guiding principle for this project (do not let it drift):** The binding constraint is DATA, not compute or time. F1 has only a few hundred thousand usable green-flag laps. A 48GB GPU does not justify a bigger model — it justifies the *right* model trained well, fast iteration, and the compute-hungry simulation showcase. Fewer things, done excellently and evaluated honestly, beats many shallow models. If a suggestion adds scope without adding rigor, push back.

---

## 1. Who I am

Data Science / AI undergraduate building a full-stack AI-powered F1 race strategy system as my primary portfolio project for internship applications (targeting McLaren Applied, Williams Racing, Stats Perform, general ML engineering roles).

Built with Claude's help — coding, architecture, debugging, writing, research.

**Constraints (v3 — updated):**
- Solo project
- **Time:** sprint mode — can commit most of the day. No hard deadline. **Quality over speed.**
- **Compute:** college server GPU — **NVIDIA A6000, 48GB** (for training). Small experiments on Kaggle / Colab free GPU when convenient.
- **Serving is separate from training:** the trained models are small at inference and run fine on CPU. The A6000 is a *training* box, not a public web host.
- **Tooling:** Claude Code is used for this project. Claude may issue file changes as precise Claude Code prompts ("in `src/...`, do X") — Claude Code reads and edits the real repo files, so it's preferred over pasting whole files for edits to existing code.
- **Terminal is bash** (Git Bash locally, Linux bash on the GPU server). `&&`, `touch`, and Unix coreutils all work. Never emit PowerShell syntax.
- Python 3.13.5, venv at `e:\f1-strategist\venv` (local dev); GPU server has its own env
- VS Code with Jupyter extension
- Public GitHub repo: `Shreyansh262/f1-strategist`

---

## 2. Project overview

**Name:** F1 AI Race Strategist
**One-line:** An AI system that predicts lap times with a Temporal Fusion Transformer across regulation eras, models tyre degradation with physics-informed curves, and feeds both — *with their uncertainties* — into a Monte Carlo race-simulation engine that recommends pit strategy as win-probability distributions, all delivered through a deployed AI Analyst dashboard.

**The headline differentiator:** Most F1 ML projects stop at "I trained a model on a table." This project composes uncertainty-aware models into a **simulation engine that outputs probability distributions over strategies, not point estimates** — which is what real motorsport strategy teams actually do. That is the thing that separates "I trained a model" from "I built a system."

**Why this stands out:**
- Handles the 2022→2026 regulation shift (most projects ignore era boundaries)
- A real sequence model (TFT), not "RF on tabular data" — with native quantile uncertainty
- A Monte Carlo simulation engine that propagates model uncertainty into strategy decisions
- Honest evaluation: per-circuit MAE, per-era MAE, green-flag vs all-laps split, calibration, model cards
- Production pipeline: FastAPI + Streamlit dashboard, with a polished static showcase landing page

**Depth over breadth: 3 core models done excellently + 1 integrative simulation showcase + 1 scoped RL stretch. Deployed. Evaluated honestly.**

---

## 3. Packages (keep updated)

**Already installed (venv):**
```
fastf1>=3.8.1
pandas
numpy
pandera>=0.19
scikit-learn
scipy
mlflow
shap
matplotlib
joblib
pytest
fastapi
uvicorn
pydantic
streamlit
torch
plotly
lightgbm
optuna
```

**Phase 2 TFT — PINNED WORKING TRIANGLE (smoke-validated end-to-end on Colab GPU, 2026-06-10):**
```
pytorch-forecasting==1.7.0   # TFT implementation
lightning==2.6.5             # unified package; import root is `lightning.pytorch`, NOT pytorch_lightning
torch 2.11 (cu128)           # Kaggle GPU torch — pf 1.1.1 is INCOMPATIBLE with torch 2.11, must use pf>=1.7
```
v3.2: pf 1.7.0 + lightning 2.6.5 are now ALSO installed in the LOCAL venv (CPU torch 2.11) —
recalibrate.py and the TFT tests run locally; only training needs the Kaggle GPU.
Notes: pf 1.7.0 dropped `stop_randomization` → use `from_dataset(..., predict=True)` for val/test.
mlflow 3.x blocks the file:// store unless `MLFLOW_ALLOW_FILE_STORE=true` (set in train_tft_data.py).

**TO INSTALL later (only those NOT above):**
```
# RL STRETCH ONLY (do not install until Phase 5):
stable-baselines3        # PPO/DQN for pit-strategy agent
gymnasium                # RL environment API
```

**RULE: Before any pip install, check this list. Only install what's NOT here. Update this list when something new is installed. Watch the torch / lightning / pytorch-forecasting version triangle — it breaks easily. The working combo is now pinned above — do not drift from it without re-running the quicktest.**

---

## 4. Scope decisions (v3) — what changed from v2 and why

| Decision | v2 | v3 | Why |
|---|---|---|---|
| Lap-time headline | tiny TabTransformer (~50K params, Colab-forced) | **Temporal Fusion Transformer** | GPU removes the size limit; laps within a stint are a sequence, so a sequence model is the right tool. Native quantile outputs give calibrated uncertainty. |
| Tree baseline | RandomForest | **LightGBM (tuned via Optuna)**, RF kept as a documented earlier rung | LightGBM is strictly stronger for tabular and is the standard "model to beat." |
| Car/team pace | not modeled | **`TeamEncoded` added** (frozen mapping) | Biggest accuracy gap in v2 — without it the model can't tell a fast car from a slow one at the same circuit. Team is known at race time → no leakage. |
| Integrative model | none | **Monte Carlo race-simulation engine** | The differentiator. Composes all models + uncertainty → strategy win-probability distributions. |
| Race-outcome classifier | planned (old HTML vision) | **CUT** — finishing-order distributions come free from the simulation | A standalone "XGBoost predicts finishing order" is generic and mostly learns "grid ≈ finish." The sim does it more rigorously. |
| Pit strategy | MDP | **MDP first**, then **RL agent as a stretch** (only after everything else works) | RL is high-ceiling, high-risk. A fragile RL agent is worse than none. MDP is the interpretable baseline. |
| Frontend | Streamlit | **Streamlit functional app + reuse existing HTML as static showcase landing page** | For these roles the frontend framework is low-leverage; React would cost days better spent on the sim engine. The HTML artifact already gives visual polish for free. |
| Live demo | Render free tier | **Local + recorded demo first; public deploy later, lower priority** | Build the system right first. Deployment data/model bundling (see Section 12) is a known gap to solve at deploy time, not now. |

---

## 5. File-by-file specification (exact contracts)

Ground truth for every file. New code must match. Do not rename columns or change return types without updating this section.

### KEEP AS-IS (already solid — do not rewrite)
- `src/pipeline/ingest.py` — **COMPLETE.** Output columns are law (Section 6). FastF1 cache enabled at module level. `session_type="R"` loads the Grand Prix, not the sprint.
- `src/pipeline/validate.py` — **COMPLETE.** Filter order: TyreLife==1 out-laps → LapTimeSeconds in [70,130] → invalid/"None" Compound → sprint safety-net (mean>105s) → season whitelist → Pandera `lazy=True`. (v3.1: removed an accidental duplicate Pandera validation call.)
- `tests/` — **COMPLETE: 50 tests passing locally** (test_features 44 incl. mapping-freeze contract, test_tyre 9, test_tft skips cleanly when lightning absent via `pytest.importorskip`).
- `conftest.py` — **COMPLETE.**

### `src/pipeline/features.py` — STATUS: COMPLETE (v3.2 — engine + driver features)
```python
def add_circuit_encoding(df, mapping=None) -> pd.DataFrame  # mapping=None builds IN-MEMORY only (no save)
def add_team_encoding(df, mapping=None) -> pd.DataFrame     # same; unseen -> -1
def add_driver_encoding(df, mapping=None) -> pd.DataFrame   # NEW v3.2 — frozen like team; rookies -> -1
def add_engine_maker(df) -> pd.DataFrame                    # NEW v3.2 — EngineMaker str + EngineEncoded int
def engine_for(team, season) -> str | None                  # season-aware team→PU dictionary
def freeze_encoding_mappings(train_df) -> (dict, dict, dict)  # THE ONLY writer of data/mappings/*.json
def load_encoding_mappings() -> (dict, dict, dict)          # circuit, team, driver — for eval/serving
def add_era_feature(df) -> pd.DataFrame                     # 2022-2025=0, 2026+=1
def add_stint_id(df) -> pd.DataFrame                        # per driver per race
```
**EngineMaker design:** `_ENGINE_2022_2025` + `_ENGINE_2026_OVERRIDES` dicts (Aston Mercedes→Honda,
Alpine Renault→Mercedes, RBR Honda→Red Bull Ford, Audi=Audi, Cadillac=Ferrari customer...).
`ENGINE_ORDER` is fixed a-priori over ALL known engines (incl. 2026) — domain knowledge like the
Era boundary, NOT frozen-from-train (no leakage: supplier contracts are public pre-season). Its
purpose is cross-era transfer: Cadillac is unseen as a Team but a known Ferrari customer.
`MODEL_FEATURE_COLUMNS` is now 16 features (added DriverEncoded, EngineEncoded); `FEATURE_COLUMNS`
also carries the `EngineMaker` string through for the TFT.
**Contract (v3.1, Errors #17):** `mapping=None` must NEVER persist to disk — tests/notebooks
used to clobber the production mappings. Only `freeze_encoding_mappings(train_rows)` writes
JSON, and it must be fed TRAIN seasons only so unseen 2026 teams (Audi, Cadillac, Racing
Bulls) hit the −1 sentinel. `train.py` does this; `evaluate.py`/serving load the frozen maps.

**Feature columns:**
```python
MODEL_FEATURE_COLUMNS: list[str] = [
    "CircuitEncoded", "TeamEncoded", "CompoundEncoded", "Era",
    "TyreLife", "TyreAgeSq", "TyreAgeCubed",
    "CompoundXTyreLife", "FuelLoad", "FuelEffect",
    "TrackTemp", "AirTemp", "NormLapNumber", "StintPhase",
]
```
For the **TFT**, features are split differently (static / known / observed) — see Section 11. `MODEL_FEATURE_COLUMNS` above is the flat list used by the linear/tree models.

### `src/pipeline/splits.py` — STATUS: COMPLETE (v3.1)
- Season-aware temporal splits only, never shuffle.
- **v3 split:** train = 2022–2024, val = 2025, test = **2026 (6 GP races available as of June 2026: Australia, China, Japan, Miami, Canada, Monaco).** Re-ingest 2026 before every evaluation as more races run. Bahrain & Saudi 2026 were cancelled — they will never exist.
- `assert_no_leakage()` checks season disjointness AND that any CircuitEncoded/TeamEncoded
  code appearing in val/test but not train is exactly −1 (mapping-freeze proof).

### `src/models/lap_time/train.py` — STATUS: COMPLETE (baseline-ladder trainer, v3.1 retrained)
Trains the **baseline ladder** and logs all to MLflow:
1. `BayesianRidge` pipeline (OneHot cats + scaled conts) — linear baseline.
2. `RandomForest` — kept as the documented era-collapse rung.
3. **`LightGBM` tuned with Optuna (60 trials)** — the model to beat.
- `load_data()` freezes mappings from TRAIN seasons via `freeze_encoding_mappings` then
  encodes the full frame with them (v3.1).
- MLflow: `MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()`; experiment
  `"lap_time_predictor"`; `MLFLOW_ALLOW_FILE_STORE=true` set at import (mlflow 3.x, Errors #13).
- Saves: `models/{bayesian_ridge,rf,lgbm,chosen}_lap.joblib`.
- Does NOT import `ingest.py` — `load_data()` reads per-round parquet directly (Errors #10).

### NEW — `src/models/lap_time/train_tft.py` — STATUS: BUILT + SMOKE-VALIDATED (Phase 2; full A6000 run pending)
TFT trainer. Runs on the A6000. See Section 11 for architecture. Saves Lightning ckpt via
ModelCheckpoint (+ exported `models/tft_lap.pt`, CPU-loadable, for serving). Logs to MLflow.
- Loads via `load_tft_data()` (see below), NOT `train.load_data()` — needs raw cols build_features drops.
- `prepare()` adds `GroupID` (Season_Round_Driver_StintID), contiguous `time_idx` (lap-in-stint),
  `EraStr`/`CompoundStr`; imputes weather.
- Feature roles BY NAME (Section 11): static_categoricals = CircuitKey, Team, EraStr, CompoundStr;
  known reals = time_idx, NormLapNumber, FuelLoad, TyreLife; unknown reals = TrackTemp, AirTemp (+target).
- **target_normalizer = `EncoderNormalizer()`** — NOT GroupNormalizer(GroupID): GroupID is unique per
  stint, so under season splits every val/test group is unseen → KeyError. See Errors log #12.
- `make_datasets()` returns (training, val|None, test|None) — val/test None when that season absent.
- `green_mae()` uses `predict(mode="prediction", return_index=True)` (point=median), handles both
  namedtuple and DataFrame return shapes; merges yhat to raw on (GroupID, time_idx) for actual + TrackStatus.
- Smoke-validated on Colab (fast=True): data contract, GPU train, val loop, checkpoint, green_mae
  predict+merge, CPU export, mlflow all run clean. Full 100-epoch run on A6000 still pending.

### NEW — `src/models/lap_time/train_tft_data.py` — STATUS: DONE (Phase 2 carry-back loader)
`load_tft_data()`: per-round dedup glob (Error 10) → `validate_laps` (once) → `add_stint_id` →
`build_features` → merge `Team`/`Compound`/`StintID`/`TrackStatus` back on
`LAP_KEYS=[Season,RoundNumber,Driver,LapNumber]`. build_features strips these (only FEATURE_COLUMNS
survive), so sequence/eval models re-attach them here. Same trick as `evaluate.load_eval_data()`.
Re-exports `MLFLOW_TRACKING_URI` (so train_tft.py needn't import train.py) and sets
`MLFLOW_ALLOW_FILE_STORE=true` at import (mlflow 3.x guard, Error #13). Verified: ~87k laps, 92.4% green.

### NEW — `notebooks/03_tft_quicktest.ipynb` — STATUS: DONE
Colab/Kaggle smoke notebook: pins the triangle, clones + always `git pull`, runs `main(fast=True)`.
Used to confirm the version triangle + data contract before committing the A6000 to a full run.

### `src/models/lap_time/evaluate.py` — STATUS: COMPLETE (v3.1)
Outputs to `reports/lap_time/`: `per_circuit_mae.csv`, `per_era_mae.csv`, `greenflag_vs_alllaps_mae.csv`, `learning_curve.png`, `shap_summary.png` + `shap_importance.csv`, and `model_comparison.csv` — now the **four-model table**: tree rows computed live, TFT rows folded in from `tft_breakdown.csv` via `_tft_comparison_rows()` (skips silently if TFT not trained). Encodes eval data with `load_encoding_mappings()` — never rebuilds mappings. Imports `MLFLOW_TRACKING_URI` from `train.py`.

### `src/models/tyre/fit.py` — STATUS: ✅ DONE (Phase 3, v3.1 — full rewrite of the stale v2 file)
Model: fuel-corrected within-stint delta, `delta(age) = b·(age−age0) + c·(age²−age0²)` —
the quadratic `a + b·age + c·age²` with the per-stint intercept differenced out against a
baseline = median of the stint's first 3 green laps. **Fuel correction first** (subtract
`FuelEffect`): fuel burn ~0.045 s/lap otherwise cancels real degradation and biases b low.
**Green-flag slick laps only** (SC laps are +20–40s outliers; INT/WET excluded — drying track
makes wet laps faster with age, wrong model). **Hierarchical pooling:** (compound×circuit×era)
[≥5 stints] → (compound×era) [≥8] → (compound); chosen level recorded in `pool_level`.
Loads via `load_tft_data()` (per-round glob + dedup + StintID/TrackStatus carry-back).
`predict_degradation(curves, compound, circuit, era, age) -> (mid, lo, hi)` is the Phase 4
sim's interface (has compound-level fallback for unknown cells; raises only on unknown compound).
RESULTS: 84 cells (76 circuit-level, 8 era-pooled — all 2026); era-0 mean b: HARD 0.024 /
MED 0.033 / SOFT 0.025 s/lap, SOFT largest cliff c≈0.0033; circuit-level mean r²=0.156 (honest:
per-lap noise from traffic/management dominates — b,c CIs are what matter). Saves
`models/tyre_curves.joblib`, `reports/tyre/tyre_curves.csv`, `degradation_curves_era{0,1}.png`;
MLflow experiment `tyre_degradation`. Tests: `tests/test_tyre.py` (synthetic b,c recovery,
pooling fallback, fuel-correction, predict band). Model card: `src/models/model_cards/tyre_degradation.md`.

### `src/models/lap_time/recalibrate.py` — STATUS: ✅ DONE (Phase 3.5, v3.2)
Era-aware split-conformal widening of the TFT 0.1–0.9 band, GREEN-FLAG LAPS ONLY (Error 20).
Calib/eval disjoint: era0 = odd/even 2025 rounds; era1 = 2026 R1 / R2-6. Loads tft_lap.ckpt
with CPU-safe overrides (`loss=QuantileLoss(...)`, `logging_metrics=ModuleList()` — pickled
metrics remember device=cuda, Error 21); rebuilds datasets via `model.dataset_parameters` +
`TimeSeriesDataSet.from_parameters` so encoders match training exactly. Outputs
`models/tft_calibration.json` (era0_shift_s / era1_shift_s) + `reports/lap_time/tft_recalibration.csv`.
Rerun INSIDE the Kaggle notebook whenever the TFT is retrained.

### `src/models/pit_strategy/pit_loss.py` — STATUS: ✅ DONE (Phase 4a, v3.2)
Measures pit loss per green-flag pit event: (in-lap + out-lap) − 2×driver race green median;
both laps must be green; loss clipped to [5,60]s. 2,153 events → per-circuit medians (min 8
events, else `_global` 23.05s) in `data/pit_loss.json`. API: `load_pit_loss()`, `pit_loss_for()`.

### `src/models/pit_strategy/mdp.py` — STATUS: ✅ DONE (Phase 4a, v3.2)
Exact backward induction (finite horizon = exact in one sweep). State (lap, compound, age≤45,
used_two_compounds); pit happens at END of lap; fresh stint age = 2 (validate drops TyreLife==1).
Two-compound rule via terminal V=inf if not satisfied. `build_race_model()` wires tyre curves +
pit table; `solve()` returns strategy + policy/value grids; `plot_policy()` heatmaps. SINGLE-CAR
by design — interaction lives in the engine. Monaco final-lap-pit artifact documented in the
model card (`pit_strategy_simulation.md`).

### `src/simulation/engine.py` — STATUS: ✅ DONE (Phase 4b, v3.2 — THE SHOWCASE)
`simulate(RaceSpec, n_rollouts, seed, device) -> SimResult` (win/podium/mean-finish + per-rollout
finish matrix). Samples: per-lap σ from RECALIBRATED band (default_sigma reads
tft_calibration.json + tft_recalibration.csv), tyre CI z per (rollout,driver,stint), pit
N(loss, 0.8s), overtaking Bernoulli by circuit class. Deterministic strategy schedules
precomputed per driver (`_stint_schedule`). `recommend()` ranks candidate ego strategies with
common random numbers. Does NOT call the TFT net per lap (distributional summaries — document
why). `src/simulation/validate_sim.py` = replay protocol (leave-one-race-out driver pace,
grid proxy = lap-1 cumulative order); writes reports/simulation/validation.{csv,_summary.md}.
Tests: `tests/test_simulation.py` (9: MDP rule/monotonicity/replay-consistency; engine
ordering/probability-sum/seed/overtake-freeze/extra-stop-cost).

### NEW — `src/models/pit_strategy/rl_agent.py` — STATUS: STRETCH (Phase 5, only after all else works)
PPO/DQN trained inside the simulation engine (wrapped as a Gymnasium env). Compared head-to-head with the MDP. If it doesn't beat the MDP, that's a documented finding, not a failure.

### NEW — `src/api/main.py` — STATUS: NOT STARTED (Phase 6)
FastAPI. Endpoints in Section 12.

### NEW — `dashboard/` — STATUS: NOT STARTED (Phase 6)
Streamlit multi-page app. Pages in Section 12. Existing HTML artifact becomes the static showcase landing page.

---

## 6. Data contracts — column names are law

| Concept | Column | Set in | Notes |
|---|---|---|---|
| Season year | `Season` | ingest.py | int, 2022–2026+ |
| Round number | `RoundNumber` | ingest.py | int, >= 1 |
| Circuit name | `CircuitKey` | ingest.py | str, EventName |
| Team | `Team` | ingest.py | str — **NEW passthrough from FastF1, needed for TeamEncoded** |
| Lap time (s) | `LapTimeSeconds` | ingest.py | float, target |
| Compound | `Compound` | FastF1 passthrough | str |
| Compound encoded | `CompoundEncoded` | features.py | int, SOFT0 MED1 HARD2 INT3 WET4 |
| Circuit encoded | `CircuitEncoded` | features.py | int, frozen, -1 unseen |
| Team encoded | `TeamEncoded` | features.py | int, frozen, -1 unseen — **NEW** |
| Regulation era | `Era` | features.py | int, 0=2022-2025, 1=2026+ |
| Tyre age | `TyreLife` | FastF1 passthrough | int |
| Track / Air temp | `TrackTemp` / `AirTemp` | ingest.py | float, nullable |
| Fuel load / effect | `FuelLoad` / `FuelEffect` | features.py | float |
| Tyre age² / ³ | `TyreAgeSq` / `TyreAgeCubed` | features.py | float |
| Compound × TyreLife | `CompoundXTyreLife` | features.py | float |
| Norm lap number | `NormLapNumber` | features.py | float 0–1 |
| Stint phase | `StintPhase` | features.py | int |
| Pit lap flag | `IsPitLap` | ingest.py | bool, stint segmentation only |
| Track status | `TrackStatus` | ingest.py | str, FastF1 codes ("1" = all green). Eval-only metadata for green-flag MAE, NOT a model feature |
| Stint ID | `StintID` | features.py | int, per driver per race |

Constants (unchanged): `FUEL_LOAD_START_KG=110.0`, `FUEL_BURN_PER_LAP=1.5`, `FUEL_EFFECT_PER_KG=0.03`.

---

## 7. Data assumptions & known limitations (interview gold — document honestly)

Carried from v2, still true: safety-car/VSC laps not filtered (report green-flag vs all-laps MAE — Section 13); red-flag restarts not filtered; track evolution not modeled; no tyre-temperature sensors; no ERS deployment data (critical in 2026); no active-aero state (2026); fuel load estimated (linear 110kg burn); driver skill not a continuous variable; SOFT/MED/HARD hide circuit-varying C1–C5 compounds.

**New v3 notes:**
- **TFT uncertainty is only as good as its calibration** — must verify quantile coverage on the val set, not assume it.
- **The simulation is only as good as its weakest input model** — sim outputs inherit lap-time, tyre, and pit-loss uncertainty. State this; don't present sim probabilities as ground truth.
- **2026 test set is small (6 races) and under brand-new regs** — the cross-era generalization number is the headline but is statistically noisy. Report it with that caveat and re-run as more races happen.

What the model DOES capture well: circuit baselines (CircuitEncoded), **car/team pace (TeamEncoded — new)**, per-compound degradation, fuel effect, weather, regulation-era shift (Era + sim re-fit per era), non-linear tyre age.

---

## 8. Progress tracker

```
DONE (from v2 — keep):
  [x] Repo, FastF1 exploration, EDA
  [x] Data pipeline: ingest, validate, features (pre-Team), splits, 24 tests passing
  [x] Lap-time RF baseline + evaluate.py + model card scaffold

v3 PHASES (no fixed weeks — quality first, sprint pace):
  [x] Phase 0 — Migration: re-ingested 2022-2026 with Team passthrough, TeamEncoded wired
                (incl. FEATURE_COLUMNS projection), Monaco floor -> 70.0 + per-filter row logging,
                splits/leakage updated, TabTransformer retired. Tests passing. (A6000 setup deferred to Phase 2.)
  [x] Phase 1 — Lap-time baseline ladder DONE: Ridge(one-hot)/RF/LightGBM+Optuna on 103,732 deduped laps.
                Chosen = LightGBM: green MAE 2.18s val(era0) / 2.14s test(era1) — generalizes across the
                2026 boundary (within 6-race noise). RF kept as the era-collapse counter-example.
                TeamEncoded validated (SHAP 6/14). Model card filled. per-era/green-flag/per-circuit/SHAP written.
  [x] Phase 2 — TFT lap model. DONE (full run on Kaggle T4, all seasons 2022-26).
                RESULT: TFT BEATS LightGBM on green MAE both splits — val 1.11s (LGBM 2.18) / test 1.67s
                (LGBM 2.14), i.e. ~halved. Win is legitimate: encoder uses past in-stint lap times (autoregressive,
                no leakage) — the "laps are a sequence" thesis. ~85.6K params, hidden_size 32, ~26 epochs.
                Bugs found+fixed across smoke+full: carry-back cols, EncoderNormalizer, mlflow file-store,
                version compat, P100/cu128 arch, eval-subset predict=True->False (Errors 11-16).
                CALIBRATION: in-era (val) intervals roughly honest (central 0.76 vs 0.80 nominal); across the
                2026 boundary OVERCONFIDENT (central 0.69, heavy lower tail) — recalibrate or document before
                the sim trusts the bands. Per-circuit: great on conventional tracks (Japan 0.46s green), poor
                on street circuits (Canada ~3s, Monaco ~2-2.8s). Reports: tft_breakdown.csv + tft_calibration.csv.
                Model card filled. Artifacts (models/*.pt,*.ckpt) now gitignored — download from Kaggle manually.
  [x] Phase 3 — Tyre degradation DONE (v3.1, 2026-06-10): full rewrite of stale v2 fit.py.
                Fuel-corrected green-flag deltas, intercept-free quadratic, hierarchical pooling.
                84 cells (76 circuit-level, 8 era-pooled = all thin 2026 cells). Era-0 b: HARD
                0.024 / MED 0.033 / SOFT 0.025 s/lap; SOFT largest cliff. r²=0.156 circuit-mean
                (per-lap noise — honest). Tests (9) + model card + MLflow + era plots done.
  [x] v3.1 hardening (same session): mapping-freeze contract (Errors #17) + ladder retrain
                (numbers moved ≤0.03s — contract fix, not results change); mlflow 3.x guards
                everywhere; four-model comparison table; validate.py double-validation removed;
                test_tft collection fix; portfolio README written.
  [x] Phase 3.5 — TFT conformal recalibration DONE (v3.2, local CPU on the v1 ckpt).
                Green-flag, era-aware, calib/eval never share laps. Era0: shift 0.082s,
                coverage 0.747->0.800 (= nominal). Era1 (calib on 2026 R1, eval R2-6):
                shift 0.868s, 0.659->0.879 (conservative — safe direction).
                models/tft_calibration.json + reports/lap_time/tft_recalibration.csv.
                NOTE: SC laps in calibration blow the shift to ~8s — green-only is correct
                (the sim samples green pace). pf+lightning now ALSO in the local venv (CPU).
  [x] v3.2 features — EngineMaker (season-aware team→PU dict, domain knowledge, ENGINE_ORDER
                fixed a priori) + DriverEncoded (frozen from train, rookies → -1). Ablation:
                val green 2.16->2.14 but test green 2.11->2.22 (drivers switch teams across
                the boundary; rookies unseen). Selection stays val-based; both rows in the
                model card. EngineEncoded SHAP ≈0 in-era (era-0 engine ⊂ team) — its real
                test is TFT v2. SHAP shift: TeamEncoded 0.43->0.08, DriverEncoded 0.32
                (driver nested in team — attribution not identifiable, good interview point).
  [x] Phase 4a — Pit MDP DONE (v3.2): exact backward induction over (lap × compound ×
                age × two-compound-rule), pit at end of lap, OUT_LAP_AGE=2. Inputs: tyre
                curves + measured pit loss + fuel delta. Bahrain/Spain 2-stop, Monaco/Japan
                1-stop (matches reality). Known artifact: Monaco pits on the FINAL lap to
                satisfy the rule — optimal in a deterministic rival-free world, absurd under
                SC risk; kept as the cleanest argument for the sim layer. Policy heatmaps +
                optimal_strategies.csv + mdp_vs_actual.csv in reports/pit_strategy/.
  [x] Phase 4b — Monte Carlo engine DONE (v3.2): torch [rollouts × drivers], samples
                per-lap σ from RECALIBRATED TFT band, tyre CI z-draw per (rollout,driver,
                stint), pit N(loss,0.8s), circuit-class overtaking (street .08/med .20/
                power .30, strike gap 1.5s, hold 0.6s). recommend() = paired common-random-
                number comparison. MDP-optimal Bahrain 2-stop wins the sim table (0.772 win
                prob) — layers agree. VALIDATION (validate_sim.py, leave-one-race-out pace):
                4 races 2025, 52 drivers, central-80% coverage 0.79 (Japan 1.00, Hungary
                0.78, Monaco 1.00, Bahrain 0.53 — SC-affected; no SC model in v1, documented).
                pit_loss.py: 2,153 green pit events, per-circuit medians 19.6-29.7s.
  [ ] Kaggle next — notebooks/05_tft_v2_kaggle.ipynb: TFT v2 (Driver+EngineMaker static
                cats) + recalibration in one session (~2-4h). Bars: green 1.12 val / 1.62
                test. After download: evaluate.py, pytest, update card + Section 8.
  [ ] Phase 5 — STRETCH: RL pit agent in the simulator, vs MDP. Only if dashboard not more
                valuable; Section 10.5. (~10 Kaggle GPU hrs)
  [ ] Phase 6 — FastAPI + Streamlit dashboard + HTML showcase page. Section 10.6.
  [ ] Phase 7 — Deploy (light host), technical report, final cross-era evaluation. Section 10.7.
```

---

## 9. Current blockers / next actions

**Phases 0–4 DONE and validated** (see tracker). 71 tests passing locally.

**NEXT (user action): run `notebooks/05_tft_v2_kaggle.ipynb` on Kaggle (GPU T4, ~2–4h).**
It trains TFT v2 (Driver + EngineMaker static categoricals), evaluates, recalibrates, and zips
artifacts. Bars: v1 green 1.12 val / 1.62 test. Either outcome is reportable. After download:
unzip into repo root → `python -m src.models.lap_time.evaluate` → `pytest -q` → update model
card + Section 8.

**Then choose: Phase 6 (dashboard — recommended next, biggest portfolio visual) or Phase 5
(RL stretch, ~10 Kaggle GPU hrs).** Playbooks: Sections 10.5 / 10.6.

**Watch:** TFT win rests on autoregressive in-stint history (thesis, not leakage). 2026 test =
6 races, noisy. Kaggle: T4 not P100 (Error 15). Model binaries gitignored — download from Kaggle
output. Recalibration must stay GREEN-ONLY (Error 20). Read TUTOR.md before any interview.

---

## 10. PHASE PLAYBOOKS — step-by-step, executable by any model

Phases 0–3 are done (Section 8). What follows is prescriptive enough to execute without
re-deriving design decisions. **Rules that apply to every playbook below:**
- Read Sections 3 (packages), 6 (columns are law), 16 (errors log) BEFORE writing code.
- Every trainer logs to MLflow; every model saves an artifact immediately; tests alongside code.
- Local shell is PowerShell-hosted but Claude Code has a Bash tool; on Kaggle it's Linux bash.
- Run `pytest -q` after every file change; all tests must pass before moving on.
- When a phase finishes: update Section 8, Section 9, and the model card, then commit.

### 10.0 Kaggle GPU budget (~28 hours) — spend it exactly like this
| Item | Hours | Phase | Notes |
|---|---|---|---|
| TFT v2 (add Driver static categorical + 6-config mini-sweep) | ~6 | optional, after 3.5 | Only if Phase 4 is on track; success bar = beat val green 1.12 |
| RL agent training runs (PPO in the sim env) | ~10 | 5 | The only genuinely GPU-hungry remaining item |
| Large sim sweeps / sensitivity analysis (N=10k rollouts × strategies) | ~4 | 4 | Engine must run CPU-first; GPU is a speedup, not a dependency |
| Buffer for re-runs / mistakes | ~8 | — | Kaggle sessions die; checkpoints every epoch |
**Everything else (recalibration, MDP, sim engine dev, API, dashboard) is LOCAL CPU work.**
Kaggle rules: GPU **T4** only (P100 = sm_60, unsupported by cu128 torch — Error 15); pin
`pytorch-forecasting==1.7.0 lightning==2.6.5`; keep Kaggle's GPU-matched torch (don't reinstall
torch); clone repo + `git pull` in cell 1 (pattern in `notebooks/04_tft_kaggle_fullrun.ipynb`);
download model artifacts from the run output (models/ is gitignored).

### 10.1 ✅ Phases 1–3 — done, see Section 8

### 10.2 ✅ Phase 3.5 — DONE in v3.2 (kept as reference; recalibrate.py implements this,
### with one amendment: GREEN-FLAG LAPS ONLY for both calibration and evaluation — Error 20)
**Why:** measured coverage 0.756 (val) / 0.690 (test) vs 0.80 nominal. The sim samples lap
times from these bands; overconfident bands → overconfident strategy probabilities → the
headline feature is wrong. Conformal scaling fixes this with zero retraining.

**File: `src/models/lap_time/recalibrate.py`** (new). Steps:
1. Load the CPU artifact `models/tft_lap.pt` (state_dict + hparams saved by `export_cpu`) —
   rebuild via `TemporalFusionTransformer.load_from_checkpoint` is NOT possible from .pt;
   instead reconstruct datasets with `make_datasets(load_tft_data())` and
   `TemporalFusionTransformer.from_dataset(training, **saved_hparams)` then `load_state_dict`.
   (If pf/lightning aren't in the local venv, run this step in the Kaggle CPU notebook or
   install the pinned triangle locally — CPU wheels are fine, no CUDA needed.)
2. Predict quantiles on the **val** set (predict=False — Error 16). Reuse
   `quantile_coverage()` from train_tft.py.
3. **Split-conformal scaling:** for each val lap compute the nonconformity score
   `s_i = max(q0.1_i − y_i, y_i − q0.9_i)` (positive = outside band). Take the
   (1−α)-quantile ŝ with α=0.20 → widen both band edges by ŝ:
   `q0.1' = q0.1 − ŝ, q0.9' = q0.9 + ŝ`. This guarantees ≥0.80 marginal coverage on
   exchangeable data.
4. **Era-aware widening for 2026:** val is era-0 only, so conformal on val does NOT cover the
   era shift. Compute a second, more conservative scale: multiply ŝ by the ratio of test/val
   raw miscoverage (0.31/0.244 ≈ 1.27) and SAY SO in the writeup — or, better, hold out the
   FIRST 2026 race (Australia R1) as a calibration slice for era-1 and conformalize on it,
   evaluating coverage on R2–R6 only. The second option is more defensible; implement it,
   and document that R1 is sacrificed from the test set for calibration.
5. Save `models/tft_calibration.json`: `{"era0_shift_s": ..., "era1_shift_s": ...,
   "quantiles": [0.1, 0.5, 0.9], "method": "split_conformal", "calibrated_on": ...}`.
6. Re-measure coverage with shifted bands → append rows (`split`, `recalibrated=True`) to
   `reports/lap_time/tft_calibration.csv`. Success = central coverage in [0.78, 0.85] on both.
7. Update model card calibration section + Section 8. Tests: synthetic check that the
   conformal shift achieves nominal coverage on a held-out synthetic sample.
**Pitfalls:** never calibrate on test laps you'll also report coverage on; keep predict=False;
the .pt artifact stores hparams — don't hardcode hidden_size.

### 10.3 ✅ Phase 4a — DONE in v3.2 (spec below kept as reference; one deliberate deviation:
### the MDP is single-car — undercut/interaction modeling moved wholly into the engine,
### which is the cleaner separation. See pit_strategy_simulation.md.)
**File: `src/models/pit_strategy/mdp.py`** (exists, empty). Interpretable baseline BEFORE the sim.
**State:** `(lap, tyre_age, compound, position, gap_ahead_s, laps_remaining)` — discretize:
tyre_age 1–40, gap_ahead bucketed [0–1s, 1–3s, 3–10s, >10s]. Era is fixed per race (input).
**Actions:** `{stay_out, pit_soft, pit_medium, pit_hard}` (compound choice matters — not just pit/no-pit).
**Transition (write this function FIRST, as `transition(state, action) -> (next_state, reward)`):**
- stay_out: tyre_age+1; lap time = circuit baseline (from LightGBM/median of green laps at that
  circuit) + `predict_degradation(curves, compound, circuit, era, age)` mid value; gap updates
  vs a rival assumed on the *historical median* strategy for that circuit.
- pit_X: lap time += pit loss (per-circuit table — build `data/pit_loss.json` from FastF1:
  median (pit-in + pit-out lap delta vs driver's green median) per circuit; fallback 22s);
  tyre_age=1, compound=X.
- **Undercut model:** when pitting, position vs the car ahead is recomputed by projecting both
  cars 3 laps forward (fresh-tyre delta from the tyre curves vs their old-tyre delta) — if the
  projected cumulative gap flips sign, positions swap. This is explicit, not hand-waved.
**Reward:** −(total race time); terminal bonus for each position gained. Value iteration over
the discretized grid in NumPy (<1s). **Deliverables:** optimal policy heatmap (lap × tyre_age →
action) per circuit/compound — a killer dashboard visual; comparison vs greedy threshold AND vs
the actual historical strategies of 3 races (did the MDP recommend what winners did?). Tests:
transition determinism, value iteration convergence, undercut sign-flip case. Model card.

### 10.4 ✅ Phase 4b — DONE in v3.2 (spec kept as reference; implemented + validated, 0.79
### replay coverage. Remaining optional: Kaggle GPU sweep, SC-hazard v2 with separate ablation.)
**File: `src/simulation/engine.py`** (+ `src/simulation/__init__.py`, tests).
**Contract (define before coding):**
```python
@dataclass
class RaceSpec:    # circuit, era, n_laps, drivers: list[DriverSpec], pit_loss_s
@dataclass
class DriverSpec:  # driver, team, grid_pos, strategy: list[(pit_lap, compound)]
def simulate(spec: RaceSpec, n_rollouts: int = 1000, seed: int | None = None,
             device: str = "cpu") -> SimResult
# SimResult: finish_position_counts [driver × position], win_prob, podium_prob,
#            mean_race_time, per-lap position traces (for the dashboard replay)
```
**Per-lap loop, vectorized across rollouts (torch tensors, shape [n_rollouts, n_drivers]):**
1. Base lap time per driver = TFT median prediction context-free fallback: circuit+team green
   median from train data (simple, defensible), THEN + tyre delta sampled from
   `predict_degradation` treating (hi−lo)/2 as ~1.96σ Gaussian, THEN + fuel-corrected offset,
   THEN + per-lap noise σ from the (recalibrated!) TFT band width for that circuit.
   (v1 of the engine does NOT need to call the TFT net per lap — sampling from its
   *distributional summaries* per (circuit, era) is the right cost/benefit. Document this.)
2. Pit stops per strategy: add pit loss, reset tyre age/compound.
3. Track position by cumulative time; overtaking friction: a car within 1s of the car ahead
   only passes with probability p_overtake (circuit-dependent: low Monaco ~0.05/lap, high
   Spa ~0.35/lap — start with a 3-value table {street: .08, medium: .2, power: .3}, refine later).
4. Optional v2: SC hazard per lap (Poisson, rate from historical SC frequency per circuit) —
   include only after v1 validates; it changes everything and must be ablated separately.
**Validation (the credibility step, non-negotiable):** replay ≥3 historical races (e.g.
Bahrain 2024, Monaco 2025, Suzuka 2025) with the ACTUAL strategies → check the actual finishing
order sits inside the simulated distribution (report per-driver percentile of actual finish;
calibration histogram across drivers/races). Write `reports/simulation/validation.md` with
these numbers — honest, even where it misses.
**Strategy recommendation API:** `recommend(spec, candidate_strategies) -> ranked table with
win/podium/points probabilities + uncertainty`. The MDP supplies candidate strategies; the sim
ranks them. This composition (MDP proposes, sim disposes) is the line for the writeup.
**Then:** GPU sweep on Kaggle (~4h budget): N=10k rollouts × all 1/2-stop strategies × 6
2026 races → `reports/simulation/strategy_tables/`.

### 10.5 Phase 5 — RL pit agent (STRETCH; Kaggle ~10h)
Only start if 3.5 + 4a + 4b are merged and validated. `pip install stable-baselines3 gymnasium`
(Section 3 ordering). Wrap the engine as a Gymnasium env (single ego driver, others on fixed
strategies; obs = MDP state vector + recent lap deltas; action = MDP action set; reward =
−race_time/1000 + position bonus at terminal). PPO, 2–5M steps, 3 seeds; checkpoint every 500k.
Compare to the MDP policy in the SAME sim (paired rollouts, same seeds). **A documented "RL
matched but didn't beat the MDP" is a legitimate result** — write it up either way.

### 10.6 Phase 6 — API + dashboard (LOCAL CPU, ~2–3 days)
FastAPI (`src/api/main.py`): endpoints exactly as Section 12; pydantic request/response models;
all models loaded once at startup from `models/` (joblib + tyre curves + calibration json; TFT
optional behind a feature flag since serving may lack pf). Streamlit (`dashboard/app.py` +
`pages/`): Pre-Race / Live-Replay / Post-Race as Section 12; replay mode reads historical
parquet — NO live-timing dependency. Visual bar: dark theme, F1-style red/white accents, the
MDP policy heatmap, sim win-probability distributions as horizontal stacked bars, tyre curves
with CI ribbons. Screenshot everything for the README.

### 10.7 Phase 7 — Deploy + write-up (~2 days)
Bundle: small curated data subset + CPU artifacts into the image (models/ and data/ are
gitignored — this is the known gap, solve it HERE). HF Spaces preferred over Render (RAM).
Re-ingest all 2026 races run by then; final cross-era eval; update every number in README +
model cards. 3–5 page technical report PDF: problem → architecture → honest results →
limitations → what-I'd-do-differently. Record a 2-min demo video/GIF of the dashboard.

---

## 11. The models (architecture detail)

### Model 1 — Lap Time Predictor
**Baseline ladder (Phase 1):** BayesianRidge (proper one-hot/target encoding for the linear model) → RandomForest → **LightGBM tuned with Optuna** (the bar to beat). Target `LapTimeSeconds`, features = `MODEL_FEATURE_COLUMNS`.

**Headline (Phase 2): Temporal Fusion Transformer.** Use `pytorch-forecasting`'s `TemporalFusionTransformer` (pragmatic path; custom PyTorch fallback if the version triangle fights back).
- **Sequence unit:** laps within a stint (`GroupID` = Season_Round_Driver_StintID), `time_idx` = contiguous lap-in-stint.
- **Static categoricals:** `CircuitKey`, `Team`, `EraStr`, `CompoundStr` — raw STRING cols with `NaNLabelEncoder(add_nan=True)` (library owns cardinalities; add_nan = unseen→sentinel). NOT the int-encoded versions (those are for the tree models).
- **Time-varying known reals:** `time_idx`, `NormLapNumber`, `FuelLoad`, `TyreLife`.
- **Time-varying unknown (observed) reals:** `TrackTemp`, `AirTemp` (+ target `LapTimeSeconds`).
- **Target normalizer:** `EncoderNormalizer()` (per-stint, from its own encoder window) — see Errors #12.
- **Output:** quantile predictions (e.g. 0.1/0.5/0.9) → median + calibrated interval. This replaces v2's MC-dropout hack with native uncertainty.
- **Why TFT not TabTransformer:** laps in a stint are a genuine sequence with degradation dynamics; the A6000 makes a real sequence model trivial; quantile loss gives honest uncertainty; variable-length stints are handled by the library's encoder/decoder + masking.
- **Feature roles by NAME, never positional index.** Declare `static_categoricals` / `time_varying_known_reals` / `time_varying_observed_reals` by column name; pytorch-forecasting builds the categorical encoders and cardinalities itself. The retired TabTransformer used positional constants (`CAT_FEATURE_INDICES`, `CONT_FEATURE_INDICES`, `CAT_CARDINALITIES`) that silently broke when a column was inserted — those are removed in v3 and must not return.
- **Why TFT not a giant transformer:** data is small. TFT is the right *size* and is interpretable (variable-selection + attention weights are inspectable — great for the writeup).
- **Honesty rule:** keep TFT as headline only if it beats LightGBM on val. Report per-era MAE either way — the cross-regulation story is the real result, not raw aggregate MAE.

### Model 2 — Tyre Degradation
`scipy.optimize.curve_fit`, quadratic `lap_time = a + b·age + c·age²`, per (compound × circuit × era), 95% CI from covariance. Hierarchical pooling for thin cells. Interpretable coefficients (a=baseline, b=linear deg, c=cliff). Physics-informed, works with tiny 2026 data, analytic CIs.

### Model 3 — Pit Strategy MDP
- **State:** (lap, tyre_age, compound, position, gap_ahead, laps_remaining, era)
- **Actions:** {pit_now, stay_out}
- **Transition / undercut model (DEFINE EXPLICITLY, don't hand-wave):** staying out → tyre_age+1, lap time from tyre+lap models (degrading); pitting → ~22s pit loss (per-circuit adjustable) + fresh tyres, position recomputed by comparing cumulative race time to rivals' projected times. The undercut is modeled as the time gained on fresh vs old tyres over the next 2–3 laps relative to the car ahead. Write this transition function down before coding value iteration.
- **Reward:** position gain minus pit loss. Value iteration in NumPy, <1s. Compare vs greedy + historical.

### Model 4 — Monte Carlo Race-Simulation Engine (THE SHOWCASE)
For a given race: simulate lap-by-lap for all cars. Each lap, sample lap time from the lap-time model's predictive distribution; apply tyre degradation sampled from the tyre model's CI; execute pit stops per a candidate strategy; update positions from cumulative time with a simple traffic/overtaking penalty. Run N≥1000 rollouts. Aggregate → win-probability and finishing-position distributions per strategy. **Vectorize across rollouts with torch tensors** (GPU for batch sims, CPU for serving). **Validation:** replay historical races and check the simulated outcome distribution covers what actually happened. Finishing-order prediction (the cut classifier) falls out of this for free.

### Model 5 — RL Pit Agent (STRETCH)
Gymnasium env = the simulator. PPO or DQN. State as in the MDP. Reward = final position / negative race time. Compare to MDP head-to-head. A documented "RL didn't beat the MDP here's why" is a legitimate result.

---

## 12. Dashboard, API, deployment

**Streamlit 3-page app** (functional) + the existing **HTML artifact as a static showcase landing page** (visual polish, near-zero cost).
- **Pre-Race:** predicted lap-time distributions by team/driver; MDP/sim strategy recommendation (1/2/3-stop with win probabilities); tyre degradation curves + CIs; optional Claude-API analyst summary.
- **Live Race (replay mode):** lap-by-lap actual vs predicted for top drivers; compound/tyre-life tracker; pit-window alerts; position-change probabilities; optional live Claude-API commentary. Replay of historical data is the demo mode — perfectly acceptable and better for demos than fragile live timing.
- **Post-Race:** predicted vs actual MAE per driver; "was the actual strategy optimal vs the sim?"; tyre deg actual vs fitted; error-spike moments (likely SC/incidents); optional Claude-API post-race report.

**FastAPI endpoints:** `POST /predict/lap-time` (with interval), `POST /predict/strategy` (sim → win-prob distribution), `GET /tyre/degradation/{circuit}/{compound}`, `POST /simulate/race`, `GET /health`. JSON with prediction + confidence.

**Deployment (Phase 7, lower priority — and the v2 gap to solve):** the A6000 trains; a light CPU host serves. **`data/` and `models/` are gitignored, so the deployed app has nothing to load.** Solve by: bundling a small curated data subset + the exported CPU model artifacts into the Docker image (or Git LFS / external object store). Decide this at deploy time. Render free tier sleeps and has ~512MB RAM — verify the TFT CPU artifact + FastAPI fit, or use HF Spaces. Claude API in the dashboard needs a key + costs money — keep it optional/feature-flagged.

---

## 13. Evaluation rules — always apply

1. Season-aware splits only. Never shuffle.
2. Per-circuit MAE always (aggregate hides circuit failures).
3. Per-era MAE — 2022-2025 separate from 2026. **This is the headline metric.**
4. **Green-flag vs all-laps MAE split** — report both; the SC/restart impact is a talking point. Green = `TrackStatus == "1"` for the whole lap (eval segmentation only, never a model feature — SC isn't known in advance).
5. Baseline ladder: TFT must beat LightGBM, LightGBM beats RF, RF beats Ridge — or report honestly why not.
6. Every prediction needs uncertainty: TFT quantiles (verify calibration), tyre CIs, sim distributions.
7. **Validate the simulation against historical races** — does its outcome distribution cover reality?
8. Log everything to MLflow.
9. FastF1 cache always enabled.
10. Realistic targets: once circuit + team + fuel + tyre are captured, green-flag per-circuit MAE should be well under 1s. Treat 3s as a ceiling, not a goal.

---

## 14. Compute rules (v3 — updated for GPU)

1. FastF1 cache enabled — non-negotiable.
2. **A6000 for training** (TFT, Optuna sweeps, batch sims, RL). Kaggle/Colab for quick experiments.
3. **Right-size models to the data, not the GPU.** A bigger model because VRAM is free is the wrong instinct — watch for overfitting on small data.
4. Save every model immediately: `joblib.dump` (sklearn/lgbm/tyre), `torch.save`/Lightning ckpt (TFT), and **export a CPU-loadable TFT artifact for serving.**
5. `n_jobs=-1` on all sklearn/lgbm.
6. Serving runs on CPU — all models are small at inference. Never assume the serving host has a GPU.
7. Pin the torch / lightning / pytorch-forecasting versions once they work; record in Section 3.

---

## 15. Key interview answers

**"Why a TFT and not just LightGBM?"** LightGBM is my baseline-to-beat and on data-rich circuits it's very strong — I report that honestly. The TFT earns its place two ways: laps within a stint are a sequence with degradation dynamics a tree treats as independent rows, and its quantile outputs give calibrated uncertainty the simulation engine needs. The cross-era generalization, not raw MAE, is where it justifies itself.

**"Why not a giant transformer — you have a 48GB GPU?"** Because the data is small. The GPU lets me train the *right* model fast and run thousands of simulations; it doesn't make a bigger model correct. Over-parameterizing a few hundred thousand laps would overfit. I sized the TFT to the data and put the GPU's real value into the simulation engine and Optuna sweeps.

**"What's the simulation engine and why does it matter?"** It composes the lap-time, tyre, and pit-loss models *with their uncertainties* and runs thousands of full-race rollouts to output strategy choices as win-probability distributions, not point estimates. That's what real strategy teams do — and it's the difference between "I trained a model" and "I built a system."

**"How did you prevent leakage?"** Temporal splits by season (train 2022-24, val 2025, test 2026). Circuit and team encodings frozen from train; unseen → sentinel -1. Every feature is known at the start of the lap predicted. Test set untouched until final evaluation.

**"How do you handle the 2026 regulation change?"** An explicit `Era` feature, era as a static input to the TFT, tyre curves fit separately per era, and per-era MAE reported separately so I can state honestly how much accuracy degrades across the boundary. The 2026 test set is small and noisy and I say so.

**"Biggest weakness?"** No live telemetry — tyre temps, fuel flow, ERS deployment, active-aero state. The 2026 energy-management and active-aero changes are especially impactful and invisible in public data. And the 2026 test set is only six races.

**"What would you build next?"** The RL pit agent trained in the simulator (if not already done), and a safety-car probability model feeding the simulation so strategy recommendations account for SC risk.

---

## 16. Errors log — fixed bugs, DO NOT reintroduce

Read before writing any code. (Carried from v2 — all still valid.)

1. **Wrong FastF1 event key** — use `session.event["EventDate"].year`, not `["year"]`. FIXED.
2. **Wrong column names** — `Season`/`RoundNumber`/`CircuitKey`, never `Year`/`Round`/`Circuit`. FIXED.
3. **TyreLife <= LapNumber check too strict** — removed; carried-over tyres legitimately exceed lap number. `checks=[]`. Never add it back. FIXED.
4. **train.py imported nonexistent `fetch_race`** — train.py does NOT import ingest; `load_data()` reads parquet. FIXED.
5. **MLflow Windows path** — always `(PROJECT_ROOT / "mlruns").as_uri()`. FIXED.
6. **pytest.approx on Series returns np.False_** — loop element-wise. FIXED.
7. **Null Compound = string "None"** — mask `isna() | == "None" | ~isin(VALID)`. FIXED.
8. **Sprint laps contaminating data** — primary defense is `session_type="R"` at ingestion; the `mean>105s` filter is a safety net only and may catch wet races, so don't rely on it. FIXED (with caveat).
9. **RF not beating baseline** — original root cause was no circuit encoding. v3 added CircuitEncoded + TeamEncoded, but RF still overfits (sklearn can't split integer-coded categoricals natively) and collapses across the era boundary (test MAE >> val, because it never saw Era=1 / new 2026 teams in train). This is now EXPECTED and KEPT ON PURPOSE: RF is the naive-tree rung and the cross-era counter-example — LightGBM (native categorical handling + Optuna + regularization) holds val≈test across the regulation boundary, which is the project's headline contrast. RESOLVED — do not "fix" RF, do not one-hot it (muddies the lesson). LightGBM is the selected lap model.

**v3 watch-items (not yet bugs, prevent them):** Monaco 75s floor clipping; pytorch-forecasting version conflicts; tyre curve_fit failing on single short stints (needs pooling fallback); deployment host missing data/models (gitignored).

10. **`load_data` double-counted every lap** — the `data/raw/*.parquet` glob matched both per-round files (`laps_YYYY_rNN.parquet`) AND `laps_YYYY_full.parquet` aggregates, so every lap loaded twice. Pre-fix MAEs/tuning/row-counts were on duplicated data. Fix: glob per-round only (`laps_*_r*.parquet`) + `.drop_duplicates(["Season","RoundNumber","Driver","LapNumber"])` guard after concat; `_full` aggregates must never be globbed by the trainer. FIXED in Phase 1.

11. **TFT can't reuse `build_features()` output directly** — it restricts to `FEATURE_COLUMNS` and never calls `add_stint_id()`, so raw `Team`/`Compound` strings, `StintID`, and `TrackStatus` are gone. The first TFT draft assumed they survived and crashed immediately (`build_features(validate_laps(load_data()))` also double-validated a frame whose Compound string was already dropped). Fix: `train_tft_data.load_tft_data()` carries those cols back by merging on `LAP_KEYS` after build_features (validate runs once). Mirror this for any new lap sequence/eval code. FIXED in Phase 2.

12. **TFT `GroupNormalizer(groups=["GroupID"])` KeyErrors on val/test** — GroupID is unique per stint, so a per-group target scale learned on train is missing for every unseen 2025/2026 stint → `KeyError: "Unknown category '2025_10_ALB_1'"`. Worked on a 2023-only smoke only because all groups were in train. Fix: `target_normalizer=EncoderNormalizer()` (scales each stint from its own encoder window, generalizes). General rule: no group-keyed encoder on a per-series id under this project's season splits. FIXED in Phase 2.

13. **mlflow 3.x blocks the file:// store** — raises `MlflowException: filesystem tracking backend ... in maintenance mode` on the project's `mlruns` URI. Fix: `os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE","true")` at import in train_tft_data.py (before any mlflow call). The same will hit `train.py`/`evaluate.py` on mlflow 3.x — set it there too or migrate to `sqlite:///mlflow.db`. FIXED in Phase 2 (train_tft path).

14. **pytorch-forecasting / torch version mismatch** — pf 1.1.1 (an early pin) is incompatible with torch 2.11 (the server/venv torch). Use pf 1.7.0 + lightning 2.6.5 (Section 3). pf 1.7.0 also dropped `stop_randomization` and changed the `predict()` return shape (handled defensively in `_extract_index`). RESOLVED in Phase 2.

15. **Kaggle P100 + cu128 torch = `CUDA error: no kernel image`** — Kaggle's stock torch (2.10+cu128) is built for sm_70+ only; the P100 is sm_60, so it has no kernels at all. Fix: select **GPU T4** (sm_75, in the build), not P100. cell 1 of `notebooks/04_tft_kaggle_fullrun.ipynb` asserts the GPU arch to fail fast. RESOLVED in Phase 2.

16. **TFT eval scored only the last lap per stint** — `from_dataset(..., predict=True)` keeps one window per series (the most-degraded last lap), so green MAE looked ~3-4s and lost to LightGBM. It also made `val_loss` (hence EarlyStopping/checkpoint) track that hard subset, halting training early at a worse model. Fix: `predict=False` for val/test → one prediction per decodable lap = fair per-lap comparison. After the fix TFT beats LightGBM (val 1.11 / test 1.67 green). One change fixed both the metric and the training. RESOLVED in Phase 2.

17. **Production mappings clobbered + built from full data** — `add_circuit_encoding/add_team_encoding(mapping=None)` wrote `data/mappings/*.json` on EVERY call, so any test/notebook run on a tiny frame overwrote the production mappings (found on disk: `{"bahrain": 0}` and a 5-team map). Additionally `train.load_data()` built mappings on ALL seasons, so 2026-only teams (Audi, Cadillac, Racing Bulls) got real codes instead of −1 — predictively equivalent (codes unseen in train behave like −1) but violates the frozen-from-train contract and would corrupt serving. Fix (v3.1): `mapping=None` never persists; `freeze_encoding_mappings(train_rows)` is the only JSON writer (called by train.py on TRAIN seasons); `evaluate.py`/serving use `load_encoding_mappings()`; `assert_no_leakage` now asserts val/test-only codes are exactly −1. Ladder retrained post-fix: all MAEs moved ≤0.03s. FIXED in v3.1.

18. **test_tft.py broke the whole pytest collection locally** — module-level import of `lightning` (absent from local venv; TFT deps live on Kaggle) made `pytest tests` ERROR during collection, killing all other tests. Fix: `pytest.importorskip("lightning")`/`("pytorch_forecasting")` at the top of test_tft.py. Also removed `stop_randomization=True` from a test (dropped in pf 1.7.0 — would have failed on the pinned Kaggle env too, Error 14). FIXED in v3.1.

19. **Stale v2 tyre fit.py reintroduced Error #10** — the pre-v3.1 `src/models/tyre/fit.py` ("Week 5" header) globbed `*.parquet` (double-counting via `_full` aggregates), fit raw deltas with no fuel correction, no green-flag filter, no pooling, and ignored the stint's starting age (biased b). Fully rewritten in Phase 3 (Section 5). Lesson: when a phase ships, grep for older drafts of its files. FIXED in v3.1.

20. **Conformal calibration on all laps produced a useless 8.4s shift** — 2026 R1 (the era-1 calibration slice) is SC-heavy; SC laps put +20-40s errors in the nonconformity tail, so the 80th-percentile shift exploded (band width 20.7s, era-1 "coverage" 0.91 by absurd width). Fix: calibrate AND evaluate on green-flag laps only — that is the distribution the sim engine samples from. After fix: era0 shift 0.082s (coverage 0.800 = nominal), era1 0.868s (0.879). Rule: every quantile-band consumer in this project is green-scoped. FIXED in v3.2.

21. **GPU-trained ckpt crashes on CPU load: "Torch not compiled with CUDA enabled"** — the Lightning checkpoint pickles torchmetrics objects (loss, logging_metrics) whose `_device` attribute is cuda; any `.to()` then tries to create a cuda tensor. `map_location="cpu"` does NOT fix attributes. Fix: pass fresh CPU objects as `load_from_checkpoint` kwargs (`loss=QuantileLoss(quantiles=...)`, `logging_metrics=torch.nn.ModuleList()`); fallback monkeypatch pins `torchmetrics.Metric.device` to cpu. FIXED in v3.2 (recalibrate.py).

22. **Driver/engine feature ablation (not a bug — a finding to not "fix")** — DriverEncoded improves val green (2.16→2.14) but degrades test green (2.11→2.22): drivers switch teams across the 2026 boundary and rookies are unseen. Do NOT remove the features to chase the test number (that is test-set selection); the model card documents both rows and the mechanism. EngineEncoded ≈0 SHAP in-era is EXPECTED (era-0 engine is constant within team) — judge it on the TFT v2 run. RECORDED in v3.2.

---

## 17. Architecture diagram

```
data/raw/*.parquet (2022-2026, 6 races in 2026)
        |
   ingest.py  -> Season, RoundNumber, CircuitKey, Team, LapTimeSeconds,
        |         Compound, TyreLife, TrackTemp, AirTemp, IsPitLap
   validate.py -> filters + Pandera (verify Monaco floor)
        |
   features.py -> CircuitEncoded, TeamEncoded(NEW), CompoundEncoded, Era,
        |          Fuel*, TyreAge*, CompoundXTyreLife, NormLapNumber, StintPhase, StintID
   splits.py   -> train 2022-24 / val 2025 / test 2026 (season-aware)
        |
        +---------------------+--------------------+
        v                     v                    v
  lap_time (ladder)     tyre/fit.py          (StintID feeds tyre + TFT)
   Ridge->RF->LGBM       curve_fit per
        |                compound x circuit x era + CI
   train_tft.py (A6000)        |
   TFT + quantiles             |
        |                      |
   evaluate.py (per-era, green-flag, calibration, comparison)
        |                      |
        +----------+-----------+
                   v
        simulation/engine.py  <-- THE SHOWCASE
          composes lap-time + tyre + pit-loss WITH uncertainty
          N>=1000 vectorized rollouts -> win-probability distributions
          validated against historical races
                   |
        +----------+-----------+
        v                      v
   mdp.py (pit baseline)   rl_agent.py (STRETCH, trained in engine)
        |
        v
   api/main.py (FastAPI): /predict/lap-time, /predict/strategy,
        |                 /simulate/race, /tyre/degradation, /health
        v
   Streamlit dashboard (Pre/Live/Post) + static HTML showcase page
        + optional Claude API analyst commentary
```

---

## 18. How Claude should behave

**Tooling:** Claude Code edits the real repo. Prefer issuing file changes as precise Claude Code prompts over pasting whole files. Net-new functions can be given as code; edits to existing files should target them by location. All shell commands are bash, never PowerShell.

**Before writing code:** check Section 3 (packages), Section 6 (column names are law), Section 8 (what's done), Section 16 (don't reintroduce bugs).

**Code style:** production Python — type hints, docstrings, `logging` not `print`, modular functions. Follow Section 5 specs. MLflow logging in all trainers. `joblib.dump`/`torch.save` + a CPU-export for the TFT. Tests alongside new code.

**When writing new code:** state signatures + data contracts first; check leakage (every feature available at prediction time); include MLflow logging + serialization; flag what to unit-test.

**When reviewing code:** column names vs Section 6 first; leakage; frozen mappings (not fit-on-predict); season-boundary splits; right-sized models (push back on over-engineering).

**Tone:** capable student learning properly. Explain the "why." Be honest when something is out of scope or when a suggestion adds scope without rigor — say so.