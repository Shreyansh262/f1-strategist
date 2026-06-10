# F1 AI Race Strategist — Master Context v3
### Paste this at the start of every new Claude conversation
### v3 supersedes v2. Rewritten after gaining access to an A6000 48GB GPU. Scope = "depth, then one ambitious showcase."

---

## CLAUDE INSTRUCTIONS — READ THIS FIRST

When the user says **"update master context"**, do the following without asking questions:
1. Read this entire file
2. Update Section 8 (progress tracker) based on what was discussed
3. Update Section 9 (next actions) to reflect the immediate next step
4. Add any new bugs to Section 17 (errors log)
5. Update Section 5 (file specs) with any fixes made
6. Output the complete updated file — no truncation

When the user says **"start phase N"**:
1. Read Section 8 to confirm what's done
2. Read Section 5 for all file specs before writing any code
3. Read Section 3 before suggesting any pip install
4. Read Section 17 before writing any code — do not reintroduce fixed bugs
5. State the plan and data contracts before writing any code

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
torch 2.11 (cu128)           # server/venv torch — pf 1.1.1 is INCOMPATIBLE with torch 2.11, must use pf>=1.7
```
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

### KEEP AS-IS (GPU-agnostic, already solid — do not rewrite)
- `src/pipeline/ingest.py` — **COMPLETE.** Output columns are law (Section 6). FastF1 cache enabled at module level. `session_type="R"` loads the Grand Prix, not the sprint.
- `src/pipeline/validate.py` — **COMPLETE** (one fix pending, see below). Filter order: TyreLife==1 out-laps → LapTimeSeconds in [75,130] → invalid/"None" Compound → sprint safety-net (mean>105s) → Pandera `lazy=True`.
- `tests/test_features.py` — **COMPLETE, 24 tests passing.**
- `conftest.py` — **COMPLETE.**

### CHANGE — `src/pipeline/validate.py`
- **Monaco floor check:** Monaco race laps can dip toward ~74–75s; the `[75.0, 130.0]` band may clip legitimate laps. Verify against a real Monaco race file before trusting it; widen the lower bound (e.g. 70.0) if laps are being dropped. Log how many rows each filter removes.

### CHANGE — `src/pipeline/features.py` — STATUS: NEEDS UPDATE
Add to the existing functions:
```python
def add_circuit_encoding(df, mapping=None) -> pd.DataFrame   # frozen mapping, -1 for unseen (spec unchanged from v2)
def add_team_encoding(df, mapping=None) -> pd.DataFrame       # NEW — frozen mapping, -1 for unseen team
def add_era_feature(df) -> pd.DataFrame                       # 2022-2025=0, 2026+=1 (spec unchanged from v2)
def add_stint_id(df) -> pd.DataFrame                          # per driver per race (spec unchanged from v2)
```
**`add_team_encoding`** mirrors `add_circuit_encoding` exactly: build sorted mapping from training set if `mapping is None`, save to `data/mappings/team_map.json`, map with `.fillna(-1).astype(int)`. Team comes from FastF1 (`Team` column on laps) — add it to `extract_laps()` passthrough if not already present.

**Updated feature columns:**
```python
MODEL_FEATURE_COLUMNS: list[str] = [
    "CircuitEncoded", "TeamEncoded", "CompoundEncoded", "Era",
    "TyreLife", "TyreAgeSq", "TyreAgeCubed",
    "CompoundXTyreLife", "FuelLoad", "FuelEffect",
    "TrackTemp", "AirTemp", "NormLapNumber", "StintPhase",
]
```
For the **TFT**, features are split differently (static / known / observed) — see Section 11. `MODEL_FEATURE_COLUMNS` above is the flat list used by the linear/tree models.

### CHANGE — `src/pipeline/splits.py` — STATUS: NEEDS UPDATE
- Season-aware temporal splits only, never shuffle.
- **v3 split:** train = 2022–2024, val = 2025, test = **2026 (6 GP races available as of June 2026: Australia, China, Japan, Miami, Canada, Monaco).** Re-ingest 2026 before every evaluation as more races run. Bahrain & Saudi 2026 were cancelled — they will never exist.
- `assert_no_leakage()` must also check team mappings are frozen from train.

### CHANGE — `src/models/lap_time/train.py` — STATUS: NEEDS UPDATE (becomes the baseline-ladder trainer)
Trains the **baseline ladder** and logs all to MLflow:
1. `BayesianRidge` + `StandardScaler` — linear baseline. **NOTE:** label-encoded `CircuitEncoded`/`TeamEncoded` are meaningless to a linear model. For the linear baseline only, one-hot (or target-) encode circuit & team. Trees are fine with the integer codes.
2. `RandomForest` — kept as a documented rung (already built).
3. **`LightGBM` tuned with Optuna** — the real "model to beat." `n_jobs=-1`.
- MLflow: `MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()` (`.as_uri()` required on Windows). Experiment `"lap_time_predictor"`.
- Save: `models/bayesian_ridge_lap.joblib`, `models/rf_lap.joblib`, `models/lgbm_lap.joblib`, `models/scaler_lap.joblib`.
- Does NOT import `ingest.py` — `load_data()` reads parquet directly.

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

### CHANGE — `src/models/lap_time/evaluate.py` — STATUS: NEEDS UPDATE
Outputs to `reports/lap_time/`: `per_circuit_mae.csv`, `per_era_mae.csv` (NEW), `greenflag_vs_alllaps_mae.csv` (NEW — MAE on green-flag laps vs all laps, the SC-impact story), `calibration.png` (NEW — do the TFT quantiles cover actual values?), `learning_curve.png`, `shap_summary.png`, `model_comparison.csv` (Ridge vs RF vs LGBM vs TFT, per-era). Imports `MLFLOW_TRACKING_URI` from `train.py` — do not redefine.

### NEW — `src/models/tyre/fit.py` — STATUS: NOT STARTED (Phase 3)
`scipy.optimize.curve_fit`, quadratic `a + b*age + c*age²`, per (compound × circuit × era), 95% CI from the covariance. **Hierarchical pooling required:** if a (compound × circuit × era) cell has too few stints (e.g. thin 2026 cells), back off to (compound × era), then (compound). Wide CIs in low-data regimes, never false precision. Uses `StintID`. Saves `models/tyre_curves.joblib`.

### NEW — `src/models/pit_strategy/mdp.py` — STATUS: NOT STARTED (Phase 4)
Value iteration in NumPy. State, actions, reward in Section 11. **The transition/undercut model is the hard part and must be defined explicitly** (see Section 11) — do not hand-wave it. Compare vs greedy threshold + actual historical strategies.

### NEW — `src/simulation/engine.py` — STATUS: NOT STARTED (Phase 4 — THE SHOWCASE)
Monte Carlo race-simulation engine. See Section 11. Vectorized with torch tensors so N=1000+ rollouts are fast on GPU (also runs on CPU for serving). Saves nothing (stateless) but its outputs feed the dashboard and API.

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
  [~] Phase 2 — TFT lap model. IN PROGRESS. Trainer + carry-back loader + quicktest notebook BUILT and
                SMOKE-VALIDATED end-to-end on Colab GPU (fast run: train/val/checkpoint/green_mae/export/mlflow
                all clean). Version triangle pinned (Section 3). Bugs found+fixed via smoke: carry-back cols,
                EncoderNormalizer, mlflow file-store, version compat (Errors 11-14).
                REMAINING: full 100-epoch run on A6000 (all seasons) → beat LightGBM's 2.14s green-test
                honestly or report why not + calibrate quantiles.   <-- CURRENT
  [ ] Phase 3 — Tyre degradation model: curve_fit + hierarchical pooling + CIs.
  [ ] Phase 4 — Pit MDP + Monte Carlo simulation engine (THE SHOWCASE). Validate sim vs history.
  [ ] Phase 5 — STRETCH: RL pit agent in the simulator, vs MDP. Only if Phases 0-4 are solid.
  [ ] Phase 6 — FastAPI + Streamlit dashboard + HTML showcase page.
  [ ] Phase 7 — Deploy (light host), README, technical report, final cross-era evaluation.
```

---

## 9. Current blockers / next actions

**Phase 2 — TFT full run on A6000 (CURRENT). Trainer is built + smoke-validated; remaining is the real run:**
1. Connect to A6000 server; clone/pull repo; create env; `pip install` the pinned triangle (Section 3) + project deps.
2. Verify `torch.cuda.is_available()` and GPU name.
3. Ensure full data present on server: `data/raw/laps_*_r*.parquet` for 2022-2026 (gitignored — not in the
   GitHub clone; rsync/upload or re-ingest). Smoke used a 28-file 2025+2026 zip; the full run needs all seasons.
4. Full run: `python -m src.models.lap_time.train_tft` (100 epochs, EarlyStopping on val_loss, bs 512).
5. Read VAL/TEST `mae_all` + `mae_green`; compare green to LightGBM bar (val 2.18s era0 / test 2.14s era1).
6. Calibrate quantiles (0.1/0.5/0.9 coverage on val) — Section 13 rule 6.
7. Wire TFT into `evaluate.py` comparison table (Ridge/RF/LGBM/TFT per-era + green) using `load_tft_data()`.
8. Commit results; update model card; then "update master context".

**Watch on server:** num_workers (Colab warned 4>2 cores — fine on A6000); confirm `green_mae` predict
column shape on pf 1.7.0 (smoke ran it but on tiny data); mlflow file-store env (handled in code).

---

## 10. Phase plan (detailed)

**Phase 1 — Lap-time baseline ladder.** Fix the linear baseline encoding. Train Ridge → RF → LightGBM(Optuna). Log all to MLflow. Generate per-circuit, per-era, and green-flag-vs-all-laps MAE. Establish LightGBM as the number to beat. Restore the "candidate must beat baseline" safety check. Fill the lap-time model card with real numbers.

**Phase 2 — TFT.** Build on the A6000 (Section 11). Sequence = laps within a stint. Train 2022–2024, val 2025, test 2026. Success = beats LightGBM on val AND generalizes across the 2026 era boundary (per-era MAE), with calibrated quantiles. Export a CPU-loadable artifact for serving. Honest comparison table is the deliverable even if TFT loses.

**Phase 3 — Tyre degradation.** curve_fit per (compound × circuit × era) with hierarchical pooling fallback and 95% CIs. Diagnostic: plot predicted vs actual degradation for 3–4 circuits. Fill tyre model card.

**Phase 4 — MDP + Simulation engine (the showcase).** MDP first (interpretable baseline). Then the Monte Carlo engine that composes lap-time + tyre + pit-loss models with uncertainty. **Validate the sim by replaying historical races** — does the simulated finishing-order distribution cover the actual result? Output: strategy win-probability distributions. Fill pit-strategy model card.

**Phase 5 — RL stretch.** Wrap the simulator as a Gymnasium env, train PPO/DQN, compare to MDP. Document honestly. Skip cleanly if time/quality isn't there.

**Phase 6 — API + Dashboard.** FastAPI endpoints (Section 12). Streamlit 3-page app. HTML artifact as static showcase landing page. Optional Claude API for natural-language analyst commentary.

**Phase 7 — Deploy + write-up.** Solve the deployment data/model bundling problem (Section 12). Public live demo (lower priority). README with architecture diagram, honest results tables, dashboard screenshots, "what I'd do differently." 3–5 page technical report PDF for applications. Final cross-era evaluation on all 2026 races available by then.

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

14. **pytorch-forecasting / torch version mismatch** — pf 1.1.1 (an early pin) is incompatible with torch 2.11 (the server/venv torch). Use pf 1.7.0 + lightning 2.6.5 (Section 3). pf 1.7.0 also dropped `stop_randomization` (use `from_dataset(..., predict=True)`) and changed the `predict()` return shape (handled defensively in `green_mae`). RESOLVED in Phase 2.

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