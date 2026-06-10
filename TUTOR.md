# TUTOR.md — How this project works, end to end

*A teaching walkthrough of the F1 AI Race Strategist: every design decision, why it was made, how the pieces connect, and the answers an interviewer will probe for. Read top to bottom once, then use the interview drill at the end.*

---

## 0. The one-paragraph story (memorize this)

> "I built a system, not a model. A Temporal Fusion Transformer predicts lap times with calibrated uncertainty across the 2026 regulation change; physics-informed curves model tyre degradation with confidence intervals; an exact dynamic-programming MDP proposes pit strategies; and a Monte Carlo engine rolls out full races thousands of times — sampling from each model's *measured* uncertainty — to rank strategies as win-probability distributions. That last part is what real F1 strategy teams output, and it's validated by replaying historical races: 79% of drivers' actual finishes fall inside their simulated 80% band."

Every section below unpacks one clause of that paragraph.

---

## 1. The data and why it's the binding constraint

**Source:** FastF1 (public F1 timing data), race sessions only, 2022–2026. ~104K laps after validation. Stored as one parquet per race round (`data/raw/laps_YYYY_rNN.parquet`).

**Key insight that shaped everything: data is the constraint, not compute.** F1 produces a few hundred thousand usable laps *total*. A bigger GPU does not justify a bigger model — it justifies the *right* model trained quickly, plus compute-hungry simulation. This is why the TFT has only ~86K parameters (hidden_size 32) and why the interview answer to "why not a giant transformer?" is "because the data is small and a giant model would memorize it."

**The 2026 regulation change is the project's spine.** New aero, new power units, three new teams (Audi, Cadillac, Racing Bulls rebrand lineage). It gives a natural, *honest* generalization test: train on 2022–24, validate on 2025, test on 2026 — a real distribution shift, not a random split. Almost every design choice below exists to handle, measure, or exploit this boundary.

**Pipeline order (each step's output is the next step's contract):**
```
ingest.py → validate.py → features.py → splits.py → models
```
- `ingest.py` pulls laps per session via FastF1 (cache enabled — non-negotiable, the API is slow), keeps a fixed column set. Column names are law (Section 6 of MASTER_CONTEXT): `Season`, `RoundNumber`, `CircuitKey`, `LapTimeSeconds`, etc. Renaming a column without updating the contract is how silent bugs happen.
- `validate.py` (Pandera schema + filters): drops pit out-laps (`TyreLife==1`, ~107s outliers), lap times outside [70, 130]s, invalid compounds (including the string `"None"` — FastF1 quirk), sprint contamination (mean>105s safety net), then schema-validates with `lazy=True` so *all* violations are reported at once.
- `features.py` builds the model features (next section).
- `splits.py` splits **by season, never shuffled**. Shuffling would mix 2026 laps into training and fake the generalization result. `assert_no_leakage()` enforces disjoint seasons AND that encoded values appearing only in val/test are exactly the −1 sentinel.

---

## 2. Feature engineering — what the model knows at prediction time

Every feature must be knowable *before the lap happens*. That's the leakage test applied to each one:

| Feature | What it captures | Why no leakage |
|---|---|---|
| `CircuitEncoded` | track identity (~80% of lap-time variance — Monaco ~74s, Spa ~107s) | circuit known pre-race |
| `TeamEncoded` | car pace | team known pre-race |
| `DriverEncoded` (v3.2) | driver skill within a team | driver known pre-race |
| `EngineEncoded`/`EngineMaker` (v3.2) | power-unit supplier | supplier contracts are public pre-season |
| `Era` | regulation period (0: 2022–25, 1: 2026+) | fixed per season |
| `TyreLife`, `TyreAgeSq`, `TyreAgeCubed` | degradation, non-linear | tyre age is known |
| `CompoundEncoded`, `CompoundXTyreLife` | compound + its interaction with age | tyre already fitted |
| `FuelLoad`, `FuelEffect` | car gets ~0.045s/lap faster as fuel burns | estimated from lap number (110kg, 1.5kg/lap, 0.03s/kg) |
| `TrackTemp`, `AirTemp` | grip conditions | live sensors |
| `NormLapNumber`, `StintPhase` | race phase | known |

**The frozen-mapping contract (a real bug we fixed — great interview story).** Categorical encodings (circuit/team/driver → int) are built from the *training seasons only* and persisted once by `freeze_encoding_mappings()`. Anything unseen at val/test/serving time gets −1. Two failure modes we actually hit and fixed:
1. The encoders used to write their JSON on *every* call — so unit tests running on toy data silently overwrote the production mappings (we found `{"bahrain": 0}` on disk).
2. Mappings were built on *all* seasons, so 2026-only teams got real codes instead of −1. Predictively harmless (a code never seen in training behaves like −1 to LightGBM), but it violated the frozen-from-train contract that the serving path depends on. We fixed it, retrained, and verified the numbers moved ≤0.03s — which is itself evidence the fix was a contract correction, not a results change.

**EngineMaker is different from the other categoricals — and knowing why matters.** It is *not* learned from data; it's a hand-written, season-aware dictionary (Aston Martin: Mercedes→Honda in 2026; Alpine: Renault→Mercedes; Red Bull: Honda→Red Bull Ford; Audi=Audi; Cadillac=Ferrari customer). It's domain knowledge fixed before any data is seen, like the era boundary — so it needs no train-freezing and carries no leakage. Its *purpose* is cross-era transfer: Cadillac is unseen as a Team in 2026, but the model can still know "Ferrari engine."

**The driver/engine ablation (v3.2) — honesty in action.** Adding driver+engine improved val green MAE (2.16→2.14) but *worsened* test green (2.11→2.22). Why: driver identity is partly a proxy for the car; drivers switch teams across the boundary and 2026 rookies are unseen. The crucial discipline: we did **not** remove the features to chase the test number — that would be *selecting on the test set*, the exact sin the season-split exists to prevent. Selection stays val-based; both rows live in the model card with the mechanism explained. SHAP also showed `DriverEncoded` (0.32) absorbing most of `TeamEncoded`'s credit (0.43→0.08) — driver is nested inside team per season, so the attribution between "car" and "driver" is not identifiable from this data. Saying that out loud is an interview win, not a weakness.

---

## 3. The lap-time models — a ladder, not a leap

**Philosophy: every model must beat the rung below it, or the failure gets documented.** The ladder also produces the interview-ready contrasts.

1. **BayesianRidge** (linear baseline). Integer category codes are meaningless to a linear model, so it gets one-hot encoding + scaling, bundled in one sklearn Pipeline (fit on train only — no leakage, no separate scaler artifact to desync). Green MAE ~3.1s.
2. **RandomForest — deliberately kept broken.** It collapses across the era boundary (green 5.63 val → 8.30 test) because sklearn trees can't treat integer codes as categories (they split them as *numbers*, learning nonsense like "team code < 7.5") and it never saw era-1 values. **Do not fix it, do not one-hot it.** It is the documented counter-example that motivates the next rung.
3. **LightGBM + Optuna (60 trials)** — the "model to beat." Native categorical handling + regularization = holds flat across the boundary (green 2.14 val / 2.22 test with the 16-feature set). This flatness *is* the cross-era headline for tabular models.
4. **TFT — the headline** (next section). Roughly halves LightGBM's green MAE.

Evaluation always reports: per-circuit MAE (aggregates hide circuit failures), per-era MAE (the headline split), and **green-flag vs all-laps** (`TrackStatus == "1"`). Safety-car laps are unpredictable from pre-lap features — ~92% of laps are green; the SC laps account for most of the all-laps vs green gap (~3.5 vs ~2.1 for LightGBM). Reporting both, instead of quietly filtering, is the honest version.

---

## 4. The Temporal Fusion Transformer — the thesis model

**The thesis: laps within a stint are a sequence, not independent rows.** A tree sees `TyreLife=15` as a static number. The TFT's encoder consumes the stint's *actual past lap times* — it watches *this* car's degradation unfolding *today* and extrapolates one lap forward. That information is 100% available at prediction time (the laps already happened), so it's legitimate autoregression, not leakage. This single difference is why the TFT halves LightGBM's error (green 1.12 vs 2.14 val).

**Setup specifics that matter:**
- Sequence unit = stint: `GroupID = Season_Round_Driver_StintID`, `time_idx` = lap-in-stint. Encoder ≤12 laps, decoder = 1 lap ahead.
- **Static categoricals** (constant per stint): `CircuitKey`, `Team`, `EraStr`, `CompoundStr` — raw *strings* with `NaNLabelEncoder(add_nan=True)` (the library owns cardinalities; unseen → NaN class ≈ the −1 sentinel). v2 adds `Driver` + `EngineMaker`.
- **Known-future reals** (the model may use the predicted lap's value): `time_idx`, `NormLapNumber`, `FuelLoad`, `TyreLife` — all deterministic given the lap number. **Observed reals** (past only): `TrackTemp`, `AirTemp`, and the target itself.
- **Quantile loss** at 0.1/0.5/0.9 → native prediction intervals, which the simulation engine needs. This is why a TFT and not, say, a plain LSTM with MSE.
- ~86K parameters. Right-sized to the data, deliberately.

**Three bugs we hit that are worth retelling (all in the errors log):**
1. **`GroupNormalizer` keyed on GroupID** — every val/test stint is a *new* group under season splits → KeyError on all of them. Fix: `EncoderNormalizer` (scales each stint from its own encoder window — generalizes to any new stint). General rule: never key an encoder on a per-series ID when splits guarantee unseen series.
2. **`predict=True` evaluated only the last lap of each stint** — the most-degraded lap — making the TFT look 3–4s bad AND making EarlyStopping optimize the wrong thing. Fix: `predict=False` → one prediction per decodable lap = fair per-lap comparison vs LightGBM. One flag fixed both the metric and the training.
3. **Kaggle P100 = sm_60, cu128 torch has no sm_60 kernels** → "no kernel image". Use T4. The notebook asserts the GPU arch in cell 1 to fail fast.

**Where the TFT is weak (report it):** street circuits. Suzuka green MAE 0.46s; Canada ~3s, Monaco ~2–2.8s. Walls, traffic, lower grip evolution — pace is less determined by the features we have. The sim treats street circuits as higher-variance, which is the correct downstream response to a known model weakness.

---

## 5. Calibration and conformal recalibration (Phase 3.5) — the most underrated part

**The question nobody else asks of their model: when you say 80% interval, do 80% of outcomes actually land inside?** Measured: 0.75 in-era, 0.66 cross-era (green laps) — the TFT is *overconfident*, worse under the regulation shift. If the sim sampled from those raw bands, every win-probability it outputs would be overconfident too. Garbage uncertainty in → confident-sounding garbage out.

**The fix: split-conformal widening.** For each calibration lap, compute the nonconformity score `s = max(q10 − y, y − q90)` (how far outside the band the truth fell; negative = inside). Take the finite-sample-corrected 80th percentile of those scores, ŝ, and widen both band edges by ŝ. Theorem: on exchangeable data the widened band has ≥80% coverage — distribution-free, no retraining.

**The two implementation decisions that make it defensible:**
1. **Calibration and evaluation never share laps.** Era 0: calibrate on odd 2025 rounds, report coverage on even rounds. Era 1: calibrate on 2026 R1, report on R2–R6 (R1 is sacrificed from coverage reporting and we say so). Scoring coverage on the data you calibrated on is circular — the guarantee makes it trivially true.
2. **Green-flag laps only.** First attempt calibrated on all laps; 2026 R1 was safety-car-heavy, the SC laps' +20–40s errors landed in the score tail, and the shift exploded to 8.4s (a 20.7s-wide band that "covers" by being useless). The sim samples *green racing pace* from these bands, so green is the right target distribution. Result after fix: era-0 shift +0.082s → coverage **0.800 exactly**; era-1 shift +0.868s → 0.879 (overshoot from a one-race calibration set — conservative, the safe direction for decision-making).

Output: `models/tft_calibration.json` — two numbers the engine reads to widen its sampling σ.

---

## 6. Tyre degradation (Phase 3) — physics-informed, uncertainty-honest

**Model:** within a stint, fuel-corrected lap-time delta vs the stint baseline:
`delta(age) = b·(age − age₀) + c·(age² − age₀²)` — which is the quadratic `a + b·age + c·age²` with the per-stint intercept `a` *differenced out*. So driver/car/track-day baselines can never contaminate the degradation estimate; only the *shape* over age is fit. `b` = linear wear (s/lap), `c` = the cliff.

**Three corrections that separate this from a naive fit:**
1. **Fuel correction first.** Cars get ~0.045 s/lap faster as fuel burns — almost exactly cancelling typical degradation. Fit raw lap times and you'll *underestimate* wear. Subtract `FuelEffect` before fitting.
2. **Green-flag laps only** — one SC lap is +30s; least squares would chase it.
3. **Baseline = median of the stint's first 3 green laps** — a single-lap baseline injects that lap's noise (~0.3s) into every delta as an offset. Switching to the median *doubled* mean r² (0.085→0.156).

**Hierarchical pooling — the small-data answer.** Fit per (compound × circuit × era) when ≥5 stints; else pool to (compound × era) [≥8]; else (compound). The chosen level is recorded per cell (`pool_level`). All 8 era-pooled cells are 2026 — exactly the thin regime pooling exists for. Wide CIs in low-data regimes; never false precision.

**Reading the results like a scientist:** mean r²=0.156 looks "bad" until you understand what r² measures here — per-lap deltas are dominated by traffic and tyre management, noise the curve is *not supposed to explain*. What matters is whether `b` and `c` are well-determined (tight CIs where stints are plentiful: Bahrain HARD b=0.075±0.013) and physically ordered (era-0 means: HARD 0.024 < MEDIUM 0.033 s/lap; SOFT has the largest cliff term c≈0.0033). Some cells legitimately fit *negative* b with positive c — warm-up + track evolution beating wear early in stints (e.g. Hungary SOFT) — "improves, then falls off a cliff" is a real shape.

Excluded: INTERMEDIATE/WET. A drying track makes wet-tyre laps *faster* with age — a monotone wear curve is the wrong model class. Out of scope, documented.

**Interface to the sim:** `predict_degradation(curves, compound, circuit, era, ages) → (mid, lo, hi)` with compound-level fallback for unknown cells.

---

## 7. Pit-loss measurement — small, but it grounds everything

A constant "pit loss ≈ 22s" is hand-waving. Instead, measured per circuit from 2,153 green-flag pit events:
`loss = (in-lap + out-lap) − 2 × driver's green median that race`, both laps green (pitting under SC is genuinely cheaper and would bias the loss low), clipped to [5, 60]s. Per-circuit medians: Spa 19.6s … Silverstone 29.4s, Qatar 29.7s; global fallback 23.05s. Output: `data/pit_loss.json`.

---

## 8. The MDP (Phase 4a) — exact, interpretable, deliberately single-car

**Formulation:** finite-horizon deterministic MDP, solved by backward induction (with a fixed race length one backward sweep is *exact*, runs in milliseconds — calling it "value iteration" is fine; it's the finite-horizon special case).

- **State:** (lap, compound, tyre_age≤45, used_two_compounds). The last bit encodes the F1 rule that you must run two different slick compounds — terminal value is +∞ if violated, which elegantly forces ≥1 stop.
- **Actions:** stay_out, pit→SOFT, pit→MEDIUM, pit→HARD (pit at the *end* of the lap; new stint starts at age 2 because validation drops out-laps).
- **Lap cost:** circuit green median + fuel delta (vs race-average fuel) + tyre delta from the Phase-3 curves (+ measured pit loss if pitting).

**Results check against reality:** Bahrain → 2-stop (classically a 2–3 stop race ✓), Spain → 2-stop ✓, Monaco → 1-stop ✓, Japan 2026 → 1-stop. The pit *windows* (e.g. Bahrain M→H at lap 17, H again at 37) are plausible.

**The Monaco artifact — keep it, it's the best slide in the deck.** The MDP pits on the *final lap* at Monaco: the two-compound rule must be satisfied, pit loss is pure cost with no degradation payoff there, so the optimum defers it to the last possible lap. Legal and optimal *in a deterministic world with no rivals and no safety cars* — and absurd in reality (a late SC or any traffic destroys you). This single artifact is the cleanest demonstration of **why the probabilistic simulation layer must exist on top of the deterministic optimizer.**

**Why single-car is a feature, not a cop-out:** the MDP's job is to be exact and interpretable (its full policy is a heatmap: lap × tyre-age → action). Interaction effects — traffic, undercuts, variance — belong in the simulation, where they can be *sampled* rather than crudely discretized. The composition is: **MDP proposes, simulation disposes.**

---

## 9. The Monte Carlo engine (Phase 4b) — THE SHOWCASE

**What it does:** simulates the whole field lap by lap, N≥1000 times, with every uncertain quantity sampled from this project's *own measured* models, and aggregates finishing positions into probabilities per strategy.

**What's sampled, and the provenance of each distribution (the table interviewers should see):**

| Quantity | Distribution | Source |
|---|---|---|
| per-lap pace noise | N(0, σ), σ = recalibrated band width / (2·z₀.₈) | TFT quantiles + conformal shift (§5) |
| tyre degradation | curve mid + z·(CI width/2·1.96), one z per (rollout, driver, stint) | Phase-3 covariance CIs (§6) |
| pit execution | N(circuit median, 0.8s) | measured pit-loss table (§7) |
| overtaking | Bernoulli per lap when within 1.5s: street 0.08 / medium 0.20 / power 0.30; failed pass ⇒ held 0.6s back | the one frankly heuristic piece — said openly |

Note the tyre draw is per-*stint*, not per-lap: a tyre that degrades fast degrades fast all stint — that correlation is what makes a gamble on softs actually risky in the sim rather than averaging out.

**Vectorization:** tensors shaped [rollouts × drivers]; the per-lap loop walks the field front-to-back applying the pass/hold rule. CPU does thousands of rollouts in seconds; `device="cuda"` for big sweeps. **Deliberate choice:** the engine does *not* call the TFT network per lap — at 1000 rollouts × 60 laps × 20 drivers that's 1.2M net calls for no strategy-relevant gain, since within a rollout the *band*, not the median path, carries the signal. It samples from the model's distributional summaries instead. (Say this before the interviewer asks.)

**`recommend()` uses common random numbers:** the same seed across candidate strategies, so the difference between strategies is strategy, not sampling luck — a classic variance-reduction trick worth naming.

**The engine and the MDP agree independently:** the MDP's optimal Bahrain 2-stop ((17,H),(37,H)) also tops the sim's recommendation table (win prob 0.772 vs 0.673 for the 1-stop). Two different formalisms, same answer — that's mutual validation.

**Historical-replay validation (the credibility step):** for 4 contrasting 2025 races, take the drivers' *actual* strategies, estimate each driver's pace **leave-one-race-out** (their green-median delta vs the field, averaged over the season's *other* races — this race's laps never inform its own sim, except the field-median anchor for track conditions), simulate 2000 rollouts, and check where the actual finish falls in each driver's simulated distribution. Result: **79% of 52 drivers inside their central-80% band (target 80%)**. Japan 1.00, Hungary 0.78, Monaco 1.00, Bahrain 0.53 — the Bahrain miss is SC contamination, which the v1 engine deliberately excludes (top of the v2 list, to be added with its own ablation).

---

## 10. How everything connects (the dependency map)

```
ingest → validate → features ──────────────┐
   │                                        │
   │     frozen mappings (train-only)       │
   ▼                                        ▼
LightGBM ladder (tabular bar)        TFT (sequence headline)
   │                                        │ quantiles
   │ green medians (base pace)              ▼
   │                              conformal recalibration ── tft_calibration.json
   │                                        │ σ per era
   ▼                                        ▼
tyre curves (b,c ± CI) ──────────► Monte Carlo engine ◄── pit_loss.json (measured)
   │                                   ▲         │
   └────────► MDP (proposes) ──────────┘         ▼
                                       win-probability distributions
                                       validated vs historical races
```

Trace one number through the system: *"should FAST1 one-stop or two-stop at Bahrain?"* → MDP says 2-stop saves ~9s deterministically → engine samples 2000 races where pace noise comes from the recalibrated TFT band, tyre risk from the curve CIs, pit risk from the measured table → 2-stop wins 77% of rollouts vs 67% → recommendation with a probability, not a promise.

---

## 11. Honesty infrastructure (what makes this portfolio-credible)

- **Model cards** (`src/models/model_cards/`) with real numbers, limitations, and known artifacts — including the unflattering ones (RF collapse, driver-feature test degradation, Monaco artifact, Bahrain validation miss).
- **Errors log** (MASTER_CONTEXT §16): 22 numbered bugs/findings with root causes and rules ("never key an encoder on a per-series ID", "every quantile consumer is green-scoped"). Re-introducing a fixed bug is the cardinal sin.
- **71 unit tests**, including synthetic-recovery tests (the tyre fit must recover known b,c from synthetic stints; the conformal shift must restore nominal coverage; the sim must preserve grid order when overtaking probability is zero).
- **MLflow** for every trainer; artifacts saved immediately; model binaries gitignored (download from Kaggle outputs).
- **Selection rules:** chosen model = best *val*; test is touched for reporting only. Feature choices are never made on test numbers (see the driver ablation).

---

## 12. Interview drill — likely questions, strong answers

**Q: Why a TFT instead of just LightGBM?**
A: LightGBM is my model-to-beat and I report it honestly — it's excellent on tabular features. The TFT earns headline status two ways: laps in a stint are a sequence with degradation dynamics a tree treats as i.i.d. rows (its encoder reads the stint's actual past lap times — that's what halves the error), and quantile loss gives native prediction intervals, which the simulation needs. Without the uncertainty story I might not bother with the TFT.

**Q: Isn't using past lap times leakage?**
A: No — those laps have already happened when the prediction is made. The decoder never sees the target lap's own time. It's autoregression, the same information a race engineer has on the pit wall. The leakage discipline is elsewhere: season splits, train-frozen encodings, every feature knowable pre-lap.

**Q: How do you know your uncertainty is real?**
A: I measured it — coverage of the 0.1–0.9 band was 0.75 in-era and 0.66 cross-era against 0.80 nominal, i.e. overconfident. I fixed it with era-aware split-conformal widening, calibrated on rounds disjoint from the rounds I score coverage on, green-flag laps only. Era-0 coverage landed on 0.800 exactly. The sim samples from the *recalibrated* bands.

**Q: Your tyre model r² is 0.15. Isn't that bad?**
A: r² there measures per-lap noise — traffic, management, track evolution — which the curve is not supposed to explain. The questions that matter are: are b and c well-determined (yes where data is rich — Bahrain HARD 0.075±0.013), are they physically ordered (yes — hard < medium wear, soft has the biggest cliff), and do the CIs widen honestly where data is thin (yes — 2026 cells pool to era level by design).

**Q: What's the cleverest thing in the project?**
A: The composition. The MDP is exact and interpretable but deterministic and single-car — at Monaco it "optimally" pits on the final lap, which is absurd under safety-car risk. The Monte Carlo engine fixes exactly that class of error by sampling from measured uncertainty. MDP proposes, sim disposes — and at Bahrain they independently agree on the same 2-stop.

**Q: Biggest weakness?**
A: No safety-car model — it's why Bahrain validation coverage is 0.53 while overall is 0.79, and the first thing I'd add (as a per-circuit Poisson hazard, ablated separately). No live telemetry: tyre temps, fuel flow, ERS, 2026 active-aero state. And the 2026 test set is six races — every cross-era number carries that small-sample caveat.

**Q: What did the driver-feature experiment teach you?**
A: Three things. Driver skill is real signal (SHAP 0.32). Driver and team are not separately identifiable from this data — driver is nested in team per season, and SHAP credit just moved between them. And in-era gains can cost cross-era generalization (val improved, test degraded, because drivers switch teams over the boundary) — which I report as an ablation instead of silently picking whichever looked better on test, because choosing features on the test set invalidates it.

**Q: Why didn't you call the TFT inside the simulation?**
A: Cost-benefit. 1000 rollouts × 60 laps × 20 drivers is over a million forward passes per strategy; within a rollout the strategy-relevant signal is the spread, not the median path. So the engine samples from the model's distributional summaries — recalibrated band width as σ — and spends the compute on more rollouts instead.

**Q: How would you deploy this?**
A: Training and serving are separate. Everything serves on CPU — the TFT exports a CPU artifact, the rest is joblib/JSON. Plan: FastAPI loading models once at startup, Streamlit dashboard with a historical-replay mode (no fragile live-timing dependency), bundled curated data + artifacts in the image since `data/` and `models/` are gitignored, HF Spaces over Render for RAM.

---

## 13. Glossary (fast lookups)

- **Era 0 / Era 1** — 2022–25 ground-effect regs / 2026+ active-aero + new PU regs.
- **Green flag** — `TrackStatus == "1"`, normal racing; everything else (SC/VSC/yellow) is excluded from pace metrics and reported separately.
- **Stint** — laps between pit stops on one tyre set; the TFT's sequence unit; `StintID` increments on tyre-age reset or compound change.
- **Frozen mapping** — categorical→int dict built from train seasons only, persisted once; unseen → −1.
- **Split conformal** — distribution-free interval widening using held-out nonconformity scores; guarantees marginal coverage.
- **Backward induction** — exact DP for finite-horizon problems; here, one sweep from the last lap to the first.
- **Common random numbers** — same seed across compared alternatives so the difference is the treatment, not noise.
- **Leave-one-race-out pace** — driver pace estimated from all *other* races of the season, so a replayed race never informs its own simulation.
- **Pool level** — which hierarchy level a tyre cell's curve was fit at (circuit / era / compound).
