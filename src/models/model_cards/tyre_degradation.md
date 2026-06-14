# Model Card — Tyre Degradation Curves (Phase 3)

## Overview
Physics-informed quadratic degradation curves fit per **(compound × circuit × era)** with `scipy.optimize.curve_fit`, 95% CIs from the parameter covariance, and **hierarchical pooling** for thin cells. Feeds the Monte Carlo simulation engine (Phase 4) with per-lap tyre time-loss *and its uncertainty*.

**Model.** Fuel-corrected lap-time delta within a stint, relative to the stint's first observed tyre age `age0`:

```
delta(age) = b·(age − age0) + c·(age² − age0²)
```

This is `f(age) = a + b·age + c·age²` with the per-stint intercept `a` eliminated by differencing — so driver/car/track-day baselines never contaminate the degradation estimate. `b` = linear degradation (s/lap), `c` = cliff acceleration.

**Three design decisions that matter:**
1. **Fuel correction first.** Cars get ~0.045 s/lap faster as fuel burns; raw lap-time trends *understate* degradation by that amount. `FuelEffect` (project constant, 0.03 s/kg × 1.5 kg/lap) is subtracted before fitting.
2. **Green-flag laps only** (`TrackStatus == "1"`). SC/VSC laps are +20–40 s outliers that destroy least-squares fits.
3. **Stint baseline = median of the first 3 green laps** — a single-lap baseline injects that lap's noise (~0.3 s) into every delta of the stint.

**Hierarchical pooling.** Cells back off when stints are thin (esp. 2026): `(compound × circuit × era)` [≥5 stints] → `(compound × era)` [≥8] → `(compound)`. The chosen level is recorded in `pool_level` — wide CIs in low-data regimes, never false precision.

## Data
- 67,649 green-flag slick laps in 3,732 stints, 23 circuits, 2022–2026 (per-round parquet, deduped — same loader as the TFT, `load_tft_data()`).
- Slicks only (SOFT/MEDIUM/HARD). INTERMEDIATE/WET excluded: a drying track makes wet-tyre laps *faster* with age — a monotone degradation curve is the wrong model. Documented out of scope.
- Stints ≥ 5 green laps; |delta| ≤ 8 s outlier guard (traffic/off-track).

## Results
- **84 cells fitted: 76 at circuit level, 8 pooled to era level** (all era-pooled cells are 2026 — exactly the regime pooling exists for).
- Era-0 mean linear degradation by compound: HARD 0.024, MEDIUM 0.033, SOFT 0.025 s/lap, with SOFT carrying the largest cliff term (mean c ≈ 0.0033 — quadratic loss dominates soft stints past ~15 laps). Physically sensible ordering.
- Best-determined cells (era 0): Bahrain MEDIUM b=0.122±0.018 (r²=0.43), Imola HARD b=0.049±0.008 (r²=0.42), Bahrain HARD b=0.075±0.013 (r²=0.39) — the classic high-degradation circuits, as expected.
- **Mean circuit-level r² = 0.156.** Read this honestly: per-lap deltas are dominated by traffic, tyre management, and track evolution — noise the curve is *not supposed* to explain. The fitted trend (b, c) is well-determined where stints are plentiful (tight CIs); r² measures lap noise, not curve quality.
- Some street/low-deg cells fit negative b (tyre warm-up + track evolution outweigh wear, e.g. Hungary SOFT b=−0.13 with a big cliff c=0.013) — the quadratic captures the "improves then falls off a cliff" shape. 2026 SOFT/MEDIUM cells trend negative-b on very thin data; treat era-1 curves as provisional and re-fit as races accumulate.

## Temperature sensitivity (Phase 9.4)
The curves carry an optional **track-temperature adjustment** so the sim can answer "this circuit, but hotter/colder". `predict_degradation(..., track_temp=T)` shifts the central slope/curvature:

```
b_eff = max(b + b_temp·(T − temp_ref), 0)     c_eff = c + c_temp·(T − temp_ref)
```

- `temp_ref` is the cell's own historical mean track temp (so the base `b, c` are "degradation at the temperature this cell was actually run at"); `b_temp`/`c_temp` are the **(compound × era)** temperature slopes, fit once by `fit_temp_sensitivity()` over all stints in that pool (`reports/tyre/tyre_temp_sensitivity.csv`).
- **The signal is cross-sectional, and we say so.** Ingest stores one session-median `TrackTemp` per race, so the temperature effect is estimated from the *same compound run across races/circuits at different temperatures*, not from within-stint temperature change. It is therefore confounded with circuit characteristics that correlate with temperature (abrasiveness, layout). The slope is bounded (`|b_temp| ≤ 0.01 s/lap/°C`) and used only as a gentle adjustment; the CI band keeps the cell's fitted half-width (temperature shifts the centre, not the spread).
- **Fitted values (2022–2026, 6 pools):** era-0 HARD `b_temp≈0.0024`, MEDIUM `0.0043` s/lap/°C (tight CIs, n>1300 stints) — robustly positive, i.e. hotter track ⇒ faster degradation, physically correct. era-0 SOFT is slightly negative (`−0.0014`, CI overlaps 0) and era-1 pools are thin/wide (2026, n≤112) — treat as provisional. Example: MEDIUM at Bahrain, age 20 — 1.4 s loss at 20 °C vs 2.6 s at 40 °C.
- **Backward compatible:** `track_temp=None` (or pre-Phase-9 curves without the temp columns) reproduces the original behaviour exactly. Threaded through `RaceSpec.track_temp` (engine) and `build_race_model(..., track_temp=)` (MDP); surfaced on the Pre-Race dashboard as a track-temperature slider.

**Track-temp estimate for forecasts.** `src/pipeline/weather.py` approximates track surface temp from air temp + Open-Meteo solar radiation: `TrackTemp ≈ AirTemp + 1.775·(radiation/100) + 2.84`, calibrated against FastF1's measured gap over 94 races (RMSE 4.1 °C, `reports/tyre/track_temp_calibration.json`).

## Intended use / out of scope
**For:** the Phase 4 simulation engine via `predict_degradation(curves, compound, circuit, era, age, track_temp=None) -> (mid, lo, hi)`; strategy what-ifs (incl. temperature); dashboard degradation plots. **Not for:** wet-running, lap-1 warm-up effects, or as a standalone lap-time predictor (it predicts *delta within a stint*, not absolute pace).

## Limitations
No tyre-temperature or compound-spec (C1–C5) data — SOFT at Monaco ≠ SOFT at Silverstone in actual compound; circuit dimension partially absorbs this. Track evolution not modeled (rubbering-in over a weekend looks like negative degradation). Fuel correction uses the project's linear-burn estimate, not measured fuel flow. 2026 cells are thin (6 races) — most pool to era level by design.

## Reproduce
```bash
python -m src.models.tyre.fit     # fits curves, writes models/ + reports/, logs to MLflow
pytest tests/test_tyre.py -v      # synthetic-recovery + pooling tests
```
Artifacts: `models/tyre_curves.joblib` (now carries `temp_ref/b_temp/c_temp`), `reports/tyre/tyre_curves.csv`, `reports/tyre/tyre_temp_sensitivity.csv`, `reports/tyre/degradation_curves_era{0,1}.png`. MLflow experiment: `tyre_degradation`. Weather backfill + track-temp calibration: `python -m scripts.enrich_weather`.
