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

## Intended use / out of scope
**For:** the Phase 4 simulation engine via `predict_degradation(curves, compound, circuit, era, age) -> (mid, lo, hi)`; strategy what-ifs; dashboard degradation plots. **Not for:** wet-running, lap-1 warm-up effects, or as a standalone lap-time predictor (it predicts *delta within a stint*, not absolute pace).

## Limitations
No tyre-temperature or compound-spec (C1–C5) data — SOFT at Monaco ≠ SOFT at Silverstone in actual compound; circuit dimension partially absorbs this. Track evolution not modeled (rubbering-in over a weekend looks like negative degradation). Fuel correction uses the project's linear-burn estimate, not measured fuel flow. 2026 cells are thin (6 races) — most pool to era level by design.

## Reproduce
```bash
python -m src.models.tyre.fit     # fits curves, writes models/ + reports/, logs to MLflow
pytest tests/test_tyre.py -v      # synthetic-recovery + pooling tests
```
Artifacts: `models/tyre_curves.joblib`, `reports/tyre/tyre_curves.csv`, `reports/tyre/degradation_curves_era{0,1}.png`. MLflow experiment: `tyre_degradation`.
