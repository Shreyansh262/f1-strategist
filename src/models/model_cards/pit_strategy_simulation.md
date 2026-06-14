# Model Card — Pit-Strategy MDP + Monte Carlo Race-Simulation Engine (Phase 4)

## Overview
Two composed decision layers on top of the lap-time and tyre models:

1. **MDP (`src/models/pit_strategy/mdp.py`)** — a finite-horizon deterministic MDP solved exactly by backward induction. Single car, no rivals: minimises total race time over (lap × compound × tyre-age × two-compound-rule) states with actions {stay_out, pit_SOFT, pit_MEDIUM, pit_HARD}. Lap cost = circuit green median + fuel delta + Phase-3 tyre delta (+ measured pit loss when pitting). Interpretable: the full optimal policy is a heatmap you can read.
2. **Monte Carlo engine (`src/simulation/engine.py`)** — N vectorized rollouts (torch, `[rollouts × drivers]`) of the full field with uncertainty sampled from the project's own models, plus a traffic model. Outputs **win/podium/finishing-position probability distributions per strategy** — the system's headline.

**The composition is the design: the MDP proposes candidate strategies, the simulation ranks them under uncertainty and interaction.** In the Bahrain demo they agree independently: the MDP's optimal 2-stop (L17→HARD, L37→HARD) also wins the sim's recommendation table (win prob 0.772 vs 0.673 for the 1-stop).

## Inputs (all measured by this project, none hand-waved)
| Input | Source | Uncertainty propagated |
|---|---|---|
| Base pace | circuit green-flag median (+ leave-one-race-out driver delta in validation) | per-lap σ from the **recalibrated** TFT band: σ = width/(2·z₀.₈) |
| Tyre degradation | Phase-3 quadratic curves | one z-draw per (rollout, driver, stint) scaling the 95% CI |
| Pit loss | `data/pit_loss.json` — 2,153 measured green-flag pit events, per-circuit median (Spa 19.6s … Qatar 29.7s, global 23.1s) | N(median, 0.8s) execution noise |
| Fuel effect | project constants (linear 110kg burn) | deterministic |
| Overtaking | circuit-class Bernoulli per lap within 1.5s: street 0.08 / medium 0.20 / power 0.30; failed pass ⇒ held 0.6s in dirty air | sampled |

Engine deliberately samples from the lap-time model's *distributional summaries* rather than calling the TFT network per lap — thousands of rollouts × 60 laps × 20 cars would make net inference the bottleneck for no strategy-relevant gain.

## Validation — historical replay (the credibility step)
Replayed 4 contrasting 2025 races with the drivers' **actual strategies** and leave-one-race-out pace estimates; checked where each driver's actual finish falls in their simulated distribution (N=2000):

| Race (2025) | drivers | central-80% coverage |
|---|---|---|
| Bahrain | 19 | 0.58 |
| Japan | 19 | 1.00 |
| Hungary | 9 | 0.78 |
| Monaco | 5 | 1.00 |
| **Overall** | **52** | **0.81 (target 0.80)** |

(Numbers are with the swept v3 TFT bands, 2026-06-11; the v1-band run scored 0.79 overall, Bahrain 0.53.) Read honestly: aggregate coverage is on target, but Bahrain misses badly — reality included dynamics (SC phases, first-lap incidents) the v1 engine deliberately excludes. Monaco/Hungary have few full-distance finishers (attrition), so their 1.00/0.78 are thin samples. Full table: `reports/simulation/validation.csv`.

## MDP results (sanity vs racing reality)
Bahrain: 2-stop (matches the classic 2–3 stop race). Spain: 2-stop. Monaco: 1-stop (matches). Japan (2026): 1-stop. **Known artifact:** at Monaco the MDP pits on the *final lap* to satisfy the two-compound rule at minimum cost — legal and optimal in a deterministic, rival-free world, absurd under real SC risk. Kept deliberately: it is the cleanest demonstration of why the probabilistic simulation layer must sit on top of the MDP.

## Live race & flag-following (Phase 10)
The engine can now **resume a race mid-flight and follow a flag**, so the same Monte-Carlo machinery projects the *rest* of a race from its live state — no rewrite, just two additions:

- **Mid-stint resume.** `DriverSpec.start_age` sets the opening-lap tyre age (>2 = a car already several laps into a stint), so a remaining-race `RaceSpec` (laps = laps *left*) starts every driver on the tyre and age they are actually on.
- **In-progress caution.** `RaceSpec.caution = (cause, laps_elapsed)` runs the opening laps under a flag: the whole field is pinned to one dictated pace (`CAUTION_PACE_MULT` — SC 1.40× / VSC 1.30× / yellow 1.12×), so overtaking is frozen (everyone gains the same time); a pit is cheaper (`CAUTION_PIT_FACTOR` — you lose far less relative to a crawling field); and at a **Safety-Car or red-flag restart the field concertinas nose-to-tail** (`SC_BUNCH_GAP_S`, 1.2 s spacing). VSC is milder and does **not** bunch.
- **It predicts when the flag ends — not blind-follow.** `src/simulation/sc_model.py` learns SC/VSC run-lengths from historical `TrackStatus` (SC mean 4.51 / VSC 1.97 laps across 94 races, `data/sc_duration.json`); each rollout samples its own remaining caution length, *conditioned on how long the flag has already flown* (`sample_remaining(cause, elapsed)`). So a long-running SC is more likely to clear soon, and the finish distribution honestly reflects restart-timing uncertainty.

The live state is assembled by `src/simulation/live_state.py` — `from_replay(season, rnd, lap)` (historical parquet) or `from_openf1(session_key)` (the OpenF1 client, `src/pipeline/openf1.py`) — and surfaced on the dashboard's **Live** page. Everything is **backward compatible**: `start_age=2` / `caution=None` reproduce the pre-Phase-10 engine exactly.

## Intended use / out of scope
**For:** pre-race strategy comparison, dashboard win-probability views, the RL agent's training environment (Phase 5), and (Phase 10) mid-race "resume from here" projection under an active flag. **Not for:** wagering; wet races (tyre model excludes INT/WET); and note the caution model projects a *currently-active* flag forward — it does **not** add a random future-SC hazard to an otherwise-green run.

## Limitations
The caution model (Phase 10) follows a flag that is *already flying* and samples when it ends; it does not model the *probability of a future* SC/VSC on a green race, and treats a red flag like a Safety Car (no free tyre change / standing restart). Overtaking is a 3-parameter Bernoulli, not a wheel-to-wheel model. Lap-1 chaos not modeled (grid spread is a fixed 0.3s/position). A live OpenF1 snapshot infers car pace from recent lap times (no clean-air pace feed). Base pace anchors on the race's own field median in validation (track conditions known by ~lap 5 — documented choice). All Phase 2/3 model limitations are inherited.

## Reproduce
```bash
python -m src.models.pit_strategy.pit_loss   # data/pit_loss.json from raw laps
python -m src.models.pit_strategy.mdp        # optimal strategies + policy heatmaps
python -m src.simulation.engine              # 6-car demo + recommendation table
python -m src.simulation.validate_sim        # historical replay validation
pytest tests/test_simulation.py tests/test_live_engine.py tests/test_live_state.py tests/test_sc_model.py tests/test_openf1.py -v   # 9 + 9 + 15 + 9 + 26
```
Artifacts: `reports/pit_strategy/{optimal_strategies.csv,mdp_vs_actual.csv,policy_*.png}`, `reports/simulation/{validation.csv,validation_summary.md,demo_recommendation.csv}`, `data/sc_duration.json` (SC/VSC durations).
