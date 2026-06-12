# Model Card — RL Pit-Stop Agent (PPO, Phase 5 stretch)

## Overview
A reinforcement-learning pit-stop agent trained **inside the project's own race simulator**, as a research/stretch deliverable on top of the Phase-4 strategy stack. It is a Stable-Baselines3 **PPO** policy (`MlpPolicy`) trained in `src/rl/env.py` (`RaceEnv`), a Gymnasium environment that steps a single race lap-by-lap and lets the agent decide, each lap, whether to pit and onto which compound.

The env deliberately **reuses the Phase-4b Monte Carlo engine's components** (`src/simulation/engine.py`) rather than re-deriving them — fuel/pit/overtaking constants, the Phase-3 tyre degradation curves, and the recalibrated TFT per-lap σ are all imported, so the RL env and the showcase engine cannot drift apart.

Trained on Kaggle (T4), **3 seeds × 3M timesteps** each, with checkpoints every 500k steps. Evaluated head-to-head against the Phase-4a MDP policy in identical, paired (common-random-number) episodes.

## Setup / Inputs — what the agent sees and does
All from `src/rl/env.py`, quoted from the code (not guessed):

- **Observation** (`Box(float32)`, dim 11) = the MDP state vector plus recent pace history:
  `[ norm_lap, norm_tyre_age, compound_onehot(3), norm_position, gap_ahead_s (clipped 0–30s, normalised), norm_laps_remaining, era, last-3 lap-time deltas vs the ego's own running median ]`.
- **Action** (`Discrete(4)`, exactly the MDP `ACTIONS`): `0 stay_out | 1 pit_SOFT | 2 pit_MEDIUM | 3 pit_HARD`.
- **Reward**: `-ego_lap_time / 1000` each step (so the undiscounted return ≈ `-total_race_time / 1000`), plus a **terminal position bonus** `+0.1 × (grid_pos − finish_pos)` (positive = places gained). The **two-compound rule** is enforced as a `-5.0` terminal penalty if the ego never ran a second slick compound (an illegal/DSQ-like classification) — the same rule the MDP uses.
- **Race config**: the env's `default_field()` — a realistic 6-car Bahrain field (57 laps, era 0), ego starting on MEDIUM mid-grid (~P3) with rivals on plausible 1-/2-stop strategies and ~1.5s front-to-back pace spread, so traffic makes pit timing matter.
- **PPO hyperparameters** (`src/rl/train_ppo.py`): `n_steps=2048, batch_size=256, gamma=0.999` (long-horizon race, minimal discounting), `gae_lambda=0.95, ent_coef=0.01, learning_rate=3e-4`, `MlpPolicy`, CPU (PPO+MLP is CPU-bound; Kaggle parallelism is `SubprocVecEnv` envs, not a big net on the GPU).

## Evaluation protocol
`src/rl/evaluate_rl.py` runs **paired, common-random-number** rollouts: for each seed, PPO and the MDP are rolled out in the **same** `RaceEnv` episode at the **same** seed, so any difference in race time or finish position is the policy, not luck. The MDP baseline reuses Phase 4a exactly (`build_race_model()` → `solve()`), indexed each step by the env's `(lap, compound, tyre_age, used_two_compounds)` state tuple (`RaceEnv.mdp_state()`).

- **500 paired episodes per seed** on Kaggle (base_seed 10000), 3 seeds.
- **Sign convention:** `ppo_minus_mdp_time_s = ppo_time − mdp_time`, so **positive = PPO slower**.

## Results

| seed | ppo_mean_finish | mdp_mean_finish | ppo_minus_mdp_time_s | ppo_legal_frac |
|------|-----------------|-----------------|----------------------|----------------|
| 0    | 2.574           | 2.664           | +1.501               | 1.0            |
| 1    | 2.598           | 2.664           | +2.448               | 1.0            |
| 2    | 2.576           | 2.664           | −0.354               | 1.0            |

Seed-0 aggregate raw times: `mdp_time_s` 5275.48, `ppo_time_s` 5276.98.

### Honest verdict — PPO MATCHED but did NOT BEAT the MDP
On **race time** (the MDP's own objective) the MDP is faster on **2 of 3 seeds**, mean ≈ **+1.2 s/race** in the MDP's favour. On **mean finish position** PPO is marginally better on **all 3 seeds** (~0.08 places, 2.57–2.60 vs 2.664). Both policies are **100% legal** under the two-compound rule.

This is an **objective mismatch**, not a failure: PPO optimizes the env's reward (finish position in a 6-car field with traffic), while the MDP minimizes single-car race time. Each policy wins on its own objective and neither dominates. This "matched, not beat" outcome was pre-authorized as a legitimate result in MASTER_CONTEXT Section 10.5.

## Intended use / out of scope
**For:** a research / stretch deliverable demonstrating that RL can be trained on top of the project's simulator and evaluated honestly head-to-head against the exact MDP it competes with. **Not for:** serving. **By scope decision the RL agent is NOT wired into the dashboard or API** — the production strategy stack remains **MDP (proposes) + Monte Carlo simulation (ranks)**, which is interpretable and validated. The RL agent is the "can we?" experiment, documented either way.

Local sanity-check command (a short CPU re-run vs the MDP on seed 0):
```bash
venv/Scripts/python.exe -m src.rl.evaluate_rl --model models/rl/ppo_pit_seed0.zip --episodes 200
```

## Limitations
- **Single training env config** (`default_field`): one Bahrain-shaped 6-car field, era 0. No curriculum across circuits/fields.
- **Reward shaped on finish position** (terminal bonus), not pure race time — hence the objective mismatch vs the MDP above.
- **No SC/VSC** in the env (inherited from the Phase-4 sim's biggest gap; lap-1 chaos and wet tyres also excluded).
- **3 seeds only** — small ensemble; per-seed time delta ranges −0.35 to +2.45 s.
- All Phase 2/3/4 model limitations (tyre curves, TFT bands, 3-parameter overtaking) are inherited through the shared engine components.

## Reproduce
- **Training** (Kaggle T4, per seed; final runs were 3M steps × 3 seeds):
  - Notebook: `notebooks/08_rl_ppo.ipynb`
  - CLI equivalent: `python -m src.rl.train_ppo --steps 3000000 --n-envs 8 --seed 0` (then seeds 1, 2)
- **Evaluation** (local CPU sanity, full Kaggle runs were `--episodes 500` per seed):
  ```bash
  venv/Scripts/python.exe -m src.rl.evaluate_rl --model models/rl/ppo_pit_seed0.zip --episodes 200
  ```
- **Tests:** `tests/test_rl_env.py` (env API / reward / two-compound rule / determinism).
- **Artifacts:**
  - `models/rl/ppo_pit_seed{0,1,2}.zip` (final) + `ppo_pit_seed{0,1,2}_{500000..3000000}_steps.zip` (checkpoints) + `models/rl/tb/seed{0,1,2}_1/` (tensorboard) — **binaries gitignored** (`*.zip`), download from Kaggle output.
  - `reports/rl/ppo_vs_mdp.csv` (3-seed summary) + `reports/rl/ppo_vs_mdp_seed{0,1,2}.csv` (500 paired episodes each + AGGREGATE row) — **committed**.
  - `reports/rl/ppo_vs_mdp_local200.csv` — local 200-ep sanity re-run.
