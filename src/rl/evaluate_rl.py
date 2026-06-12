"""Head-to-head: PPO agent vs the MDP policy in the SAME simulator — Phase 5.

Both policies are rolled out in identical RaceEnv episodes with COMMON SEEDS
(paired comparison / common random numbers), so any difference in race time or
finish position is the policy, not luck. A documented "RL matched but did not beat
the MDP" is a legitimate result (MASTER_CONTEXT 10.5) — this script reports it
either way.

The MDP baseline reuses Phase 4a exactly: build_race_model() for the env's race,
solve() to get the optimal policy grid, then index that grid by the env's
(lap, compound, tyre_age, used_two_compounds) state each step — the same state
tuple the env exposes via RaceEnv.mdp_state().

Output: reports/rl/ppo_vs_mdp.csv (per-seed paired means + an aggregate row).

Run (after a PPO checkpoint exists):
    python -m src.rl.evaluate_rl --model models/rl/ppo_pit_seed0.zip --episodes 300
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.pit_strategy.mdp import (
    ACTIONS, COMPOUNDS, build_race_model, solve,
)
from src.rl.env import RaceConfig, RaceEnv, default_field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports" / "rl"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Policies (a policy maps an env to a per-step action)
# ---------------------------------------------------------------------------

class MDPPolicy:
    """Wraps a solved MDP policy grid as an action function over env state.

    The MDP is single-car (no rivals), so its policy depends only on
    (lap, compound, tyre_age, used_two_compounds) — exactly RaceEnv.mdp_state().
    """

    def __init__(self, config: RaceConfig, curves: pd.DataFrame | None = None) -> None:
        base = config.ego.base_pace_s
        model = build_race_model(
            config.circuit, config.era, config.n_laps, base, curves=curves,
        )
        sol = solve(model, start_compound=config.ego.start_compound)
        self.policy = sol["policy"]            # [lap, compound, age, used_two] -> action int
        self.n_laps = config.n_laps

    def act(self, env: RaceEnv) -> int:
        lap, comp, age, used_two = env.mdp_state()
        lap = min(max(lap, 1), self.n_laps)
        return int(self.policy[lap, comp, age, used_two])


class PPOPolicy:
    """Wraps a trained SB3 PPO model."""

    def __init__(self, model_path: str | Path) -> None:
        from stable_baselines3 import PPO

        self.model = PPO.load(str(model_path), device="cpu")

    def act(self, env: RaceEnv) -> int:
        obs = env._observation()
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout(env: RaceEnv, policy, seed: int) -> dict:
    """One episode under `policy` at `seed`. Returns race time + finish + legality."""
    env.reset(seed=seed)
    done = False
    info: dict = {}
    while not done:
        action = policy.act(env)
        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return {
        "race_time_s": info["race_time_s"],
        "finish_position": info["finish_position"],
        "used_two_compounds": info["used_two_compounds"],
    }


def evaluate(
    model_path: str | Path,
    episodes: int = 300,
    config: RaceConfig | None = None,
    curves: pd.DataFrame | None = None,
    base_seed: int = 10_000,
) -> pd.DataFrame:
    """Paired PPO-vs-MDP rollouts over `episodes` common seeds."""
    config = config if config is not None else default_field()
    env = RaceEnv(config=config, curves=curves)
    mdp = MDPPolicy(config, curves=curves)
    ppo = PPOPolicy(model_path)

    rows = []
    for k in range(episodes):
        seed = base_seed + k
        r_mdp = rollout(env, mdp, seed)
        r_ppo = rollout(env, ppo, seed)
        rows.append({
            "seed": seed,
            "mdp_time_s": r_mdp["race_time_s"],
            "ppo_time_s": r_ppo["race_time_s"],
            "mdp_finish": r_mdp["finish_position"],
            "ppo_finish": r_ppo["finish_position"],
            "mdp_legal": r_mdp["used_two_compounds"],
            "ppo_legal": r_ppo["used_two_compounds"],
            "ppo_minus_mdp_time_s": r_ppo["race_time_s"] - r_mdp["race_time_s"],
        })
    df = pd.DataFrame(rows)

    agg = {
        "seed": "AGGREGATE",
        "mdp_time_s": df["mdp_time_s"].mean(),
        "ppo_time_s": df["ppo_time_s"].mean(),
        "mdp_finish": df["mdp_finish"].mean(),
        "ppo_finish": df["ppo_finish"].mean(),
        "mdp_legal": df["mdp_legal"].mean(),
        "ppo_legal": df["ppo_legal"].mean(),
        "ppo_minus_mdp_time_s": df["ppo_minus_mdp_time_s"].mean(),
    }
    out = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)

    out_path = REPORTS_DIR / "ppo_vs_mdp.csv"
    out.to_csv(out_path, index=False)
    log.info("ppo_vs_mdp.csv -> %s", out_path)
    win = float((df["ppo_minus_mdp_time_s"] < 0).mean())
    log.info(
        "Paired %d eps: mean PPO-MDP time = %+.3fs (PPO faster in %.1f%% of races). "
        "PPO legal %.1f%% / MDP legal %.1f%%",
        episodes, agg["ppo_minus_mdp_time_s"], 100 * win,
        100 * agg["ppo_legal"], 100 * agg["mdp_legal"],
    )
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO vs MDP head-to-head in the sim.")
    p.add_argument("--model", type=str, required=True, help="path to a PPO .zip")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--base-seed", type=int, default=10_000)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate(args.model, episodes=args.episodes, base_seed=args.base_seed)


if __name__ == "__main__":
    main()
