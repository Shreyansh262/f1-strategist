"""Phase 5 tests — RL race env, synthetic tyre curves only.

Run: pytest tests/test_rl_env.py -v

stable-baselines3/gymnasium live in the local venv (Section 3). If gymnasium is
absent the module skips cleanly, mirroring test_tft.py's importorskip guard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from src.rl.env import (
    COMPOUNDS, MAX_AGE, EgoSpec, RaceConfig, RaceEnv, RivalSpec,
)


# ---------------------------------------------------------------------------
# Synthetic curves (same structure as test_simulation._curves_df)
# ---------------------------------------------------------------------------

def _curves_df(circuit="Test Grand Prix"):
    rows = []
    for comp, b, c in [("SOFT", 0.10, 0.004), ("MEDIUM", 0.05, 0.002), ("HARD", 0.03, 0.001)]:
        rows.append({
            "Compound": comp, "CircuitKey": circuit, "Era": 0,
            "pool_level": "circuit", "b": b, "c": c,
            "b_ci": 0.01, "c_ci": 0.0005, "r2": 0.5, "n_stints": 20, "n_laps": 300,
        })
    return pd.DataFrame(rows)


def _config(n_laps=30):
    rivals = [
        RivalSpec("R1", 90.0, "MEDIUM", [(15, "HARD")], 0.0),
        RivalSpec("R2", 90.6, "MEDIUM", [(15, "HARD")], 0.6),
    ]
    ego = EgoSpec(name="EGO", base_pace_s=90.3, start_compound="MEDIUM", grid_gap_s=0.3)
    return RaceConfig(
        circuit="Test Grand Prix", era=0, n_laps=n_laps, ego=ego, rivals=rivals,
        pit_loss_s=22.0, sigma_lap_s=0.3, overtake_prob=0.2,
    )


def _env(n_laps=30, seed=0):
    return RaceEnv(config=_config(n_laps), curves=_curves_df(), seed=seed)


# ---------------------------------------------------------------------------
# Gymnasium contract
# ---------------------------------------------------------------------------

class TestEnvContract:
    def test_check_env_passes(self):
        from gymnasium.utils.env_checker import check_env

        # check_env runs its own resets/steps; a fresh env with synthetic curves.
        check_env(_env(), skip_render_check=True)

    def test_reset_returns_obs_in_space(self):
        env = _env()
        obs, info = env.reset(seed=1)
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)

    def test_step_signature(self):
        env = _env()
        env.reset(seed=2)
        obs, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------

class TestEnvMechanics:
    def test_episode_terminates_at_n_laps(self):
        n_laps = 25
        env = _env(n_laps=n_laps)
        env.reset(seed=3)
        steps = 0
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(0)
            done = terminated or truncated
            steps += 1
            assert steps <= n_laps + 1            # never overruns
        assert steps == n_laps
        assert "race_time_s" in info and np.isfinite(info["race_time_s"])

    def test_pit_action_changes_compound(self):
        env = _env()
        env.reset(seed=4)
        # start on MEDIUM; pit onto HARD (action 3) -> compound must change + age resets
        assert env._ego_comp == COMPOUNDS.index("MEDIUM")
        env.step(0)                               # stay out one lap, age grows
        assert env._ego_age > 2
        env.step(3)                               # pit -> HARD
        assert env._ego_comp == COMPOUNDS.index("HARD")
        assert env._ego_age == 2                  # OUT_LAP_AGE
        assert COMPOUNDS.index("HARD") in env._used_compounds

    def test_reward_is_finite_and_negative_per_lap(self):
        env = _env()
        env.reset(seed=5)
        _, reward, _, _, _ = env.step(0)
        # per-lap reward ≈ -lap_time/1000, so a fraction of a second, always finite
        assert np.isfinite(reward)
        assert reward < 0

    def test_reward_tracks_race_time_on_legal_run(self):
        # Reward is -lap_time/1000 per step plus a small terminal position bonus.
        # On a LEGAL run (a real stop -> no two-compound penalty) the summed raw
        # lap times closely match the ego's race time, so -1000*return ≈ race time.
        env = _env(n_laps=20)
        env.reset(seed=6)
        total = 0.0
        done = False
        info = {}
        lap = 0
        while not done:
            action = 3 if lap == 10 else 0          # one pit onto HARD at lap 11
            _, r, terminated, truncated, info = env.step(action)
            total += r
            lap += 1
            done = terminated or truncated
        assert info["used_two_compounds"] is True   # legal -> no -5 penalty
        approx_time = -1000.0 * total
        # bonus is O(0.1*positions) seconds; race time dominates within a few seconds
        assert abs(approx_time - info["race_time_s"]) < 30.0

    def test_two_compound_penalty_when_never_pitting(self):
        # Staying out the whole race never runs a 2nd compound -> illegal -> penalty.
        env = _env(n_laps=20)
        env.reset(seed=7)
        rewards = []
        done = False
        info = {}
        while not done:
            _, r, terminated, truncated, info = env.step(0)
            rewards.append(r)
            done = terminated or truncated
        assert info["used_two_compounds"] is False
        # terminal reward carries the -5 penalty -> markedly more negative than a lap reward
        assert rewards[-1] < -1.0

    def test_seed_reproducible(self):
        a = _env()
        b = _env()
        oa, _ = a.reset(seed=99)
        ob, _ = b.reset(seed=99)
        assert np.allclose(oa, ob)
        for action in (0, 0, 3, 0, 2):
            sa = a.step(action)
            sb = b.step(action)
            assert np.allclose(sa[0], sb[0])
            assert sa[1] == sb[1]

    def test_mdp_state_shape(self):
        env = _env()
        env.reset(seed=8)
        lap, comp, age, used = env.mdp_state()
        assert lap == 1 and comp == COMPOUNDS.index("MEDIUM")
        assert 0 <= age <= MAX_AGE and used in (0, 1)
