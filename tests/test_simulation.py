"""Phase 4 tests — MDP + Monte Carlo engine, synthetic inputs only.

Run: pytest tests/test_simulation.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.pit_strategy.mdp import (
    COMPOUNDS, MAX_AGE, OUT_LAP_AGE, RaceModel, solve,
)
from src.simulation.engine import DriverSpec, RaceSpec, SimResult, simulate


# ---------------------------------------------------------------------------
# Fixtures: synthetic tyre deltas with known structure
# ---------------------------------------------------------------------------

def _race_model(pit_loss=22.0, n_laps=50, soft_rate=0.10, med_rate=0.05, hard_rate=0.03):
    ages = np.arange(0, MAX_AGE + 1, dtype=float)
    return RaceModel(
        circuit="Test Grand Prix", era=0, n_laps=n_laps, base_lap_s=90.0,
        pit_loss_s=pit_loss,
        tyre_delta={
            "SOFT":   soft_rate * ages + 0.004 * ages**2,
            "MEDIUM": med_rate * ages + 0.002 * ages**2,
            "HARD":   hard_rate * ages + 0.001 * ages**2,
        },
    )


def _curves_df():
    rows = []
    for comp, b, c in [("SOFT", 0.10, 0.004), ("MEDIUM", 0.05, 0.002), ("HARD", 0.03, 0.001)]:
        rows.append({
            "Compound": comp, "CircuitKey": "Test Grand Prix", "Era": 0,
            "pool_level": "circuit", "b": b, "c": c,
            "b_ci": 0.01, "c_ci": 0.0005, "r2": 0.5, "n_stints": 20, "n_laps": 300,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MDP
# ---------------------------------------------------------------------------

class TestMDP:
    def test_two_compound_rule_enforced(self):
        r = solve(_race_model(), start_compound="MEDIUM")
        assert r["feasible"]
        assert r["n_stops"] >= 1                       # must stop at least once
        used = {"MEDIUM"} | {c for _, c in r["stops"]}
        assert len(used) >= 2

    def test_huge_pit_loss_minimises_stops(self):
        few = solve(_race_model(pit_loss=60.0), "MEDIUM")["n_stops"]
        many = solve(_race_model(pit_loss=8.0), "MEDIUM")["n_stops"]
        assert few <= many
        assert few == 1                                # only the mandatory stop

    def test_high_degradation_forces_more_stops(self):
        calm = solve(_race_model(soft_rate=0.02, med_rate=0.02, hard_rate=0.01), "MEDIUM")
        cliff = solve(_race_model(soft_rate=0.30, med_rate=0.20, hard_rate=0.10), "MEDIUM")
        assert cliff["n_stops"] >= calm["n_stops"]

    def test_total_time_matches_forward_replay(self):
        m = _race_model()
        r = solve(m, "MEDIUM")
        # Replay the strategy and re-price every lap independently.
        t, comp, age = 0.0, "MEDIUM", OUT_LAP_AGE
        pits = dict(r["stops"])
        for lap in range(1, m.n_laps + 1):
            t += m.lap_time(lap, comp, age)
            if lap in pits:
                t += m.pit_loss_s
                comp, age = pits[lap], OUT_LAP_AGE
            else:
                age = min(age + 1, MAX_AGE)
        assert t == pytest.approx(r["total_time_s"], abs=1e-6)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _spec(n_laps=40, overtake_prob=0.2, sigma=0.5):
    drivers = [
        DriverSpec("A", 90.0, "MEDIUM", [(18, "HARD")], 1, 0.0),
        DriverSpec("B", 90.4, "MEDIUM", [(18, "HARD")], 2, 0.4),
        DriverSpec("C", 91.0, "MEDIUM", [(18, "HARD")], 3, 0.8),
    ]
    return RaceSpec(
        circuit="Test Grand Prix", era=0, n_laps=n_laps, drivers=drivers,
        pit_loss_s=22.0, sigma_lap_s=sigma, overtake_prob=overtake_prob,
    )


class TestEngine:
    def test_faster_driver_wins_most(self):
        res = simulate(_spec(), n_rollouts=500, seed=1, curves=_curves_df())
        assert res.win_prob["A"] > res.win_prob["B"] > res.win_prob["C"]
        assert res.win_prob["A"] > 0.5

    def test_probabilities_sum_to_one(self):
        res = simulate(_spec(), n_rollouts=400, seed=2, curves=_curves_df())
        assert sum(res.win_prob.values()) == pytest.approx(1.0)
        # every rollout has exactly one car per position
        assert (np.sort(res.finish_pos, axis=1) == [1, 2, 3]).all()

    def test_seed_reproducible(self):
        a = simulate(_spec(), n_rollouts=300, seed=7, curves=_curves_df())
        b = simulate(_spec(), n_rollouts=300, seed=7, curves=_curves_df())
        assert np.array_equal(a.finish_pos, b.finish_pos)

    def test_zero_overtaking_freezes_order_when_pace_equal(self):
        spec = _spec(overtake_prob=0.0, sigma=2.0)
        for d in spec.drivers:
            d.base_pace_s = 90.0          # equal pace, big noise, no passing
        res = simulate(spec, n_rollouts=300, seed=3, curves=_curves_df())
        # grid order must be preserved in every rollout
        assert (res.finish_pos == [1, 2, 3]).all()

    def test_extra_stop_costs_time(self):
        spec1 = _spec(sigma=0.1)
        spec2 = _spec(sigma=0.1)
        spec2.drivers[0].strategy = [(14, "HARD"), (28, "MEDIUM")]   # extra stop
        r1 = simulate(spec1, n_rollouts=300, seed=4, curves=_curves_df())
        r2 = simulate(spec2, n_rollouts=300, seed=4, curves=_curves_df())
        i = r1.drivers.index("A")
        assert r2.race_time_s[:, i].mean() > r1.race_time_s[:, i].mean()
