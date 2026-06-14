"""Engine resume + flag-following tests — Phase 10.3/10.4. No network, no GPU.

Covers mid-stint resume (DriverSpec.start_age), in-progress caution slowing the
field, Safety-Car restart bunching, and full backward compatibility with the
pre-Phase-10 engine (caution=None reproduces the old behaviour).

Run: pytest tests/test_live_engine.py -v
"""
from __future__ import annotations

import numpy as np

from src.simulation import engine as E
from src.simulation.engine import DriverSpec, RaceSpec, simulate, _stint_schedule


def _drivers(n=4, spread=0.0):
    return [
        DriverSpec(f"D{i}", 90.0 + 0.1 * i, "MEDIUM", [(15, "HARD")], i + 1, spread * i)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 10.3 mid-stint resume
# ---------------------------------------------------------------------------

class TestStartAge:
    def test_default_age_is_two(self):
        spec = RaceSpec("Bahrain Grand Prix", 0, 20, _drivers())
        _, age, _ = _stint_schedule(spec, spec.drivers[0])
        assert age[0] == 2                       # unchanged opening-lap convention

    def test_resume_age_carries_into_first_stint(self):
        d = DriverSpec("X", 90.0, "MEDIUM", [], 1, 0.0, start_age=18)
        spec = RaceSpec("Bahrain Grand Prix", 0, 5, [d])
        _, age, stint = _stint_schedule(spec, d)
        assert age[0] == 18 and list(age) == [18, 19, 20, 21, 22]
        assert (stint == 0).all()

    def test_pit_resets_age_to_two(self):
        d = DriverSpec("X", 90.0, "MEDIUM", [(2, "HARD")], 1, 0.0, start_age=18)
        spec = RaceSpec("Bahrain Grand Prix", 0, 4, [d])
        _, age, stint = _stint_schedule(spec, d)
        assert list(age) == [18, 19, 2, 3]       # fresh tyre after the stop
        assert list(stint) == [0, 0, 1, 1]

    def test_unknown_compound_falls_back_not_crash(self):
        d = DriverSpec("X", 90.0, "INTERMEDIATE", [], 1, 0.0)
        spec = RaceSpec("Bahrain Grand Prix", 0, 3, [d])
        comp, _, _ = _stint_schedule(spec, d)
        assert comp[0] == E.COMPOUNDS.index("MEDIUM")


# ---------------------------------------------------------------------------
# 10.4 flag following
# ---------------------------------------------------------------------------

class TestCaution:
    def test_backward_compatible_no_caution(self):
        spec = RaceSpec("Bahrain Grand Prix", 0, 25, _drivers())
        res = simulate(spec, n_rollouts=200, seed=1)
        assert res.finish_pos.shape == (200, 4)
        assert set(res.win_prob) == {"D0", "D1", "D2", "D3"}

    def test_safety_car_slows_the_race(self):
        drivers = _drivers()
        green = simulate(RaceSpec("Bahrain Grand Prix", 0, 25, drivers),
                         n_rollouts=300, seed=7)
        sc = simulate(RaceSpec("Bahrain Grand Prix", 0, 25, drivers, caution=("SC", 1)),
                      n_rollouts=300, seed=7)
        assert sc.race_time_s.mean() > green.race_time_s.mean()

    def test_restart_bunches_the_field(self, monkeypatch):
        # Force the entire race under Safety Car so the only restart lands on the
        # last lap: the final spread must collapse to the fixed bunch spacing,
        # regardless of the large grid gaps we start with.
        drivers = _drivers(n=4, spread=10.0)     # 0,10,20,30s grid spread
        L = 12
        monkeypatch.setattr(E, "_sample_caution_lengths",
                            lambda cause, elapsed, n, seed: np.full(n, L, dtype=int))
        res = simulate(RaceSpec("Bahrain Grand Prix", 0, L, drivers, caution=("SC", 1)),
                       n_rollouts=50, seed=3)
        spread = res.race_time_s.max(axis=1) - res.race_time_s.min(axis=1)
        assert np.allclose(spread, (len(drivers) - 1) * E.SC_BUNCH_GAP_S, atol=1e-3)

    def test_vsc_does_not_bunch(self, monkeypatch):
        drivers = _drivers(n=4, spread=10.0)
        L = 12
        monkeypatch.setattr(E, "_sample_caution_lengths",
                            lambda cause, elapsed, n, seed: np.full(n, L, dtype=int))
        res = simulate(RaceSpec("Bahrain Grand Prix", 0, L, drivers, caution=("VSC", 1)),
                       n_rollouts=50, seed=3)
        spread = res.race_time_s.max(axis=1) - res.race_time_s.min(axis=1)
        # VSC holds gaps (no concertina) -> spread stays near the 30s grid spread.
        assert spread.mean() > 20.0


def test_sample_caution_lengths_floor_and_shape():
    out = E._sample_caution_lengths("SC", 2, 64, seed=0)
    assert out.shape == (64,) and (out >= 1).all()
    # Already longer than anything observed -> remaining floored at 1.
    assert (E._sample_caution_lengths("VSC", 99, 16, seed=0) == 1).all()
