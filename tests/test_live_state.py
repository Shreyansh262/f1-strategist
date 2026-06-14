"""Live-state assembler tests — Phase 10.2. No network, no GPU, no parquet.

from_snapshot is exercised from a plain dict; the replay path is exercised via
spec_from_race_df on a synthetic race DataFrame.

Run: pytest tests/test_live_state.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation import live_state as LS
from src.simulation.engine import simulate


# ---------------------------------------------------------------------------
# circuit metadata
# ---------------------------------------------------------------------------

class TestCircuitMeta:
    def test_known_short_name(self):
        event, laps = LS._circuit_meta("Sakhir")
        assert event == "Bahrain Grand Prix" and laps == 57

    def test_variant_and_accent(self):
        assert LS._circuit_meta("Interlagos")[0] == "São Paulo Grand Prix"
        assert LS._circuit_meta("Monaco")[1] == 78

    def test_unknown_falls_back(self):
        event, laps = LS._circuit_meta("Nürburgring")
        assert laps == LS._DEFAULT_RACE_LAPS


# ---------------------------------------------------------------------------
# from_snapshot (live path)
# ---------------------------------------------------------------------------

def _snapshot(caution=None, current_lap=20):
    return {
        "session": {"session_key": 1, "circuit": "Sakhir", "year": 2024,
                    "session_name": "Race", "current_lap": current_lap, "available": True},
        "drivers": [
            {"driver_number": 1, "position": 1, "current_lap": current_lap,
             "compound": "HARD", "start_age": 15, "gap_to_leader_s": 0.0, "last_lap_s": 95.0},
            {"driver_number": 44, "position": 2, "current_lap": current_lap,
             "compound": "MEDIUM", "start_age": 8, "gap_to_leader_s": 4.2, "last_lap_s": 95.3},
            {"driver_number": 16, "position": 3, "current_lap": current_lap,
             "compound": "SOFT", "start_age": 3, "gap_to_leader_s": 9.9, "last_lap_s": 95.6},
        ],
        "caution": caution,
    }


class TestFromSnapshot:
    def test_basic_assembly(self):
        spec = LS.from_snapshot(_snapshot())
        assert spec.circuit == "Bahrain Grand Prix"
        assert spec.era == 0                       # 2024 -> era 0
        assert spec.n_laps == 57 - 20              # remaining laps
        assert len(spec.drivers) == 3
        lead = spec.drivers[0]
        assert lead.driver == "1" and lead.start_compound == "HARD"
        assert lead.start_age == 15 and lead.grid_gap_s == 0.0
        assert spec.drivers[1].grid_gap_s == 4.2

    def test_caution_threaded(self):
        spec = LS.from_snapshot(_snapshot(caution={"cause": "SC", "elapsed_laps": 2}))
        assert spec.caution == ("SC", 2)

    def test_total_laps_override(self):
        spec = LS.from_snapshot(_snapshot(current_lap=10), total_laps=30)
        assert spec.n_laps == 20

    def test_empty_drivers_returns_none(self):
        assert LS.from_snapshot({"session": {}, "drivers": []}) is None

    def test_non_slick_compound_safe(self):
        snap = _snapshot()
        snap["drivers"][0]["compound"] = "INTERMEDIATE"
        spec = LS.from_snapshot(snap)
        assert spec.drivers[0].start_compound == "MEDIUM"

    def test_spec_actually_simulates(self):
        spec = LS.from_snapshot(_snapshot(caution={"cause": "SC", "elapsed_laps": 1}))
        res = simulate(spec, n_rollouts=100, seed=0)
        assert res.finish_pos.shape == (100, 3)


# ---------------------------------------------------------------------------
# spec_from_race_df (replay path)
# ---------------------------------------------------------------------------

def _race_df(n_laps=30, drivers=("VER", "HAM", "LEC"), sc_laps=()):
    rows = []
    pace = {"VER": 90.0, "HAM": 90.4, "LEC": 90.8}
    for d in drivers:
        for lap in range(1, n_laps + 1):
            status = "4" if lap in sc_laps else "1"
            lt = pace[d] + (20.0 if lap in sc_laps else 0.0)
            rows.append({
                "Season": 2024, "RoundNumber": 1, "Driver": d, "LapNumber": lap,
                "CircuitKey": "Bahrain Grand Prix", "Compound": "MEDIUM",
                "TyreLife": lap + 1, "TrackStatus": status,
                "LapTimeSeconds": lt, "PitInTime": np.nan, "PitOutTime": np.nan,
            })
    return pd.DataFrame(rows)


class TestReplay:
    def test_remaining_and_gaps(self):
        spec = LS.spec_from_race_df(_race_df(), current_lap=10)
        assert spec.n_laps == 30 - 10
        assert spec.circuit == "Bahrain Grand Prix" and spec.era == 0
        # VER fastest -> leader, gap 0; field ordered by cumulative time.
        assert spec.drivers[0].driver == "VER"
        assert spec.drivers[0].grid_gap_s == 0.0
        assert spec.drivers[1].grid_gap_s > 0

    def test_start_age_from_tyrelife(self):
        spec = LS.spec_from_race_df(_race_df(), current_lap=10)
        assert spec.drivers[0].start_age == 11    # TyreLife at lap 10 = lap+1

    def test_caution_detected(self):
        spec = LS.spec_from_race_df(_race_df(sc_laps=(9, 10)), current_lap=10)
        assert spec.caution == ("SC", 2)          # two consecutive SC laps

    def test_no_caution_when_green(self):
        spec = LS.spec_from_race_df(_race_df(), current_lap=10)
        assert spec.caution is None

    def test_retired_driver_excluded(self):
        df = _race_df()
        df = df[~((df["Driver"] == "LEC") & (df["LapNumber"] >= 8))]  # LEC out after lap 7
        spec = LS.spec_from_race_df(df, current_lap=10)
        assert {d.driver for d in spec.drivers} == {"VER", "HAM"}

    def test_empty_returns_none(self):
        assert LS.spec_from_race_df(pd.DataFrame(), current_lap=5) is None
