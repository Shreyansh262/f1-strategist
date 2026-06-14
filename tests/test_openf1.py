"""OpenF1 live-timing client tests — Phase 10.  No network, no GPU.

All OpenF1 API calls are monkeypatched via ``_get_json``; no HTTP traffic is
made.  Each test class covers a distinct slice of the public contract.

Run: pytest tests/test_openf1.py -v
"""
from __future__ import annotations

import pytest

import src.pipeline.openf1 as OF


# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

# A minimal session record returned by /sessions
SESSION_ROW = {
    "session_key": 9158,
    "circuit_short_name": "Bahrain",
    "country_name": "Bahrain",
    "year": 2024,
    "date_start": "2024-03-02T15:00:00+03:00",
    "gmt_offset": "+03:00",
    "session_name": "Race",
}

# Two drivers: #1 on lap 42, #44 on lap 41
LAPS = [
    # Driver 1 — two completed laps
    {"driver_number": 1, "lap_number": 41, "lap_duration": 90.1, "is_pit_out_lap": False, "date_start": "2024-03-02T15:00:00"},
    {"driver_number": 1, "lap_number": 42, "lap_duration": 91.5, "is_pit_out_lap": False, "date_start": "2024-03-02T15:01:30"},
    # Driver 44 — one lap
    {"driver_number": 44, "lap_number": 41, "lap_duration": 92.0, "is_pit_out_lap": False, "date_start": "2024-03-02T15:00:00"},
]

INTERVALS = [
    {"driver_number": 1, "gap_to_leader": 0.0, "interval": 0.0, "date": "2024-03-02T15:05:00"},
    {"driver_number": 44, "gap_to_leader": 5.3, "interval": 5.3, "date": "2024-03-02T15:05:00"},
]

# Driver 1: SOFT, started lap 30, tyre_age_at_start=3 => at lap 42: age = 3+(42-30)=15
# Driver 44: MEDIUM, started lap 1, tyre_age_at_start=0 => at lap 41: age = 0+(41-1)=40
STINTS = [
    {"driver_number": 1, "compound": "soft", "lap_start": 30, "lap_end": 99, "tyre_age_at_start": 3},
    {"driver_number": 44, "compound": "medium", "lap_start": 1, "lap_end": 99, "tyre_age_at_start": 0},
]

POSITIONS = [
    {"driver_number": 1, "position": 1, "date": "2024-03-02T15:05:00"},
    {"driver_number": 44, "position": 2, "date": "2024-03-02T15:05:00"},
]

# Race control: SC deployed lap 5, then GREEN on lap 8 -> caution should clear
RACE_CONTROL_SC_CLEARED = [
    {"category": "SafetyCar", "flag": None, "message": "SAFETY CAR DEPLOYED", "lap_number": 5, "date": "2024-03-02T15:10:00", "scope": "Track"},
    {"category": "Flag", "flag": "GREEN", "message": "", "lap_number": 8, "date": "2024-03-02T15:15:00", "scope": "Track"},
]

# Race control: SC deployed lap 5, NO green — still active at lap 42
RACE_CONTROL_SC_ACTIVE = [
    {"category": "SafetyCar", "flag": None, "message": "SAFETY CAR DEPLOYED", "lap_number": 5, "date": "2024-03-02T15:10:00", "scope": "Track"},
]

# VSC deployed lap 10, no green
RACE_CONTROL_VSC_ACTIVE = [
    {"category": None, "flag": None, "message": "VIRTUAL SAFETY CAR DEPLOYED", "lap_number": 10, "date": "2024-03-02T15:10:00", "scope": "Track"},
]

# RED flag lap 5, then a YELLOW on lap 6 (RED still takes precedence)
RACE_CONTROL_RED_PRECEDENCE = [
    {"category": "Flag", "flag": "RED", "message": "", "lap_number": 5, "date": "2024-03-02T15:10:00", "scope": "Track"},
    {"category": "Flag", "flag": "YELLOW", "message": "", "lap_number": 6, "date": "2024-03-02T15:11:00", "scope": "Track"},
]


def _make_get_json(
    sessions=None,
    laps=None,
    intervals=None,
    stints=None,
    position=None,
    race_control=None,
):
    """Return a fake _get_json that dispatches by URL suffix."""
    _map = {
        "sessions": sessions if sessions is not None else [SESSION_ROW],
        "laps": laps if laps is not None else LAPS,
        "intervals": intervals if intervals is not None else INTERVALS,
        "stints": stints if stints is not None else STINTS,
        "position": position if position is not None else POSITIONS,
        "race_control": race_control if race_control is not None else [],
    }

    def fake(url: str, params: dict, use_cache: bool = True) -> list:
        for key, data in _map.items():
            if url.endswith(f"/{key}"):
                return data
        return []

    return fake


# ---------------------------------------------------------------------------
# start_age calculation
# ---------------------------------------------------------------------------

class TestStartAge:
    def test_correct_start_age_driver1(self, monkeypatch):
        """Driver 1: tyre_age_at_start=3, lap_start=30, current_lap=42 -> age=15."""
        monkeypatch.setattr(OF, "_get_json", _make_get_json())
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        d1 = next(d for d in snap["drivers"] if d["driver_number"] == 1)
        # 3 + (42 - 30) = 15
        assert d1["start_age"] == 15

    def test_correct_start_age_driver44(self, monkeypatch):
        """Driver 44: tyre_age_at_start=0, lap_start=1, current_lap=41 -> age=40."""
        monkeypatch.setattr(OF, "_get_json", _make_get_json())
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        d44 = next(d for d in snap["drivers"] if d["driver_number"] == 44)
        # 0 + (41 - 1) = 40
        assert d44["start_age"] == 40

    def test_start_age_floor_at_1(self, monkeypatch):
        """start_age floors at 1 when the formula would give 0."""
        stints_edge = [
            {"driver_number": 1, "compound": "SOFT", "lap_start": 42, "lap_end": 99, "tyre_age_at_start": 0},
            {"driver_number": 44, "compound": "MEDIUM", "lap_start": 1, "lap_end": 99, "tyre_age_at_start": 0},
        ]
        monkeypatch.setattr(OF, "_get_json", _make_get_json(stints=stints_edge))
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        d1 = next(d for d in snap["drivers"] if d["driver_number"] == 1)
        # 0 + (42 - 42) = 0 -> floor to 1
        assert d1["start_age"] == 1


# ---------------------------------------------------------------------------
# gap_to_leader_s
# ---------------------------------------------------------------------------

class TestGapToLeader:
    def test_numeric_gap(self, monkeypatch):
        """Numeric gap_to_leader passes through as float."""
        monkeypatch.setattr(OF, "_get_json", _make_get_json())
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        d44 = next(d for d in snap["drivers"] if d["driver_number"] == 44)
        assert d44["gap_to_leader_s"] == pytest.approx(5.3)

    def test_string_gap_is_none(self, monkeypatch):
        """String gaps like '+1 LAP' map to None."""
        intervals_lapped = [
            {"driver_number": 1, "gap_to_leader": 0.0, "interval": 0.0, "date": "2024-03-02T15:05:00"},
            {"driver_number": 44, "gap_to_leader": "+1 LAP", "interval": "+1 LAP", "date": "2024-03-02T15:05:00"},
        ]
        monkeypatch.setattr(OF, "_get_json", _make_get_json(intervals=intervals_lapped))
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        d44 = next(d for d in snap["drivers"] if d["driver_number"] == 44)
        assert d44["gap_to_leader_s"] is None

    def test_null_gap_is_none(self, monkeypatch):
        """null gap_to_leader maps to None."""
        intervals_null = [
            {"driver_number": 1, "gap_to_leader": None, "interval": None, "date": "2024-03-02T15:05:00"},
        ]
        monkeypatch.setattr(OF, "_get_json", _make_get_json(intervals=intervals_null))
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        d1 = next(d for d in snap["drivers"] if d["driver_number"] == 1)
        assert d1["gap_to_leader_s"] is None


# ---------------------------------------------------------------------------
# Compound normalisation & stint selection
# ---------------------------------------------------------------------------

class TestCompoundAndStint:
    def test_compound_uppercased(self, monkeypatch):
        """Lowercase compounds from API must be uppercased."""
        monkeypatch.setattr(OF, "_get_json", _make_get_json())
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        d1 = next(d for d in snap["drivers"] if d["driver_number"] == 1)
        d44 = next(d for d in snap["drivers"] if d["driver_number"] == 44)
        assert d1["compound"] == "SOFT"
        assert d44["compound"] == "MEDIUM"

    def test_current_stint_chosen_by_lap_range(self, monkeypatch):
        """Stint whose lap_start<=current_lap<=lap_end is selected."""
        stints_multi = [
            # Driver 1: first stint 1-20 HARD, second stint 21-99 SOFT
            {"driver_number": 1, "compound": "hard", "lap_start": 1, "lap_end": 20, "tyre_age_at_start": 0},
            {"driver_number": 1, "compound": "soft", "lap_start": 21, "lap_end": 99, "tyre_age_at_start": 0},
            {"driver_number": 44, "compound": "medium", "lap_start": 1, "lap_end": 99, "tyre_age_at_start": 0},
        ]
        monkeypatch.setattr(OF, "_get_json", _make_get_json(stints=stints_multi))
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        d1 = next(d for d in snap["drivers"] if d["driver_number"] == 1)
        # current_lap = 42, which is in the SOFT stint (21-99)
        assert d1["compound"] == "SOFT"

    def test_fallback_to_last_stint(self, monkeypatch):
        """When no stint's lap range covers current_lap, use the last stint."""
        stints_old = [
            # Driver 1: stint ended at lap 20 but driver is on lap 42
            {"driver_number": 1, "compound": "hard", "lap_start": 1, "lap_end": 20, "tyre_age_at_start": 0},
            {"driver_number": 44, "compound": "medium", "lap_start": 1, "lap_end": 99, "tyre_age_at_start": 0},
        ]
        monkeypatch.setattr(OF, "_get_json", _make_get_json(stints=stints_old))
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        d1 = next(d for d in snap["drivers"] if d["driver_number"] == 1)
        assert d1["compound"] == "HARD"  # falls back to only/last stint


# ---------------------------------------------------------------------------
# Caution parsing
# ---------------------------------------------------------------------------

class TestCautionParsing:
    def test_sc_deployed_then_no_green_is_active(self, monkeypatch):
        """SC DEPLOYED with no subsequent GREEN -> cause='SC', elapsed_laps correct."""
        monkeypatch.setattr(OF, "_get_json", _make_get_json(race_control=RACE_CONTROL_SC_ACTIVE))
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        caution = snap["caution"]
        assert caution is not None
        assert caution["cause"] == "SC"
        # current_lap=42, deployed_lap=5 -> elapsed = max(1, 42-5+1) = 38
        assert caution["elapsed_laps"] == 38

    def test_sc_then_green_clears_caution(self, monkeypatch):
        """SC DEPLOYED followed by GREEN -> caution is None."""
        monkeypatch.setattr(OF, "_get_json", _make_get_json(race_control=RACE_CONTROL_SC_CLEARED))
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        assert snap["caution"] is None

    def test_vsc_detected(self, monkeypatch):
        """VIRTUAL SAFETY CAR DEPLOYED -> cause='VSC'."""
        monkeypatch.setattr(OF, "_get_json", _make_get_json(race_control=RACE_CONTROL_VSC_ACTIVE))
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        caution = snap["caution"]
        assert caution is not None
        assert caution["cause"] == "VSC"
        # deployed lap 10, current_lap 42 -> elapsed = 42-10+1 = 33
        assert caution["elapsed_laps"] == 33

    def test_red_precedence_over_yellow(self, monkeypatch):
        """RED + simultaneous YELLOW -> RED wins."""
        monkeypatch.setattr(OF, "_get_json", _make_get_json(race_control=RACE_CONTROL_RED_PRECEDENCE))
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        caution = snap["caution"]
        assert caution is not None
        assert caution["cause"] == "RED"

    def test_vsc_ending_clears_vsc(self, monkeypatch):
        """VSC DEPLOYED then VSC ENDING -> caution is None."""
        rc = [
            {"category": None, "flag": None, "message": "VIRTUAL SAFETY CAR DEPLOYED", "lap_number": 10, "date": "2024-03-02T15:10:00", "scope": "Track"},
            {"category": None, "flag": None, "message": "VIRTUAL SAFETY CAR ENDING", "lap_number": 12, "date": "2024-03-02T15:12:00", "scope": "Track"},
        ]
        monkeypatch.setattr(OF, "_get_json", _make_get_json(race_control=rc))
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        assert snap["caution"] is None

    def test_elapsed_laps_floor_at_1(self, monkeypatch):
        """elapsed_laps is at least 1 even when current_lap == deployed_lap."""
        # SC deployed on lap 42 (same as current_lap)
        rc = [
            {"category": "SafetyCar", "flag": None, "message": "SAFETY CAR DEPLOYED", "lap_number": 42, "date": "2024-03-02T15:10:00", "scope": "Track"},
        ]
        monkeypatch.setattr(OF, "_get_json", _make_get_json(race_control=rc))
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        assert snap["caution"]["elapsed_laps"] == 1


# ---------------------------------------------------------------------------
# Fully empty API (all endpoints return [])
# ---------------------------------------------------------------------------

class TestEmptyAPI:
    def _empty_get_json(self, url, params, use_cache=True):
        return []

    def test_empty_returns_valid_shape(self, monkeypatch):
        """All endpoints empty -> available=False, drivers=[], caution=None, no exception."""
        monkeypatch.setattr(OF, "_get_json", self._empty_get_json)
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        assert snap["session"]["available"] is False
        assert snap["drivers"] == []
        assert snap["caution"] is None

    def test_empty_has_required_keys(self, monkeypatch):
        monkeypatch.setattr(OF, "_get_json", self._empty_get_json)
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        assert set(snap) == {"session", "drivers", "caution"}
        session_keys = {"session_key", "circuit", "year", "session_name", "current_lap", "available"}
        assert session_keys <= set(snap["session"])

    def test_no_exception_raised(self, monkeypatch):
        """live_snapshot must not raise on fully empty API."""
        monkeypatch.setattr(OF, "_get_json", self._empty_get_json)
        try:
            OF.live_snapshot(session_key="latest", use_cache=False)
        except Exception as exc:
            pytest.fail(f"live_snapshot raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# resolve_session
# ---------------------------------------------------------------------------

class TestResolveSession:
    def test_returns_dict_on_success(self, monkeypatch):
        monkeypatch.setattr(OF, "_get_json", _make_get_json())
        result = OF.resolve_session(session_key=9158, use_cache=False)
        assert isinstance(result, dict)
        assert result["session_key"] == 9158

    def test_returns_none_when_empty(self, monkeypatch):
        """resolve_session returns None when /sessions returns []."""
        monkeypatch.setattr(OF, "_get_json", lambda url, params, use_cache=True: [])
        result = OF.resolve_session(session_key=9999, use_cache=False)
        assert result is None


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------

class TestListSessions:
    def test_passthrough(self, monkeypatch):
        monkeypatch.setattr(OF, "_get_json", _make_get_json())
        rows = OF.list_sessions(year=2024, use_cache=False)
        assert isinstance(rows, list)

    def test_empty_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(OF, "_get_json", lambda url, params, use_cache=True: [])
        rows = OF.list_sessions(year=2099, use_cache=False)
        assert rows == []


# ---------------------------------------------------------------------------
# Driver sorting by position
# ---------------------------------------------------------------------------

class TestDriverSorting:
    def test_sorted_by_position(self, monkeypatch):
        """Drivers must be sorted by position ascending; None position goes last."""
        positions_rev = [
            {"driver_number": 44, "position": 1, "date": "2024-03-02T15:05:00"},
            {"driver_number": 1, "position": 2, "date": "2024-03-02T15:05:00"},
        ]
        monkeypatch.setattr(OF, "_get_json", _make_get_json(position=positions_rev))
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        driver_nums = [d["driver_number"] for d in snap["drivers"]]
        # 44 is P1, 1 is P2
        assert driver_nums.index(44) < driver_nums.index(1)

    def test_none_position_goes_last(self, monkeypatch):
        """Drivers with no position entry are sorted after positioned ones."""
        positions_partial = [
            {"driver_number": 1, "position": 1, "date": "2024-03-02T15:05:00"},
            # driver 44 has no position
        ]
        # add a driver-only-in-laps with no position
        extra_laps = LAPS + [
            {"driver_number": 99, "lap_number": 10, "lap_duration": 95.0, "is_pit_out_lap": False, "date_start": "2024-03-02T15:00:00"},
        ]
        monkeypatch.setattr(OF, "_get_json", _make_get_json(
            laps=extra_laps,
            position=positions_partial,
        ))
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        positioned = [d for d in snap["drivers"] if d["position"] is not None]
        unpositioned = [d for d in snap["drivers"] if d["position"] is None]
        assert all(
            snap["drivers"].index(p) < snap["drivers"].index(u)
            for p in positioned for u in unpositioned
        )


# ---------------------------------------------------------------------------
# Session availability flag
# ---------------------------------------------------------------------------

class TestSessionAvailable:
    def test_available_true_when_data_present(self, monkeypatch):
        monkeypatch.setattr(OF, "_get_json", _make_get_json())
        snap = OF.live_snapshot(session_key=9158, use_cache=False)
        assert snap["session"]["available"] is True

    def test_available_false_when_no_session_row(self, monkeypatch):
        """available=False when /sessions returns nothing (unknown session key)."""
        def fake(url, params, use_cache=True):
            if "sessions" in url:
                return []
            return LAPS  # data present but session unknown

        monkeypatch.setattr(OF, "_get_json", fake)
        snap = OF.live_snapshot(session_key=9999, use_cache=False)
        assert snap["session"]["available"] is False
