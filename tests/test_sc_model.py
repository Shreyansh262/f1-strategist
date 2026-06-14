"""Offline tests for src/simulation/sc_model.py — no real parquets required."""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.simulation.sc_model import (
    build_distributions,
    load_distributions,
    sample_remaining,
    sample_total,
    save_distributions,
    _FALLBACK,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_race_df(status_seq: list[str], season: int = 2000, rnd: int = 1) -> pd.DataFrame:
    """Build a minimal one-driver DataFrame from a sequence of TrackStatus strings."""
    rows = []
    for lap, ts in enumerate(status_seq, start=1):
        rows.append({
            "Season": season,
            "RoundNumber": rnd,
            "LapNumber": lap,
            "TrackStatus": ts,
            "Driver": "TST",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: build_distributions recovers a known SC run of 4 and VSC run of 2
# ---------------------------------------------------------------------------

def test_build_distributions_known_runs():
    # SC of length 4, then green, then VSC of length 2
    statuses = ["1", "4", "4", "4", "4", "1", "6", "6", "1"]
    df = _make_race_df(statuses)
    dist = build_distributions(df)

    assert dist["SC"]["n"] == 1
    assert dist["SC"]["lengths"] == [4]
    assert dist["SC"]["max"] == 4

    assert dist["VSC"]["n"] == 1
    assert dist["VSC"]["lengths"] == [2]
    assert dist["VSC"]["max"] == 2

    assert dist["generated_from_races"] == 1


# ---------------------------------------------------------------------------
# Test 2: build_distributions with multiple races accumulates correctly
# ---------------------------------------------------------------------------

def test_build_distributions_two_races():
    df1 = _make_race_df(["1", "4", "4", "1"], season=2000, rnd=1)
    df2 = _make_race_df(["1", "6", "6", "6", "1"], season=2000, rnd=2)
    df = pd.concat([df1, df2], ignore_index=True)
    dist = build_distributions(df)

    assert dist["SC"]["lengths"] == [2]
    assert dist["VSC"]["lengths"] == [3]
    assert dist["generated_from_races"] == 2


# ---------------------------------------------------------------------------
# Test 3: sample_remaining with elapsed=2 and fixed lengths [5,5,5] always → 3
# ---------------------------------------------------------------------------

def test_sample_remaining_deterministic():
    dist = {
        "SC": {"lengths": [5, 5, 5], "n": 3, "mean": 5.0, "max": 5},
        "VSC": {"lengths": [3], "n": 1, "mean": 3.0, "max": 3},
    }
    for seed in range(20):
        rng = np.random.default_rng(seed)
        result = sample_remaining("SC", elapsed_laps=2, dist=dist, rng=rng)
        assert result == 3, f"seed={seed}: expected 3, got {result}"


# ---------------------------------------------------------------------------
# Test 4: sample_remaining when elapsed exceeds all observed lengths → 1
# ---------------------------------------------------------------------------

def test_sample_remaining_elapsed_exceeds_all():
    dist = {
        "SC": {"lengths": [2, 3], "n": 2, "mean": 2.5, "max": 3},
        "VSC": {"lengths": [2], "n": 1, "mean": 2.0, "max": 2},
    }
    rng = np.random.default_rng(0)
    result = sample_remaining("SC", elapsed_laps=10, dist=dist, rng=rng)
    assert result == 1


# ---------------------------------------------------------------------------
# Test 5: sample_remaining always >= 1, sample_total always >= 1
# ---------------------------------------------------------------------------

def test_sample_always_positive():
    dist = {
        "SC": {"lengths": [3, 4, 5, 6, 7], "n": 5, "mean": 5.0, "max": 7},
        "VSC": {"lengths": [2, 2, 3], "n": 3, "mean": 2.33, "max": 3},
    }
    rng = np.random.default_rng(42)
    for _ in range(50):
        assert sample_remaining("SC", elapsed_laps=0, dist=dist, rng=rng) >= 1
        assert sample_remaining("VSC", elapsed_laps=5, dist=dist, rng=rng) >= 1
        assert sample_total("SC", dist=dist, rng=rng) >= 1
        assert sample_total("VSC", dist=dist, rng=rng) >= 1


# ---------------------------------------------------------------------------
# Test 6: load_distributions on missing path → fallback (n>0 for SC and VSC)
# ---------------------------------------------------------------------------

def test_load_distributions_missing_path_returns_fallback():
    result = load_distributions(path="/nonexistent/path/sc_duration.json")
    assert result["SC"]["n"] > 0
    assert result["VSC"]["n"] > 0
    assert isinstance(result["SC"]["lengths"], list)
    assert isinstance(result["VSC"]["lengths"], list)


# ---------------------------------------------------------------------------
# Test 7: save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip():
    dist = {
        "SC": {"lengths": [3, 5, 7], "n": 3, "mean": 5.0, "max": 7},
        "VSC": {"lengths": [2, 4], "n": 2, "mean": 3.0, "max": 4},
        "generated_from_races": 42,
    }
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sc_duration.json")
        save_distributions(dist, path=path)
        loaded = load_distributions(path=path)
    assert loaded["SC"]["lengths"] == [3, 5, 7]
    assert loaded["VSC"]["n"] == 2
    assert loaded["generated_from_races"] == 42


# ---------------------------------------------------------------------------
# Test 8: determinism — same rng seed → same sample
# ---------------------------------------------------------------------------

def test_determinism():
    dist = {
        "SC": {"lengths": [3, 4, 5, 6], "n": 4, "mean": 4.5, "max": 6},
        "VSC": {"lengths": [2, 3], "n": 2, "mean": 2.5, "max": 3},
    }
    r1 = sample_remaining("SC", elapsed_laps=1, dist=dist, rng=np.random.default_rng(99))
    r2 = sample_remaining("SC", elapsed_laps=1, dist=dist, rng=np.random.default_rng(99))
    assert r1 == r2

    t1 = sample_total("VSC", dist=dist, rng=np.random.default_rng(7))
    t2 = sample_total("VSC", dist=dist, rng=np.random.default_rng(7))
    assert t1 == t2


# ---------------------------------------------------------------------------
# Test 9: concatenated TrackStatus codes ("46" means both SC and VSC codes)
# ---------------------------------------------------------------------------

def test_concatenated_status_codes():
    # "46" lap: both '4' and '6' present — counted in both SC and VSC runs
    statuses = ["1", "4", "46", "6", "1"]
    df = _make_race_df(statuses)
    dist = build_distributions(df)
    # SC run: laps 2,3 (length 2)
    assert 2 in dist["SC"]["lengths"]
    # VSC run: laps 3,4 (length 2)
    assert 2 in dist["VSC"]["lengths"]
