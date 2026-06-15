"""Tyre degradation model tests — synthetic data, no FastF1/GPU needed (Phase 3).

Run: pytest tests/test_tyre.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.tyre.fit import (
    MIN_STINTS_CIRCUIT,
    degradation_delta,
    degradation_model,
    fit_all_curves,
    fit_cell,
    predict_degradation,
    prepare_stint_laps,
)

B_TRUE, C_TRUE = 0.06, 0.002


def _stint_rows(season, rnd, driver, stint, circuit, compound, era,
                n_laps=18, age0=2, b=B_TRUE, c=C_TRUE, noise=0.0, seed=0,
                track_status="1"):
    """One synthetic stint following delta = b·(age−age0) + c·(age²−age0²)."""
    rng = np.random.default_rng(seed)
    rows = []
    base = 90.0
    for i in range(n_laps):
        age = age0 + i
        lap_number = 10 + i
        fuel_load = max(110.0 - (lap_number - 1) * 1.5, 0.0)
        fuel_effect = fuel_load * 0.03
        delta = b * (age - age0) + c * (age**2 - age0**2)
        rows.append({
            "Season": season, "RoundNumber": rnd, "Driver": driver,
            "StintID": stint, "LapNumber": lap_number,
            "CircuitKey": circuit, "Compound": compound, "Era": era,
            "TyreLife": age, "TrackStatus": track_status,
            "FuelEffect": fuel_effect,
            "LapTimeSeconds": base + delta + fuel_effect + rng.normal(0, noise),
        })
    return rows


def _cell_df(n_stints=8, **kw):
    rows = []
    for s in range(n_stints):
        rows += _stint_rows(2023, 1 + s, f"D{s%5:02d}", 1, seed=s, **kw)
    return pd.DataFrame(rows)


class TestPrepare:
    def test_fuel_correction_and_delta(self):
        df = _cell_df(n_stints=2, circuit="Bahrain Grand Prix", compound="SOFT", era=0)
        out = prepare_stint_laps(df)
        # FuelCorrected removes the fuel term exactly
        expected = df["LapTimeSeconds"] - df["FuelEffect"]
        got = out.sort_values(["RoundNumber", "LapNumber"])["FuelCorrected"].to_numpy()
        want = expected.to_numpy()[: len(got)]
        assert np.allclose(np.sort(got), np.sort(want))
        # Baseline = median of the first 3 laps -> that median lap's delta is 0
        med3 = out.groupby(["Season", "RoundNumber", "Driver", "StintID"])["Delta"].apply(
            lambda s: s.head(3).median()
        )
        assert np.allclose(med3, 0.0)

    def test_non_green_laps_dropped(self):
        df = pd.DataFrame(
            _stint_rows(2023, 1, "VER", 1, "X Grand Prix", "SOFT", 0, track_status="4")
        )
        assert prepare_stint_laps(df).empty

    def test_wet_compounds_now_included(self):
        # Wet support (Phase 11): INTERMEDIATE/WET are kept and modelled with
        # their own curves, no longer silently dropped.
        df = pd.DataFrame(
            _stint_rows(2023, 1, "VER", 1, "X Grand Prix", "INTERMEDIATE", 0)
        )
        out = prepare_stint_laps(df)
        assert not out.empty
        assert (out["Compound"] == "INTERMEDIATE").all()

    def test_unknown_compound_still_excluded(self):
        # A genuinely unrecognised compound string is still dropped.
        df = pd.DataFrame(
            _stint_rows(2023, 1, "VER", 1, "X Grand Prix", "ULTRASOFT", 0)
        )
        assert prepare_stint_laps(df).empty

    def test_short_stints_dropped(self):
        df = pd.DataFrame(
            _stint_rows(2023, 1, "VER", 1, "X Grand Prix", "SOFT", 0, n_laps=3)
        )
        assert prepare_stint_laps(df).empty


class TestFitCell:
    def test_recovers_known_coefficients(self):
        df = _cell_df(n_stints=10, circuit="Bahrain Grand Prix", compound="SOFT",
                      era=0, noise=0.05)
        prepared = prepare_stint_laps(df)
        fit = fit_cell(prepared)
        assert fit is not None
        assert fit["b"] == pytest.approx(B_TRUE, abs=0.02)
        assert fit["c"] == pytest.approx(C_TRUE, abs=0.001)
        assert fit["r2"] > 0.9

    def test_ci_widens_with_noise(self):
        quiet = fit_cell(prepare_stint_laps(
            _cell_df(n_stints=8, circuit="X Grand Prix", compound="SOFT", era=0, noise=0.05)))
        loud = fit_cell(prepare_stint_laps(
            _cell_df(n_stints=8, circuit="X Grand Prix", compound="SOFT", era=0, noise=0.5)))
        assert loud["b_ci"] > quiet["b_ci"]


class TestPooling:
    def test_thin_cell_falls_back_to_era_level(self):
        # Rich circuit (8 stints) + thin circuit (2 stints) for the same compound/era
        rich = _cell_df(n_stints=8, circuit="Rich Grand Prix", compound="SOFT", era=0, noise=0.1)
        thin = pd.DataFrame(
            _stint_rows(2023, 30, "TH1", 1, "Thin Grand Prix", "SOFT", 0, seed=91, noise=0.1)
            + _stint_rows(2023, 31, "TH2", 1, "Thin Grand Prix", "SOFT", 0, seed=92, noise=0.1)
        )
        prepared = prepare_stint_laps(pd.concat([rich, thin], ignore_index=True))
        curves = fit_all_curves(prepared)

        rich_row = curves[curves["CircuitKey"] == "Rich Grand Prix"].iloc[0]
        thin_row = curves[curves["CircuitKey"] == "Thin Grand Prix"].iloc[0]
        assert rich_row["pool_level"] == "circuit"
        assert rich_row["n_stints"] >= MIN_STINTS_CIRCUIT
        assert thin_row["pool_level"] in ("era", "compound")
        # Pooled fit still recovers the shared ground truth
        assert thin_row["b"] == pytest.approx(B_TRUE, abs=0.03)


class TestPredict:
    def test_predict_band_and_fallback(self):
        curves = pd.DataFrame([{
            "Compound": "SOFT", "CircuitKey": "Bahrain Grand Prix", "Era": 0,
            "pool_level": "circuit", "b": B_TRUE, "c": C_TRUE,
            "b_ci": 0.01, "c_ci": 0.0005, "r2": 0.95, "n_stints": 10, "n_laps": 150,
        }])
        ages = np.array([1.0, 10.0, 20.0])
        mid, lo, hi = predict_degradation(curves, "SOFT", "Bahrain Grand Prix", 0, ages)
        assert mid[0] == pytest.approx(0.0)
        assert (lo <= mid).all() and (mid <= hi).all()
        assert mid[2] == pytest.approx(degradation_delta(B_TRUE, C_TRUE, 20.0))
        # Unknown circuit falls back to the compound row, never raises
        mid2, _, _ = predict_degradation(curves, "SOFT", "Nowhere Grand Prix", 0, ages)
        assert np.allclose(mid, mid2)
        with pytest.raises(KeyError):
            predict_degradation(curves, "HARD", "Bahrain Grand Prix", 0, ages)


def test_model_intercept_free_differencing():
    """delta at age0 must be exactly 0 — the per-stint intercept is differenced out."""
    x = np.vstack([np.array([5.0, 6.0]), np.array([5.0, 5.0])])
    out = degradation_model(x, 0.1, 0.01)
    assert out[0] == pytest.approx(0.0)
