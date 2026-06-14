"""Weather / temperature-robustness tests — Phase 9. No network, no GPU.

Open-Meteo calls are monkeypatched or the pure helpers are exercised directly;
the temperature-aware tyre model is checked for monotonicity, backward
compatibility, and synthetic slope recovery.

Run: pytest tests/test_weather.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline import weather as W
from src.pipeline.circuits import CIRCUIT_COORDS, coords_for
from src.models.tyre.fit import (
    attach_temp_sensitivity,
    degradation_delta,
    fit_all_curves,
    fit_temp_sensitivity,
    predict_degradation,
    prepare_stint_laps,
)


# ---------------------------------------------------------------------------
# 9.1 circuit coordinates
# ---------------------------------------------------------------------------

class TestCircuits:
    def test_known_circuit(self):
        lat, lon = coords_for("Bahrain Grand Prix")
        assert 25 < lat < 27 and 49 < lon < 52

    def test_suffix_and_case_tolerant(self):
        assert coords_for("monaco") == CIRCUIT_COORDS["Monaco Grand Prix"]
        assert coords_for("BAHRAIN GRAND PRIX") == CIRCUIT_COORDS["Bahrain Grand Prix"]

    def test_unknown_circuit(self):
        assert coords_for("Nürburgring Grand Prix") is None

    def test_all_coords_plausible(self):
        for name, (lat, lon) in CIRCUIT_COORDS.items():
            assert -90 <= lat <= 90 and -180 <= lon <= 180, name


# ---------------------------------------------------------------------------
# 9.2 Open-Meteo client (pure helpers + monkeypatched fetch)
# ---------------------------------------------------------------------------

class TestTrackTempEstimate:
    def test_monotonic_in_radiation(self):
        cold = W.estimate_track_temp(25.0, 100.0)
        hot = W.estimate_track_temp(25.0, 900.0)
        assert hot > cold

    def test_monotonic_in_air(self):
        assert W.estimate_track_temp(30.0, 500.0) > W.estimate_track_temp(15.0, 500.0)

    def test_track_above_air_under_sun(self):
        # Asphalt under strong sun is hotter than the air.
        assert W.estimate_track_temp(20.0, 700.0) > 20.0

    def test_negative_radiation_clamped(self):
        # No physical "negative sun" contribution.
        assert W.estimate_track_temp(20.0, -50.0) == pytest.approx(20.0 + W.TRACK_TEMP_BIAS)


def _fake_hourly(temps, rads, hums, winds, precs, day="2023-03-05"):
    times = [f"{day}T{h:02d}:00" for h in range(24)]
    return {
        "time": times,
        "temperature_2m": temps,
        "shortwave_radiation": rads,
        "relative_humidity_2m": hums,
        "wind_speed_10m": winds,
        "precipitation": precs,
    }


class TestSessionSummary:
    def test_afternoon_window_default(self):
        temps = [10.0] * 24
        for h in range(13, 17):
            temps[h] = 30.0                      # hot afternoon
        hourly = _fake_hourly(temps, [600.0] * 24, [40.0] * 24, [5.0] * 24, [0.0] * 24)
        out = W.summarize_session(hourly)
        assert out["AirTemp"] == pytest.approx(30.0)
        assert out["Rainfall"] is False
        assert out["TrackTempEst"] > 30.0

    def test_night_session_hour_picks_right_window(self):
        # Cold afternoon, warm evening — a night race must read the evening.
        temps = [20.0] * 24
        for h in (17, 18, 19):
            temps[h] = 28.0
        hourly = _fake_hourly(temps, [0.0] * 24, [50.0] * 24, [8.0] * 24, [0.0] * 24)
        day = W.summarize_session(hourly)                      # afternoon default
        night = W.summarize_session(hourly, session_hour=18)   # 18:00 race
        assert night["AirTemp"] == pytest.approx(28.0)
        assert night["AirTemp"] > day["AirTemp"]

    def test_rainfall_flag(self):
        hourly = _fake_hourly([25.0] * 24, [500.0] * 24, [80.0] * 24, [10.0] * 24,
                              [0.0] * 13 + [2.0] * 4 + [0.0] * 7)
        assert W.summarize_session(hourly)["Rainfall"] is True

    def test_session_weather_monkeypatched(self, monkeypatch):
        captured = {}

        def fake_get(url, params, use_cache=True):
            captured["url"] = url
            return {"hourly": _fake_hourly([22.0] * 24, [550.0] * 24, [60.0] * 24,
                                           [6.0] * 24, [0.0] * 24)}

        monkeypatch.setattr(W, "_get_json", fake_get)
        out = W.session_weather("Bahrain Grand Prix", "2023-03-05", mode="archive")
        assert out["source"] == "archive"
        assert out["AirTemp"] == pytest.approx(22.0)
        assert "archive" in captured["url"]

    def test_session_weather_unknown_circuit(self):
        out = W.session_weather("Nowhere Grand Prix", "2023-03-05")
        assert out["source"] == "unavailable"


# ---------------------------------------------------------------------------
# 9.4 temperature-aware tyre degradation
# ---------------------------------------------------------------------------

def _temp_curves():
    """Minimal curves table carrying Phase-9 temperature columns."""
    return pd.DataFrame([{
        "Compound": "MEDIUM", "CircuitKey": "Bahrain Grand Prix", "Era": 0,
        "pool_level": "circuit", "b": 0.06, "c": 0.002,
        "b_ci": 0.01, "c_ci": 0.0005, "r2": 0.9, "n_stints": 10, "n_laps": 150,
        "temp_ref": 30.0, "b_temp": 0.004, "c_temp": 0.0,
    }])


class TestTempAwarePredict:
    def test_hotter_degrades_more(self):
        curves = _temp_curves()
        age = np.array([20.0])
        cold = predict_degradation(curves, "MEDIUM", "Bahrain Grand Prix", 0, age, track_temp=20.0)[0]
        hot = predict_degradation(curves, "MEDIUM", "Bahrain Grand Prix", 0, age, track_temp=45.0)[0]
        assert hot[0] > cold[0]

    def test_track_temp_none_is_baseline(self):
        curves = _temp_curves()
        age = np.array([1.0, 10.0, 25.0])
        none = predict_degradation(curves, "MEDIUM", "Bahrain Grand Prix", 0, age)[0]
        at_ref = predict_degradation(curves, "MEDIUM", "Bahrain Grand Prix", 0, age, track_temp=30.0)[0]
        base = degradation_delta(0.06, 0.002, age)
        assert np.allclose(none, base)
        assert np.allclose(at_ref, base)          # at temp_ref the adjustment is zero

    def test_backward_compatible_without_temp_columns(self):
        # Pre-Phase-9 curves (no temp_ref/b_temp/c_temp) must still work and
        # ignore track_temp entirely.
        curves = _temp_curves().drop(columns=["temp_ref", "b_temp", "c_temp"])
        age = np.array([15.0])
        a = predict_degradation(curves, "MEDIUM", "Bahrain Grand Prix", 0, age)[0]
        b = predict_degradation(curves, "MEDIUM", "Bahrain Grand Prix", 0, age, track_temp=50.0)[0]
        assert np.allclose(a, b)

    def test_slope_cannot_go_negative(self):
        # Strong negative extrapolation clamps the slope at 0, not below.
        curves = _temp_curves()
        age = np.array([1.0, 30.0])
        mid = predict_degradation(curves, "MEDIUM", "Bahrain Grand Prix", 0, age, track_temp=-50.0)[0]
        assert mid[1] >= mid[0] - 1e-9          # non-decreasing (b>=0, c~0)


# ---------------------------------------------------------------------------
# fit_temp_sensitivity recovers a synthetic temperature slope
# ---------------------------------------------------------------------------

def _temp_stint_rows(season, rnd, driver, circuit, compound, era, track_temp,
                     b0=0.05, c0=0.0015, b_temp=0.003, n_laps=18, age0=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    b = b0 + b_temp * (track_temp - 30.0)        # true sensitivity around Tref=30
    for i in range(n_laps):
        age = age0 + i
        lap = 10 + i
        fuel_effect = max(110.0 - (lap - 1) * 1.5, 0.0) * 0.03
        delta = b * (age - age0) + c0 * (age ** 2 - age0 ** 2)
        rows.append({
            "Season": season, "RoundNumber": rnd, "Driver": driver, "StintID": 1,
            "LapNumber": lap, "CircuitKey": circuit, "Compound": compound, "Era": era,
            "TyreLife": age, "TrackStatus": "1", "TrackTemp": track_temp,
            "FuelEffect": fuel_effect,
            "LapTimeSeconds": 90.0 + delta + fuel_effect + rng.normal(0, 0.03),
        })
    return rows


def test_fit_temp_sensitivity_recovers_slope():
    # Many races for one compound across a temperature sweep 15..50°C.
    rows = []
    for i, temp in enumerate(np.linspace(15, 50, 16)):
        rows += _temp_stint_rows(2023, i + 1, f"D{i:02d}", f"C{i}", "MEDIUM", 0,
                                 float(temp), b_temp=0.003, seed=i)
    prepared = prepare_stint_laps(pd.DataFrame(rows))
    sens = fit_temp_sensitivity(prepared)
    assert not sens.empty
    row = sens[(sens["Compound"] == "MEDIUM") & (sens["Era"] == 0)].iloc[0]
    assert row["b_temp"] == pytest.approx(0.003, abs=0.0015)
    assert row["temp_min"] <= 16 and row["temp_max"] >= 49


def test_attach_temp_sensitivity_fills_and_is_backward_safe():
    rows = []
    for i, temp in enumerate(np.linspace(20, 45, 10)):
        rows += _temp_stint_rows(2023, i + 1, f"D{i:02d}", "Bahrain Grand Prix",
                                 "MEDIUM", 0, float(temp), seed=i)
    prepared = prepare_stint_laps(pd.DataFrame(rows))
    curves = fit_all_curves(prepared)
    sens = fit_temp_sensitivity(prepared)
    enriched = attach_temp_sensitivity(curves, prepared, sens)
    assert {"temp_ref", "b_temp", "c_temp"} <= set(enriched.columns)
    assert enriched["temp_ref"].notna().all()
    assert enriched["b_temp"].notna().all()
