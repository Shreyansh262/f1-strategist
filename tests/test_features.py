"""
tests/test_features.py

Unit tests for src/pipeline/features.py.

Design philosophy:
- One test per transform — makes failures easy to localise
- Use minimal synthetic DataFrames — don't depend on real FastF1 data
- Test both the happy path and the edge cases (missing values, unknown compounds)
- All assertions are deterministic — no randomness in test data

Run with:
    pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.features import (
    COMPOUND_ORDER,
    FUEL_BURN_PER_LAP,
    FUEL_EFFECT_PER_KG,
    FUEL_LOAD_START_KG,
    MODEL_FEATURE_COLUMNS,
    add_compound_encoding,
    add_fuel_load,
    add_interaction_features,
    add_lap_position_features,
    add_tyre_age_features,
    build_features,
    impute_weather,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_minimal_df(n: int = 5) -> pd.DataFrame:
    """Minimal valid DataFrame that satisfies all feature transforms."""
    return pd.DataFrame(
        {
            "Driver":         ["VER", "HAM", "LEC", "NOR", "ALO"][:n],
            "LapNumber":      list(range(1, n + 1)),
            "LapTimeSeconds": [93.0 + i * 0.1 for i in range(n)],
            "Compound":       ["SOFT", "MEDIUM", "HARD", "SOFT", "MEDIUM"][:n],
            "TyreLife":       [3, 7, 12, 5, 9][:n],
            "TrackTemp":      [38.0, 39.0, 40.0, 38.5, 39.5][:n],
            "AirTemp":        [28.0, 27.5, 28.5, 27.0, 29.0][:n],
            "Season":         [2022] * n,
            "RoundNumber":    [1] * n,
            "CircuitKey":     ["bahrain"] * n,
        }
    )


# ---------------------------------------------------------------------------
# add_compound_encoding
# ---------------------------------------------------------------------------

class TestAddCompoundEncoding:

    def test_known_compounds_encoded(self):
        df = make_minimal_df()
        result = add_compound_encoding(df)
        assert "CompoundEncoded" in result.columns
        assert result.loc[result["Compound"] == "SOFT",   "CompoundEncoded"].iloc[0] == 0
        assert result.loc[result["Compound"] == "MEDIUM", "CompoundEncoded"].iloc[0] == 1
        assert result.loc[result["Compound"] == "HARD",   "CompoundEncoded"].iloc[0] == 2

    def test_unknown_compound_is_nan(self):
        df = make_minimal_df(1)
        df["Compound"] = "SUPERSOFT"   # not in COMPOUND_ORDER
        result = add_compound_encoding(df)
        assert result["CompoundEncoded"].isna().all()

    def test_original_compound_column_preserved(self):
        df = make_minimal_df()
        result = add_compound_encoding(df)
        assert "Compound" in result.columns

    def test_does_not_modify_input(self):
        df = make_minimal_df()
        original_cols = set(df.columns)
        _ = add_compound_encoding(df)
        assert set(df.columns) == original_cols, "Input DataFrame should not be mutated"


# ---------------------------------------------------------------------------
# add_fuel_load
# ---------------------------------------------------------------------------

class TestAddFuelLoad:

    def test_fuel_load_decreases_each_lap(self):
        df = make_minimal_df()
        result = add_fuel_load(df)
        assert result["FuelLoad"].is_monotonic_decreasing

    def test_fuel_load_on_lap_1(self):
        df = make_minimal_df(1)
        df["LapNumber"] = 1
        result = add_fuel_load(df)
        expected = FUEL_LOAD_START_KG  # no burn on lap 0 → 1
        assert result["FuelLoad"].iloc[0] == pytest.approx(expected)

    def test_fuel_load_never_negative(self):
        """Race longer than fuel capacity should floor at 0, not go negative."""
        df = make_minimal_df(1)
        df["LapNumber"] = 200   # absurdly long race
        result = add_fuel_load(df)
        assert result["FuelLoad"].iloc[0] >= 0.0

    def test_fuel_effect_proportional_to_fuel_load(self):
        df = make_minimal_df()
        result = add_fuel_load(df)
        expected_effect = result["FuelLoad"] * FUEL_EFFECT_PER_KG
        pd.testing.assert_series_equal(
            result["FuelEffect"].reset_index(drop=True),
            expected_effect.reset_index(drop=True),
            check_names=False,
        )

    def test_burn_rate(self):
        """Check the burn rate constant is applied correctly over multiple laps."""
        df = pd.DataFrame({
            "LapNumber": [1, 2, 3],
            "LapTimeSeconds": [93.0] * 3,
            "Compound": ["SOFT"] * 3,
            "TyreLife": [1, 2, 3],
            "TrackTemp": [38.0] * 3,
            "AirTemp": [28.0] * 3,
            "Season": [2022] * 3,
            "RoundNumber": [1] * 3,
            "CircuitKey": ["bahrain"] * 3,
            "Driver": ["VER"] * 3,
        })
        result = add_fuel_load(df)
        delta = result["FuelLoad"].diff().dropna()
        for val in delta:
            assert val == pytest.approx(-FUEL_BURN_PER_LAP)


# ---------------------------------------------------------------------------
# add_tyre_age_features
# ---------------------------------------------------------------------------

class TestAddTyreAgeFeatures:

    def test_squared_term_correct(self):
        df = make_minimal_df()
        result = add_tyre_age_features(df)
        expected_sq = df["TyreLife"] ** 2
        pd.testing.assert_series_equal(
            result["TyreAgeSq"].reset_index(drop=True),
            expected_sq.reset_index(drop=True),
            check_names=False,
        )

    def test_cubed_term_correct(self):
        df = make_minimal_df()
        result = add_tyre_age_features(df)
        expected_cu = df["TyreLife"] ** 3
        pd.testing.assert_series_equal(
            result["TyreAgeCubed"].reset_index(drop=True),
            expected_cu.reset_index(drop=True),
            check_names=False,
        )

    def test_new_columns_added(self):
        df = make_minimal_df()
        result = add_tyre_age_features(df)
        assert "TyreAgeSq" in result.columns
        assert "TyreAgeCubed" in result.columns


# ---------------------------------------------------------------------------
# add_interaction_features
# ---------------------------------------------------------------------------

class TestAddInteractionFeatures:

    def test_interaction_is_product(self):
        df = make_minimal_df()
        df = add_compound_encoding(df)
        result = add_interaction_features(df)
        expected = df["CompoundEncoded"] * df["TyreLife"]
        pd.testing.assert_series_equal(
            result["CompoundXTyreLife"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_soft_has_lowest_interaction_at_equal_tyre_life(self):
        """Soft (0) × TyreLife should produce smallest interaction value."""
        df = pd.DataFrame({
            "CompoundEncoded": [0, 1, 2],  # Soft, Medium, Hard
            "TyreLife": [10, 10, 10],
            "LapNumber": [10, 10, 10],
        })
        result = add_interaction_features(df)
        assert result["CompoundXTyreLife"].iloc[0] < result["CompoundXTyreLife"].iloc[1]
        assert result["CompoundXTyreLife"].iloc[1] < result["CompoundXTyreLife"].iloc[2]


# ---------------------------------------------------------------------------
# impute_weather
# ---------------------------------------------------------------------------

class TestImputeWeather:

    def test_no_imputation_needed_when_complete(self):
        df = make_minimal_df()
        result = impute_weather(df)
        assert result["TrackTemp"].isna().sum() == 0
        assert result["AirTemp"].isna().sum() == 0

    def test_missing_track_temp_imputed(self):
        df = make_minimal_df()
        df.loc[0, "TrackTemp"] = np.nan
        result = impute_weather(df)
        assert not result["TrackTemp"].isna().any()

    def test_imputation_uses_circuit_median(self):
        """Missing value should be filled with the median of its CircuitKey."""
        df = pd.DataFrame({
            "Driver": ["VER", "HAM", "LEC"],
            "LapNumber": [1, 2, 3],
            "LapTimeSeconds": [93.0] * 3,
            "Compound": ["SOFT"] * 3,
            "TyreLife": [3, 4, 5],
            "TrackTemp": [np.nan, 40.0, 42.0],  # first row missing
            "AirTemp": [28.0] * 3,
            "Season": [2022] * 3,
            "RoundNumber": [1] * 3,
            "CircuitKey": ["bahrain"] * 3,
        })
        result = impute_weather(df)
        expected_median = np.median([40.0, 42.0])
        assert result["TrackTemp"].iloc[0] == pytest.approx(expected_median)

    def test_missing_column_handled_gracefully(self):
        """If TrackTemp is entirely absent, should fill with global fallback."""
        df = make_minimal_df()
        df = df.drop(columns=["TrackTemp"])
        result = impute_weather(df)
        assert "TrackTemp" in result.columns
        assert result["TrackTemp"].notna().all()


# ---------------------------------------------------------------------------
# add_lap_position_features
# ---------------------------------------------------------------------------

class TestAddLapPositionFeatures:

    def test_norm_lap_number_between_0_and_1(self):
        df = make_minimal_df()
        result = add_lap_position_features(df)
        assert result["NormLapNumber"].between(0, 1).all()

    def test_last_lap_is_1(self):
        df = make_minimal_df()
        result = add_lap_position_features(df)
        max_lap = df["LapNumber"].max()
        assert result.loc[result["LapNumber"] == max_lap, "NormLapNumber"].iloc[0] == pytest.approx(1.0)

    def test_stint_phase_values(self):
        """TyreLife ≤8 → 0, 9–15 → 1, >15 → 2."""
        df = pd.DataFrame({
            "Driver": ["VER", "HAM", "LEC"],
            "LapNumber": [5, 10, 20],
            "LapTimeSeconds": [93.0] * 3,
            "Compound": ["SOFT"] * 3,
            "TyreLife": [5, 12, 20],
            "TrackTemp": [38.0] * 3,
            "AirTemp": [28.0] * 3,
            "Season": [2022] * 3,
            "RoundNumber": [1] * 3,
            "CircuitKey": ["bahrain"] * 3,
        })
        result = add_lap_position_features(df)
        phases = result["StintPhase"].tolist()
        assert phases[0] == 0   # TyreLife 5 → early
        assert phases[1] == 1   # TyreLife 12 → mid
        assert phases[2] == 2   # TyreLife 20 → late


# ---------------------------------------------------------------------------
# build_features (integration test)
# ---------------------------------------------------------------------------

class TestBuildFeatures:

    def test_returns_dataframe(self):
        df = make_minimal_df()
        result = build_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_all_model_features_present(self):
        df = make_minimal_df()
        result = build_features(df)
        for col in MODEL_FEATURE_COLUMNS:
            assert col in result.columns, f"Missing model feature: {col}"

    def test_no_nan_in_model_features(self):
        df = make_minimal_df()
        result = build_features(df)
        nan_counts = result[MODEL_FEATURE_COLUMNS].isna().sum()
        assert nan_counts.sum() == 0, f"NaN found in model features:\n{nan_counts[nan_counts > 0]}"

    def test_row_count_unchanged_for_clean_input(self):
        df = make_minimal_df()
        result = build_features(df)
        assert len(result) == len(df)

    def test_rows_with_unknown_compound_dropped(self):
        """Unknown compound → NaN in CompoundEncoded → row should be dropped."""
        df = make_minimal_df()
        df.loc[0, "Compound"] = "HYPERSOFT"
        result = build_features(df)
        assert len(result) == len(df) - 1

    def test_does_not_mutate_input(self):
        df = make_minimal_df()
        original_shape = df.shape
        _ = build_features(df)
        assert df.shape == original_shape, "build_features must not mutate the input"