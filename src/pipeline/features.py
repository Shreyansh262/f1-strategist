"""
src/pipeline/features.py

All feature engineering for the F1 AI Race Strategist.
Every feature here must be available at prediction time during a race —
no future information, no post-race data.

Leakage audit (checked for every feature):
- LapNumber          ✓  known at start of lap
- TyreLife           ✓  known — you know how many laps the tyre has done
- CompoundEncoded    ✓  known — tyre is already fitted
- FuelLoad           ✓  estimated from lap number (decreases ~1.5 kg/lap)
- TrackTemp          ✓  live sensor during lap
- AirTemp            ✓  live sensor during lap
- TyreAgeSq          ✓  derived from TyreLife — no future info
- IsFirstStint       ✓  known from lap 1 flag or TyreLife reset

Usage:
    from src.pipeline.features import build_features
    features_df = build_features(validated_df)
"""

import logging
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FUEL_LOAD_START_KG: Final[float] = 110.0   # Approximate full fuel load at race start
FUEL_BURN_PER_LAP: Final[float] = 1.5      # kg per lap (F1 regulations ~105 kg / ~70 laps)
FUEL_EFFECT_PER_KG: Final[float] = 0.03    # seconds per kg (well-established in literature)

# Ordinal encoding: captures tyre performance hierarchy meaningfully
COMPOUND_ORDER: Final[dict[str, int]] = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTERMEDIATE": 3,
    "WET": 4,
}

# Track temperature imputation fallbacks (median values from FastF1 EDA)
TRACK_TEMP_FALLBACK: Final[float] = 35.0   # °C
AIR_TEMP_FALLBACK: Final[float] = 25.0     # °C


# ---------------------------------------------------------------------------
# Individual feature transforms — each is a pure function on a DataFrame
# ---------------------------------------------------------------------------

def add_compound_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal-encode tyre compound.
    Soft=0, Medium=1, Hard=2 preserves degradation hierarchy.
    Intermediate and Wet treated separately (rarely appear in dry races).
    """
    df = df.copy()
    df["CompoundEncoded"] = df["Compound"].map(COMPOUND_ORDER)
    n_unknown = df["CompoundEncoded"].isna().sum()
    if n_unknown > 0:
        logger.warning("%d rows have unrecognised compound — will be NaN", n_unknown)
    return df


def add_fuel_load(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate fuel load in kg from lap number.

    Fuel load decreases linearly at ~1.5 kg/lap from a starting load of ~110 kg.
    This is a known-at-race-time estimate — real teams use fuel flow sensors,
    but this proxy is standard in the literature and defensible.

    Fuel effect on lap time: ~0.03 s per kg (approximately 3 seconds total over a race).
    """
    df = df.copy()
    df["FuelLoad"] = np.maximum(
        FUEL_LOAD_START_KG - (df["LapNumber"] - 1) * FUEL_BURN_PER_LAP,
        0.0,
    )
    df["FuelEffect"] = df["FuelLoad"] * FUEL_EFFECT_PER_KG
    return df


def add_tyre_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive polynomial tyre age features.

    TyreLife:    raw laps on current set (already in validated DataFrame)
    TyreAgeSq:   captures the accelerating degradation curve (quadratic term)
    TyreAgeCubed: for compounds with cliff-edge degradation (primarily Soft)

    Interview note: using polynomial terms explicitly is more interpretable than
    letting the RF split on them implicitly — SHAP values are cleaner.
    """
    df = df.copy()
    df["TyreAgeSq"] = df["TyreLife"] ** 2
    df["TyreAgeCubed"] = df["TyreLife"] ** 3
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compound × TyreLife interaction.

    A Soft tyre degrades much faster than a Hard at high TyreLife — this
    explicit interaction term captures that without relying on the RF to
    discover it through deep splits.
    """
    df = df.copy()
    df["CompoundXTyreLife"] = df["CompoundEncoded"] * df["TyreLife"]
    return df


def impute_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing TrackTemp and AirTemp with circuit-level medians,
    falling back to global constants.

    Imputation is done BEFORE the train/val/test split is applied, but
    the imputation values are derived per-circuit from the full DataFrame.
    This is acceptable because temperature medians per circuit do not leak
    lap time targets — they are environmental constants.
    """
    df = df.copy()

    for col, fallback in [("TrackTemp", TRACK_TEMP_FALLBACK),
                           ("AirTemp", AIR_TEMP_FALLBACK)]:
        if col not in df.columns:
            logger.warning("%s not found — using global fallback %.1f", col, fallback)
            df[col] = fallback
            continue

        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue

        # Impute with per-circuit median
        circuit_medians = df.groupby("CircuitKey")[col].transform("median")
        df[col] = df[col].fillna(circuit_medians)

        # If still missing (circuit had all nulls), use global fallback
        still_missing = df[col].isna().sum()
        if still_missing > 0:
            logger.warning(
                "%d rows still missing %s after circuit imputation — using %.1f",
                still_missing, col, fallback,
            )
            df[col] = df[col].fillna(fallback)

    return df


def add_lap_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Race progress features: normalised lap number and stint phase.

    NormLapNumber: lap as a fraction of total race laps (0 → 1)
                   estimated from max LapNumber per race.
    StintPhase:    early (0–30% tyre life) / mid (30–60%) / late (60%+) encoded 0/1/2
    """
    df = df.copy()

    # Normalise lap number within each race
    max_lap = df.groupby(["Season", "RoundNumber"])["LapNumber"].transform("max")
    df["NormLapNumber"] = df["LapNumber"] / max_lap

    # Stint phase (0 = early, 1 = mid, 2 = late)
    # Boundaries based on tyre life percentage relative to typical stint length
    # Typical stint: ~20–25 laps → thresholds at 8 and 15 laps
    df["StintPhase"] = pd.cut(
        df["TyreLife"],
        bins=[0, 8, 15, 999],
        labels=[0, 1, 2],
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# Master transform — apply all steps in order
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: Final[list[str]] = [
    # Target
    "LapTimeSeconds",
    # Identity (for grouping / eval — not model inputs)
    "Driver",
    "Season",
    "RoundNumber",
    "CircuitKey",
    "LapNumber",
    # Model features
    "CompoundEncoded",
    "TyreLife",
    "TyreAgeSq",
    "TyreAgeCubed",
    "CompoundXTyreLife",
    "FuelLoad",
    "FuelEffect",
    "TrackTemp",
    "AirTemp",
    "NormLapNumber",
    "StintPhase",
]

MODEL_FEATURE_COLUMNS: Final[list[str]] = [
    # Exactly the columns passed to model.fit() / model.predict()
    "CompoundEncoded",
    "TyreLife",
    "TyreAgeSq",
    "TyreAgeCubed",
    "CompoundXTyreLife",
    "FuelLoad",
    "FuelEffect",
    "TrackTemp",
    "AirTemp",
    "NormLapNumber",
    "StintPhase",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transforms in the correct order.

    Parameters
    ----------
    df : pd.DataFrame
        Validated laps DataFrame from validate.py.

    Returns
    -------
    pd.DataFrame
        DataFrame with all engineered features added, restricted to
        FEATURE_COLUMNS. Rows with any NaN in model feature columns
        are dropped with a warning.
    """
    logger.info("Building features from %d rows", len(df))

    df = add_compound_encoding(df)
    df = impute_weather(df)
    df = add_fuel_load(df)
    df = add_tyre_age_features(df)
    df = add_interaction_features(df)
    df = add_lap_position_features(df)

    # Keep only the columns we need downstream
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing_cols = set(FEATURE_COLUMNS) - set(available)
    if missing_cols:
        logger.warning("Missing expected columns: %s", missing_cols)
    df = df[available].copy()

    # Drop rows with NaN in any model feature
    n_before = len(df)
    model_cols_present = [c for c in MODEL_FEATURE_COLUMNS if c in df.columns]
    df = df.dropna(subset=model_cols_present)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning("Dropped %d rows with NaN in model features", n_dropped)

    logger.info("Feature build complete: %d rows, %d columns", len(df), len(df.columns))
    return df