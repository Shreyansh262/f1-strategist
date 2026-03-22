"""
src/pipeline/validate.py

Pandera schema validation for the raw laps DataFrame produced by ingest.py.
Raises SchemaError immediately if the data violates any contract — fail fast,
never silently pass bad data downstream.

Usage:
    from src.pipeline.validate import validate_laps
    clean_df = validate_laps(raw_df)
"""

import logging

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known tyre compounds in FastF1.  UNKNOWN / TEST_UNKNOWN are excluded
# upstream but we whitelist here defensively.
# ---------------------------------------------------------------------------
VALID_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}

# Lap time bounds derived from EDA (Bahrain 2023):
# - Clean lap floor  ~90 s  (fast circuits can go ~80 s — keep 75 s as hard floor)
# - Outlier ceiling  ~115 s (pit-in / safety car laps spike to 130 s+, filtered upstream)
LAP_TIME_MIN = 75.0   # seconds
LAP_TIME_MAX = 130.0  # seconds — includes slow laps but not obvious sensor errors

TYRE_AGE_MAX = 80     # No tyre runs more than ~60 laps in a race

LAPS_SCHEMA = DataFrameSchema(
    columns={
        # ---- identity -------------------------------------------------------
        "Driver": Column(
            str,
            Check(lambda s: s.str.len() == 3, element_wise=False,
                  error="Driver codes must be 3 characters"),
            nullable=False,
        ),
        "LapNumber": Column(
            int,
            [
                Check.greater_than_or_equal_to(1),
                Check.less_than_or_equal_to(100),
            ],
            nullable=False,
        ),
        # ---- timing ---------------------------------------------------------
        "LapTimeSeconds": Column(
            float,
            [
                Check.greater_than_or_equal_to(LAP_TIME_MIN),
                Check.less_than_or_equal_to(LAP_TIME_MAX),
            ],
            nullable=False,
        ),
        # ---- tyre -----------------------------------------------------------
        "Compound": Column(
            str,
            Check(lambda s: s.isin(VALID_COMPOUNDS),
                  element_wise=False,
                  error=f"Compound must be one of {VALID_COMPOUNDS}"),
            nullable=False,
        ),
        
        "TyreLife": Column(
            int,
            [
                Check.greater_than_or_equal_to(1),
                Check.less_than_or_equal_to(TYRE_AGE_MAX),
            ],
            nullable=False,
        ),
        # ---- environment ----------------------------------------------------
        "TrackTemp": Column(
            float,
            [
                Check.greater_than_or_equal_to(0.0),
                Check.less_than_or_equal_to(70.0),   # °C — record track temp ~66 °C
            ],
            nullable=True,   # Sometimes missing in FastF1; imputed in features.py
        ),
        "AirTemp": Column(
            float,
            [
                Check.greater_than_or_equal_to(-10.0),
                Check.less_than_or_equal_to(50.0),
            ],
            nullable=True,
        ),
        # ---- session metadata -----------------------------------------------
        "Season": Column(int, Check.isin([2021, 2022, 2023, 2024, 2025]), nullable=False),
        "RoundNumber": Column(int, Check.greater_than_or_equal_to(1), nullable=False),
        "CircuitKey": Column(str, nullable=False),
    },
    checks=[],
    coerce=True,       # Cast dtypes where safe (e.g. int64 → int)
    strict=False,      # Allow extra columns (telemetry data etc.)
)


def validate_laps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a raw laps DataFrame against LAPS_SCHEMA.

    Filters out pit out-laps (TyreLife == 1) before validation — these are
    outliers (~107 s) confirmed in EDA and should never enter model training.

    Parameters
    ----------
    df : pd.DataFrame
        Raw laps DataFrame from ingest.py.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with pit out-laps removed.

    Raises
    ------
    pandera.errors.SchemaError
        If any column contract is violated after filtering.
    """
    n_raw = len(df)

    # --- filter pit out-laps ------------------------------------------------
    df = df[df["TyreLife"] > 1].copy()
    n_filtered = n_raw - len(df)
    if n_filtered > 0:
        logger.info("Removed %d pit out-laps (TyreLife == 1)", n_filtered)

    # --- filter extreme lap times -------------------------------------------
    # Belt-and-suspenders: ingest.py may already filter, but validate again
    # filter extreme lap times
    mask_time = df["LapTimeSeconds"].between(LAP_TIME_MIN, LAP_TIME_MAX)
    n_time = (~mask_time).sum()
    if n_time > 0:
        logger.warning("Dropping %d laps outside time bounds [%s, %s]",
                   n_time, LAP_TIME_MIN, LAP_TIME_MAX)
    df = df[mask_time].copy()

    # ADD THIS — drop laps with null compound
    mask_compound = df["Compound"].isna() | (df["Compound"] == "None") | (~df["Compound"].isin(VALID_COMPOUNDS))
    n_bad_compound = mask_compound.sum()
    if n_bad_compound > 0:
        logger.warning("Dropping %d laps with invalid/null Compound", n_bad_compound)
    df = df[~mask_compound].copy()

    # --- filter sprint race laps -------------------------------------------
    race_mean = df.groupby(["Season", "RoundNumber"])["LapTimeSeconds"].transform("mean")
    n_sprint = (race_mean > 105).sum()
    if n_sprint > 0:
        logger.warning("Dropping %d laps from sprint races (mean lap time > 105s)", n_sprint)
    df = df[race_mean <= 105].copy()

    # --- pandera validation -------------------------------------------------
    validated = LAPS_SCHEMA.validate(df, lazy=True)
    # --- pandera validation -------------------------------------------------
    validated = LAPS_SCHEMA.validate(df, lazy=True)   # lazy=True collects all errors at once
    logger.info("Validation passed: %d laps retained from %d raw", len(validated), n_raw)
    return validated