"""
src/pipeline/features.py

All feature engineering for the F1 AI Race Strategist.
Every feature here must be available at prediction time during a race —
no future information, no post-race data.

Leakage audit:
- CircuitEncoded    ✓  circuit identity known before race start
- CompoundEncoded   ✓  tyre is already fitted
- Era               ✓  regulation era fixed for entire season
- TyreLife          ✓  known — laps done on current set
- FuelLoad          ✓  estimated from lap number (~1.5 kg/lap)
- TrackTemp         ✓  live sensor during lap
- AirTemp           ✓  live sensor during lap
- TyreAgeSq         ✓  derived from TyreLife
- StintPhase        ✓  derived from TyreLife
- StintID           ✓  derived from tyre resets (tyre model only)

Usage:
    from src.pipeline.features import build_features
    features_df = build_features(validated_df)
"""

import json
import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPINGS_DIR = PROJECT_ROOT / "data" / "mappings"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FUEL_LOAD_START_KG: Final[float] = 110.0
FUEL_BURN_PER_LAP:  Final[float] = 1.5
FUEL_EFFECT_PER_KG: Final[float] = 0.03

COMPOUND_ORDER: Final[dict[str, int]] = {
    "SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4,
}

# 0 = ground-effect era (2022-2025), 1 = active aero era (2026+)
ERA_BOUNDARY: Final[int] = 2026

TRACK_TEMP_FALLBACK: Final[float] = 35.0
AIR_TEMP_FALLBACK:   Final[float] = 25.0

# ---------------------------------------------------------------------------
# Engine (power-unit) supplier — DOMAIN KNOWLEDGE, not learned from data.
# Known before every race weekend, so there is no leakage. The value of this
# feature is cross-era transfer: a brand-new 2026 team (Cadillac) is unseen as
# a Team, but its PU lineage (Ferrari customer) is known and shared with teams
# the model HAS seen. Season-aware because suppliers change (Aston Martin
# Mercedes->Honda in 2026, Alpine Renault->Mercedes, Sauber Ferrari->Audi...).
# Default era covers 2022-2025; the 2026 dict overrides where changed.
# ---------------------------------------------------------------------------
_ENGINE_2022_2025: Final[dict[str, str]] = {
    "Red Bull Racing": "Honda RBPT",
    "AlphaTauri":      "Honda RBPT",
    "RB":              "Honda RBPT",
    "Racing Bulls":    "Honda RBPT",
    "Mercedes":        "Mercedes",
    "McLaren":         "Mercedes",
    "Aston Martin":    "Mercedes",
    "Williams":        "Mercedes",
    "Ferrari":         "Ferrari",
    "Haas F1 Team":    "Ferrari",
    "Alfa Romeo":      "Ferrari",
    "Kick Sauber":     "Ferrari",
    "Alpine":          "Renault",
}
_ENGINE_2026_OVERRIDES: Final[dict[str, str]] = {
    "Red Bull Racing": "Red Bull Ford",
    "Racing Bulls":    "Red Bull Ford",
    "Aston Martin":    "Honda",
    "Alpine":          "Mercedes",
    "Audi":            "Audi",
    "Cadillac":        "Ferrari",
}
# Fixed a-priori encoding over ALL known engine makers (incl. 2026 entrants).
# Unlike Team/Circuit this is NOT frozen-from-train: the dictionary is domain
# knowledge fixed before any data is seen, like the Era boundary.
ENGINE_ORDER: Final[dict[str, int]] = {
    e: i for i, e in enumerate(sorted({
        *_ENGINE_2022_2025.values(), *_ENGINE_2026_OVERRIDES.values()
    }))
}


def engine_for(team: str, season: int) -> str | None:
    """Power-unit supplier for a team in a season; None if unknown."""
    if season >= 2026 and team in _ENGINE_2026_OVERRIDES:
        return _ENGINE_2026_OVERRIDES[team]
    return _ENGINE_2022_2025.get(team)


# ---------------------------------------------------------------------------
# Individual feature transforms
# ---------------------------------------------------------------------------

def add_compound_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode tyre compound. SOFT=0 MEDIUM=1 HARD=2 INT=3 WET=4."""
    df = df.copy()
    df["CompoundEncoded"] = df["Compound"].map(COMPOUND_ORDER)
    n_unknown = df["CompoundEncoded"].isna().sum()
    if n_unknown > 0:
        logger.warning("%d rows have unrecognised compound — will be NaN", n_unknown)
    return df


def _freeze_mapping(values: pd.Series, filename: str, save: bool) -> dict[str, int]:
    """Build an alphabetical label mapping; optionally persist it to data/mappings/.

    Persisting is opt-in (save=True) so that tests, notebooks, and ad-hoc runs on
    small frames can never clobber the production mapping artifacts. Only the
    trainer's freeze step (freeze_encoding_mappings) should save.
    """
    unique = sorted(values.dropna().unique())
    mapping = {v: i for i, v in enumerate(unique)}
    if save:
        MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
        map_path = MAPPINGS_DIR / filename
        with open(map_path, "w") as f:
            json.dump(mapping, f, indent=2)
        logger.info("Froze mapping (%d entries) → %s", len(mapping), map_path)
    return mapping


def freeze_encoding_mappings(train_df: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    """Build AND persist circuit/team mappings from the TRAINING split only.

    This is the single sanctioned writer of data/mappings/*.json. Mappings must be
    frozen from train so unseen val/test circuits/teams hit the -1 sentinel path —
    the same path a genuinely new team takes at serving time (Section 5/6).

    Returns (circuit_mapping, team_mapping).
    """
    circuit_map = _freeze_mapping(train_df["CircuitKey"], "circuit_map.json", save=True)
    team_map    = _freeze_mapping(train_df["Team"],       "team_map.json",    save=True)
    return circuit_map, team_map


def load_encoding_mappings() -> tuple[dict[str, int], dict[str, int]]:
    """Load the frozen circuit/team mappings for inference/serving."""
    with open(MAPPINGS_DIR / "circuit_map.json") as f:
        circuit_map = json.load(f)
    with open(MAPPINGS_DIR / "team_map.json") as f:
        team_map = json.load(f)
    return circuit_map, team_map


def add_circuit_encoding(
    df: pd.DataFrame,
    mapping: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Label-encode CircuitKey using a frozen mapping dictionary.

    mapping=None builds an in-memory mapping from df (does NOT save — use
    freeze_encoding_mappings() to persist train-frozen mappings).
    mapping=dict applies the frozen mapping; unseen circuits → sentinel -1.

    Leakage check: circuit identity is known before race start — no future
    information used. Frozen mapping ensures train/inference consistency.
    """
    df = df.copy()

    if mapping is None:
        mapping = _freeze_mapping(df["CircuitKey"], "circuit_map.json", save=False)

    df["CircuitEncoded"] = df["CircuitKey"].map(mapping).fillna(-1).astype(int)

    n_unseen = (df["CircuitEncoded"] == -1).sum()
    if n_unseen > 0:
        logger.warning(
            "%d rows have unseen CircuitKey (encoded as -1). "
            "New circuits may need adding to circuit_map.json.",
            n_unseen,
        )
    return df


def add_team_encoding(
    df: pd.DataFrame,
    mapping: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Label-encode Team using a frozen mapping dictionary.

    mapping=None builds an in-memory mapping from df (does NOT save — use
    freeze_encoding_mappings() to persist train-frozen mappings).
    mapping=dict applies the frozen mapping; unseen teams → sentinel -1.

    Leakage check: team identity is known before race start — no future
    information used. Frozen mapping ensures train/inference consistency.
    """
    df = df.copy()

    if mapping is None:
        mapping = _freeze_mapping(df["Team"], "team_map.json", save=False)

    df["TeamEncoded"] = df["Team"].map(mapping).fillna(-1).astype(int)

    n_unseen = (df["TeamEncoded"] == -1).sum()
    if n_unseen > 0:
        logger.warning(
            "%d rows have unseen Team (encoded as -1). "
            "New teams may need adding to team_map.json.",
            n_unseen,
        )
    return df


def add_era_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary regulation era feature.
        Era 0: ground-effect regulations (2022-2025)
        Era 1: active aero regulations   (2026+)

    Leakage check: era is fixed for the whole season, known before race start.
    """
    df = df.copy()
    df["Era"] = (df["Season"] >= ERA_BOUNDARY).astype(int)
    return df


def add_fuel_load(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate fuel load (kg) from lap number.
    ~1.5 kg/lap burn from 110 kg starting load.
    FuelEffect: lap time penalty in seconds (~0.03 s/kg → ~3s total over race).
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
    Polynomial tyre age features.
    TyreAgeSq:    accelerating degradation (quadratic term).
    TyreAgeCubed: cliff-edge degradation on Soft compounds.
    Explicit polynomial terms → cleaner SHAP values than implicit RF splits.
    """
    df = df.copy()
    df["TyreAgeSq"]    = df["TyreLife"] ** 2
    df["TyreAgeCubed"] = df["TyreLife"] ** 3
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compound × TyreLife interaction.
    Soft degrades faster than Hard at the same TyreLife — captured explicitly.
    """
    df = df.copy()
    df["CompoundXTyreLife"] = df["CompoundEncoded"] * df["TyreLife"]
    return df


def impute_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing TrackTemp / AirTemp with per-circuit medians,
    falling back to global constants if a circuit has all nulls.
    """
    df = df.copy()
    for col, fallback in [("TrackTemp", TRACK_TEMP_FALLBACK),
                           ("AirTemp",   AIR_TEMP_FALLBACK)]:
        if col not in df.columns:
            logger.warning("%s not found — using global fallback %.1f", col, fallback)
            df[col] = fallback
            continue
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        circuit_medians = df.groupby("CircuitKey")[col].transform("median")
        df[col] = df[col].fillna(circuit_medians)
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
    NormLapNumber: lap as fraction of race length (0 → 1).
    StintPhase:    0=early (TyreLife ≤8), 1=mid (9-15), 2=late (>15).
    """
    df = df.copy()
    max_lap = df.groupby(["Season", "RoundNumber"])["LapNumber"].transform("max")
    df["NormLapNumber"] = df["LapNumber"] / max_lap
    df["StintPhase"] = pd.cut(
        df["TyreLife"], bins=[0, 8, 15, 999], labels=[0, 1, 2],
    ).astype(int)
    return df


def add_stint_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign stint IDs per driver per race.
    New stint when TyreLife resets (decreases) OR Compound changes.

    Used by the tyre degradation model (fit.py) to isolate continuous stints.
    NOT included in MODEL_FEATURE_COLUMNS — not a lap time predictor feature.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Season, RoundNumber, Driver, LapNumber, TyreLife, Compound.

    Returns
    -------
    pd.DataFrame with 'StintID' column added (int, 1-indexed per driver per race).
    """
    df = df.copy()
    df = df.sort_values(["Season", "RoundNumber", "Driver", "LapNumber"])
    grp            = df.groupby(["Season", "RoundNumber", "Driver"])
    tyre_reset     = grp["TyreLife"].diff() < 0
    compound_change = grp["Compound"].shift() != df["Compound"]
    new_stint      = (tyre_reset | compound_change).fillna(True)
    df["StintID"]  = (
        new_stint
        .groupby([df["Season"], df["RoundNumber"], df["Driver"]])
        .cumsum()
        .astype(int)
    )
    return df


# ---------------------------------------------------------------------------
# Column lists — source of truth
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: Final[list[str]] = [
    # Target
    "LapTimeSeconds",
    # Identity (grouping / eval — NOT model inputs)
    "Driver", "Season", "RoundNumber", "CircuitKey", "LapNumber",
    # Model features
    "CircuitEncoded", "TeamEncoded", "CompoundEncoded", "Era",
    "TyreLife", "TyreAgeSq", "TyreAgeCubed",
    "CompoundXTyreLife", "FuelLoad", "FuelEffect",
    "TrackTemp", "AirTemp", "NormLapNumber", "StintPhase",
]

MODEL_FEATURE_COLUMNS: Final[list[str]] = [
    # Exactly the columns passed to model.fit() / model.predict()
    "CircuitEncoded", "TeamEncoded", "CompoundEncoded", "Era",
    "TyreLife", "TyreAgeSq", "TyreAgeCubed",
    "CompoundXTyreLife", "FuelLoad", "FuelEffect",
    "TrackTemp", "AirTemp", "NormLapNumber", "StintPhase",
]


# ---------------------------------------------------------------------------
# Master transform
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    circuit_mapping: dict[str, int] | None = None,
    team_mapping: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Apply all feature engineering transforms in the correct order.

    Parameters
    ----------
    df : pd.DataFrame
        Validated laps DataFrame from validate.py.
    circuit_mapping : dict[str, int] | None
        Training mode  (None)   : builds mapping from df, saves circuit_map.json.
        Inference mode (dict)   : uses supplied mapping; unseen circuits → -1.
    team_mapping : dict[str, int] | None
        Training mode  (None)   : builds mapping from df, saves team_map.json.
        Inference mode (dict)   : uses supplied mapping; unseen teams → -1.

    Returns
    -------
    pd.DataFrame
        Restricted to FEATURE_COLUMNS. Rows with NaN in model features dropped.
    """
    logger.info("Building features from %d rows", len(df))

    df = add_compound_encoding(df)
    df = add_circuit_encoding(df, mapping=circuit_mapping)
    df = add_team_encoding(df, mapping=team_mapping)
    df = add_era_feature(df)
    df = impute_weather(df)
    df = add_fuel_load(df)
    df = add_tyre_age_features(df)
    df = add_interaction_features(df)
    df = add_lap_position_features(df)
    # NOTE: add_stint_id() is called separately in the tyre model pipeline only.

    available    = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing_cols = set(FEATURE_COLUMNS) - set(available)
    if missing_cols:
        logger.warning("Missing expected columns: %s", missing_cols)
    df = df[available].copy()

    n_before           = len(df)
    model_cols_present = [c for c in MODEL_FEATURE_COLUMNS if c in df.columns]
    df                 = df.dropna(subset=model_cols_present)
    n_dropped          = n_before - len(df)
    if n_dropped > 0:
        logger.warning("Dropped %d rows with NaN in model features", n_dropped)

    logger.info("Feature build complete: %d rows, %d columns", len(df), len(df.columns))
    return df