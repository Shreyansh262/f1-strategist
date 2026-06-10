"""TFT data loader — carry-back pattern (Section 5, Error log item).

build_features() emits only FEATURE_COLUMNS; add_stint_id() is tyre-model-only.
Sequence models need Team/Compound/StintID/TrackStatus. This module re-attaches
them after feature engineering by merging on LAP_KEYS.

Imported by train_tft.py. Also usable by evaluate.py for TFT eval.

MLFLOW_TRACKING_URI re-exported here so train_tft.py has a single import source
and doesn't import train.py (avoids triggering that module's train logic on import).
"""
from __future__ import annotations

import glob
import logging
from pathlib import Path

import pandas as pd

from src.pipeline.validate import validate_laps
from src.pipeline.features import add_stint_id, build_features

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Re-export so train_tft.py only needs one import
MLFLOW_TRACKING_URI: str = (PROJECT_ROOT / "mlruns").as_uri()

# Columns that uniquely identify a lap — merge key for carry-back
LAP_KEYS = ["Season", "RoundNumber", "Driver", "LapNumber"]

# Columns to carry back that build_features() drops
CARRY_BACK_COLS = ["Team", "Compound", "StintID", "TrackStatus"]


def load_tft_data() -> pd.DataFrame:
    """Load, validate, feature-engineer, and carry-back sequence-model columns.

    Steps (in order — do not reorder):
      1. Glob per-round parquet only (laps_*_r*.parquet) — avoids double-count (Error 10)
      2. Concat + drop_duplicates on LAP_KEYS as safety net
      3. validate_laps()
      4. add_stint_id() — tyre-model helper; needed before build_features for StintID
      5. Snapshot carry-back cols on LAP_KEYS
      6. build_features() — restricts to FEATURE_COLUMNS, drops Compound/Team/StintID
      7. Merge carry-back cols back on LAP_KEYS

    Returns DataFrame with FEATURE_COLUMNS + CARRY_BACK_COLS + LAP_KEYS.
    """
    raw_dir = PROJECT_ROOT / "data" / "raw"
    files = sorted(glob.glob(str(raw_dir / "laps_*_r*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}. Run ingest first.")

    frames = [pd.read_parquet(f) for f in files]
    raw = pd.concat(frames, ignore_index=True)
    before = len(raw)
    raw = raw.drop_duplicates(subset=LAP_KEYS)
    log.info("Loaded %d files | %d rows -> %d after dedup", len(files), before, len(raw))

    validated = validate_laps(raw)
    log.info("After validate_laps: %d rows", len(validated))

    with_stints = add_stint_id(validated)

    # Snapshot before build_features drops these cols
    carry = with_stints[LAP_KEYS + CARRY_BACK_COLS].copy()
    # TrackStatus may not exist in older parquets — add null column if absent
    for col in CARRY_BACK_COLS:
        if col not in carry.columns:
            carry[col] = None
            log.warning("Column %s missing from raw data — filling with None", col)

    features = build_features(with_stints)
    log.info("After build_features: %d rows, %d cols", len(features), features.shape[1])

    merged = features.merge(carry, on=LAP_KEYS, how="left")
    null_stints = merged["StintID"].isna().sum()
    if null_stints:
        log.warning("%d rows have null StintID after carry-back — check add_stint_id()", null_stints)

    log.info(
        "load_tft_data complete | %d rows | cols: %s",
        len(merged), sorted(merged.columns.tolist())
    )
    return merged
