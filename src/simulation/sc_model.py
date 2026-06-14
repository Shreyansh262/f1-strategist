"""
Safety-Car / Virtual-Safety-Car duration model.

Learns from historical race parquets how many laps an SC or VSC period lasts,
and can sample the REMAINING laps of an in-progress caution period.

TrackStatus FIA codes (strings, may be concatenated per lap):
  '1'=green, '2'=yellow, '4'=Safety Car, '5'=red, '6'=Virtual SC,
  '7'=VSC ending
"""
from __future__ import annotations

import glob
import json
import os
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fallback distribution used when data/sc_duration.json is absent
# ---------------------------------------------------------------------------
_FALLBACK: dict = {
    "SC": {
        "lengths": [3, 4, 4, 5, 6, 4, 5, 3, 6, 4],
        "n": 10,
        "mean": 4.4,
        "max": 6,
    },
    "VSC": {
        "lengths": [2, 2, 3, 3, 2, 3, 2],
        "n": 7,
        "mean": 2.43,
        "max": 3,
    },
    "generated_from_races": 0,
}

_DEFAULT_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "sc_duration.json"
)
_DEFAULT_JSON_PATH = os.path.normpath(_DEFAULT_JSON_PATH)


def build_distributions(parquet_glob_or_df=None) -> dict:
    """Build SC and VSC run-length distributions from parquet files or a DataFrame.

    Parameters
    ----------
    parquet_glob_or_df : str | list[str] | pd.DataFrame | None
        - None   -> glob data/raw/*.parquet relative to repo root
        - str    -> treated as a glob pattern
        - list   -> list of parquet file paths
        - DataFrame -> used directly (must have Season, RoundNumber, LapNumber, TrackStatus)

    Returns
    -------
    dict with keys "SC", "VSC", "generated_from_races"
    """
    if isinstance(parquet_glob_or_df, pd.DataFrame):
        df = parquet_glob_or_df.copy()
        paths = None
    else:
        if parquet_glob_or_df is None:
            # Default: repo root / data / raw / per-race parquets (exclude full-season files)
            root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
            pattern = os.path.join(root, "data", "raw", "laps_*_r*.parquet")
        elif isinstance(parquet_glob_or_df, list):
            paths = parquet_glob_or_df
            pattern = None
        else:
            pattern = parquet_glob_or_df

        if not isinstance(parquet_glob_or_df, list):
            paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No parquet files found matching pattern")

        frames = []
        for p in paths:
            try:
                f = pd.read_parquet(p, columns=["Season", "RoundNumber", "LapNumber", "TrackStatus"])
                frames.append(f)
            except Exception:
                pass
        if not frames:
            raise ValueError("Could not read any parquet files")
        df = pd.concat(frames, ignore_index=True)

    # Ensure TrackStatus is string
    df["TrackStatus"] = df["TrackStatus"].astype(str)

    # Reduce to one representative row per (Season, RoundNumber, LapNumber)
    # Use the first driver's value — field-wide status, all drivers same
    lap_status = (
        df.groupby(["Season", "RoundNumber", "LapNumber"], sort=True)["TrackStatus"]
        .first()
        .reset_index()
    )

    sc_lengths: list[int] = []
    vsc_lengths: list[int] = []
    race_count = 0

    for (season, rnd), grp in lap_status.groupby(["Season", "RoundNumber"]):
        grp = grp.sort_values("LapNumber").reset_index(drop=True)
        race_count += 1

        sc_run = 0
        vsc_run = 0
        for _, row in grp.iterrows():
            status = row["TrackStatus"]
            sc_active = "4" in status
            vsc_active = "6" in status

            # SC run
            if sc_active:
                sc_run += 1
            else:
                if sc_run > 0:
                    sc_lengths.append(sc_run)
                sc_run = 0

            # VSC run
            if vsc_active:
                vsc_run += 1
            else:
                if vsc_run > 0:
                    vsc_lengths.append(vsc_run)
                vsc_run = 0

        # Flush runs that extend to last lap
        if sc_run > 0:
            sc_lengths.append(sc_run)
        if vsc_run > 0:
            vsc_lengths.append(vsc_run)

    def _stats(lengths: list[int]) -> dict:
        n = len(lengths)
        return {
            "lengths": lengths,
            "n": n,
            "mean": float(round(sum(lengths) / n, 3)) if n else 0.0,
            "max": int(max(lengths)) if lengths else 0,
        }

    return {
        "SC": _stats(sc_lengths),
        "VSC": _stats(vsc_lengths),
        "generated_from_races": race_count,
    }


def save_distributions(dist: dict, path: str = None) -> None:
    """Save the distribution dict to a JSON file."""
    if path is None:
        path = _DEFAULT_JSON_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dist, fh, indent=2)


def load_distributions(path: str = None) -> dict:
    """Load distribution dict from JSON.  Returns a sensible fallback if missing."""
    if path is None:
        path = _DEFAULT_JSON_PATH
    if not os.path.exists(path):
        return _FALLBACK
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sample_remaining(
    cause: str,
    elapsed_laps: int,
    dist: Optional[dict] = None,
    rng=None,
) -> int:
    """Sample the REMAINING laps for an in-progress caution period.

    Parameters
    ----------
    cause : "SC" or "VSC"
    elapsed_laps : int  — laps the caution has already been active
    dist : distribution dict (from load_distributions / build_distributions);
           if None, load_distributions() is called.
    rng : numpy random generator; default fresh default_rng()

    Returns
    -------
    int >= 1 (remaining laps)
    """
    if dist is None:
        dist = load_distributions()
    if rng is None:
        rng = np.random.default_rng()

    lengths: list[int] = dist[cause]["lengths"]
    threshold = max(elapsed_laps, 1)
    candidates = [l for l in lengths if l >= threshold]
    if not candidates:
        return 1
    total = int(rng.choice(candidates))
    return max(total - elapsed_laps, 1)


def sample_total(cause: str, dist: Optional[dict] = None, rng=None) -> int:
    """Sample an unconditional total duration for a caution period (laps), floored at 1."""
    if dist is None:
        dist = load_distributions()
    if rng is None:
        rng = np.random.default_rng()

    lengths: list[int] = dist[cause]["lengths"]
    if not lengths:
        return 1
    return max(int(rng.choice(lengths)), 1)


# ---------------------------------------------------------------------------
# CLI entry-point: python -m src.simulation.sc_model
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building SC/VSC distributions from race parquets …")
    dist = build_distributions()
    sc = dist["SC"]
    vsc = dist["VSC"]
    print(f"SC  — n={sc['n']}, mean={sc['mean']:.2f}, max={sc['max']}")
    print(f"VSC — n={vsc['n']}, mean={vsc['mean']:.2f}, max={vsc['max']}")
    print(f"Races processed: {dist['generated_from_races']}")
    save_distributions(dist)
    print(f"Saved to {_DEFAULT_JSON_PATH}")
