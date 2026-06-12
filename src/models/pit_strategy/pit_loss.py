"""Per-circuit pit-loss estimation from raw lap data (Phase 4 input).

Pit loss = time lost by pitting vs staying out, measured per pit event as:

    loss = (in-lap time + out-lap time) − 2 × driver's green-lap median that race

Both laps of the event must be green (TrackStatus == "1") — pitting under a
safety car is genuinely cheaper and would bias the green-racing loss low; the
simulation models SC pits separately (if at all). Events outside [5, 60] s are
discarded as data errors / red-flag artifacts.

Outputs data/pit_loss.json:
    {"<CircuitKey>": {"pit_loss_s": float, "n_events": int, "iqr_s": float}, ...,
     "_global": {...}}   # fallback for circuits with < MIN_EVENTS

Run:
    python -m src.models.pit_strategy.pit_loss
"""
from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Final

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR  = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROJECT_ROOT / "data" / "pit_loss.json"

MIN_EVENTS: Final[int] = 8       # below this, fall back to the global median
LOSS_MIN_S: Final[float] = 5.0
LOSS_MAX_S: Final[float] = 60.0
DEFAULT_LOSS_S: Final[float] = 22.0   # used only if the data yields nothing

RACE_KEYS = ["Season", "RoundNumber"]


def _load_raw() -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW_DIR / "laps_*_r*.parquet")))
    if not files:
        raise FileNotFoundError(f"No per-round parquet in {RAW_DIR}")
    raw = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    raw = raw.drop_duplicates(subset=["Season", "RoundNumber", "Driver", "LapNumber"])
    raw["TrackStatus"] = raw["TrackStatus"].astype(str)
    return raw


def measure_pit_events(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per usable green-flag pit event with its measured loss."""
    df = raw.sort_values(["Season", "RoundNumber", "Driver", "LapNumber"]).copy()
    grp = df.groupby(["Season", "RoundNumber", "Driver"])

    # In-lap (PitInTime set) followed by the same driver's out-lap (next lap).
    df["NextLapTime"]     = grp["LapTimeSeconds"].shift(-1)
    df["NextTrackStatus"] = grp["TrackStatus"].shift(-1)
    df["NextLapNumber"]   = grp["LapNumber"].shift(-1)

    in_laps = df[
        df["PitInTime"].notna()
        & df["LapTimeSeconds"].notna() & df["NextLapTime"].notna()
        & (df["NextLapNumber"] == df["LapNumber"] + 1)
        & (df["TrackStatus"] == "1") & (df["NextTrackStatus"] == "1")
    ].copy()

    # Driver's green non-pit median for that race = the counterfactual lap time.
    clean = df[
        (df["TrackStatus"] == "1")
        & df["PitInTime"].isna() & df["PitOutTime"].isna()
        & df["LapTimeSeconds"].between(60, 150)
    ]
    medians = (
        clean.groupby(["Season", "RoundNumber", "Driver"])["LapTimeSeconds"]
        .median().rename("GreenMedian")
    )
    in_laps = in_laps.join(medians, on=["Season", "RoundNumber", "Driver"], how="inner")

    in_laps["loss_s"] = (
        in_laps["LapTimeSeconds"] + in_laps["NextLapTime"] - 2 * in_laps["GreenMedian"]
    )
    events = in_laps[in_laps["loss_s"].between(LOSS_MIN_S, LOSS_MAX_S)]
    log.info(
        "Measured %d usable green pit events (from %d raw in-laps)",
        len(events), int(df["PitInTime"].notna().sum()),
    )
    return events[["Season", "RoundNumber", "CircuitKey", "Driver", "LapNumber", "loss_s"]]


def build_table(events: pd.DataFrame) -> dict:
    """Per-circuit median pit loss with a global fallback."""
    table: dict[str, dict] = {}
    global_loss = float(events["loss_s"].median()) if len(events) else DEFAULT_LOSS_S
    for circuit, g in events.groupby("CircuitKey"):
        if len(g) >= MIN_EVENTS:
            q1, q3 = g["loss_s"].quantile([0.25, 0.75])
            table[str(circuit)] = {
                "pit_loss_s": round(float(g["loss_s"].median()), 2),
                "n_events": int(len(g)),
                "iqr_s": round(float(q3 - q1), 2),
            }
    table["_global"] = {
        "pit_loss_s": round(global_loss, 2),
        "n_events": int(len(events)),
        "iqr_s": round(float(events["loss_s"].quantile(0.75) - events["loss_s"].quantile(0.25)), 2)
        if len(events) else 0.0,
    }
    return table


def load_pit_loss() -> dict:
    """Load the pit-loss table (for the MDP / sim engine)."""
    with open(OUT_PATH) as f:
        return json.load(f)


def pit_loss_for(table: dict, circuit: str) -> float:
    """Circuit pit loss with global fallback."""
    return table.get(circuit, table["_global"])["pit_loss_s"]


def main() -> dict:
    events = measure_pit_events(_load_raw())
    table = build_table(events)
    with open(OUT_PATH, "w") as f:
        json.dump(table, f, indent=2, sort_keys=True)
    log.info("Pit-loss table (%d circuits + _global) -> %s", len(table) - 1, OUT_PATH)
    for k in sorted(table):
        log.info("  %-35s %6.2fs  (n=%d)", k, table[k]["pit_loss_s"], table[k]["n_events"])
    return table


if __name__ == "__main__":
    main()
