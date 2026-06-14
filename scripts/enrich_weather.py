"""Backfill weather enrichment onto existing race parquets — Phase 9.3.

The per-round parquets in ``data/raw/`` were ingested with only FastF1's
session-median TrackTemp/AirTemp. This script adds the Phase-9 weather channels
from Open-Meteo's ERA5 archive (FREE, no key), keyed by circuit coordinates +
the actual local race date/hour (from the FastF1 event schedule, so night races
are sampled correctly):

    Humidity, WindSpeed, Rainfall, SolarRadiation   (Open-Meteo, session window)
    AirTempOM, TrackTempEst                          (cross-check only)

FastF1's measured TrackTemp/AirTemp are left untouched as the authoritative
trackside values; the Open-Meteo columns are additive enrichment (old parquets
had none of these). Forward ingests (ingest.py) populate Humidity/WindSpeed/
Rainfall from FastF1's own sensors instead — same column names, documented
provenance difference.

The run also CALIBRATES the track-temp offset model in weather.py by regressing
the measured (TrackTemp − AirTemp) gap on SolarRadiation across all races, and
writes the fitted constants to reports/tyre/track_temp_calibration.json.

Run:
    python -m scripts.enrich_weather            # enrich missing, then calibrate
    python -m scripts.enrich_weather --force    # re-fetch all
    python -m scripts.enrich_weather --calibrate-only
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CALIB_PATH = PROJECT_ROOT / "reports" / "tyre" / "track_temp_calibration.json"

ENRICH_COLS = ["Humidity", "WindSpeed", "Rainfall", "SolarRadiation",
               "AirTempOM", "TrackTempEst"]


def _race_schedule() -> dict[tuple[int, int], tuple[str, int]]:
    """(season, round) -> (local ISO date, local session hour) from FastF1."""
    import fastf1

    out: dict[tuple[int, int], tuple[str, int]] = {}
    seasons = sorted({int(re.search(r"laps_(\d+)_r", Path(f).stem).group(1))
                      for f in glob.glob(str(RAW_DIR / "laps_*_r*.parquet"))})
    for year in seasons:
        try:
            sch = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:                       # pragma: no cover - network
            log.warning("schedule fetch failed for %d: %s", year, e)
            continue
        for _, row in sch.iterrows():
            s5 = row.get("Session5Date")
            if pd.isna(s5):
                continue
            ts = pd.Timestamp(s5)
            out[(year, int(row["RoundNumber"]))] = (ts.strftime("%Y-%m-%d"), int(ts.hour))
    return out


def enrich(force: bool = False) -> pd.DataFrame:
    """Enrich every per-round parquet; return a per-race calibration frame."""
    from src.pipeline.weather import session_weather

    schedule = _race_schedule()
    files = sorted(glob.glob(str(RAW_DIR / "laps_*_r*.parquet")))
    calib_rows = []

    for f in files:
        m = re.search(r"laps_(\d+)_r(\d+)", Path(f).stem)
        season, rnd = int(m.group(1)), int(m.group(2))
        df = pd.read_parquet(f)
        circuit = str(df["CircuitKey"].iloc[0])

        already = all(c in df.columns for c in ENRICH_COLS) and df["SolarRadiation"].notna().any()
        if already and not force:
            log.info("skip (already enriched): %s %02d %s", season, rnd, circuit)
        else:
            sched = schedule.get((season, rnd))
            if sched is None:
                log.warning("no schedule date for %d r%02d (%s) — skipping", season, rnd, circuit)
                continue
            date, hour = sched
            wx = session_weather(circuit, date, mode="archive", session_hour=hour)
            if wx.get("source") == "unavailable":
                log.warning("weather unavailable: %d r%02d %s", season, rnd, circuit)
                continue
            df["Humidity"]       = wx["Humidity"]
            df["WindSpeed"]      = wx["WindSpeed"]
            df["Rainfall"]       = wx["Rainfall"]
            df["SolarRadiation"] = wx["SolarRadiation"]
            df["AirTempOM"]      = wx["AirTemp"]
            df["TrackTempEst"]   = wx["TrackTempEst"]
            df.to_parquet(f, index=False)
            log.info("enriched %d r%02d %-26s air(F1=%.1f,OM=%.1f) rad=%.0f hum=%.0f",
                     season, rnd, circuit[:26],
                     df["AirTemp"].median(), wx["AirTemp"] or np.nan,
                     wx["SolarRadiation"] or np.nan, wx["Humidity"] or np.nan)

        # collect calibration point (measured gap vs solar radiation)
        if {"TrackTemp", "AirTemp", "SolarRadiation"} <= set(df.columns):
            tt, at, rad = df["TrackTemp"].median(), df["AirTemp"].median(), df["SolarRadiation"].median()
            if pd.notna(tt) and pd.notna(at) and pd.notna(rad):
                calib_rows.append({"season": season, "round": rnd, "circuit": circuit,
                                   "track_minus_air": float(tt - at), "radiation": float(rad)})
    return pd.DataFrame(calib_rows)


def calibrate(calib: pd.DataFrame) -> dict:
    """Fit TrackTemp − AirTemp = coef·(radiation/100) + bias across all races."""
    if len(calib) < 5:
        log.warning("not enough points to calibrate track-temp model (%d)", len(calib))
        return {}
    x = calib["radiation"].to_numpy() / 100.0
    y = calib["track_minus_air"].to_numpy()
    coef, bias = np.polyfit(x, y, 1)
    resid = y - (coef * x + bias)
    out = {
        "TRACK_TEMP_RAD_COEF": round(float(coef), 4),
        "TRACK_TEMP_BIAS": round(float(bias), 4),
        "n_races": int(len(calib)),
        "rmse_c": round(float(np.sqrt(np.mean(resid ** 2))), 3),
        "note": "TrackTemp ≈ AirTemp + COEF·(shortwave_radiation/100) + BIAS, "
                "fit on FastF1 session-median gap vs Open-Meteo ERA5 radiation.",
    }
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIB_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    log.info("track-temp calibration: coef=%.3f bias=%.3f rmse=%.2f°C (n=%d) -> %s",
             out["TRACK_TEMP_RAD_COEF"], out["TRACK_TEMP_BIAS"], out["rmse_c"],
             out["n_races"], CALIB_PATH)
    log.info("  → set weather.py: TRACK_TEMP_RAD_COEF=%.3f  TRACK_TEMP_BIAS=%.3f",
             out["TRACK_TEMP_RAD_COEF"], out["TRACK_TEMP_BIAS"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-fetch even if enriched")
    ap.add_argument("--calibrate-only", action="store_true",
                    help="skip fetching; recompute calibration from existing columns")
    args = ap.parse_args()

    if args.calibrate_only:
        rows = []
        for f in sorted(glob.glob(str(RAW_DIR / "laps_*_r*.parquet"))):
            df = pd.read_parquet(f)
            if {"TrackTemp", "AirTemp", "SolarRadiation"} <= set(df.columns):
                tt, at, rad = df["TrackTemp"].median(), df["AirTemp"].median(), df["SolarRadiation"].median()
                if pd.notna(tt) and pd.notna(at) and pd.notna(rad):
                    rows.append({"track_minus_air": float(tt - at), "radiation": float(rad)})
        calibrate(pd.DataFrame(rows))
        return

    calib = enrich(force=args.force)
    calibrate(calib)


if __name__ == "__main__":
    main()
