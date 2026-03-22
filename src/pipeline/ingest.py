"""FastF1 data ingestion with local caching.

Fetches lap-level telemetry for a given season and round.
Outputs a clean DataFrame saved to data/raw/.
"""

import logging
import os
from pathlib import Path

import fastf1
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Cache — enable FIRST, before any session load ────────────────────────────
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_session(
    year: int,
    round_number: int,
    session_type: str = "R",
) -> fastf1.core.Session:
    """Load a FastF1 session (uses cache if available).

    Args:
        year: Championship season, e.g. 2023.
        round_number: Round number within the season, e.g. 1.
        session_type: 'R' = Race, 'Q' = Qualifying, 'FP1/FP2/FP3'.

    Returns:
        Loaded FastF1 Session object.
    """
    logger.info(f"Loading session: {year} Round {round_number} ({session_type})")
    session = fastf1.get_session(year, round_number, session_type)
    session.load(telemetry=False, weather=True, messages=False)
    logger.info(f"Session loaded: {session.event['EventName']}")
    return session


def extract_laps(session: fastf1.core.Session) -> pd.DataFrame:
    """Extract lap-level features from a loaded session.

    Columns are named to match validate.py schema:
        Season, RoundNumber, CircuitKey, TrackTemp, AirTemp
    """
    laps = session.laps.copy()

    laps = laps[laps["IsPersonalBest"].notna() | laps["LapTime"].notna()]

    keep_cols = [
        "Driver", "DriverNumber", "Team",
        "LapNumber", "LapTime",
        "Compound", "TyreLife", "FreshTyre",
        "Stint", "IsPersonalBest",
        "PitOutTime", "PitInTime",
        "TrackStatus",
    ]
    keep_cols = [c for c in keep_cols if c in laps.columns]
    laps = laps[keep_cols].copy()

    if "LapTime" in laps.columns:
        laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

    # ── Column names must match validate.py schema ────────────────────────
    laps["Season"]      = session.event["EventDate"].year
    laps["RoundNumber"] = session.event["RoundNumber"]
    laps["CircuitKey"]  = session.event["EventName"]

    laps["IsPitLap"] = laps["PitInTime"].notna() | laps["PitOutTime"].notna()

    # ── Merge weather (TrackTemp, AirTemp) onto laps ─────────────────────
    # Weather is time-series; take the median values for the session
    # (lap-accurate merge requires lap timestamps — median is a safe fallback)
    weather = session.weather_data
    if weather is not None and len(weather) > 0:
        laps["TrackTemp"] = weather["TrackTemp"].median()
        laps["AirTemp"]   = weather["AirTemp"].median()
    else:
        laps["TrackTemp"] = None
        laps["AirTemp"]   = None

    # Cast TyreLife to int (FastF1 returns float)
    laps["TyreLife"] = laps["TyreLife"].fillna(0).astype(int)
    # Cast LapNumber to int
    laps["LapNumber"] = laps["LapNumber"].fillna(0).astype(int)

    logger.info(f"Extracted {len(laps)} laps from {session.event['EventName']}")
    return laps

def fetch_weather(session: fastf1.core.Session) -> pd.DataFrame:
    """Extract weather data (track/air temp, humidity) from session.

    Args:
        session: A loaded FastF1 Session.

    Returns:
        DataFrame with weather snapshots indexed by session time.
    """
    weather = session.weather_data.copy()
    weather.columns = [c.strip() for c in weather.columns]
    return weather


def save_laps(df: pd.DataFrame, year: int, round_number: int) -> Path:
    """Save extracted laps DataFrame as parquet.

    Args:
        df: Laps DataFrame.
        year: Season year.
        round_number: Round number.

    Returns:
        Path to saved file.
    """
    out_path = RAW_DIR / f"laps_{year}_r{round_number:02d}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved: {out_path} ({len(df)} rows)")
    return out_path


def ingest_season(year: int, rounds: list[int] | None = None) -> pd.DataFrame:
    """Ingest all rounds for a season and return combined DataFrame.

    Args:
        year: Championship season.
        rounds: List of round numbers. If None, fetches rounds 1–22.

    Returns:
        Combined laps DataFrame for the full season.
    """
    if rounds is None:
        rounds = list(range(1, 23))

    all_laps: list[pd.DataFrame] = []

    for rnd in rounds:
        try:
            session = fetch_session(year, rnd)
            laps = extract_laps(session)
            save_laps(laps, year, rnd)
            all_laps.append(laps)
        except Exception as e:
            logger.warning(f"Skipped {year} round {rnd}: {e}")
            continue

    combined = pd.concat(all_laps, ignore_index=True)
    combined_path = RAW_DIR / f"laps_{year}_full.parquet"
    combined.to_parquet(combined_path, index=False)
    logger.info(f"Season {year}: {len(combined)} total laps → {combined_path}")
    return combined


if __name__ == "__main__":
    # Quick smoke test — fetch 2023 Round 1 (Bahrain)
    session = fetch_session(2023, 1)
    laps = extract_laps(session)
    weather = fetch_weather(session)
    save_laps(laps, 2023, 1)

    print(laps[["Driver", "LapNumber", "LapTimeSeconds", "Compound", "TyreLife"]].head(10))
    print(f"\nWeather shape: {weather.shape}")
    print(f"Track temp range: {weather['TrackTemp'].min():.1f}–{weather['TrackTemp'].max():.1f}°C")