"""Open-Meteo weather client — Phase 9 (weather robustness, Section 10.8).

A tiny, dependency-free client over the FREE Open-Meteo APIs (no key required):

    * Archive / ERA5  (``archive-api.open-meteo.com``) — historical reanalysis,
      used to backfill weather for races already in the dataset.
    * Forecast        (``api.open-meteo.com``)         — up to 16 days ahead,
      used to fetch conditions for an UPCOMING race in the dashboard.

Channels pulled (hourly): ``temperature_2m``, ``precipitation``,
``relative_humidity_2m``, ``wind_speed_10m``, ``shortwave_radiation``.

Track temperature is not provided by any free API, so we approximate it from
air temperature + incoming solar radiation with a simple linear offset model
(``estimate_track_temp``). The constants were calibrated against the FastF1
session-median TrackTemp we already ingest (see ``calibrate_track_temp``); the
estimate is deliberately rough and documented as such.

Network access is required only when a request misses the on-disk JSON cache
(``data/weather_cache/``). Tests never hit the network — they monkeypatch
``_get_json`` or call the pure helpers directly.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "weather_cache"

ARCHIVE_URL: Final[str] = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL: Final[str] = "https://api.open-meteo.com/v1/forecast"

HOURLY_CHANNELS: Final[list[str]] = [
    "temperature_2m",
    "precipitation",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
]

# Race sessions run in the afternoon; sample the hottest part of the day so the
# medians reflect on-track conditions rather than the 24h mean. Local time.
SESSION_HOURS: Final[range] = range(13, 17)   # 13:00–16:59 local

# ── Track-temperature offset model ───────────────────────────────────────────
# TrackTemp ≈ AirTemp + TRACK_TEMP_RAD_COEF · (shortwave_radiation / 100) + bias.
# Calibrated by scripts/enrich_weather.py: linear fit of the measured FastF1
# (TrackTemp − AirTemp) gap on Open-Meteo ERA5 shortwave radiation over 94 races
# (2022–2026), RMSE 4.1°C. Values mirrored from
# reports/tyre/track_temp_calibration.json — re-run enrich_weather to refresh.
TRACK_TEMP_RAD_COEF: Final[float] = 1.775   # °C per 100 W/m²
TRACK_TEMP_BIAS: Final[float] = 2.836       # °C baseline track-vs-air offset

_REQUEST_TIMEOUT_S: Final[int] = 30


# ---------------------------------------------------------------------------
# Low-level fetch with on-disk cache
# ---------------------------------------------------------------------------

def _cache_key(url: str, params: dict) -> Path:
    safe = urllib.parse.urlencode(sorted(params.items()))
    digest = abs(hash(url + safe))
    stub = url.split("//", 1)[-1].split("/", 1)[0].split(".")[0]
    return CACHE_DIR / f"{stub}_{digest}.json"


def _get_json(url: str, params: dict, use_cache: bool = True) -> dict:
    """GET a JSON payload, caching successful responses on disk.

    Tests monkeypatch this function so no network call is made.
    """
    cache_path = _cache_key(url, params)
    if use_cache and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    query = urllib.parse.urlencode(params, doseq=True)
    full = f"{url}?{query}"
    logger.debug("Open-Meteo GET %s", full)
    with urllib.request.urlopen(full, timeout=_REQUEST_TIMEOUT_S) as resp:
        payload = json.load(resp)

    if "error" in payload and payload.get("error"):
        raise RuntimeError(f"Open-Meteo error: {payload.get('reason', payload)}")

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    return payload


# ---------------------------------------------------------------------------
# Public fetchers
# ---------------------------------------------------------------------------

def fetch_archive(lat: float, lon: float, date: str) -> dict[str, list]:
    """Historical hourly weather (ERA5) for a single local calendar day.

    Args:
        lat, lon: circuit coordinates.
        date: ISO date "YYYY-MM-DD" (local race day).

    Returns:
        dict {channel -> list of hourly values} plus "time" (list of ISO hours).
    """
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": date, "end_date": date,
        "hourly": ",".join(HOURLY_CHANNELS),
        "timezone": "auto",
    }
    payload = _get_json(ARCHIVE_URL, params)
    return payload.get("hourly", {})


def fetch_forecast(lat: float, lon: float, date: str | None = None) -> dict[str, list]:
    """Hourly weather forecast (≤16 days ahead) for an upcoming race day.

    Args:
        lat, lon: circuit coordinates.
        date: ISO "YYYY-MM-DD". If None, returns the default 7-day forecast.

    Returns:
        dict {channel -> list of hourly values} plus "time".
    """
    params: dict = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(HOURLY_CHANNELS),
        "timezone": "auto",
    }
    if date is not None:
        params["start_date"] = date
        params["end_date"] = date
    # Forecasts change; cache them only when pinned to a specific date.
    payload = _get_json(FORECAST_URL, params, use_cache=date is not None)
    return payload.get("hourly", {})


# ---------------------------------------------------------------------------
# Track-temperature estimate (air + solar radiation -> track surface temp)
# ---------------------------------------------------------------------------

def estimate_track_temp(air_temp: float, shortwave_radiation: float) -> float:
    """Approximate track-surface temperature from air temp + solar radiation.

    Asphalt under sun runs well above air temperature; the gap scales with
    incoming shortwave radiation. Linear offset model — see module docstring.
    """
    return float(
        air_temp
        + TRACK_TEMP_RAD_COEF * (max(shortwave_radiation, 0.0) / 100.0)
        + TRACK_TEMP_BIAS
    )


def _session_window(session_hour: int | None) -> set[int]:
    """Hours to sample: ±1h around a known session start, else the afternoon."""
    if session_hour is None:
        return set(SESSION_HOURS)
    return {(session_hour + d) % 24 for d in (-1, 0, 1)}


def _session_median(
    hourly: dict[str, list], channel: str, session_hour: int | None = None
) -> float | None:
    """Median of one channel over the session window (afternoon by default).

    Many F1 races (Bahrain, Saudi, Singapore, Abu Dhabi, Las Vegas, Qatar) run
    at dusk/night, so when the actual local session hour is known it is passed
    here to sample the right window instead of the generic afternoon.
    """
    times = hourly.get("time")
    values = hourly.get(channel)
    if not times or not values:
        return None
    window = _session_window(session_hour)
    picked = [
        v for t, v in zip(times, values)
        if v is not None and int(str(t)[11:13]) in window
    ]
    if not picked:
        picked = [v for v in values if v is not None]
    if not picked:
        return None
    picked.sort()
    n = len(picked)
    mid = n // 2
    return float(picked[mid] if n % 2 else (picked[mid - 1] + picked[mid]) / 2)


def summarize_session(
    hourly: dict[str, list], session_hour: int | None = None
) -> dict[str, float | bool | None]:
    """Collapse an hourly payload to the session-median weather a race uses.

    Returns a dict with: AirTemp, Humidity, WindSpeed, Precipitation,
    SolarRadiation, TrackTempEst (derived), Rainfall (bool: any precip in window).
    Pass ``session_hour`` (local 0–23) for night/dusk races; defaults to afternoon.
    """
    air = _session_median(hourly, "temperature_2m", session_hour)
    rad = _session_median(hourly, "shortwave_radiation", session_hour)
    precip = _session_median(hourly, "precipitation", session_hour)
    out: dict[str, float | bool | None] = {
        "AirTemp": air,
        "Humidity": _session_median(hourly, "relative_humidity_2m", session_hour),
        "WindSpeed": _session_median(hourly, "wind_speed_10m", session_hour),
        "Precipitation": precip,
        "SolarRadiation": rad,
        "Rainfall": bool(precip and precip > 0.1),
    }
    out["TrackTempEst"] = (
        estimate_track_temp(air, rad) if air is not None and rad is not None else None
    )
    return out


def session_weather(
    circuit: str, date: str, mode: str = "auto", session_hour: int | None = None
) -> dict[str, float | bool | None]:
    """Session-median weather for a race, by circuit name + local date.

    Args:
        circuit: CircuitKey (FastF1 EventName) — resolved to coords via circuits.py.
        date: ISO "YYYY-MM-DD" race day (local).
        mode: "archive" (ERA5), "forecast", or "auto" (forecast first, archive
              fallback) — "auto" suits an upcoming race that may also be recent.
        session_hour: local hour (0–23) of the race start, for night/dusk races.

    Returns:
        Summary dict from summarize_session, plus "source" and "circuit".
        All channels None (with source="unavailable") if the circuit is unknown.
    """
    from src.pipeline.circuits import coords_for

    coord = coords_for(circuit)
    if coord is None:
        logger.warning("No coordinates for circuit %r — weather unavailable", circuit)
        return {"source": "unavailable", "circuit": circuit}

    lat, lon = coord
    order = {"archive": ["archive"], "forecast": ["forecast"]}.get(
        mode, ["forecast", "archive"]
    )
    last_err: Exception | None = None
    for src in order:
        try:
            hourly = fetch_forecast(lat, lon, date) if src == "forecast" else fetch_archive(lat, lon, date)
            summary = summarize_session(hourly, session_hour)
            if summary.get("AirTemp") is not None:
                summary["source"] = src
                summary["circuit"] = circuit
                return summary
        except Exception as e:                       # pragma: no cover - network
            last_err = e
            logger.debug("%s fetch failed for %s: %s", src, circuit, e)
    logger.warning("Weather fetch failed for %s @ %s: %s", circuit, date, last_err)
    return {"source": "unavailable", "circuit": circuit}


__all__ = [
    "CIRCUIT_COORDS_HELP", "HOURLY_CHANNELS",
    "fetch_archive", "fetch_forecast",
    "estimate_track_temp", "summarize_session", "session_weather",
]

# Tiny docstring constant so callers can surface the channel list in the UI.
CIRCUIT_COORDS_HELP = ", ".join(HOURLY_CHANNELS)
