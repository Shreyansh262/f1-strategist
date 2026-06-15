"""OpenF1 live-timing client — Phase 10 (live-strategy feed).

A dependency-free client over the public OpenF1 REST API (no key required):
    https://api.openf1.org/v1

Endpoints consumed:
    /sessions     — session metadata (circuit, year, date, gmt_offset, …)
    /laps         — per-driver per-lap times
    /intervals    — gap-to-leader & interval
    /stints       — tyre compound + stint bounds
    /position     — race order
    /race_control — safety car, VSC, flags

All network calls are cached on disk under ``data/openf1_cache/`` using the
same ``_get_json`` helper pattern as ``src/pipeline/weather.py``.  Tests never
hit the network — they monkeypatch ``_get_json`` directly.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CACHE_DIR: Final[Path] = PROJECT_ROOT / "data" / "openf1_cache"

BASE_URL: Final[str] = "https://api.openf1.org/v1"

_REQUEST_TIMEOUT_S: Final[int] = 20


# ---------------------------------------------------------------------------
# Low-level fetch with on-disk cache (same idiom as weather.py)
# ---------------------------------------------------------------------------

def _cache_key(url: str, params: dict) -> Path:
    """Deterministic file path for a (url, params) pair."""
    safe = urllib.parse.urlencode(sorted(params.items()))
    digest = abs(hash(url + safe))
    # Use the last path segment as a human-readable prefix, e.g. "laps".
    stub = url.rstrip("/").rsplit("/", 1)[-1]
    return CACHE_DIR / f"{stub}_{digest}.json"


def _get_json(url: str, params: dict, use_cache: bool = True) -> list:
    """GET an OpenF1 endpoint, returning a JSON list.

    Caches successful (non-empty) responses on disk.  Returns an empty list
    and logs a warning on any network or JSON error — callers must tolerate [].

    Tests monkeypatch this function so no network call is made.
    """
    cache_path = _cache_key(url, params)
    if use_cache and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    query = urllib.parse.urlencode(params, doseq=True)
    full = f"{url}?{query}" if params else url
    logger.debug("OpenF1 GET %s", full)
    try:
        with urllib.request.urlopen(full, timeout=_REQUEST_TIMEOUT_S) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("OpenF1 fetch failed %s: %s", full, exc)
        return []

    if not isinstance(payload, list):
        logger.warning("OpenF1 unexpected payload type (%s) for %s", type(payload).__name__, full)
        return []

    if use_cache and payload:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    return payload


def _fetch(endpoint: str, params: dict | None = None, use_cache: bool = True) -> list:
    """Build the full URL and delegate to _get_json."""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    return _get_json(url, params or {}, use_cache=use_cache)


# ---------------------------------------------------------------------------
# Public API — session discovery
# ---------------------------------------------------------------------------

def list_sessions(
    year: int | None = None,
    country: str | None = None,
    use_cache: bool = True,
) -> list[dict]:
    """Discover sessions, optionally filtered by year and/or country name.

    Thin wrapper over GET /sessions.  Returns a list of session dicts, each
    containing at least: session_key, circuit_short_name, country_name, year,
    date_start, gmt_offset, session_name.
    """
    params: dict = {}
    if year is not None:
        params["year"] = year
    if country is not None:
        params["country_name"] = country
    return _fetch("sessions", params, use_cache=use_cache)


def resolve_session(
    session_key: str | int = "latest",
    use_cache: bool = True,
) -> dict | None:
    """Return the single /sessions record for *session_key*, or None.

    Passing ``"latest"`` resolves to the most-recent (or ongoing) session as
    the OpenF1 API defines it.
    """
    rows = _fetch("sessions", {"session_key": session_key}, use_cache=use_cache)
    if not rows:
        return None
    return rows[-1]  # API may return multiple if key is ambiguous; take last


# ---------------------------------------------------------------------------
# Driver identity (names / teams)
# ---------------------------------------------------------------------------

def fetch_driver_map(
    session_key: str | int = "latest",
    use_cache: bool = True,
) -> dict[int, dict]:
    """Build a ``driver_number -> {name_acronym, full_name, team_name}`` map.

    Thin wrapper over GET /drivers.  Always returns a dict (empty on failure);
    never raises.  Later records for the same driver_number win (the API may
    emit one row per session lap-source).
    """
    rows = _fetch("drivers", {"session_key": session_key}, use_cache=use_cache)
    out: dict[int, dict] = {}
    for row in rows:
        dn = row.get("driver_number")
        if dn is None:
            continue
        try:
            dn_int = int(dn)
        except (TypeError, ValueError):
            continue
        out[dn_int] = {
            "name_acronym": row.get("name_acronym"),
            "full_name": row.get("full_name"),
            "team_name": row.get("team_name"),
        }
    return out


def _driver_identity(dn: int, driver_map: dict[int, dict]) -> dict:
    """Resolve display name / full name / team for a driver number.

    name preference: name_acronym -> full_name -> "#<driver_number>".
    """
    info = driver_map.get(dn) or {}
    acronym = (info.get("name_acronym") or "").strip()
    full_name = (info.get("full_name") or "").strip()
    team = (info.get("team_name") or "").strip()
    if acronym:
        name = acronym
    elif full_name:
        name = full_name
    else:
        name = f"#{dn}"
    return {"name": name, "full_name": full_name or name, "team": team}


# ---------------------------------------------------------------------------
# Caution parser (race-control replay)
# ---------------------------------------------------------------------------

def _parse_caution(
    race_control: list[dict], current_lap: int
) -> dict | None:
    """Replay race-control messages in time order and return the active caution.

    Returns None if no caution is active, otherwise:
        {"cause": "SC" | "VSC" | "RED" | "YELLOW", "elapsed_laps": int}

    Precedence when several flags coexist: RED > SC > VSC > YELLOW.
    """
    # Sort by date string (ISO-8601 so lexicographic == chronological).
    sorted_msgs = sorted(race_control, key=lambda r: r.get("date", ""))

    # Track each caution type independently; we pick the highest-precedence
    # one still active at the end.
    sc_lap: int | None = None      # lap when SC was deployed
    vsc_lap: int | None = None
    red_lap: int | None = None
    yellow_lap: int | None = None

    for row in sorted_msgs:
        category = (row.get("category") or "").strip()
        flag = (row.get("flag") or "").strip().upper()
        message = (row.get("message") or "").strip().upper()
        lap = row.get("lap_number") or 0

        # -- GREEN clears everything --
        if category == "Flag" and flag == "GREEN":
            sc_lap = None
            vsc_lap = None
            red_lap = None
            yellow_lap = None
            continue

        # -- RED flag --
        if category == "Flag" and flag == "RED":
            red_lap = lap
            # Red supersedes ongoing SC/VSC/yellow
            sc_lap = None
            vsc_lap = None
            yellow_lap = None
            continue

        # -- Safety Car --
        if category == "SafetyCar":
            if "DEPLOYED" in message or "IN THIS LAP" in message:
                sc_lap = lap
                vsc_lap = None   # SC replaces VSC if both somehow active
            elif "ENDING" in message:
                sc_lap = None
            continue

        # -- Virtual Safety Car --
        if "VIRTUAL SAFETY CAR" in message:
            if "DEPLOYED" in message:
                vsc_lap = lap
            elif "ENDING" in message or "VIRTUAL SAFETY CAR ENDING" in message:
                vsc_lap = None
            continue

        # -- Yellow flags (no SC/VSC/RED active) --
        if flag in ("YELLOW", "DOUBLE YELLOW"):
            yellow_lap = lap
            continue

        # If a GREEN flag clears yellow specifically:
        if flag == "GREEN" and not (sc_lap or vsc_lap or red_lap):
            yellow_lap = None

    # Pick highest-precedence active caution.
    active_lap: int | None = None
    cause: str | None = None

    if red_lap is not None:
        cause, active_lap = "RED", red_lap
    elif sc_lap is not None:
        cause, active_lap = "SC", sc_lap
    elif vsc_lap is not None:
        cause, active_lap = "VSC", vsc_lap
    elif yellow_lap is not None:
        cause, active_lap = "YELLOW", yellow_lap

    if cause is None or active_lap is None:
        return None

    elapsed = max(1, current_lap - active_lap + 1)
    return {"cause": cause, "elapsed_laps": elapsed}


# ---------------------------------------------------------------------------
# Driver snapshot builder
# ---------------------------------------------------------------------------

def _build_drivers(
    laps: list[dict],
    intervals: list[dict],
    stints: list[dict],
    positions: list[dict],
    current_lap: int,
    driver_map: dict[int, dict] | None = None,
) -> list[dict]:
    """Combine per-endpoint lists into a per-driver snapshot list.

    Returns drivers sorted by position (None-position drivers go last).
    *driver_map* (driver_number -> identity) adds readable name/team fields.
    """
    driver_map = driver_map or {}
    # ---- latest position per driver ----------------------------------------
    pos_map: dict[int, int | None] = {}
    for row in sorted(positions, key=lambda r: r.get("date", "")):
        dn = row.get("driver_number")
        if dn is not None:
            pos_map[int(dn)] = row.get("position")

    # ---- latest gap_to_leader per driver ------------------------------------
    gap_map: dict[int, float | None] = {}
    for row in sorted(intervals, key=lambda r: r.get("date", "")):
        dn = row.get("driver_number")
        if dn is None:
            continue
        raw_gap = row.get("gap_to_leader")
        if raw_gap is None:
            gap_map[int(dn)] = None
        elif isinstance(raw_gap, (int, float)):
            gap_map[int(dn)] = float(raw_gap)
        else:
            # String like "+1 LAP" — not a numeric gap
            gap_map[int(dn)] = None

    # ---- per-driver max lap & last lap_duration ----------------------------
    driver_laps: dict[int, list[dict]] = {}
    for row in laps:
        dn = row.get("driver_number")
        if dn is None:
            continue
        driver_laps.setdefault(int(dn), []).append(row)

    driver_max_lap: dict[int, int] = {}
    driver_last_dur: dict[int, float | None] = {}
    for dn, dl in driver_laps.items():
        sorted_dl = sorted(dl, key=lambda r: r.get("lap_number") or 0)
        driver_max_lap[dn] = max((r.get("lap_number") or 0) for r in sorted_dl)
        # Last lap_duration (may be None for an in-progress lap)
        driver_last_dur[dn] = None
        for r in reversed(sorted_dl):
            dur = r.get("lap_duration")
            if dur is not None:
                driver_last_dur[dn] = float(dur)
                break

    # ---- stint selection per driver ----------------------------------------
    # Current stint = latest stint whose lap_start <= driver_lap <= lap_end;
    # if none match (e.g. historical blanks), fall back to the last stint.
    def _pick_stint(dn: int, d_lap: int) -> dict | None:
        driver_stints = [s for s in stints if int(s.get("driver_number", -1)) == dn]
        if not driver_stints:
            return None
        # Sort by lap_start so "last" is well-defined.
        driver_stints.sort(key=lambda s: s.get("lap_start") or 0)
        for s in driver_stints:
            ls = s.get("lap_start") or 0
            le = s.get("lap_end") or 9999
            if ls <= d_lap <= le:
                return s
        # Fallback: most recent by lap_start.
        return driver_stints[-1]

    # ---- collect all known drivers -----------------------------------------
    all_drivers = (
        set(driver_max_lap.keys())
        | set(pos_map.keys())
        | set(gap_map.keys())
    )

    results: list[dict] = []
    for dn in all_drivers:
        d_lap = driver_max_lap.get(dn, 0)
        stint = _pick_stint(dn, d_lap)

        compound: str | None = None
        start_age = 1
        if stint is not None:
            raw_cpd = stint.get("compound")
            compound = raw_cpd.upper() if isinstance(raw_cpd, str) else None
            tyre_age_at_start = int(stint.get("tyre_age_at_start") or 0)
            lap_start = int(stint.get("lap_start") or 0)
            start_age = max(1, tyre_age_at_start + (d_lap - lap_start))

        identity = _driver_identity(dn, driver_map)
        results.append({
            "driver_number": dn,
            "name": identity["name"],
            "full_name": identity["full_name"],
            "team": identity["team"],
            "position": pos_map.get(dn),
            "current_lap": d_lap,
            "compound": compound,
            "start_age": start_age,
            "gap_to_leader_s": gap_map.get(dn),
            "last_lap_s": driver_last_dur.get(dn),
        })

    # Sort by position (None last).
    results.sort(key=lambda d: (d["position"] is None, d["position"] or 0))
    return results


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------

def live_snapshot(
    session_key: str | int = "latest",
    use_cache: bool = True,
) -> dict:
    """Fetch and assemble a point-in-time race snapshot from OpenF1.

    Calls /sessions, /laps, /intervals, /stints, /position, /race_control for
    the given *session_key*, then merges them into the canonical shape:

    {
        "session": {
            "session_key": int,
            "circuit": str,       # circuit_short_name
            "year": int,
            "session_name": str,
            "current_lap": int,   # max lap_number seen across all drivers (0 if none)
            "available": bool,    # False when the API returned nothing usable
        },
        "drivers": [...],         # see _build_drivers for per-driver schema
        "caution": None | {"cause": "SC"|"VSC"|"YELLOW"|"RED", "elapsed_laps": int},
    }

    Never raises — any network / parsing failure results in available=False and
    empty drivers.  Completed (historical) sessions return the final-lap state.
    """
    params = {"session_key": session_key}

    session_row = resolve_session(session_key, use_cache=use_cache)

    # Even if session metadata is missing, still try the data endpoints using
    # session_key directly (for numeric keys where /sessions is optional).
    sk_for_data: int | str = session_key
    session_meta: dict = {}
    if session_row is not None:
        sk_for_data = session_row.get("session_key", session_key)
        session_meta = session_row

    data_params = {"session_key": sk_for_data}

    # Fetch all endpoints; each returns [] on failure (fail-soft).
    laps = _fetch("laps", data_params, use_cache=use_cache)
    intervals = _fetch("intervals", data_params, use_cache=use_cache)
    stints = _fetch("stints", data_params, use_cache=use_cache)
    positions = _fetch("position", data_params, use_cache=use_cache)
    race_control = _fetch("race_control", data_params, use_cache=use_cache)
    driver_map = fetch_driver_map(sk_for_data, use_cache=use_cache)

    # current_lap = max lap_number across all drivers' laps
    current_lap = 0
    for lap in laps:
        ln = lap.get("lap_number") or 0
        if ln > current_lap:
            current_lap = ln

    available = bool(session_row is not None and (laps or stints or positions))

    drivers = _build_drivers(laps, intervals, stints, positions, current_lap, driver_map)
    caution = _parse_caution(race_control, current_lap)

    return {
        "session": {
            "session_key": int(session_meta.get("session_key", 0) or 0),
            "circuit": session_meta.get("circuit_short_name", ""),
            "year": int(session_meta.get("year", 0) or 0),
            "session_name": session_meta.get("session_name", ""),
            "current_lap": current_lap,
            "available": available,
        },
        "drivers": drivers,
        "caution": caution,
    }


# ---------------------------------------------------------------------------
# Live-race detection
# ---------------------------------------------------------------------------

_RACE_SESSION_NAMES: Final[frozenset[str]] = frozenset({"Race", "Sprint"})
_DEFAULT_RACE_DURATION = timedelta(hours=3)


def _parse_iso8601(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp into a tz-aware UTC datetime, or None.

    OpenF1 sends offsets like ``2024-03-02T15:00:00+03:00``.  A naive timestamp
    (no offset) is assumed to already be UTC.  Never raises.
    """
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    # Python <3.11 fromisoformat rejects a trailing "Z"; normalise it.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _readable_session(row: dict) -> dict:
    """A readable, possibly-partial session summary for the public contract."""
    return {
        "session_name": row.get("session_name", "") or "",
        "circuit": row.get("circuit_short_name", "") or "",
        "year": int(row.get("year", 0) or 0),
        "date_start": row.get("date_start", "") or "",
        "date_end": row.get("date_end", "") or "",
    }


def live_race_status(use_cache: bool = False) -> dict:
    """Decide whether a real RACE/SPRINT is live right now.

    Resolves the latest session and checks whether it is a Race/Sprint whose
    [date_start, date_end] window contains the current UTC time (date_end is
    assumed to be date_start + 3h when missing).

    Returns::

        {"state": "live" | "no_live",
         "session": {session_name, circuit, year, date_start, date_end},
         "session_key": int,
         "message": str}

    Never raises.  On any offline / parse error -> a safe ``no_live`` result
    with an empty session and session_key 0.
    """
    try:
        row = resolve_session("latest", use_cache=use_cache)
    except Exception as exc:  # defensive: resolve_session shouldn't raise
        logger.warning("live_race_status: resolve failed: %s", exc)
        row = None

    if not row:
        return {
            "state": "no_live",
            "session": {},
            "session_key": 0,
            "message": "No live race right now. Live timing is unavailable.",
        }

    session = _readable_session(row)
    session_key = int(row.get("session_key", 0) or 0)
    session_name = session["session_name"]

    start = _parse_iso8601(row.get("date_start"))
    end = _parse_iso8601(row.get("date_end"))
    if start is not None and end is None:
        end = start + _DEFAULT_RACE_DURATION

    now = datetime.now(timezone.utc)
    is_race = session_name in _RACE_SESSION_NAMES
    in_window = (
        start is not None and end is not None and start <= now <= end
    )

    label = f"{session.get('circuit') or 'Unknown'} {session.get('year') or ''}".strip()

    if is_race and in_window:
        return {
            "state": "live",
            "session": session,
            "session_key": session_key,
            "message": f"Live now: {label} {session_name}.".replace("  ", " "),
        }

    return {
        "state": "no_live",
        "session": session,
        "session_key": session_key,
        "message": f"No live race right now. Most recent: {label} (finished).",
    }


__all__ = [
    "list_sessions",
    "resolve_session",
    "fetch_driver_map",
    "live_snapshot",
    "live_race_status",
]
