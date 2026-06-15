"""Live / mid-race state assembler — Phase 10.2.

Turns a *point-in-time* race state into a `RaceSpec` for the remaining laps, so
the Monte-Carlo engine (Phase 4) can project the rest of the race from where it
actually is — including any flag that is currently flying (Phase 10.4).

Two entry points, one shared builder:
    from_openf1(session_key)        — true-live, via src.pipeline.openf1
    from_replay(season, rnd, lap)   — replay a historical race up to `lap`

Both produce a `RaceSpec` whose laps are the *remaining* laps, with each
`DriverSpec` carrying its current compound, current tyre age (`start_age`), and a
`grid_gap_s` set to the live time-gap to the leader. `RaceSpec.caution` is set if
a Safety Car / VSC / flag is active right now.

Pure module — no Streamlit, no network at import. `from_openf1` is a thin shell
over `from_snapshot`, which is fully unit-testable from a plain dict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.engine import COMPOUNDS, DriverSpec, RaceSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# OpenF1 reports a short circuit name (e.g. "Sakhir"); the tyre curves and pit
# tables are keyed by the FastF1 EventName (e.g. "Bahrain Grand Prix"). This
# also carries the FIA scheduled race distance so a live feed (which never sends
# "total laps") knows how many laps remain. Unknown circuits fall back to the
# short name (the curves pool to era level) and a 57-lap default.
_CIRCUIT_META: dict[str, tuple[str, int]] = {
    "sakhir":            ("Bahrain Grand Prix", 57),
    "jeddah":            ("Saudi Arabian Grand Prix", 50),
    "melbourne":         ("Australian Grand Prix", 58),
    "suzuka":            ("Japanese Grand Prix", 53),
    "shanghai":          ("Chinese Grand Prix", 56),
    "miami":             ("Miami Grand Prix", 57),
    "imola":             ("Emilia Romagna Grand Prix", 63),
    "monaco":            ("Monaco Grand Prix", 78),
    "catalunya":         ("Spanish Grand Prix", 66),
    "barcelona":         ("Spanish Grand Prix", 66),
    "montreal":          ("Canadian Grand Prix", 70),
    "spielberg":         ("Austrian Grand Prix", 71),
    "silverstone":       ("British Grand Prix", 52),
    "hungaroring":       ("Hungarian Grand Prix", 70),
    "spa-francorchamps": ("Belgian Grand Prix", 44),
    "spa":               ("Belgian Grand Prix", 44),
    "zandvoort":         ("Dutch Grand Prix", 72),
    "monza":             ("Italian Grand Prix", 53),
    "baku":              ("Azerbaijan Grand Prix", 51),
    "singapore":         ("Singapore Grand Prix", 62),
    "marina bay":        ("Singapore Grand Prix", 62),
    "austin":            ("United States Grand Prix", 56),
    "mexico city":       ("Mexico City Grand Prix", 71),
    "mexico":            ("Mexico City Grand Prix", 71),
    "sao paulo":         ("São Paulo Grand Prix", 71),
    "interlagos":        ("São Paulo Grand Prix", 71),
    "las vegas":         ("Las Vegas Grand Prix", 50),
    "lusail":            ("Qatar Grand Prix", 57),
    "losail":            ("Qatar Grand Prix", 57),
    "yas marina":        ("Abu Dhabi Grand Prix", 58),
}
_DEFAULT_RACE_LAPS = 57


def _circuit_meta(short_name: str) -> tuple[str, int]:
    """(EventName, scheduled race laps) for an OpenF1 circuit_short_name."""
    key = (short_name or "").strip().lower()
    if key in _CIRCUIT_META:
        return _CIRCUIT_META[key]
    # tolerate minor variants ("Sao Paulo" vs "São Paulo", trailing words)
    for k, v in _CIRCUIT_META.items():
        if k in key or key in k:
            return v
    return (short_name or "Unknown", _DEFAULT_RACE_LAPS)


def _era_for_year(year: int) -> int:
    return 0 if (year or 0) <= 2025 else 1


# ---------------------------------------------------------------------------
# Live (OpenF1) path
# ---------------------------------------------------------------------------

def from_snapshot(snapshot: dict, total_laps: int | None = None) -> RaceSpec | None:
    """Build a remaining-race `RaceSpec` from an openf1.live_snapshot() dict.

    Pure (no network). Returns None if the snapshot has no usable drivers.
    base_pace is approximated from each driver's most recent lap time (falling
    back to the field median) — a live feed does not expose clean-air pace, so
    relative car pace is inferred from current lap times.
    """
    sess = snapshot.get("session", {})
    drivers_in = snapshot.get("drivers", [])
    if not drivers_in:
        return None

    circuit_short = sess.get("circuit", "")
    event_name, race_laps = _circuit_meta(circuit_short)
    race_laps = int(total_laps) if total_laps else race_laps
    current_lap = int(sess.get("current_lap", 0) or 0)
    remaining = max(race_laps - current_lap, 1)
    era = _era_for_year(int(sess.get("year", 0) or 0))

    # Field median lap time as the base-pace fallback.
    last_laps = [d["last_lap_s"] for d in drivers_in if d.get("last_lap_s")]
    field_pace = float(np.median(last_laps)) if last_laps else 90.0

    drivers: list[DriverSpec] = []
    for i, d in enumerate(drivers_in):
        base = float(d["last_lap_s"]) if d.get("last_lap_s") else field_pace
        compound = d.get("compound") or "MEDIUM"
        if compound not in COMPOUNDS:
            compound = "MEDIUM"
        # Prefer a readable name/acronym; fall back to "#<driver_number>".
        name = (d.get("name") or "").strip()
        if not name:
            name = f"#{d.get('driver_number')}"
        drivers.append(DriverSpec(
            driver=name,
            base_pace_s=base,
            start_compound=compound,
            strategy=[],
            grid_pos=int(d["position"]) if d.get("position") else i + 1,
            grid_gap_s=float(d["gap_to_leader_s"]) if d.get("gap_to_leader_s") else 0.0,
            start_age=int(d.get("start_age", 2) or 2),
        ))

    caution = _caution_tuple(snapshot.get("caution"))
    return RaceSpec(
        circuit=event_name, era=era, n_laps=remaining,
        drivers=drivers, caution=caution,
    )


def from_openf1(session_key: str | int = "latest", total_laps: int | None = None,
                use_cache: bool = True) -> RaceSpec | None:
    """Fetch the live OpenF1 snapshot and assemble a remaining-race RaceSpec.

    Works on any session_key: a live GP gives a genuine mid-race state, a
    completed session gives the final-lap state (remaining laps -> 1).
    """
    from src.pipeline.openf1 import live_snapshot
    snap = live_snapshot(session_key, use_cache=use_cache)
    if not snap["session"].get("available") and not snap.get("drivers"):
        return None
    return from_snapshot(snap, total_laps=total_laps)


def _caution_tuple(caution: dict | None) -> tuple[str, int] | None:
    if not caution:
        return None
    return (str(caution.get("cause", "SC")), int(caution.get("elapsed_laps", 1) or 1))


# ---------------------------------------------------------------------------
# Replay path (historical parquet)
# ---------------------------------------------------------------------------

def _green_base_pace(df: pd.DataFrame) -> dict[str, float]:
    """Per-driver green-flag median lap time (clean pit-free racing laps)."""
    green = df[
        (df["TrackStatus"].astype(str) == "1")
        & df.get("PitInTime", pd.Series(index=df.index, dtype=float)).isna()
        & df.get("PitOutTime", pd.Series(index=df.index, dtype=float)).isna()
        & df["LapTimeSeconds"].between(60, 150)
    ]
    med = green.groupby("Driver")["LapTimeSeconds"].median()
    return med.to_dict()


def _caution_at_lap(df: pd.DataFrame, current_lap: int) -> tuple[str, int] | None:
    """Active caution at `current_lap` from the per-lap TrackStatus string.

    elapsed_laps = how many consecutive laps (including this one) the same
    caution class has been flying.
    """
    by_lap = (df.groupby("LapNumber")["TrackStatus"]
              .agg(lambda s: "".join(str(x) for x in s)))   # codes present that lap

    def _cause(codes: str) -> str | None:
        if "5" in codes:
            return "RED"
        if "4" in codes:
            return "SC"
        if "6" in codes:
            return "VSC"
        if "2" in codes:
            return "YELLOW"
        return None

    cause = _cause(by_lap.get(current_lap, "1"))
    if cause is None:
        return None
    elapsed = 0
    lap = current_lap
    while lap >= 1 and _cause(by_lap.get(lap, "1")) == cause:
        elapsed += 1
        lap -= 1
    return (cause, elapsed)


def spec_from_race_df(df: pd.DataFrame, current_lap: int,
                      season: int | None = None) -> RaceSpec | None:
    """Pure builder: a remaining-race RaceSpec from a race lap DataFrame.

    Only drivers still circulating at `current_lap` are included. Each driver's
    cumulative race time to `current_lap` sets the gap to the leader; their tyre
    age and compound at that lap set start_age / start_compound.
    """
    if df is None or df.empty or current_lap < 1:
        return None
    df = df.copy()
    df["TrackStatus"] = df["TrackStatus"].astype(str)
    circuit = str(df["CircuitKey"].iloc[0])
    if season is None:
        season = int(df["Season"].iloc[0])
    era = 0 if season <= 2025 else 1
    total_laps = int(df["LapNumber"].max())
    remaining = max(total_laps - current_lap, 1)

    base_pace = _green_base_pace(df)
    field_pace = float(np.median(list(base_pace.values()))) if base_pace else 90.0

    up_to = df[df["LapNumber"] <= current_lap]
    rows = []
    for driver, g in up_to.groupby("Driver"):
        at = g[g["LapNumber"] == current_lap]
        if at.empty:                                  # retired / lapped out before now
            continue
        cum = float(g["LapTimeSeconds"].dropna().sum())
        cur = at.iloc[0]
        compound = str(cur.get("Compound", "MEDIUM") or "MEDIUM").upper()
        age = int(cur.get("TyreLife", 2) or 2)
        rows.append({
            "driver": driver, "cum": cum,
            "base": float(base_pace.get(driver, field_pace)),
            "compound": compound, "age": max(age, 1),
        })
    if not rows:
        return None

    leader_cum = min(r["cum"] for r in rows)
    rows.sort(key=lambda r: r["cum"])
    drivers = [
        DriverSpec(
            driver=str(r["driver"]), base_pace_s=r["base"],
            start_compound=r["compound"] if r["compound"] in COMPOUNDS else "MEDIUM",
            strategy=[], grid_pos=i + 1,
            grid_gap_s=r["cum"] - leader_cum, start_age=r["age"],
        )
        for i, r in enumerate(rows)
    ]
    caution = _caution_at_lap(df, current_lap)
    return RaceSpec(circuit=circuit, era=era, n_laps=remaining,
                    drivers=drivers, caution=caution)


def from_replay(season: int, rnd: int, current_lap: int,
                data_raw: Path | None = None) -> RaceSpec | None:
    """Load a historical race parquet and build its state at `current_lap`."""
    base = data_raw or DATA_RAW
    p = base / f"laps_{season}_r{rnd:02d}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = df.drop_duplicates(subset=["Season", "RoundNumber", "Driver", "LapNumber"])
    return spec_from_race_df(df, current_lap, season=season)


__all__ = ["from_snapshot", "from_openf1", "from_replay", "spec_from_race_df"]
