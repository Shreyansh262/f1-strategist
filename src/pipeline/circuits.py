"""Static circuit geography — Phase 9 (weather robustness, Section 10.8).

A small, hand-curated table of (latitude, longitude) per circuit, keyed by
``CircuitKey`` (== FastF1 ``EventName``, the column that is law in Section 6).
Used by ``src/pipeline/weather.py`` to query Open-Meteo for a race's weather —
both historical backfill (ERA5 archive) and upcoming-race forecasts.

Coordinates are the circuit centre (good to ~1 km — far finer than the ~9 km
ERA5 grid and the ~1 km forecast grid, so sub-km precision is irrelevant).
Keys must match the EventName strings FastF1 returns; ``coords_for`` also
tolerates the bare circuit name without the " Grand Prix" suffix.
"""
from __future__ import annotations

from typing import Final

# CircuitKey (FastF1 EventName) -> (latitude, longitude) in decimal degrees.
CIRCUIT_COORDS: Final[dict[str, tuple[float, float]]] = {
    "Bahrain Grand Prix":         (26.0325,  50.5106),   # Sakhir
    "Saudi Arabian Grand Prix":   (21.6319,  39.1044),   # Jeddah Corniche
    "Australian Grand Prix":      (-37.8497, 144.9680),  # Albert Park, Melbourne
    "Japanese Grand Prix":        (34.8431, 136.5410),   # Suzuka
    "Chinese Grand Prix":         (31.3389, 121.2200),   # Shanghai
    "Miami Grand Prix":           (25.9581, -80.2389),   # Miami Gardens
    "Emilia Romagna Grand Prix":  (44.3439,  11.7167),   # Imola
    "Monaco Grand Prix":          (43.7347,   7.4206),   # Monte Carlo
    "Canadian Grand Prix":        (45.5000, -73.5228),   # Circuit Gilles Villeneuve
    "Spanish Grand Prix":         (41.5700,   2.2611),   # Circuit de Barcelona-Catalunya
    "Austrian Grand Prix":        (47.2197,  14.7647),   # Red Bull Ring, Spielberg
    "British Grand Prix":         (52.0786,  -1.0169),   # Silverstone
    "Hungarian Grand Prix":       (47.5789,  19.2486),   # Hungaroring, Budapest
    "Belgian Grand Prix":         (50.4372,   5.9714),   # Spa-Francorchamps
    "Dutch Grand Prix":           (52.3888,   4.5409),   # Zandvoort
    "Italian Grand Prix":         (45.6156,   9.2811),   # Monza
    "Azerbaijan Grand Prix":      (40.3725,  49.8533),   # Baku City
    "Singapore Grand Prix":       (1.2914,  103.8640),   # Marina Bay
    "United States Grand Prix":   (30.1328, -97.6411),   # COTA, Austin
    "Mexico City Grand Prix":     (19.4042, -99.0907),   # Autódromo Hermanos Rodríguez
    "São Paulo Grand Prix":       (-23.7036, -46.6997),  # Interlagos
    "Las Vegas Grand Prix":       (36.1147, -115.1730),  # Las Vegas Strip
    "Qatar Grand Prix":           (25.4900,  51.4542),   # Lusail
    "Abu Dhabi Grand Prix":       (24.4672,  54.6031),   # Yas Marina
    "French Grand Prix":          (43.2506,   5.7917),   # Paul Ricard, Le Castellet
}


def coords_for(circuit: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a circuit, or None if unknown.

    Accepts either the full EventName ("Bahrain Grand Prix") or the bare name
    ("Bahrain"); matching is case-insensitive and suffix-tolerant.
    """
    if circuit in CIRCUIT_COORDS:
        return CIRCUIT_COORDS[circuit]
    key = circuit.strip().lower().removesuffix(" grand prix")
    for name, coord in CIRCUIT_COORDS.items():
        if name.lower().removesuffix(" grand prix") == key:
            return coord
    return None
