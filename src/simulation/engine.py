"""Monte Carlo race-simulation engine — Phase 4b (THE SHOWCASE).

Composes the project's models *with their uncertainties* and rolls out full
races N times, turning candidate pit strategies into probability distributions
over finishing positions — what real strategy teams output, instead of point
estimates.

What is sampled per rollout (and where it comes from):
    per-lap pace noise      σ from the RECALIBRATED TFT band (models/tft_calibration.json
                            widens reports' raw quantile width; σ = width / 2·z₀.₈)
    tyre degradation        one z-draw per (rollout, driver, stint) scaling the
                            Phase 3 curve's 95% CI around its mid line
    pit-stop execution      N(pit_loss_s, 0.8s) around the per-circuit table
    overtaking              Bernoulli per (lap, attacker) with a circuit-class
                            probability — a deliberately simple traffic model

What is deterministic: strategies (pit laps + compounds), fuel burn, base pace.

Design choice (document in interviews): the engine samples from the lap-time
model's *distributional summaries* per circuit/era rather than calling the TFT
network inside the loop — thousands of rollouts × 60 laps × 20 drivers would
make per-lap net inference the bottleneck for no accuracy gain, since within a
rollout the band, not the median path, carries the strategy-relevant signal.

CPU-first; pass device="cuda" for big sweeps (tensors are [n_rollouts, n_drivers]).

Run a demo:
    python -m src.simulation.engine
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
import torch

from src.models.pit_strategy.pit_loss import load_pit_loss, pit_loss_for
from src.models.tyre.fit import predict_degradation
from src.pipeline.features import (
    FUEL_LOAD_START_KG, FUEL_BURN_PER_LAP, FUEL_EFFECT_PER_KG,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports" / "simulation"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Indices 0/1/2 are S/M/H and MUST stay fixed — dry rollouts are byte-for-byte
# unchanged. INTERMEDIATE/WET are appended so wet stints use their own curve
# instead of being silently treated as MEDIUM.
COMPOUNDS: Final[list[str]] = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

# Per-lap overtake probability when within striking distance, by circuit class.
OVERTAKE_PROB: Final[dict[str, float]] = {"street": 0.08, "medium": 0.20, "power": 0.30}
STREET_CIRCUITS: Final[set[str]] = {
    "Monaco Grand Prix", "Singapore Grand Prix", "Azerbaijan Grand Prix",
    "Las Vegas Grand Prix", "Miami Grand Prix",
}
POWER_CIRCUITS: Final[set[str]] = {
    "Italian Grand Prix", "Belgian Grand Prix", "Austrian Grand Prix",
    "Canadian Grand Prix", "Bahrain Grand Prix", "Saudi Arabian Grand Prix",
}
STRIKE_GAP_S:  Final[float] = 1.5   # close enough to attempt a pass
HOLD_GAP_S:    Final[float] = 0.6   # stuck-in-dirty-air gap when the pass fails
PIT_SIGMA_S:   Final[float] = 0.8   # pit-stop execution noise
Z80:           Final[float] = 1.2816  # z for the 0.1/0.9 quantiles

# --- caution / flag following (Phase 10.3-10.4) ---------------------------
# A caution slows the whole field to a single dictated pace (overtaking frozen
# because every car gains the same time), makes a pit cheaper (you lose far less
# relative to a crawling field), and — for a full Safety Car — bunches the field
# nose-to-tail at the restart. VSC is milder and does not bunch.
CAUTION_PACE_MULT: Final[dict[str, float]] = {
    "SC": 1.40, "VSC": 1.30, "YELLOW": 1.12, "RED": 1.40,
}
CAUTION_PIT_FACTOR: Final[dict[str, float]] = {   # fraction of green pit loss
    "SC": 0.55, "VSC": 0.70, "YELLOW": 1.0, "RED": 0.50,
}
BUNCH_CAUSES: Final[set[str]] = {"SC", "RED"}   # field concertinas at the restart
SC_BUNCH_GAP_S: Final[float] = 1.2              # nose-to-tail spacing on restart
# Offline fallback so the engine never hard-depends on the SC duration artifact.
_CAUTION_FALLBACK: Final[dict[str, list[int]]] = {
    "SC": [3, 4, 4, 5, 6], "VSC": [2, 2, 3, 3], "YELLOW": [1, 2], "RED": [4, 5, 6],
}


def circuit_class(circuit: str) -> str:
    if circuit in STREET_CIRCUITS:
        return "street"
    if circuit in POWER_CIRCUITS:
        return "power"
    return "medium"


def _comp_idx(compound: str) -> int:
    """Compound index into COMPOUNDS, defaulting to MEDIUM for truly unknown strings.

    All five known compounds (SOFT/MEDIUM/HARD/INTERMEDIATE/WET) map to their own
    index, so a wet stint uses the wet degradation curve. The MEDIUM fallback now
    fires only for an unrecognised compound string (e.g. a malformed live feed),
    never for INTERMEDIATE/WET.
    """
    try:
        return COMPOUNDS.index(compound)
    except ValueError:
        return COMPOUNDS.index("MEDIUM")


def _sample_caution_lengths(cause: str, elapsed: int, n: int, seed: int | None) -> np.ndarray:
    """Per-rollout REMAINING laps of an in-progress caution, conditioned on `elapsed`.

    Uses the historical SC/VSC duration model (src.simulation.sc_model) when its
    artifact is present; otherwise falls back to a small built-in distribution so
    the engine runs standalone. Each value is floored at 1.
    """
    rng = np.random.default_rng(seed)
    try:                                           # pragma: no cover - artifact path
        from src.simulation.sc_model import load_distributions, sample_remaining
        dist = load_distributions()
        return np.array(
            [int(sample_remaining(cause, elapsed, dist=dist, rng=rng)) for _ in range(n)],
            dtype=int,
        )
    except Exception:
        lengths = np.array(_CAUTION_FALLBACK.get(cause, _CAUTION_FALLBACK["SC"]))
        eligible = lengths[lengths >= max(elapsed, 1)]
        if eligible.size == 0:
            return np.ones(n, dtype=int)
        totals = rng.choice(eligible, size=n)
        return np.maximum(totals - elapsed, 1).astype(int)


# ---------------------------------------------------------------------------
# Specs and result
# ---------------------------------------------------------------------------

@dataclass
class DriverSpec:
    driver: str
    base_pace_s: float                       # green-flag pace at zero tyre delta, mean fuel
    start_compound: str = "MEDIUM"
    strategy: list[tuple[int, str]] = field(default_factory=list)  # (pit at END of lap, compound)
    grid_pos: int = 1
    grid_gap_s: float = 0.0                  # cumulative-time handicap at the start
    start_age: int = 2                       # tyre age on the opening lap (>2 = mid-stint resume)


@dataclass
class RaceSpec:
    circuit: str
    era: int
    n_laps: int
    drivers: list[DriverSpec]
    pit_loss_s: float | None = None          # None -> per-circuit table
    sigma_lap_s: float | None = None         # None -> recalibrated TFT band width
    overtake_prob: float | None = None       # None -> circuit-class table
    track_temp: float | None = None          # None -> curve's historical temp (Phase 9.4)
    caution: tuple[str, int] | None = None   # (cause, laps_elapsed) of an in-progress flag (Phase 10.4)


@dataclass
class SimResult:
    drivers: list[str]
    finish_pos: np.ndarray            # [n_rollouts, n_drivers] int
    race_time_s: np.ndarray           # [n_rollouts, n_drivers]
    win_prob: dict[str, float]
    podium_prob: dict[str, float]
    mean_finish: dict[str, float]

    def table(self) -> pd.DataFrame:
        rows = [
            {
                "driver": d,
                "win_prob": self.win_prob[d],
                "podium_prob": self.podium_prob[d],
                "mean_finish": self.mean_finish[d],
                "p5_finish": float(np.percentile(self.finish_pos[:, i], 5)),
                "p95_finish": float(np.percentile(self.finish_pos[:, i], 95)),
            }
            for i, d in enumerate(self.drivers)
        ]
        return pd.DataFrame(rows).sort_values("mean_finish").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model-derived inputs
# ---------------------------------------------------------------------------

def default_sigma(circuit: str, era: int) -> float:
    """Per-lap pace σ from the recalibrated TFT quantile band.

    Raw 0.1-0.9 band width per split is in reports/lap_time/tft_recalibration.csv;
    the conformal shift (models/tft_calibration.json) widens it by 2·shift.
    σ = recalibrated_width / (2·z80). Falls back to 1.0s if artifacts are absent.
    """
    try:
        with open(MODELS_DIR / "tft_calibration.json") as f:
            cal = json.load(f)
        rep = pd.read_csv(PROJECT_ROOT / "reports" / "lap_time" / "tft_recalibration.csv")
        raw = rep[(rep["era"] == era) & (rep["bands"] == "raw")].iloc[0]
        width = float(raw["mean_band_width_s"]) + 2 * float(cal[f"era{era}_shift_s"])
        return width / (2 * Z80)
    except Exception:                          # pragma: no cover - artifact missing
        log.warning("Calibration artifacts missing — using σ=1.0s")
        return 1.0


def _stint_schedule(spec: RaceSpec, d: DriverSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-lap (compound_idx, tyre_age, stint_idx) arrays for one driver. Deterministic."""
    comp = np.empty(spec.n_laps + 1, dtype=int)
    age  = np.empty(spec.n_laps + 1, dtype=int)
    stint = np.empty(spec.n_laps + 1, dtype=int)
    pits = dict(d.strategy)
    c, a, s = _comp_idx(d.start_compound), max(int(d.start_age), 1), 0
    for lap in range(1, spec.n_laps + 1):
        comp[lap], age[lap], stint[lap] = c, a, s
        if lap in pits:
            c, a, s = _comp_idx(pits[lap]), 2, s + 1
        else:
            a += 1
    return comp[1:], age[1:], stint[1:]


def _fuel_delta(n_laps: int) -> np.ndarray:
    laps = np.arange(1, n_laps + 1)
    load = np.maximum(FUEL_LOAD_START_KG - (laps - 1) * FUEL_BURN_PER_LAP, 0.0)
    return (load - load.mean()) * FUEL_EFFECT_PER_KG


# ---------------------------------------------------------------------------
# The simulation
# ---------------------------------------------------------------------------

def simulate(
    spec: RaceSpec,
    n_rollouts: int = 1000,
    seed: int | None = 42,
    device: str = "cpu",
    curves: pd.DataFrame | None = None,
) -> SimResult:
    """Vectorized Monte Carlo rollout of one race. See module docstring."""
    if curves is None:
        curves = joblib.load(MODELS_DIR / "tyre_curves.joblib")
    pit_loss = spec.pit_loss_s
    if pit_loss is None:
        pit_loss = pit_loss_for(load_pit_loss(), spec.circuit)
    sigma = spec.sigma_lap_s if spec.sigma_lap_s is not None else default_sigma(spec.circuit, spec.era)
    p_pass = spec.overtake_prob if spec.overtake_prob is not None else OVERTAKE_PROB[circuit_class(spec.circuit)]

    R, D, L = n_rollouts, len(spec.drivers), spec.n_laps
    gen = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    dev = torch.device(device)

    # ---- deterministic per-driver per-lap components -------------------------
    fuel = torch.tensor(_fuel_delta(L), dtype=torch.float32)            # [L]
    base = torch.tensor([d.base_pace_s for d in spec.drivers])          # [D]

    deg_mid  = torch.empty(D, L)
    deg_sig  = torch.empty(D, L)
    stint_id = torch.empty(D, L, dtype=torch.long)
    is_pit   = torch.zeros(D, L, dtype=torch.bool)
    n_stints = 1
    for i, d in enumerate(spec.drivers):
        comp, age, stint = _stint_schedule(spec, d)
        mid = np.empty(L); lo = np.empty(L); hi = np.empty(L)
        for ci in range(len(COMPOUNDS)):
            m = comp == ci
            if m.any():
                mm, ll, hh = predict_degradation(
                    curves, COMPOUNDS[ci], spec.circuit, spec.era, age[m].astype(float),
                    track_temp=spec.track_temp,
                )
                mid[m], lo[m], hi[m] = mm, ll, hh
        deg_mid[i]  = torch.tensor(mid, dtype=torch.float32)
        deg_sig[i]  = torch.tensor((hi - lo) / (2 * 1.96), dtype=torch.float32).clamp(min=0.0)
        stint_id[i] = torch.tensor(stint, dtype=torch.long)
        for pit_lap, _ in d.strategy:
            if 1 <= pit_lap <= L:
                is_pit[i, pit_lap - 1] = True
        n_stints = max(n_stints, int(stint.max()) + 1)

    # ---- random draws ---------------------------------------------------------
    lap_noise = torch.randn(R, D, L, generator=gen) * sigma             # [R,D,L]
    z_stint   = torch.randn(R, D, n_stints, generator=gen)              # [R,D,S]
    deg_noise = z_stint.gather(2, stint_id.unsqueeze(0).expand(R, -1, -1)) * deg_sig.unsqueeze(0)
    pit_noise = torch.randn(R, D, L, generator=gen) * PIT_SIGMA_S
    pass_roll = torch.rand(R, D, L, generator=gen)

    lap_time = (
        base.view(1, D, 1) + fuel.view(1, 1, L)
        + deg_mid.unsqueeze(0) + deg_noise
        + lap_noise
        + is_pit.unsqueeze(0) * (pit_loss + pit_noise)
    )                                                                    # [R,D,L]

    # ---- in-progress caution: overlay SC/VSC pace on the opening laps --------
    # Per rollout we sample how many more laps the flag lasts (conditioned on how
    # long it has already run). On those laps the whole field runs at one dictated
    # pace (overtaking frozen), pitting is cheaper, and at the Safety-Car restart
    # the field is bunched nose-to-tail. is_restart marks each rollout's final
    # caution lap; the bunching is applied inside the lap loop below.
    is_restart = torch.zeros(R, L, dtype=torch.bool)
    bunch = False
    if spec.caution is not None:
        cause, elapsed = spec.caution[0], int(spec.caution[1])
        rem = torch.tensor(_sample_caution_lengths(cause, elapsed, R, seed), dtype=torch.long)
        lap_idx = torch.arange(L)
        caution_active = lap_idx.view(1, L) < rem.view(R, 1)             # [R,L]
        is_restart = (lap_idx.view(1, L) == (rem.view(R, 1) - 1)) & (rem.view(R, 1) <= L)
        field_pace = float(base.mean()) * CAUTION_PACE_MULT.get(cause, 1.40)
        caution_lap_time = (
            field_pace
            + is_pit.unsqueeze(0) * (pit_loss * CAUTION_PIT_FACTOR.get(cause, 0.55) + pit_noise)
        )                                                                # [R,D,L]
        lap_time = torch.where(caution_active.unsqueeze(1), caution_lap_time, lap_time)
        bunch = cause in BUNCH_CAUSES
    lap_time = lap_time.to(dev)
    is_restart = is_restart.to(dev)

    # ---- lap-by-lap with overtaking friction ---------------------------------
    cum = torch.tensor([d.grid_gap_s for d in spec.drivers]).expand(R, D).clone().to(dev)
    for lap in range(L):
        proposed = cum + lap_time[:, :, lap]
        # Order at the START of the lap defines who is "ahead on track".
        order = torch.argsort(cum, dim=1)                                # [R,D] indices
        resolved = proposed.clone()
        # Walk the field front to back; a follower that would jump ahead of the
        # car in front needs a successful pass roll, else is held in dirty air.
        min_ahead = torch.full((R,), -1e9, device=dev)
        for pos in range(D):
            idx = order[:, pos]                                          # [R]
            t = resolved.gather(1, idx.unsqueeze(1)).squeeze(1)
            would_pass = t < min_ahead
            close = (t - min_ahead).abs() < STRIKE_GAP_S
            allowed = pass_roll[:, :, lap].gather(1, idx.unsqueeze(1)).squeeze(1) < p_pass
            blocked = would_pass & ~(close & allowed)
            t = torch.where(blocked, min_ahead + HOLD_GAP_S, t)
            resolved.scatter_(1, idx.unsqueeze(1), t.unsqueeze(1))
            min_ahead = torch.maximum(min_ahead, t)
        cum = resolved

        # Safety-Car restart: on each rollout's final caution lap, concertina the
        # field to a fixed nose-to-tail spacing, preserving track order.
        if bunch:
            rmask = is_restart[:, lap]                                   # [R]
            if bool(rmask.any()):
                ranks = torch.argsort(torch.argsort(cum, dim=1), dim=1).to(cum.dtype)
                leader = cum.min(dim=1, keepdim=True).values
                bunched = leader + ranks * SC_BUNCH_GAP_S
                cum = torch.where(rmask.unsqueeze(1), bunched, cum)

    # ---- aggregate ------------------------------------------------------------
    race_time = cum.cpu().numpy()
    finish = np.argsort(np.argsort(race_time, axis=1), axis=1) + 1       # 1 = winner

    drivers = [d.driver for d in spec.drivers]
    win    = {d: float((finish[:, i] == 1).mean()) for i, d in enumerate(drivers)}
    podium = {d: float((finish[:, i] <= 3).mean()) for i, d in enumerate(drivers)}
    meanf  = {d: float(finish[:, i].mean()) for i, d in enumerate(drivers)}
    return SimResult(drivers, finish, race_time, win, podium, meanf)


# ---------------------------------------------------------------------------
# Strategy recommendation: MDP proposes, the simulation disposes
# ---------------------------------------------------------------------------

def recommend(
    spec: RaceSpec,
    ego_driver: str,
    candidate_strategies: dict[str, tuple[str, list[tuple[int, str]]]],
    n_rollouts: int = 1000,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Rank candidate (start_compound, stops) strategies for one driver.

    candidate_strategies: name -> (start_compound, [(pit_lap, compound), ...]).
    Rivals keep their spec strategies. Same seed across candidates = paired
    comparison (common random numbers), so differences are strategy, not luck.
    """
    rows = []
    ego_idx = next(i for i, d in enumerate(spec.drivers) if d.driver == ego_driver)
    for name, (start, stops) in candidate_strategies.items():
        drivers = [
            DriverSpec(**{**d.__dict__}) for d in spec.drivers
        ]
        drivers[ego_idx].start_compound = start
        drivers[ego_idx].strategy = stops
        res = simulate(
            RaceSpec(**{**spec.__dict__, "drivers": drivers}),
            n_rollouts=n_rollouts, seed=seed,
        )
        i = res.drivers.index(ego_driver)
        rows.append({
            "strategy": name, "start": start, "stops": str(stops),
            "win_prob": res.win_prob[ego_driver],
            "podium_prob": res.podium_prob[ego_driver],
            "mean_finish": res.mean_finish[ego_driver],
            "p50_time_s": float(np.median(res.race_time_s[:, i])),
        })
    return pd.DataFrame(rows).sort_values("mean_finish").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Three-tier 6-car demo race + a strategy recommendation table."""
    drivers = [
        DriverSpec("FAST1", 90.0, "MEDIUM", [(20, "HARD")], 1, 0.0),
        DriverSpec("FAST2", 90.1, "MEDIUM", [(22, "HARD")], 2, 0.3),
        DriverSpec("MID1",  90.5, "MEDIUM", [(20, "HARD")], 3, 0.6),
        DriverSpec("MID2",  90.6, "SOFT",   [(15, "HARD")], 4, 0.9),
        DriverSpec("SLOW1", 91.2, "MEDIUM", [(25, "HARD")], 5, 1.2),
        DriverSpec("SLOW2", 91.4, "HARD",   [(30, "MEDIUM")], 6, 1.5),
    ]
    spec = RaceSpec("Bahrain Grand Prix", 0, 57, drivers)
    res = simulate(spec, n_rollouts=2000)
    log.info("Demo result:\n%s", res.table().round(3).to_string(index=False))

    cands = {
        "1-stop M->H (L28)": ("MEDIUM", [(28, "HARD")]),
        "2-stop M->H->H":    ("MEDIUM", [(17, "HARD"), (37, "HARD")]),
        "2-stop S->H->M":    ("SOFT",   [(13, "HARD"), (35, "MEDIUM")]),
    }
    rec = recommend(spec, "FAST1", cands, n_rollouts=2000)
    log.info("Strategy recommendation for FAST1:\n%s", rec.round(3).to_string(index=False))
    rec.to_csv(REPORTS_DIR / "demo_recommendation.csv", index=False)


if __name__ == "__main__":
    _demo()
