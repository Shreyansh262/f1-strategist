"""Pit-strategy MDP — Phase 4a (interpretable baseline before the sim engine).

Formulation
-----------
Finite-horizon deterministic MDP solved exactly by backward induction (a special
case of value iteration — the horizon is the race length, so one backward sweep
is exact and runs in milliseconds).

    State   : (lap, compound, tyre_age, used_two_compounds)
    Actions : stay_out | pit_SOFT | pit_MEDIUM | pit_HARD
    Lap cost: circuit_base
              + fuel_delta(lap)                  (lighter car later = faster)
              + tyre_delta(compound, age)        (from Phase 3 curves, mid value)
              + pit_loss(circuit) if pitting
    Terminal: 0 if both slick compound groups were used (F1 two-compound rule),
              +inf otherwise (illegal race).

Single-car scope (deliberate): the MDP optimises total race time with no rivals.
Traffic, undercuts, and position dynamics are modeled in the Monte Carlo engine
(src/simulation/engine.py) — the MDP proposes candidate strategies, the sim
ranks them under uncertainty and interaction. That composition is the design.

Outputs
-------
reports/pit_strategy/optimal_strategies.csv      one row per (circuit, era, start compound)
reports/pit_strategy/policy_<circuit>_era<e>.png pit-window heatmap (lap × age → action)
reports/pit_strategy/mdp_vs_actual.csv           MDP strategy vs actual top-3 strategies

Run:
    python -m src.models.pit_strategy.mdp
"""
from __future__ import annotations

import glob
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd

from src.models.pit_strategy.pit_loss import load_pit_loss, pit_loss_for
from src.pipeline.features import (
    FUEL_LOAD_START_KG, FUEL_BURN_PER_LAP, FUEL_EFFECT_PER_KG,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports" / "pit_strategy"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

COMPOUNDS: Final[list[str]] = ["SOFT", "MEDIUM", "HARD"]
ACTIONS:   Final[list[str]] = ["stay_out", "pit_SOFT", "pit_MEDIUM", "pit_HARD"]
MAX_AGE:   Final[int] = 45          # cap tyre age in the state grid
OUT_LAP_AGE: Final[int] = 2         # validate.py drops TyreLife==1; fresh stint starts at 2


# ---------------------------------------------------------------------------
# Race model inputs
# ---------------------------------------------------------------------------

@dataclass
class RaceModel:
    """Everything the MDP needs to price one lap."""
    circuit: str
    era: int
    n_laps: int
    base_lap_s: float                       # green-flag median pace at this circuit
    pit_loss_s: float
    tyre_delta: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    # tyre_delta[compound][age] = seconds lost vs fresh tyre at that age

    def fuel_delta(self, lap: int) -> float:
        """Fuel-load lap-time delta relative to the race-average fuel load."""
        load = max(FUEL_LOAD_START_KG - (lap - 1) * FUEL_BURN_PER_LAP, 0.0)
        mean_load = FUEL_LOAD_START_KG - (self.n_laps - 1) * FUEL_BURN_PER_LAP / 2
        return (load - mean_load) * FUEL_EFFECT_PER_KG

    def lap_time(self, lap: int, compound: str, age: int) -> float:
        return (
            self.base_lap_s
            + self.fuel_delta(lap)
            + float(self.tyre_delta[compound][min(age, MAX_AGE)])
        )


def build_race_model(
    circuit: str,
    era: int,
    n_laps: int,
    base_lap_s: float,
    curves: pd.DataFrame | None = None,
    pit_table: dict | None = None,
    track_temp: float | None = None,
) -> RaceModel:
    """Assemble a RaceModel from the Phase 3 tyre curves + pit-loss table.

    ``track_temp`` (Phase 9.4): when given, degradation is evaluated at that
    track temperature so the optimal policy reflects hotter/colder conditions.
    None keeps the curve's historical temperature (original behaviour).
    """
    if curves is None:
        curves = joblib.load(MODELS_DIR / "tyre_curves.joblib")
    if pit_table is None:
        pit_table = load_pit_loss()

    from src.models.tyre.fit import predict_degradation
    ages = np.arange(0, MAX_AGE + 1, dtype=float)
    tyre_delta = {}
    for comp in COMPOUNDS:
        mid, _, _ = predict_degradation(curves, comp, circuit, era, ages, track_temp=track_temp)
        tyre_delta[comp] = np.asarray(mid, dtype=float)

    return RaceModel(
        circuit=circuit, era=era, n_laps=n_laps, base_lap_s=base_lap_s,
        pit_loss_s=pit_loss_for(pit_table, circuit), tyre_delta=tyre_delta,
    )


# ---------------------------------------------------------------------------
# Exact backward induction
# ---------------------------------------------------------------------------

def solve(model: RaceModel, start_compound: str = "MEDIUM") -> dict:
    """Solve the MDP exactly. Returns optimal strategy + value/policy grids.

    V[lap, c, a, u] = best total time from the start of `lap` given compound c,
    tyre age a, and u = whether two slick compounds were already used.
    """
    n_laps = model.n_laps
    n_c, n_a = len(COMPOUNDS), MAX_AGE + 1
    INF = np.inf

    V = np.full((n_laps + 2, n_c, n_a, 2), INF)
    policy = np.zeros((n_laps + 1, n_c, n_a, 2), dtype=np.int8)

    # Terminal: race over — legal only if two compounds were used.
    V[n_laps + 1, :, :, 1] = 0.0
    V[n_laps + 1, :, :, 0] = INF

    lap_cost = np.empty((n_laps + 1, n_c, n_a))
    for lap in range(1, n_laps + 1):
        for ci, comp in enumerate(COMPOUNDS):
            for age in range(n_a):
                lap_cost[lap, ci, age] = model.lap_time(lap, comp, age)

    for lap in range(n_laps, 0, -1):
        for ci in range(n_c):
            for age in range(n_a):
                next_age = min(age + 1, MAX_AGE)
                for u in (0, 1):
                    # action 0: stay out
                    best = lap_cost[lap, ci, age] + V[lap + 1, ci, next_age, u]
                    best_a = 0
                    # actions 1..3: pit at the END of this lap onto compound k
                    for k in range(n_c):
                        u2 = 1 if (u == 1 or k != ci) else 0
                        cost = (
                            lap_cost[lap, ci, age] + model.pit_loss_s
                            + V[lap + 1, k, OUT_LAP_AGE, u2]
                        )
                        if cost < best:
                            best, best_a = cost, k + 1
                    V[lap, ci, age, u] = best
                    policy[lap, ci, age, u] = best_a

    # --- roll the optimal trajectory forward --------------------------------
    ci = COMPOUNDS.index(start_compound)
    age, u = OUT_LAP_AGE, 0
    stops: list[tuple[int, str]] = []
    total = V[1, ci, age, u]
    for lap in range(1, n_laps + 1):
        a = policy[lap, ci, age, u]
        if a == 0:
            age = min(age + 1, MAX_AGE)
        else:
            new_ci = a - 1
            u = 1 if (u == 1 or new_ci != ci) else 0
            stops.append((lap, COMPOUNDS[new_ci]))
            ci, age = new_ci, OUT_LAP_AGE

    return {
        "circuit": model.circuit, "era": model.era, "n_laps": n_laps,
        "start_compound": start_compound,
        "n_stops": len(stops),
        "stops": stops,
        "total_time_s": float(total),
        "avg_lap_s": float(total / n_laps),
        "policy": policy, "value": V,
        "feasible": bool(np.isfinite(total)),
    }


# ---------------------------------------------------------------------------
# Policy heatmap (dashboard visual)
# ---------------------------------------------------------------------------

_ACTION_COLORS = ["#E8E8E8", "#E8002D", "#FFC906", "#B0B0B0"]  # stay, S, M, H


def plot_policy(result: dict, compound: str = "MEDIUM", used_two: int = 0) -> Path:
    """Heatmap of the optimal action over (lap, tyre age) for one compound."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ci = COMPOUNDS.index(compound)
    grid = result["policy"][1:, ci, :, used_two]          # [lap, age]
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        grid.T, aspect="auto", origin="lower", interpolation="nearest",
        cmap=ListedColormap(_ACTION_COLORS), vmin=-0.5, vmax=3.5,
        extent=(1, result["n_laps"], 0, MAX_AGE),
    )
    cbar = fig.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["stay out", "pit→SOFT", "pit→MED", "pit→HARD"])
    ax.set_xlabel("Lap")
    ax.set_ylabel(f"Tyre age on {compound}")
    two = "2-compound rule met" if used_two else "2nd compound still needed"
    ax.set_title(f"Optimal pit policy — {result['circuit']} (era {result['era']}, {two})")
    out = REPORTS_DIR / f"policy_{result['circuit'].replace(' ', '_')}_era{result['era']}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Policy heatmap -> %s", out)
    return out


# ---------------------------------------------------------------------------
# Historical comparison
# ---------------------------------------------------------------------------

def _actual_strategies(circuit: str, season: int) -> pd.DataFrame:
    """Strategies actually used by the 3 fastest full-distance finishers."""
    files = sorted(glob.glob(str(PROJECT_ROOT / "data" / "raw" / f"laps_{season}_r*.parquet")))
    if not files:
        return pd.DataFrame()
    raw = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    raw = raw[raw["CircuitKey"] == circuit]
    if raw.empty:
        return pd.DataFrame()
    raw = raw.drop_duplicates(subset=["Season", "RoundNumber", "Driver", "LapNumber"])
    totals = raw.groupby("Driver").agg(
        time=("LapTimeSeconds", "sum"), laps=("LapNumber", "max")
    )
    finishers = totals[totals["laps"] == totals["laps"].max()].nsmallest(3, "time")
    rows = []
    for drv in finishers.index:
        d = raw[raw["Driver"] == drv].sort_values("LapNumber")
        pit_laps = d.loc[d["PitInTime"].notna(), "LapNumber"].tolist()
        compounds = (
            d.groupby((d["Compound"] != d["Compound"].shift()).cumsum())["Compound"]
            .first().tolist()
        )
        rows.append({
            "Driver": drv, "race_time_s": float(finishers.loc[drv, "time"]),
            "n_stops": len(pit_laps), "pit_laps": pit_laps, "compounds": compounds,
        })
    return pd.DataFrame(rows)


def compare_vs_actual(results: list[dict], season: int = 2025) -> pd.DataFrame:
    rows = []
    for r in results:
        actual = _actual_strategies(r["circuit"], season)
        if actual.empty:
            continue
        for _, a in actual.iterrows():
            rows.append({
                "circuit": r["circuit"],
                "mdp_n_stops": r["n_stops"],
                "mdp_stops": str(r["stops"]),
                "driver": a["Driver"],
                "actual_n_stops": a["n_stops"],
                "actual_pit_laps": str(a["pit_laps"]),
                "actual_compounds": str(a["compounds"]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "mdp_vs_actual.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Entry point — solve a representative set of circuits
# ---------------------------------------------------------------------------

def _green_base_and_laps(circuit: str, seasons: list[int]) -> tuple[float, int] | None:
    """Median green pace + typical race length for a circuit from raw data."""
    files: list[str] = []
    for s in seasons:
        files += glob.glob(str(PROJECT_ROOT / "data" / "raw" / f"laps_{s}_r*.parquet"))
    if not files:
        return None
    raw = pd.concat((pd.read_parquet(f) for f in sorted(files)), ignore_index=True)
    raw = raw[raw["CircuitKey"] == circuit]
    if raw.empty:
        return None
    raw = raw.drop_duplicates(subset=["Season", "RoundNumber", "Driver", "LapNumber"])
    green = raw[
        (raw["TrackStatus"].astype(str) == "1")
        & raw["PitInTime"].isna() & raw["PitOutTime"].isna()
        & raw["LapTimeSeconds"].between(60, 150)
    ]
    n_laps = int(raw.groupby(["Season", "RoundNumber"])["LapNumber"].max().median())
    return float(green["LapTimeSeconds"].median()), n_laps


DEMO_CIRCUITS: Final[list[tuple[str, int, list[int]]]] = [
    ("Bahrain Grand Prix",  0, [2022, 2023, 2024]),
    ("Spanish Grand Prix",  0, [2022, 2023, 2024]),
    ("Monaco Grand Prix",   0, [2022, 2023, 2024]),
    ("Japanese Grand Prix", 1, [2026]),
]


def main() -> list[dict]:
    curves = joblib.load(MODELS_DIR / "tyre_curves.joblib")
    pit_table = load_pit_loss()

    results = []
    summary_rows = []
    for circuit, era, seasons in DEMO_CIRCUITS:
        info = _green_base_and_laps(circuit, seasons)
        if info is None:
            log.warning("No data for %s — skipping", circuit)
            continue
        base, n_laps = info
        model = build_race_model(circuit, era, n_laps, base, curves, pit_table)
        per_circuit = []
        for start in COMPOUNDS:
            r = solve(model, start_compound=start)
            results.append(r)
            per_circuit.append(r)
            summary_rows.append({
                "circuit": circuit, "era": era, "n_laps": n_laps,
                "base_lap_s": round(base, 2), "pit_loss_s": model.pit_loss_s,
                "start_compound": start, "n_stops": r["n_stops"],
                "stops": str(r["stops"]), "total_time_s": round(r["total_time_s"], 1),
                "feasible": r["feasible"],
            })
            log.info(
                "%-28s era%d start=%-6s -> %d stop(s) %s  total %.1fs",
                circuit, era, start, r["n_stops"], r["stops"], r["total_time_s"],
            )
        plot_policy(per_circuit[1])    # MEDIUM-start policy for this circuit

    pd.DataFrame(summary_rows).to_csv(REPORTS_DIR / "optimal_strategies.csv", index=False)
    log.info("optimal_strategies.csv -> %s", REPORTS_DIR)

    comp = compare_vs_actual(
        [r for r in results if r["start_compound"] == "MEDIUM" and r["era"] == 0]
    )
    if len(comp):
        log.info("mdp_vs_actual.csv (%d rows) -> %s", len(comp), REPORTS_DIR)
    return results


if __name__ == "__main__":
    main()
