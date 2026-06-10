"""
src/models/tyre/fit.py

Tyre degradation model — Phase 3 (v3 spec, Section 5/11).

Model
-----
Fuel-corrected lap-time delta within a stint, relative to the stint's first
observed tyre age (age0):

    delta(age) = b·(age − age0) + c·(age² − age0²)

which is the quadratic degradation curve  f(age) = a + b·age + c·age²  with the
per-stint intercept `a` eliminated by differencing against the stint baseline.
Subtracting the estimated FuelEffect first matters: fuel burn makes cars ~0.045
s/lap FASTER over a stint, which cancels real degradation and biases b low.

Hierarchical pooling (Section 5)
--------------------------------
Thin cells (esp. 2026) back off:
    (compound × circuit × era)  →  (compound × era)  →  (compound)
The chosen level is recorded per cell in `pool_level`. Wide CIs in low-data
regimes are reported honestly — never false precision.

Scope
-----
Slick compounds only (SOFT/MEDIUM/HARD). INTERMEDIATE/WET running is dominated
by track-condition evolution (a drying track makes laps FASTER with age), so a
monotone degradation curve is the wrong model — documented out of scope.
Green-flag laps only (TrackStatus == "1"): SC/VSC laps are +20-40s outliers that
destroy the fit.

Outputs
-------
models/tyre_curves.joblib       — curves DataFrame (the sim engine's input)
reports/tyre/tyre_curves.csv    — same, for inspection
reports/tyre/degradation_curves_era{0,1}.png — fitted vs actual diagnostics
MLflow run "tyre_curves" under experiment "tyre_degradation".

Entry point:
    python -m src.models.tyre.fit
"""

import logging
import os
from pathlib import Path
from typing import Final

# mlflow 3.x blocks the local file:// store unless opted in (Errors log #13).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist

from src.models.lap_time.train_tft_data import load_tft_data, MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports" / "tyre"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME: Final[str] = "tyre_degradation"

SLICK_COMPOUNDS: Final[list[str]] = ["SOFT", "MEDIUM", "HARD"]

# Pooling thresholds: minimum stints to trust a fit at each level
MIN_STINTS_CIRCUIT: Final[int] = 5    # (compound × circuit × era)
MIN_STINTS_POOLED:  Final[int] = 8    # (compound × era) and (compound)
MIN_LAPS_PER_STINT: Final[int] = 5    # need enough points for curvature
MAX_ABS_DELTA:      Final[float] = 8.0  # s — outlier guard (traffic, mistakes)
CI_LEVEL:           Final[float] = 0.95

STINT_KEYS: Final[list[str]] = ["Season", "RoundNumber", "Driver", "StintID"]


# ---------------------------------------------------------------------------
# Degradation model
# ---------------------------------------------------------------------------

def degradation_model(x: np.ndarray, b: float, c: float) -> np.ndarray:
    """delta(age) = b·(age − age0) + c·(age² − age0²).

    x is a (2, n) array: row 0 = tyre age, row 1 = the stint's first observed
    age (age0). Differencing against age0 removes the per-stint intercept.
    """
    age, age0 = x[0], x[1]
    return b * (age - age0) + c * (age ** 2 - age0 ** 2)


def degradation_delta(b: float, c: float, age: np.ndarray | float,
                      age_from: float = 1.0) -> np.ndarray | float:
    """Predicted lap-time loss (s) at `age` relative to `age_from` (fresh=1)."""
    return b * (np.asarray(age) - age_from) + c * (np.asarray(age) ** 2 - age_from ** 2)


# ---------------------------------------------------------------------------
# Stint preparation
# ---------------------------------------------------------------------------

def prepare_stint_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Green-flag slick laps with fuel-corrected per-stint deltas.

    Adds columns: FuelCorrected, Age0 (stint's first observed TyreLife),
    Delta (fuel-corrected delta vs the stint baseline lap).
    Keeps only stints with >= MIN_LAPS_PER_STINT green laps.
    """
    d = df.copy()
    d = d[d["Compound"].isin(SLICK_COMPOUNDS)]
    d = d[d["TrackStatus"].astype(str) == "1"]

    # Remove the estimated fuel effect so the remaining trend is tyre, not fuel.
    d["FuelCorrected"] = d["LapTimeSeconds"] - d["FuelEffect"]

    d = d.sort_values(STINT_KEYS + ["LapNumber"])
    grp = d.groupby(STINT_KEYS)
    d["Age0"] = grp["TyreLife"].transform("min")
    # Baseline = median of the stint's first 3 green laps — a single-lap baseline
    # injects that lap's noise (~0.3s) into every delta of the stint as an offset.
    d["Baseline"] = grp["FuelCorrected"].transform(lambda s: s.head(3).median())
    d["Delta"] = d["FuelCorrected"] - d["Baseline"]

    # Keep usable stints, drop outlier deltas (traffic / off-track moments)
    stint_len = grp["TyreLife"].transform("size")
    d = d[(stint_len >= MIN_LAPS_PER_STINT) & (d["Delta"].abs() <= MAX_ABS_DELTA)]
    return d


# ---------------------------------------------------------------------------
# Single-cell fit
# ---------------------------------------------------------------------------

def fit_cell(cell_df: pd.DataFrame) -> dict | None:
    """Fit the degradation curve to one pool of prepared stint laps.

    Returns dict(b, c, b_ci, c_ci, r2, n_stints, n_laps) or None on failure.
    """
    n_stints = cell_df.groupby(STINT_KEYS).ngroups
    if n_stints == 0 or len(cell_df) < MIN_LAPS_PER_STINT:
        return None

    x = np.vstack([
        cell_df["TyreLife"].to_numpy(dtype=float),
        cell_df["Age0"].to_numpy(dtype=float),
    ])
    y = cell_df["Delta"].to_numpy(dtype=float)

    try:
        popt, pcov = curve_fit(
            degradation_model, x, y,
            p0=[0.05, 0.001],
            bounds=([-0.5, -0.05], [2.0, 0.5]),
            maxfev=10000,
        )
    except (RuntimeError, ValueError) as e:
        logger.warning("curve_fit failed: %s", e)
        return None

    b, c = popt
    perr = np.sqrt(np.diag(pcov))
    dof = max(1, len(y) - 2)
    t_crit = t_dist.ppf((1 + CI_LEVEL) / 2, dof)

    y_pred = degradation_model(x, b, c)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "b": float(b), "c": float(c),
        "b_ci": float(t_crit * perr[0]), "c_ci": float(t_crit * perr[1]),
        "r2": round(float(r2), 4),
        "n_stints": int(n_stints), "n_laps": int(len(cell_df)),
    }


# ---------------------------------------------------------------------------
# Hierarchical fitting
# ---------------------------------------------------------------------------

def fit_all_curves(stint_laps: pd.DataFrame) -> pd.DataFrame:
    """Fit every (compound × circuit × era) cell with hierarchical pooling.

    For each cell present in the data, use the deepest level with enough stints:
        circuit  : that exact (compound, circuit, era)      >= MIN_STINTS_CIRCUIT
        era      : all circuits for (compound, era)          >= MIN_STINTS_POOLED
        compound : all eras+circuits for compound             >= MIN_STINTS_POOLED
    Returns one row per cell: Compound, CircuitKey, Era, pool_level, b, c, CIs,
    r2, n_stints, n_laps (counts are from the pool actually fitted).
    """
    # Pre-fit the pooled fallbacks once
    era_fits: dict[tuple[str, int], dict | None] = {}
    for (comp, era), g in stint_laps.groupby(["Compound", "Era"]):
        f = fit_cell(g)
        era_fits[(comp, int(era))] = f if f and f["n_stints"] >= MIN_STINTS_POOLED else None

    compound_fits: dict[str, dict | None] = {}
    for comp, g in stint_laps.groupby("Compound"):
        f = fit_cell(g)
        compound_fits[comp] = f if f and f["n_stints"] >= MIN_STINTS_POOLED else None

    rows = []
    cells = stint_laps.groupby(["Compound", "CircuitKey", "Era"])
    logger.info("Fitting %d (compound × circuit × era) cells with pooling fallback", len(cells))

    for (comp, circuit, era), g in cells:
        era = int(era)
        fit = fit_cell(g)
        if fit is not None and fit["n_stints"] >= MIN_STINTS_CIRCUIT:
            level = "circuit"
        elif era_fits.get((comp, era)) is not None:
            fit, level = era_fits[(comp, era)], "era"
        elif compound_fits.get(comp) is not None:
            fit, level = compound_fits[comp], "compound"
        else:
            logger.warning("No usable fit at any level: %s × %s × Era%d", comp, circuit, era)
            continue

        rows.append({
            "Compound": comp, "CircuitKey": circuit, "Era": era,
            "pool_level": level, **fit,
        })

    curves = pd.DataFrame(rows)
    if len(curves):
        logger.info(
            "Fitted %d cells | levels: %s",
            len(curves), curves["pool_level"].value_counts().to_dict(),
        )
    return curves


# ---------------------------------------------------------------------------
# Prediction helper (consumed by the Phase 4 simulation engine)
# ---------------------------------------------------------------------------

def predict_degradation(
    curves: pd.DataFrame,
    compound: str,
    circuit: str,
    era: int,
    age: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Degradation delta (s, relative to fresh tyre age=1) with a 95% CI band.

    Looks up the cell row (already pool-resolved by fit_all_curves); falls back
    to the closest available row for the compound if the exact cell is missing.
    Returns (mid, lo, hi) arrays.
    """
    sel = curves[
        (curves["Compound"] == compound)
        & (curves["CircuitKey"] == circuit)
        & (curves["Era"] == era)
    ]
    if sel.empty:
        sel = curves[(curves["Compound"] == compound) & (curves["Era"] == era)]
    if sel.empty:
        sel = curves[curves["Compound"] == compound]
    if sel.empty:
        raise KeyError(f"No tyre curve for compound={compound}")

    row = sel.iloc[0]
    age = np.asarray(age, dtype=float)
    mid = degradation_delta(row["b"], row["c"], age)
    lo  = degradation_delta(row["b"] - row["b_ci"], row["c"] - row["c_ci"], age)
    hi  = degradation_delta(row["b"] + row["b_ci"], row["c"] + row["c_ci"], age)
    return mid, lo, hi


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_degradation_curves(
    curves: pd.DataFrame,
    stint_laps: pd.DataFrame,
    era: int,
    circuits: list[str] | None = None,
    compounds: list[str] | None = None,
) -> Path | None:
    """Fitted curves + CI bands over actual per-stint delta traces, one era.

    Default circuits = top 4 by lap volume within the era.
    Saved to reports/tyre/degradation_curves_era{era}.png
    """
    era_laps = stint_laps[stint_laps["Era"] == era]
    if era_laps.empty:
        logger.warning("No laps for Era %d — skipping plot", era)
        return None

    if circuits is None:
        circuits = (
            era_laps.groupby("CircuitKey").size().sort_values(ascending=False)
            .head(4).index.tolist()
        )
    if compounds is None:
        compounds = SLICK_COMPOUNDS

    fig, axes = plt.subplots(
        len(circuits), len(compounds),
        figsize=(5 * len(compounds), 3.5 * len(circuits)),
        squeeze=False,
    )

    for r, circuit in enumerate(circuits):
        for col, compound in enumerate(compounds):
            ax = axes[r][col]
            actual = era_laps[
                (era_laps["CircuitKey"] == circuit) & (era_laps["Compound"] == compound)
            ]
            for _, stint in actual.groupby(STINT_KEYS):
                ax.plot(stint["TyreLife"], stint["Delta"],
                        alpha=0.18, color="steelblue", linewidth=0.8)

            cell = curves[
                (curves["CircuitKey"] == circuit) & (curves["Compound"] == compound)
                & (curves["Era"] == era)
            ]
            if len(cell):
                row = cell.iloc[0]
                max_age = max(actual["TyreLife"].max() if len(actual) else 30, 10)
                ages = np.linspace(1, max_age, 100)
                mid = degradation_delta(row["b"], row["c"], ages)
                lo  = degradation_delta(row["b"] - row["b_ci"], row["c"] - row["c_ci"], ages)
                hi  = degradation_delta(row["b"] + row["b_ci"], row["c"] + row["c_ci"], ages)
                ax.plot(ages, mid, color="#E8002D", linewidth=2,
                        label=f"fit [{row['pool_level']}] r²={row['r2']:.2f}")
                ax.fill_between(ages, lo, hi, alpha=0.25, color="#E8002D", label="95% CI")
                ax.legend(fontsize=7)

            ax.set_title(f"{circuit.replace(' Grand Prix', '')} — {compound}", fontsize=9)
            ax.set_xlabel("Tyre age (laps)", fontsize=8)
            ax.set_ylabel("Δ fuel-corrected lap time (s)", fontsize=8)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.grid(True, alpha=0.3)

    era_name = {0: "2022-2025 ground effect", 1: "2026+ active aero"}.get(era, str(era))
    plt.suptitle(f"Tyre degradation — fitted vs actual (Era {era}: {era_name})", fontsize=12)
    plt.tight_layout()
    out = REPORTS_DIR / f"degradation_curves_era{era}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved → %s", out)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def fit() -> pd.DataFrame:
    """Load data, fit all curves with pooling, save artifacts, log to MLflow."""
    df = load_tft_data()      # per-round glob + dedup + StintID/TrackStatus carry-back
    stint_laps = prepare_stint_laps(df)
    logger.info(
        "Prepared %d green slick laps in %d stints (%d circuits, eras %s)",
        len(stint_laps), stint_laps.groupby(STINT_KEYS).ngroups,
        stint_laps["CircuitKey"].nunique(), sorted(stint_laps["Era"].unique()),
    )

    curves = fit_all_curves(stint_laps)
    if curves.empty:
        raise RuntimeError("No tyre curves fitted — check input data")

    joblib.dump(curves, MODELS_DIR / "tyre_curves.joblib")
    curves.to_csv(REPORTS_DIR / "tyre_curves.csv", index=False)
    logger.info("Saved tyre_curves.joblib + tyre_curves.csv")

    plot_paths = [plot_degradation_curves(curves, stint_laps, era) for era in (0, 1)]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="tyre_curves"):
        mlflow.log_params({
            "model": "quadratic_delta_curvefit",
            "fuel_corrected": True, "green_flag_only": True,
            "compounds": ",".join(SLICK_COMPOUNDS),
            "min_stints_circuit": MIN_STINTS_CIRCUIT,
            "min_stints_pooled": MIN_STINTS_POOLED,
            "min_laps_per_stint": MIN_LAPS_PER_STINT,
        })
        circuit_lvl = curves[curves["pool_level"] == "circuit"]
        mlflow.log_metrics({
            "n_cells": len(curves),
            "n_circuit_level": int((curves["pool_level"] == "circuit").sum()),
            "n_era_level": int((curves["pool_level"] == "era").sum()),
            "n_compound_level": int((curves["pool_level"] == "compound").sum()),
            "mean_r2_circuit_level": float(circuit_lvl["r2"].mean()) if len(circuit_lvl) else float("nan"),
            "mean_b": float(curves["b"].mean()),
            "mean_c": float(curves["c"].mean()),
        })
        mlflow.log_artifact(str(REPORTS_DIR / "tyre_curves.csv"))
        for p in plot_paths:
            if p is not None:
                mlflow.log_artifact(str(p))

    logger.info("=" * 60)
    logger.info("TYRE MODEL COMPLETE — %d cells | pool levels %s | circuit-level mean r² %.3f",
                len(curves), curves["pool_level"].value_counts().to_dict(),
                circuit_lvl["r2"].mean() if len(circuit_lvl) else float("nan"))
    logger.info("=" * 60)
    return curves


if __name__ == "__main__":
    fit()
