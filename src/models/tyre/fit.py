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
All five compounds: slicks (SOFT/MEDIUM/HARD) plus wets (INTERMEDIATE/WET).
The slick fits are unchanged. The wet compounds are SPARSE (intermediate is
fittable per compound/era; wet running is only ~19 stints total), and wet pace is
also confounded by track-condition evolution (a drying track makes laps FASTER
with age), so wet curves are pooled aggressively — typically a single
per-compound fit shared across every circuit and era — and carry honestly wide
CIs. predict_degradation guarantees a finite (mid, lo, hi) for any of the five
compounds at any (circuit, era), backing off to a built-in wet fallback if even
the global pool is too thin. See WET_FALLBACK below.
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
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist

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
# Wet compounds are sparse and pooled aggressively (see fit_all_curves).
WET_COMPOUNDS:   Final[list[str]] = ["INTERMEDIATE", "WET"]
ALL_COMPOUNDS:   Final[list[str]] = SLICK_COMPOUNDS + WET_COMPOUNDS

# Built-in last-resort wet curve, used only when a wet compound has too little
# data to fit even a global pool. delta(age) = b·(age−age0) + c·(age²−age0²).
# Intentionally gentle slopes with WIDE CIs — wet pace is dominated by track
# evolution, not tyre wear, so we assert little and report large uncertainty.
WET_FALLBACK: Final[dict[str, dict[str, float]]] = {
    "INTERMEDIATE": {"b": 0.05, "c": 0.0015, "b_ci": 0.06, "c_ci": 0.003},
    "WET":          {"b": 0.04, "c": 0.0010, "b_ci": 0.08, "c_ci": 0.004},
}

# Pooling thresholds: minimum stints to trust a fit at each level
MIN_STINTS_CIRCUIT: Final[int] = 5    # (compound × circuit × era)
MIN_STINTS_POOLED:  Final[int] = 8    # (compound × era) and (compound)
# Wet running is rare; allow a global per-compound pool to form from fewer stints
# than the slick threshold so INTERMEDIATE/WET always get a real fitted curve
# before the built-in fallback is ever needed.
MIN_STINTS_WET_POOLED: Final[int] = 4
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


def degradation_model_temp(
    x: np.ndarray, b0: float, c0: float, b_temp: float, c_temp: float
) -> np.ndarray:
    """Temperature-augmented degradation (Phase 9.4).

    x is a (3, n) array: row 0 = tyre age, row 1 = age0, row 2 = dT
    (TrackTemp − pool reference temp). The slope/curvature scale linearly with
    track temperature:

        delta(age) = (b0 + b_temp·dT)·(age − age0) + (c0 + c_temp·dT)·(age² − age0²)

    so b_temp is the extra s/lap of linear degradation per °C above the pool's
    reference temperature — the quantity that lets the sim answer "this circuit,
    but hotter/colder".
    """
    age, age0, dT = x[0], x[1], x[2]
    return (b0 + b_temp * dT) * (age - age0) + (c0 + c_temp * dT) * (age ** 2 - age0 ** 2)


# ---------------------------------------------------------------------------
# Stint preparation
# ---------------------------------------------------------------------------

def prepare_stint_laps(df: pd.DataFrame, compounds: list[str] | None = None) -> pd.DataFrame:
    """Green-flag laps with fuel-corrected per-stint deltas.

    Adds columns: FuelCorrected, Age0 (stint's first observed TyreLife),
    Delta (fuel-corrected delta vs the stint baseline lap).
    Keeps only stints with >= MIN_LAPS_PER_STINT green laps.

    ``compounds`` selects which compounds to keep (default: all five — slicks and
    wets). Each compound is processed independently per stint, so widening the set
    leaves the slick rows numerically identical to a slick-only run.
    """
    if compounds is None:
        compounds = ALL_COMPOUNDS
    d = df.copy()
    d = d[d["Compound"].isin(compounds)]
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

def _pooled_threshold(comp: str) -> int:
    """Minimum stints to trust a pooled (era / compound) fit for this compound.

    Wet compounds are rare, so they use a lower bar — otherwise a thin but real
    wet pool would be discarded in favour of the synthetic fallback.
    """
    return MIN_STINTS_WET_POOLED if comp in WET_COMPOUNDS else MIN_STINTS_POOLED


def _wet_fallback_fit(comp: str) -> dict:
    """Built-in synthetic fit for a wet compound with too little data to model."""
    fb = WET_FALLBACK[comp]
    return {
        "b": fb["b"], "c": fb["c"], "b_ci": fb["b_ci"], "c_ci": fb["c_ci"],
        "r2": 0.0, "n_stints": 0, "n_laps": 0,
    }


def fit_all_curves(stint_laps: pd.DataFrame) -> pd.DataFrame:
    """Fit every (compound × circuit × era) cell with hierarchical pooling.

    For each cell present in the data, use the deepest level with enough stints:
        circuit  : that exact (compound, circuit, era)      >= MIN_STINTS_CIRCUIT
        era      : all circuits for (compound, era)          >= pooled threshold
        compound : all eras+circuits for compound            >= pooled threshold
    Wet compounds (INTERMEDIATE/WET) additionally get a GLOBAL fallback row
    (CircuitKey="*", Era=-1) so predict_degradation always resolves them for any
    circuit/era — using the global per-compound fit when one exists, else the
    built-in WET_FALLBACK. They never produce a "no usable fit" gap.

    Returns one row per cell: Compound, CircuitKey, Era, pool_level, b, c, CIs,
    r2, n_stints, n_laps (counts are from the pool actually fitted).
    """
    # Pre-fit the pooled fallbacks once
    era_fits: dict[tuple[str, int], dict | None] = {}
    for (comp, era), g in stint_laps.groupby(["Compound", "Era"]):
        f = fit_cell(g)
        era_fits[(comp, int(era))] = f if f and f["n_stints"] >= _pooled_threshold(comp) else None

    compound_fits: dict[str, dict | None] = {}
    for comp, g in stint_laps.groupby("Compound"):
        f = fit_cell(g)
        compound_fits[comp] = f if f and f["n_stints"] >= _pooled_threshold(comp) else None

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
        elif comp in WET_COMPOUNDS:
            # Sparse wet cell with no pool deep enough — synthesise a curve so the
            # cell still exists rather than vanishing (never silently → MEDIUM).
            fit, level = _wet_fallback_fit(comp), "wet_fallback"
        else:
            logger.warning("No usable fit at any level: %s × %s × Era%d", comp, circuit, era)
            continue

        rows.append({
            "Compound": comp, "CircuitKey": circuit, "Era": era,
            "pool_level": level, **fit,
        })

    # Guaranteed global wet rows so ANY (circuit, era) query for a wet compound
    # resolves — including eras/circuits that never saw wet running.
    for comp in WET_COMPOUNDS:
        if compound_fits.get(comp) is not None:
            fit, level = compound_fits[comp], "compound_global"
        else:
            fit, level = _wet_fallback_fit(comp), "wet_fallback_global"
        rows.append({
            "Compound": comp, "CircuitKey": "*", "Era": -1,
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
# Temperature sensitivity (Phase 9.4) — how degradation scales with track temp
# ---------------------------------------------------------------------------

# Plausibility bounds on the per-°C slope sensitivity (s/lap per °C). Tyre
# degradation rises with track temperature but the effect is gentle; these keep
# a thin/confounded cell from producing an absurd extrapolation.
B_TEMP_BOUND: Final[float] = 0.01
C_TEMP_BOUND: Final[float] = 0.001


def fit_temp_sensitivity(stint_laps: pd.DataFrame) -> pd.DataFrame:
    """Estimate degradation's track-temperature sensitivity per (compound × era).

    Because our ingest stores ONE session-median TrackTemp per race, the
    temperature signal is CROSS-SECTIONAL — it comes from the same compound
    being run across races/circuits at different track temperatures, not from
    within-stint temperature change. We therefore pool all circuits within a
    (compound, era) and fit the temp-augmented model once, reporting b_temp /
    c_temp with CIs. This is confounded with circuit characteristics correlated
    with temperature (documented in the model card); it is the best estimate the
    lap-median data supports and is used only as a gentle, bounded adjustment.

    Returns one row per (Compound, Era): temp_ref (pool mean TrackTemp), b_temp,
    c_temp, their CIs, temp_min/temp_max (support), n_stints, n_laps.
    """
    if "TrackTemp" not in stint_laps.columns:
        logger.warning("TrackTemp absent — skipping temperature sensitivity fit")
        return pd.DataFrame()

    rows = []
    for (comp, era), g in stint_laps.groupby(["Compound", "Era"]):
        g = g.dropna(subset=["TrackTemp"])
        n_stints = g.groupby(STINT_KEYS).ngroups
        if n_stints < MIN_STINTS_POOLED or len(g) < 10:
            continue
        temp_ref = float(g["TrackTemp"].mean())
        x = np.vstack([
            g["TyreLife"].to_numpy(dtype=float),
            g["Age0"].to_numpy(dtype=float),
            g["TrackTemp"].to_numpy(dtype=float) - temp_ref,
        ])
        y = g["Delta"].to_numpy(dtype=float)
        try:
            popt, pcov = curve_fit(
                degradation_model_temp, x, y,
                p0=[0.05, 0.001, 0.0, 0.0],
                bounds=([-0.5, -0.05, -B_TEMP_BOUND, -C_TEMP_BOUND],
                        [2.0, 0.5, B_TEMP_BOUND, C_TEMP_BOUND]),
                maxfev=20000,
            )
        except (RuntimeError, ValueError) as e:
            logger.warning("temp fit failed for %s Era%d: %s", comp, era, e)
            continue
        b0, c0, b_temp, c_temp = popt
        perr = np.sqrt(np.diag(pcov))
        dof = max(1, len(y) - 4)
        t_crit = t_dist.ppf((1 + CI_LEVEL) / 2, dof)
        rows.append({
            "Compound": comp, "Era": int(era),
            "temp_ref": round(temp_ref, 2),
            "b_temp": float(b_temp), "c_temp": float(c_temp),
            "b_temp_ci": float(t_crit * perr[2]), "c_temp_ci": float(t_crit * perr[3]),
            "temp_min": round(float(g["TrackTemp"].min()), 1),
            "temp_max": round(float(g["TrackTemp"].max()), 1),
            "n_stints": int(n_stints), "n_laps": int(len(g)),
        })
    sens = pd.DataFrame(rows)
    if len(sens):
        logger.info("Temp sensitivity fitted for %d (compound×era) pools; "
                    "b_temp range [%.4f, %.4f] s/lap/°C",
                    len(sens), sens["b_temp"].min(), sens["b_temp"].max())
    return sens


def attach_temp_sensitivity(
    curves: pd.DataFrame, stint_laps: pd.DataFrame, sensitivity: pd.DataFrame
) -> pd.DataFrame:
    """Add temp_ref / b_temp / c_temp columns to the curves table.

    temp_ref per cell = that cell's own mean TrackTemp (so the fitted b, c are
    "degradation at the cell's historical temperature"); b_temp/c_temp come from
    the (compound × era) sensitivity pool. Cells with no sensitivity get 0.0
    (i.e. predict_degradation with a track_temp simply returns the base curve).
    """
    curves = curves.copy()
    # per-cell historical mean track temp
    cell_temp = (
        stint_laps.dropna(subset=["TrackTemp"])
        .groupby(["Compound", "CircuitKey", "Era"])["TrackTemp"].mean()
        .rename("temp_ref")
    )
    curves = curves.merge(cell_temp, on=["Compound", "CircuitKey", "Era"], how="left")

    if len(sensitivity):
        sens = sensitivity[["Compound", "Era", "b_temp", "c_temp"]]
        curves = curves.merge(sens, on=["Compound", "Era"], how="left")
    else:
        curves["b_temp"] = np.nan
        curves["c_temp"] = np.nan

    # Fallbacks: cell with no historical temp → global mean; no sensitivity → 0.
    global_temp = float(stint_laps["TrackTemp"].mean()) if "TrackTemp" in stint_laps else 30.0
    curves["temp_ref"] = curves["temp_ref"].fillna(global_temp).round(2)
    curves["b_temp"] = curves["b_temp"].fillna(0.0)
    curves["c_temp"] = curves["c_temp"].fillna(0.0)
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
    track_temp: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Degradation delta (s, relative to fresh tyre age=1) with a 95% CI band.

    Looks up the cell row (already pool-resolved by fit_all_curves); falls back
    to the closest available row for the compound if the exact cell is missing.
    For wet compounds (INTERMEDIATE/WET) it never raises: if the table has no row
    for the compound at all, the built-in WET_FALLBACK curve is used so the caller
    always gets a finite, sensible band.

    If ``track_temp`` is given AND the curves carry Phase-9 temperature columns
    (temp_ref / b_temp / c_temp), the central slope/curvature are adjusted by
    (track_temp − temp_ref): hotter track → faster degradation. The CI band keeps
    the cell's fitted half-width (temperature shifts the centre, not the spread).
    With ``track_temp=None`` or pre-Phase-9 curves the result is identical to the
    original behaviour — fully backward compatible.

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
        if compound in WET_COMPOUNDS:
            # No row at all for a known wet compound — use the built-in fallback
            # so we never crash a (possibly wet) race mid-simulation.
            fb = WET_FALLBACK[compound]
            row = pd.Series({**fb, "temp_ref": np.nan, "b_temp": 0.0, "c_temp": 0.0})
            age = np.asarray(age, dtype=float)
            return (
                degradation_delta(fb["b"], fb["c"], age),
                degradation_delta(fb["b"] - fb["b_ci"], fb["c"] - fb["c_ci"], age),
                degradation_delta(fb["b"] + fb["b_ci"], fb["c"] + fb["c_ci"], age),
            )
        raise KeyError(f"No tyre curve for compound={compound}")

    row = sel.iloc[0]
    b, c = row["b"], row["c"]

    if track_temp is not None and {"temp_ref", "b_temp", "c_temp"} <= set(curves.columns):
        dT = float(track_temp) - float(row["temp_ref"])
        b = max(b + float(row["b_temp"]) * dT, 0.0)   # slope can't go negative
        c = c + float(row["c_temp"]) * dT

    age = np.asarray(age, dtype=float)
    mid = degradation_delta(b, c, age)
    lo  = degradation_delta(b - row["b_ci"], c - row["c_ci"], age)
    hi  = degradation_delta(b + row["b_ci"], c + row["c_ci"], age)
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    import mlflow
    from src.models.lap_time.train_tft_data import load_tft_data, MLFLOW_TRACKING_URI

    df = load_tft_data()      # per-round glob + dedup + StintID/TrackStatus carry-back
    stint_laps = prepare_stint_laps(df)
    logger.info(
        "Prepared %d green laps in %d stints (%d circuits, eras %s) | by compound: %s",
        len(stint_laps), stint_laps.groupby(STINT_KEYS).ngroups,
        stint_laps["CircuitKey"].nunique(), sorted(stint_laps["Era"].unique()),
        stint_laps["Compound"].value_counts().to_dict(),
    )

    curves = fit_all_curves(stint_laps)
    if curves.empty:
        raise RuntimeError("No tyre curves fitted — check input data")

    # Phase 9.4 — temperature sensitivity (cross-sectional, per compound × era).
    sensitivity = fit_temp_sensitivity(stint_laps)
    curves = attach_temp_sensitivity(curves, stint_laps, sensitivity)
    if len(sensitivity):
        sensitivity.to_csv(REPORTS_DIR / "tyre_temp_sensitivity.csv", index=False)
        logger.info("Saved tyre_temp_sensitivity.csv (%d compound×era pools)", len(sensitivity))

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
            "compounds": ",".join(ALL_COMPOUNDS),
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
        if len(sensitivity):
            mlflow.log_metrics({
                "temp_sens_pools": len(sensitivity),
                "mean_b_temp_per_C": float(sensitivity["b_temp"].mean()),
            })
            mlflow.log_artifact(str(REPORTS_DIR / "tyre_temp_sensitivity.csv"))
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
