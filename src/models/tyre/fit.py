"""
src/models/tyre/fit.py

Tyre degradation model — Week 5.

Approach: scipy.optimize.curve_fit per (compound × circuit × era).
Model:    lap_time_delta = b * tyre_age + c * tyre_age²
          (degradation relative to a fresh tyre baseline)

Outputs per (compound, circuit, era):
    - coefficients (b, c)
    - 95% confidence interval on each coefficient
    - r² goodness of fit
    - n_stints used for fitting

Key design decisions:
    - Physics-informed: tyre degradation is genuinely polynomial
    - Interpretable: b=linear degradation rate, c=cliff acceleration
    - Low-data aware: wide CIs when n_stints < 5 — no false precision
    - Era-aware: separate curves for 2022-2025 vs 2026+

Entry point:
    python -m src.models.tyre.fit

Saves: models/tyre_curves.joblib
"""

import logging
from pathlib import Path
from typing import Final

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist

from src.pipeline.features import add_stint_id, build_features
from src.pipeline.validate import validate_laps

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

# Minimum stints required to fit a curve reliably
MIN_STINTS_FOR_FIT: Final[int] = 3
# Minimum laps per stint to include it
MIN_LAPS_PER_STINT: Final[int] = 4
# Confidence level for CIs
CI_LEVEL: Final[float] = 0.95


# ---------------------------------------------------------------------------
# Degradation model
# ---------------------------------------------------------------------------

def degradation_model(tyre_age: np.ndarray, b: float, c: float) -> np.ndarray:
    """
    Tyre degradation relative to fresh tyre (tyre_age=1).
    delta_lap_time = b * tyre_age + c * tyre_age²

    b > 0: linear degradation rate (seconds per lap)
    c > 0: quadratic acceleration (cliff effect on worn tyres)
    """
    return b * tyre_age + c * tyre_age ** 2


def fit_degradation_curve(
    stint_df: pd.DataFrame,
) -> dict | None:
    """
    Fit degradation curve for a single (compound, circuit, era) group.

    Parameters
    ----------
    stint_df : pd.DataFrame
        Laps from continuous stints for one (compound, circuit, era).
        Must contain TyreLife and LapTimeSeconds.

    Returns
    -------
    dict with keys: b, c, b_ci, c_ci, r2, n_stints, n_laps
    Returns None if insufficient data or fit fails.
    """
    # Use only stints with enough laps
    valid_stints = (
        stint_df.groupby(["Season", "RoundNumber", "Driver", "StintID"])
        .filter(lambda g: len(g) >= MIN_LAPS_PER_STINT)
    )

    n_stints = valid_stints.groupby(
        ["Season", "RoundNumber", "Driver", "StintID"]
    ).ngroups

    if n_stints < MIN_STINTS_FOR_FIT:
        logger.debug(
            "Insufficient stints (%d < %d) — skipping", n_stints, MIN_STINTS_FOR_FIT
        )
        return None

    # Normalise: subtract per-stint baseline (lap time at tyre_age=1 or earliest lap)
    rows = []
    for _, stint in valid_stints.groupby(
        ["Season", "RoundNumber", "Driver", "StintID"]
    ):
        stint = stint.sort_values("TyreLife")
        baseline = stint["LapTimeSeconds"].iloc[0]
        for _, row in stint.iterrows():
            rows.append({
                "TyreAge": row["TyreLife"],
                "Delta":   row["LapTimeSeconds"] - baseline,
            })

    fit_df = pd.DataFrame(rows)
    x = fit_df["TyreAge"].values.astype(float)
    y = fit_df["Delta"].values.astype(float)

    try:
        popt, pcov = curve_fit(
            degradation_model, x, y,
            p0=[0.05, 0.001],             # sensible starting point
            bounds=([-1.0, -0.1], [2.0, 0.5]),
            maxfev=5000,
        )
    except (RuntimeError, ValueError) as e:
        logger.warning("curve_fit failed: %s", e)
        return None

    b, c = popt
    perr = np.sqrt(np.diag(pcov))         # standard errors

    # 95% CI using t-distribution
    n_pts   = len(x)
    n_params = 2
    dof     = max(1, n_pts - n_params)
    t_crit  = t_dist.ppf((1 + CI_LEVEL) / 2, dof)
    b_ci    = float(t_crit * perr[0])
    c_ci    = float(t_crit * perr[1])

    # R²
    y_pred  = degradation_model(x, b, c)
    ss_res  = np.sum((y - y_pred) ** 2)
    ss_tot  = np.sum((y - y.mean()) ** 2)
    r2      = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "b":        float(b),
        "c":        float(c),
        "b_ci":     b_ci,
        "c_ci":     c_ci,
        "r2":       round(r2, 4),
        "n_stints": n_stints,
        "n_laps":   len(fit_df),
    }


# ---------------------------------------------------------------------------
# Batch fitting
# ---------------------------------------------------------------------------

def fit_all_curves(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit degradation curves for all (compound, circuit, era) combinations.

    Parameters
    ----------
    features_df : pd.DataFrame
        Full features DataFrame from build_features() — must include StintID.

    Returns
    -------
    pd.DataFrame with one row per (compound, circuit, era), columns:
        Compound, CircuitKey, Era, b, c, b_ci, c_ci, r2, n_stints, n_laps
    """
    # Ensure StintID is present
    if "StintID" not in features_df.columns:
        logger.info("Adding StintID...")
        features_df = add_stint_id(features_df)

    # Filter out pit laps and short stints
    df = features_df[features_df["TyreLife"] > 1].copy()

    results = []
    groups = df.groupby(["Compound", "CircuitKey", "Era"])
    total  = len(groups)

    logger.info("Fitting degradation curves for %d (compound × circuit × era) groups", total)

    for (compound, circuit, era), group in groups:
        result = fit_degradation_curve(group)

        if result is None:
            logger.debug(
                "No fit: %s × %s × Era%d (n_laps=%d)",
                compound, circuit, era, len(group),
            )
            continue

        results.append({
            "Compound":   compound,
            "CircuitKey": circuit,
            "Era":        era,
            **result,
        })
        logger.info(
            "  %-12s %-35s Era%d | b=%.4f±%.4f  c=%.5f±%.5f  r²=%.3f  n=%d stints",
            compound, circuit, era,
            result["b"], result["b_ci"],
            result["c"], result["c_ci"],
            result["r2"], result["n_stints"],
        )

    curves_df = pd.DataFrame(results)
    logger.info(
        "Fitted %d/%d curves successfully", len(curves_df), total
    )
    return curves_df


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_degradation_curves(
    curves_df: pd.DataFrame,
    features_df: pd.DataFrame,
    circuits: list[str] | None = None,
    compounds: list[str] | None = None,
) -> None:
    """
    Plot fitted degradation curves vs actual data for selected circuits.
    Saved to reports/tyre/degradation_curves.png

    Default: top 4 circuits by data volume, SOFT + MEDIUM compounds.
    """
    if circuits is None:
        top_circuits = (
            features_df.groupby("CircuitKey").size()
            .sort_values(ascending=False)
            .head(4)
            .index.tolist()
        )
        circuits = top_circuits

    if compounds is None:
        compounds = ["SOFT", "MEDIUM", "HARD"]

    if "StintID" not in features_df.columns:
        features_df = add_stint_id(features_df)

    fig, axes = plt.subplots(
        len(circuits), len(compounds),
        figsize=(5 * len(compounds), 4 * len(circuits)),
        squeeze=False,
    )

    tyre_age_range = np.linspace(1, 40, 100)

    for row_idx, circuit in enumerate(circuits):
        for col_idx, compound in enumerate(compounds):
            ax = axes[row_idx][col_idx]

            # Actual data (Era 0 only for readability)
            actual = features_df[
                (features_df["CircuitKey"] == circuit) &
                (features_df["Compound"] == compound) &
                (features_df["Era"] == 0)
            ]

            if len(actual) > 0:
                # Plot normalised deltas per stint
                for _, stint in actual.groupby(
                    ["Season", "RoundNumber", "Driver", "StintID"]
                    if "StintID" in actual.columns
                    else ["Season", "RoundNumber", "Driver"]
                ):
                    stint = stint.sort_values("TyreLife")
                    if len(stint) < MIN_LAPS_PER_STINT:
                        continue
                    baseline = stint["LapTimeSeconds"].iloc[0]
                    ax.plot(
                        stint["TyreLife"],
                        stint["LapTimeSeconds"] - baseline,
                        alpha=0.2, color="steelblue", linewidth=0.8,
                    )

            # Fitted curve + CI band
            curve_row = curves_df[
                (curves_df["CircuitKey"] == circuit) &
                (curves_df["Compound"] == compound) &
                (curves_df["Era"] == 0)
            ]

            if len(curve_row) > 0:
                row = curve_row.iloc[0]
                y_fit  = degradation_model(tyre_age_range, row["b"], row["c"])
                y_upper = degradation_model(tyre_age_range, row["b"] + row["b_ci"], row["c"] + row["c_ci"])
                y_lower = degradation_model(tyre_age_range, row["b"] - row["b_ci"], row["c"] - row["c_ci"])

                ax.plot(tyre_age_range, y_fit, color="#E8002D", linewidth=2,
                        label=f"fit (r²={row['r2']:.2f})")
                ax.fill_between(tyre_age_range, y_lower, y_upper,
                                alpha=0.25, color="#E8002D", label="95% CI")
                ax.legend(fontsize=7)

            circuit_short = circuit.replace(" Grand Prix", "")
            ax.set_title(f"{circuit_short} — {compound}", fontsize=9)
            ax.set_xlabel("Tyre Age (laps)", fontsize=8)
            ax.set_ylabel("Δ Lap Time (s)", fontsize=8)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.grid(True, alpha=0.3)

    plt.suptitle("Tyre Degradation Curves — Fitted vs Actual (Era 0: 2022-2025)", fontsize=12)
    plt.tight_layout()

    out_path = REPORTS_DIR / "degradation_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Degradation curves saved → %s", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def fit() -> pd.DataFrame:
    """Load data, fit all curves, save results."""

    raw_dir = PROJECT_ROOT / "data" / "raw"
    parquet_files = list(raw_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files in {raw_dir}. Run ingest.py first."
        )

    logger.info("Loading %d parquet files...", len(parquet_files))
    dfs = [pd.read_parquet(f) for f in sorted(parquet_files)]
    raw_df = pd.concat(dfs, ignore_index=True)

    validated_df = validate_laps(raw_df)
    features_df  = build_features(validated_df)
    features_df  = add_stint_id(features_df)

    logger.info(
        "Data loaded: %d laps | %d circuits | seasons %s",
        len(features_df),
        features_df["CircuitKey"].nunique(),
        sorted(features_df["Season"].unique()),
    )

    # Fit curves
    curves_df = fit_all_curves(features_df)

    # Save curves
    curves_path = MODELS_DIR / "tyre_curves.joblib"
    joblib.dump(curves_df, curves_path)
    logger.info("Tyre curves saved → %s", curves_path)

    # Also save as CSV for inspection
    csv_path = REPORTS_DIR / "tyre_curves.csv"
    curves_df.to_csv(csv_path, index=False)
    logger.info("Tyre curves CSV  → %s", csv_path)

    # Diagnostic plots
    plot_degradation_curves(curves_df, features_df)

    # Summary stats
    logger.info("=" * 60)
    logger.info("TYRE MODEL COMPLETE")
    logger.info("  Curves fitted: %d", len(curves_df))
    logger.info(
        "  Mean r²:       %.3f",
        curves_df["r2"].mean() if len(curves_df) > 0 else float("nan"),
    )
    logger.info(
        "  Circuits covered: %d",
        curves_df["CircuitKey"].nunique() if len(curves_df) > 0 else 0,
    )
    logger.info(
        "  Compounds covered: %s",
        sorted(curves_df["Compound"].unique().tolist()) if len(curves_df) > 0 else [],
    )
    logger.info("=" * 60)

    return curves_df


if __name__ == "__main__":
    fit()