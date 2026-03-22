"""
src/models/lap_time/evaluate.py

Evaluation for the trained lap time predictor.
Generates three sets of outputs, all saved to reports/lap_time/:

    1. per_circuit_mae.csv     — MAE per circuit for both models
    2. learning_curve.png      — train vs val MAE as training size grows
    3. shap_summary.png        — SHAP feature importance for RF model

Run after train.py has completed:
    python -m src.models.lap_time.evaluate

All plots are also logged to the active MLflow run if one is open.
"""

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import learning_curve

from src.pipeline.features import MODEL_FEATURE_COLUMNS, build_features
from src.pipeline.splits import make_splits
from src.pipeline.validate import validate_laps
from src.models.lap_time.train import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    MODELS_DIR,
    PROJECT_ROOT,
    TARGET,
    get_X_y,
    load_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports" / "lap_time"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Per-circuit MAE table
# ---------------------------------------------------------------------------

def evaluate_per_circuit(
    br_model: BayesianRidge,
    br_scaler,
    rf_model: RandomForestRegressor,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute MAE for both models on both val and test sets, broken down by circuit.

    Returns a DataFrame with columns:
        CircuitKey | split | bayesian_ridge_mae | random_forest_mae
    """
    rows = []

    for split_name, df in [("val_2024", val_df), ("test_2025", test_df)]:
        if len(df) == 0:
            logger.warning("Skipping %s — empty DataFrame", split_name)
            continue

        for circuit, group in df.groupby("CircuitKey"):
            X, y = get_X_y(group)

            X_scaled = br_scaler.transform(X)
            br_mae   = mean_absolute_error(y, br_model.predict(X_scaled))
            rf_mae   = mean_absolute_error(y, rf_model.predict(X))

            rows.append({
                "CircuitKey":         circuit,
                "split":              split_name,
                "bayesian_ridge_mae": round(br_mae, 4),
                "random_forest_mae":  round(rf_mae, 4),
                "n_laps":             len(group),
            })

    results_df = pd.DataFrame(rows).sort_values(["split", "random_forest_mae"])

    out_path = REPORTS_DIR / "per_circuit_mae.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("Per-circuit MAE saved to %s", out_path)

    # Pretty print
    logger.info("\n%s", results_df.to_string(index=False))

    return results_df


# ---------------------------------------------------------------------------
# 2. Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(
    rf_model: RandomForestRegressor,
    train_df: pd.DataFrame,
) -> None:
    """
    Plot training MAE vs validation MAE as training set size increases.
    Uses sklearn's learning_curve with 5 train-size steps.

    Saved to reports/lap_time/learning_curve.png
    """
    logger.info("Computing learning curves (this may take ~2 min on CPU)...")

    X_train, y_train = get_X_y(train_df)

    # Use a fresh RF with best params (can't reuse fitted model for learning_curve)
    from sklearn.base import clone
    model_clone = clone(rf_model)

    train_sizes, train_scores, val_scores = learning_curve(
        model_clone,
        X_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="neg_mean_absolute_error",
        cv=3,                   # 3-fold — small enough to run on CPU
        n_jobs=-1,
        verbose=0,
    )

    # Convert neg MAE → MAE
    train_mae = -train_scores.mean(axis=1)
    val_mae   = -val_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_std   = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mae, "o-", color="#E8002D", label="Train MAE")
    ax.fill_between(train_sizes,
                    train_mae - train_std, train_mae + train_std,
                    alpha=0.15, color="#E8002D")
    ax.plot(train_sizes, val_mae, "o-", color="#1565C0", label="Val MAE (CV)")
    ax.fill_between(train_sizes,
                    val_mae - val_std, val_mae + val_std,
                    alpha=0.15, color="#1565C0")

    ax.set_xlabel("Training set size (laps)")
    ax.set_ylabel("MAE (seconds)")
    ax.set_title("Learning Curve — Random Forest Lap Time Predictor")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = REPORTS_DIR / "learning_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Learning curve saved to %s", out_path)

    try:
        mlflow.log_artifact(str(out_path))
    except Exception:
        pass  # fine if no active MLflow run


# ---------------------------------------------------------------------------
# 3. SHAP feature importance
# ---------------------------------------------------------------------------

def plot_shap_summary(
    rf_model: RandomForestRegressor,
    val_df: pd.DataFrame,
    max_display: int = 11,
    sample_n: int = 500,
) -> None:
    """
    Generate SHAP summary plot for the Random Forest model.

    Uses a random sample of val_df for speed — SHAP on full RF is slow on CPU.
    sample_n=500 gives a stable importance ranking in ~30s.

    Saved to reports/lap_time/shap_summary.png

    Interview point: SHAP values show *direction* of feature effect, not just
    magnitude. A positive SHAP for FuelLoad means "high fuel → slower lap",
    which matches physical intuition — good sanity check.
    """
    logger.info("Computing SHAP values (sample n=%d)...", sample_n)

    X_val, _ = get_X_y(val_df)
    feature_cols = [c for c in MODEL_FEATURE_COLUMNS if c in val_df.columns]

    # Sample for speed
    if len(X_val) > sample_n:
        X_sample = X_val.sample(n=sample_n, random_state=42)
    else:
        X_sample = X_val

    explainer   = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_cols,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Feature Importance — Random Forest Lap Time Predictor")
    plt.tight_layout()

    out_path = REPORTS_DIR / "shap_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("SHAP summary saved to %s", out_path)

    # Log mean absolute SHAP per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature":    feature_cols,
        "mean_abs_shap": mean_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    logger.info("\nFeature importance (mean |SHAP|):\n%s", shap_df.to_string(index=False))

    shap_csv = REPORTS_DIR / "shap_importance.csv"
    shap_df.to_csv(shap_csv, index=False)

    try:
        mlflow.log_artifact(str(out_path))
        mlflow.log_artifact(str(shap_csv))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate() -> None:
    """Load trained models and run all evaluation steps."""

    # ---- load models -------------------------------------------------------
    br_path = MODELS_DIR / "bayesian_ridge_lap.joblib"
    rf_path = MODELS_DIR / "rf_lap.joblib"
    sc_path = MODELS_DIR / "scaler_lap.joblib"

    for p in [br_path, rf_path, sc_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run train.py first:\n"
                "  python -m src.models.lap_time.train"
            )

    br_model  = joblib.load(br_path)
    rf_model  = joblib.load(rf_path)
    br_scaler = joblib.load(sc_path)
    logger.info("Models loaded from %s", MODELS_DIR)

    # ---- load data ---------------------------------------------------------
    features_df = load_data()
    train_df, val_df, test_df = make_splits(features_df)

    # ---- run evaluations ---------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="evaluation"):

        # 1. Per-circuit MAE
        results_df = evaluate_per_circuit(
            br_model, br_scaler, rf_model, val_df, test_df
        )
        mlflow.log_artifact(str(REPORTS_DIR / "per_circuit_mae.csv"))

        # 2. Learning curves (on train set — cross-validated)
        plot_learning_curves(rf_model, train_df)

        # 3. SHAP — only if val data is available
        if len(val_df) > 0:
            plot_shap_summary(rf_model, val_df)
        else:
            logger.warning("No val data — skipping SHAP. Fetch 2023 data first.")

    logger.info("Evaluation complete. Reports in %s", REPORTS_DIR)
    logger.info("Run: mlflow ui --port 5000  to view all runs")


if __name__ == "__main__":
    evaluate()