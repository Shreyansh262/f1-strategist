"""
src/models/lap_time/train.py

Lap time predictor — two-stage training:
    Stage 1: Bayesian Ridge (baseline — must beat this before adding complexity)
    Stage 2: Random Forest with manual grid search

Both models are logged to MLflow. Final models are saved with joblib.
Season-aware splits from splits.py are used — no random shuffling.

Usage:
    python -m src.models.lap_time.train

MLflow UI:
    mlflow ui --port 5000
    then open http://localhost:5000
"""

import logging
import time
from pathlib import Path
from typing import Final

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from src.pipeline.features import MODEL_FEATURE_COLUMNS, build_features
from src.pipeline.splits import assert_no_leakage, make_splits
from src.pipeline.validate import validate_laps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI: Final[str] = (PROJECT_ROOT / "mlruns").as_uri()
EXPERIMENT_NAME: Final[str]     = "lap_time_predictor"

TARGET: Final[str] = "LapTimeSeconds"

# ---------------------------------------------------------------------------
# Grid search space — deliberately small for CPU training
# Full Optuna sweep deferred to v2 (see roadmap)
# ---------------------------------------------------------------------------
RF_PARAM_GRID: Final[list[dict]] = [
    {"n_estimators": 100, "max_depth": 6,    "min_samples_leaf": 5},
    {"n_estimators": 100, "max_depth": 10,   "min_samples_leaf": 5},
    {"n_estimators": 200, "max_depth": 10,   "min_samples_leaf": 3},
    {"n_estimators": 200, "max_depth": 15,   "min_samples_leaf": 3},
    {"n_estimators": 300, "max_depth": 15,   "min_samples_leaf": 2},
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_data(parquet_glob: str = "data/raw/*.parquet") -> pd.DataFrame:
    """
    Load all raw parquet files, validate, and build features.

    Parameters
    ----------
    parquet_glob : str
        Glob pattern relative to project root for raw lap parquet files.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame ready for splitting.
    """
    raw_dir = PROJECT_ROOT / "data" / "raw"
    parquet_files = list(raw_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {raw_dir}. "
            "Run src/pipeline/ingest.py first to fetch race data."
        )

    logger.info("Found %d parquet files in %s", len(parquet_files), raw_dir)

    dfs = []
    for fp in sorted(parquet_files):
        logger.info("Loading %s", fp.name)
        dfs.append(pd.read_parquet(fp))

    raw_df = pd.concat(dfs, ignore_index=True)
    logger.info("Raw data: %d rows from %d files", len(raw_df), len(parquet_files))

    validated_df = validate_laps(raw_df)
    features_df  = build_features(validated_df)
    return features_df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a features DataFrame into X (model inputs) and y (target)."""
    feature_cols = [c for c in MODEL_FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols]
    y = df[TARGET]
    return X, y


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def per_circuit_mae(
    model,
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> dict[str, float]:
    """
    Compute MAE separately for each circuit in df.

    Returns a dict mapping CircuitKey → MAE (seconds).
    Logged to MLflow as individual metrics.
    """
    results = {}
    for circuit, group in df.groupby("CircuitKey"):
        X, y = get_X_y(group)
        if scaler is not None:
            X = scaler.transform(X)
        preds = model.predict(X)
        results[str(circuit)] = float(mean_absolute_error(y, preds))
    return results


# ---------------------------------------------------------------------------
# Stage 1 — Bayesian Ridge baseline
# ---------------------------------------------------------------------------

def train_bayesian_ridge(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[BayesianRidge, StandardScaler, dict]:
    """
    Train Bayesian Ridge baseline.

    Bayesian Ridge requires feature scaling (it's a regularised linear model).
    Scaler is fit on train only — never on val/test.

    Returns
    -------
    model, scaler, metrics_dict
    """
    logger.info("--- Stage 1: Bayesian Ridge baseline ---")

    X_train, y_train = get_X_y(train_df)
    X_val,   y_val   = get_X_y(val_df)

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    model = BayesianRidge(max_iter=500)

    t0 = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - t0

    train_mae = mean_absolute_error(y_train, model.predict(X_train_scaled))
    val_mae   = mean_absolute_error(y_val,   model.predict(X_val_scaled))
    val_circuit_mae = per_circuit_mae(model, val_df, scaler)

    metrics = {
        "train_mae":    train_mae,
        "val_mae":      val_mae,
        "train_time_s": train_time,
        **{f"val_mae_{c}": v for c, v in val_circuit_mae.items()},
    }

    logger.info("Bayesian Ridge  train MAE: %.4f s  val MAE: %.4f s  (%.1f s)",
                train_mae, val_mae, train_time)

    return model, scaler, metrics


# ---------------------------------------------------------------------------
# Stage 2 — Random Forest grid search
# ---------------------------------------------------------------------------

def train_random_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    baseline_val_mae: float,
) -> tuple[RandomForestRegressor, dict, dict]:
    """
    Manual grid search over RF_PARAM_GRID.
    Each configuration is logged as a separate MLflow child run.

    Returns best model, its params, and its metrics.
    Raises ValueError if no configuration beats the baseline.
    """
    logger.info("--- Stage 2: Random Forest grid search (%d configs) ---",
                len(RF_PARAM_GRID))

    X_train, y_train = get_X_y(train_df)
    X_val,   y_val   = get_X_y(val_df)

    best_val_mae    = float("inf")
    best_model      = None
    best_params     = None
    best_metrics    = None

    for i, params in enumerate(RF_PARAM_GRID):
        logger.info("Config %d/%d: %s", i + 1, len(RF_PARAM_GRID), params)

        with mlflow.start_run(run_name=f"rf_config_{i+1}", nested=True):
            mlflow.log_params(params)

            model = RandomForestRegressor(
                **params,
                n_jobs=-1,       # use all CPU cores
                random_state=42,
            )

            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            train_mae = mean_absolute_error(y_train, model.predict(X_train))
            val_mae   = mean_absolute_error(y_val,   model.predict(X_val))
            circuit_mae = per_circuit_mae(model, val_df)

            metrics = {
                "train_mae":    train_mae,
                "val_mae":      val_mae,
                "train_time_s": train_time,
                **{f"val_mae_{c}": v for c, v in circuit_mae.items()},
            }
            mlflow.log_metrics({k: v for k, v in metrics.items()
                                 if isinstance(v, float)})

            logger.info("  train MAE: %.4f s  val MAE: %.4f s  (%.1f s)",
                        train_mae, val_mae, train_time)

            if val_mae < best_val_mae:
                best_val_mae  = val_mae
                best_model    = model
                best_params   = params
                best_metrics  = metrics

    if best_model is None:
        raise ValueError(
            f"No RF configuration (best val MAE: {best_val_mae:.4f}s) "
            f"beat the Bayesian Ridge baseline ({baseline_val_mae:.4f}s). "
            "Check feature engineering before adding model complexity."
        )

    logger.info("Best RF config: %s  val MAE: %.4f s  (vs baseline %.4f s)",
                best_params, best_val_mae, baseline_val_mae)

    return best_model, best_params, best_metrics


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(parquet_glob: str = "data/raw/*.parquet") -> None:
    """
    Full training pipeline:
        1. Load + validate + build features
        2. Season-aware splits
        3. Bayesian Ridge baseline (logged to MLflow)
        4. RF grid search (each config logged as child run)
        5. Save best models with joblib
    """
    # ---- data --------------------------------------------------------------
    features_df = load_data(parquet_glob)

    train_df, val_df, test_df = make_splits(
        features_df,
        train_seasons=[2022, 2023],
        val_seasons=[2024],
        test_seasons=[2025],
    )
    assert_no_leakage(train_df, val_df, test_df)

    if len(train_df) == 0:
        raise ValueError("Training set is empty — fetch 2022/2023 data first.")
    if len(val_df) == 0:
        logger.warning("Validation set is empty — fetch 2024 data to enable proper eval.")

    # ---- MLflow setup ------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ---- Stage 1: Bayesian Ridge ------------------------------------------
    with mlflow.start_run(run_name="bayesian_ridge_baseline") as br_run:
        mlflow.log_param("model_type", "BayesianRidge")
        mlflow.log_param("train_seasons", [2022, 2023])
        mlflow.log_param("val_seasons",   [2024])
        mlflow.log_param("n_train_laps",  len(train_df))
        mlflow.log_param("n_val_laps",    len(val_df))
        mlflow.log_param("features",      MODEL_FEATURE_COLUMNS)

        br_model, br_scaler, br_metrics = train_bayesian_ridge(train_df, val_df)

        mlflow.log_metrics({k: v for k, v in br_metrics.items()
                            if isinstance(v, float)})
        mlflow.sklearn.log_model(br_model, "bayesian_ridge")

        baseline_val_mae = br_metrics["val_mae"]

    # ---- Stage 2: Random Forest -------------------------------------------
    with mlflow.start_run(run_name="random_forest_grid_search") as rf_run:
        mlflow.log_param("model_type",    "RandomForestRegressor")
        mlflow.log_param("train_seasons", [2022, 2023])
        mlflow.log_param("val_seasons",   [2024])
        mlflow.log_param("n_train_laps",  len(train_df))
        mlflow.log_param("n_val_laps",    len(val_df))
        mlflow.log_param("features",      MODEL_FEATURE_COLUMNS)
        mlflow.log_param("grid_size",     len(RF_PARAM_GRID))

        rf_model, rf_params, rf_metrics = train_random_forest(
            train_df, val_df, baseline_val_mae
        )

        mlflow.log_params(rf_params)
        mlflow.log_metrics({k: v for k, v in rf_metrics.items()
                            if isinstance(v, float)})
        mlflow.sklearn.log_model(rf_model, "random_forest")

    # ---- Save models -------------------------------------------------------
    br_path = MODELS_DIR / "bayesian_ridge_lap.joblib"
    rf_path = MODELS_DIR / "rf_lap.joblib"
    sc_path = MODELS_DIR / "scaler_lap.joblib"

    joblib.dump(br_model,  br_path)
    joblib.dump(rf_model,  rf_path)
    joblib.dump(br_scaler, sc_path)

    logger.info("Saved Bayesian Ridge → %s", br_path)
    logger.info("Saved Random Forest  → %s", rf_path)
    logger.info("Saved Scaler         → %s", sc_path)

    # ---- Summary -----------------------------------------------------------
    improvement = baseline_val_mae - rf_metrics["val_mae"]
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("  Baseline (Bayesian Ridge) val MAE : %.4f s", baseline_val_mae)
    logger.info("  Best RF val MAE             : %.4f s", rf_metrics["val_mae"])
    logger.info("  Improvement over baseline   : %.4f s", improvement)
    logger.info("  Best RF params              : %s", rf_params)
    logger.info("  Models saved to             : %s", MODELS_DIR)
    logger.info("=" * 60)
    logger.info("Run: mlflow ui --port 5000  to inspect all runs")


if __name__ == "__main__":
    train()