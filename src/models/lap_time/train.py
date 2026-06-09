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
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    {"n_estimators": 200, "max_depth": 15,   "min_samples_leaf": 3},
    {"n_estimators": 200, "max_depth": 20,   "min_samples_leaf": 2},
    {"n_estimators": 300, "max_depth": 20,   "min_samples_leaf": 2},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},  # fully grown
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5},  # grown but regularised
]

# ---------------------------------------------------------------------------
# Categorical features
#   Linear baseline : one-hot encoded (see build_linear_pipeline)
#   LightGBM        : native categoricals; -1 sentinel -> treated as missing
#   RF              : raw integer codes (unchanged)
# ---------------------------------------------------------------------------
CAT_FEATURES: Final[list[str]] = [
    "CircuitEncoded", "TeamEncoded", "CompoundEncoded", "Era",
]
CONT_FEATURES: Final[list[str]] = [
    c for c in MODEL_FEATURE_COLUMNS if c not in CAT_FEATURES
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
    # Per-round files ONLY. The laps_YYYY_full.parquet aggregates are a superset
    # of the per-round files — globbing "*.parquet" loaded every lap twice and
    # doubled the dataset. Match laps_*_r*.parquet so the _full files are never
    # picked up by the trainer.
    parquet_files = list(raw_dir.glob("laps_*_r*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No per-round parquet files (laps_*_r*.parquet) found in {raw_dir}. "
            "Run src/pipeline/ingest.py first to fetch race data."
        )

    logger.info("Found %d per-round parquet files in %s", len(parquet_files), raw_dir)

    dfs = []
    for fp in sorted(parquet_files):
        logger.info("Loading %s", fp.name)
        dfs.append(pd.read_parquet(fp))

    raw_df = pd.concat(dfs, ignore_index=True)
    # Guard against any residual duplication (e.g. a stray re-ingest of a round).
    n_before = len(raw_df)
    raw_df = raw_df.drop_duplicates(
        subset=["Season", "RoundNumber", "Driver", "LapNumber"]
    ).reset_index(drop=True)
    n_dups = n_before - len(raw_df)
    if n_dups > 0:
        logger.warning("Dropped %d duplicate laps after concat", n_dups)
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

def per_circuit_mae(model, df: pd.DataFrame) -> dict[str, float]:
    """
    Compute MAE separately for each circuit in df.

    `model` must accept the raw MODEL_FEATURE_COLUMNS DataFrame directly — the
    linear baseline is a Pipeline (encoder+scaler bundled in), RF/LGBM use raw
    integer codes. Returns a dict mapping CircuitKey → MAE (seconds).
    """
    results = {}
    for circuit, group in df.groupby("CircuitKey"):
        X, y = get_X_y(group)
        preds = model.predict(X)
        results[str(circuit)] = float(mean_absolute_error(y, preds))
    return results


# ---------------------------------------------------------------------------
# Stage 1 — Bayesian Ridge baseline
# ---------------------------------------------------------------------------

def build_linear_pipeline() -> Pipeline:
    """
    Linear baseline = one-hot(categorical codes) + scale(continuous) → BayesianRidge.

    The categorical codes (CircuitEncoded, TeamEncoded, CompoundEncoded, Era) are
    arbitrary integer labels — treating them as continuous is wrong. OneHotEncoder
    makes them proper indicator columns; handle_unknown='ignore' maps unseen codes
    (e.g. the -1 sentinel for an unseen circuit/team) to an all-zero row. The
    remaining continuous features are standardised. Everything is bundled in one
    Pipeline so the encoder+scaler are fit on train only and reused for val/test —
    no leakage, and no separate scaler artefact to keep in sync.
    """
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
            ("num", StandardScaler(), CONT_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("model", BayesianRidge(max_iter=500))])


def train_bayesian_ridge(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[Pipeline, dict]:
    """
    Train Bayesian Ridge baseline as a self-contained sklearn Pipeline.

    The pipeline (OneHotEncoder + StandardScaler + BayesianRidge) is fit on train
    only; val/test are transformed through the same fitted pipeline.

    Returns
    -------
    pipeline, metrics_dict
    """
    logger.info("--- Stage 1: Bayesian Ridge baseline ---")

    X_train, y_train = get_X_y(train_df)
    X_val,   y_val   = get_X_y(val_df)

    pipeline = build_linear_pipeline()

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    train_mae = mean_absolute_error(y_train, pipeline.predict(X_train))
    val_mae   = mean_absolute_error(y_val,   pipeline.predict(X_val))
    val_circuit_mae = per_circuit_mae(pipeline, val_df)

    metrics = {
        "train_mae":    train_mae,
        "val_mae":      val_mae,
        "train_time_s": train_time,
        **{f"val_mae_{c}": v for c, v in val_circuit_mae.items()},
    }

    logger.info("Bayesian Ridge  train MAE: %.4f s  val MAE: %.4f s  (%.1f s)",
                train_mae, val_mae, train_time)

    return pipeline, metrics


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
# Stage 3 — LightGBM with Optuna hyperparameter search
# ---------------------------------------------------------------------------

def _lgbm_objective(trial, X_tr, y_tr, X_val, y_val):
    params = {
        "objective": "regression_l1", "metric": "mae", "verbosity": -1, "n_jobs": -1,
        "n_estimators": 2000,
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0), "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="mae",
          categorical_feature=CAT_FEATURES, callbacks=[lgb.early_stopping(50, verbose=False)])
    return mean_absolute_error(y_val, m.predict(X_val))


def train_lightgbm(X_tr, y_tr, X_val, y_val, n_trials=60):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: _lgbm_objective(t, X_tr, y_tr, X_val, y_val),
                   n_trials=n_trials, show_progress_bar=True)
    best = {**study.best_params, "objective": "regression_l1", "metric": "mae",
            "n_jobs": -1, "verbosity": -1, "n_estimators": 2000}
    model = lgb.LGBMRegressor(**best)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="mae",
              categorical_feature=CAT_FEATURES, callbacks=[lgb.early_stopping(50, verbose=False)])
    return model, study


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
        train_seasons=[2022, 2023, 2024],
        val_seasons=[2025],
        test_seasons=[2026],
    )
    assert_no_leakage(train_df, val_df, test_df)

    if len(train_df) == 0:
        raise ValueError("Training set is empty — fetch 2022/2023 data first.")
    if len(val_df) == 0:
        logger.warning("Validation set is empty — fetch 2024 data to enable proper eval.")

    # ---- MLflow setup ------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ---- test-set helper (held-out 2026) ----------------------------------
    has_test = len(test_df) > 0
    if not has_test:
        logger.warning("Test set is empty — test MAE will be reported as nan.")
    X_test, y_test = get_X_y(test_df) if has_test else (None, None)

    def _test_mae(model) -> float:
        if not has_test:
            return float("nan")
        return float(mean_absolute_error(y_test, model.predict(X_test)))

    # ---- Stage 1: Bayesian Ridge ------------------------------------------
    with mlflow.start_run(run_name="bayesian_ridge_baseline") as br_run:
        mlflow.log_param("model_type", "BayesianRidge")
        mlflow.log_param("train_seasons", [2022, 2023, 2024])
        mlflow.log_param("val_seasons",   [2025])
        mlflow.log_param("n_train_laps",  len(train_df))
        mlflow.log_param("n_val_laps",    len(val_df))
        mlflow.log_param("features",      MODEL_FEATURE_COLUMNS)

        br_model, br_metrics = train_bayesian_ridge(train_df, val_df)
        br_metrics["test_mae"] = _test_mae(br_model)
        br_test_mae = br_metrics["test_mae"]

        mlflow.log_metrics({k: v for k, v in br_metrics.items()
                            if isinstance(v, float)})
        mlflow.sklearn.log_model(br_model, "bayesian_ridge")

        baseline_val_mae = br_metrics["val_mae"]

    # ---- Stage 2: Random Forest -------------------------------------------
    with mlflow.start_run(run_name="random_forest_grid_search") as rf_run:
        mlflow.log_param("model_type",    "RandomForestRegressor")
        mlflow.log_param("train_seasons", [2022, 2023, 2024])
        mlflow.log_param("val_seasons",   [2025])
        mlflow.log_param("n_train_laps",  len(train_df))
        mlflow.log_param("n_val_laps",    len(val_df))
        mlflow.log_param("features",      MODEL_FEATURE_COLUMNS)
        mlflow.log_param("grid_size",     len(RF_PARAM_GRID))

        rf_model, rf_params, rf_metrics = train_random_forest(
            train_df, val_df, baseline_val_mae
        )
        rf_metrics["test_mae"] = _test_mae(rf_model)

        mlflow.log_params(rf_params)
        mlflow.log_metrics({k: v for k, v in rf_metrics.items()
                            if isinstance(v, float)})
        mlflow.sklearn.log_model(rf_model, "random_forest")

    # ---- Stage 3: LightGBM + Optuna ---------------------------------------
    with mlflow.start_run(run_name="lightgbm_optuna") as lgbm_run:
        mlflow.log_param("model_type",    "LGBMRegressor")
        mlflow.log_param("train_seasons", [2022, 2023, 2024])
        mlflow.log_param("val_seasons",   [2025])
        mlflow.log_param("n_train_laps",  len(train_df))
        mlflow.log_param("n_val_laps",    len(val_df))
        mlflow.log_param("features",      MODEL_FEATURE_COLUMNS)
        mlflow.log_param("cat_features",  CAT_FEATURES)

        X_tr, y_tr   = get_X_y(train_df)
        X_val, y_val = get_X_y(val_df)

        logger.info("--- Stage 3: LightGBM + Optuna search ---")
        lgbm_model, lgbm_study = train_lightgbm(X_tr, y_tr, X_val, y_val, n_trials=60)

        lgbm_val_mae  = float(mean_absolute_error(y_val, lgbm_model.predict(X_val)))
        lgbm_test_mae = _test_mae(lgbm_model)
        lgbm_metrics = {
            "train_mae": float(mean_absolute_error(y_tr, lgbm_model.predict(X_tr))),
            "val_mae":   lgbm_val_mae,
            "test_mae":  lgbm_test_mae,
            **{f"val_mae_{c}": v for c, v in per_circuit_mae(lgbm_model, val_df).items()},
        }

        mlflow.log_params(lgbm_study.best_params)
        mlflow.log_metrics({k: v for k, v in lgbm_metrics.items()
                            if isinstance(v, float)})
        mlflow.sklearn.log_model(lgbm_model, "lightgbm")

        logger.info("LightGBM  best params: %s", lgbm_study.best_params)
        logger.info("LightGBM  val MAE: %.4f s  test MAE: %.4f s",
                    lgbm_val_mae, lgbm_test_mae)

    # ---- Safety check + model selection -----------------------------------
    # No candidate (RF / LGBM) may be selected unless it beats the linear baseline.
    candidates = {
        "RandomForest": (rf_model,   rf_metrics["val_mae"]),
        "LightGBM":     (lgbm_model, lgbm_metrics["val_mae"]),
    }
    best_candidate_val = min(v for _, v in candidates.values())
    if best_candidate_val >= baseline_val_mae:
        raise ValueError(
            f"No candidate beat the Bayesian Ridge baseline "
            f"(val MAE {baseline_val_mae:.4f}s). "
            f"Best candidate val MAE: {best_candidate_val:.4f}s. "
            "Check feature engineering before shipping a more complex model."
        )

    # Chosen = lowest val MAE across all three models.
    all_models = {
        "BayesianRidge": (br_model,   baseline_val_mae),
        **candidates,
    }
    chosen_name = min(all_models, key=lambda k: all_models[k][1])
    chosen_model, chosen_val_mae = all_models[chosen_name]
    logger.info("Chosen lap model: %s (val MAE %.4f s)", chosen_name, chosen_val_mae)

    # ---- Save models -------------------------------------------------------
    # scaler_lap.joblib is gone — the linear pipeline now bundles encoder+scaler.
    br_path     = MODELS_DIR / "bayesian_ridge_lap.joblib"
    rf_path     = MODELS_DIR / "rf_lap.joblib"
    lgbm_path   = MODELS_DIR / "lgbm_lap.joblib"
    chosen_path = MODELS_DIR / "chosen_lap.joblib"

    joblib.dump(br_model,     br_path)
    joblib.dump(rf_model,     rf_path)
    joblib.dump(lgbm_model,   lgbm_path)
    joblib.dump(chosen_model, chosen_path)

    # Drop the now-redundant scaler artefact if it lingers from a prior run.
    stale_scaler = MODELS_DIR / "scaler_lap.joblib"
    if stale_scaler.exists():
        stale_scaler.unlink()
        logger.info("Removed redundant scaler artefact → %s", stale_scaler)

    logger.info("Saved Bayesian Ridge → %s", br_path)
    logger.info("Saved Random Forest  → %s", rf_path)
    logger.info("Saved LightGBM       → %s", lgbm_path)
    logger.info("Saved chosen (%s) → %s", chosen_name, chosen_path)

    # ---- Summary -----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("  Bayesian Ridge  val MAE: %.4f s  test MAE: %.4f s",
                baseline_val_mae, br_test_mae)
    logger.info("  Random Forest   val MAE: %.4f s  test MAE: %.4f s",
                rf_metrics["val_mae"], rf_metrics["test_mae"])
    logger.info("  LightGBM        val MAE: %.4f s  test MAE: %.4f s",
                lgbm_metrics["val_mae"], lgbm_metrics["test_mae"])
    logger.info("  Chosen model            : %s (val MAE %.4f s)",
                chosen_name, chosen_val_mae)
    logger.info("  Models saved to         : %s", MODELS_DIR)
    logger.info("=" * 60)
    logger.info("Run: mlflow ui --port 5000  to inspect all runs")


if __name__ == "__main__":
    train()