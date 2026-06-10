"""
src/models/lap_time/evaluate.py

Evaluation for the trained lap time predictor — per-era + green-flag breakdown.
All outputs saved to reports/lap_time/:

    1. model_comparison.csv          — all 3 models × val/test (all-laps + green + per-era)
    2. per_era_mae.csv               — MAE per regulation era (all-laps + green)
    3. greenflag_vs_alllaps_mae.csv  — green-flag vs all-laps MAE per model × split
    4. per_circuit_mae.csv           — MAE per circuit, split by era
    5. shap_summary.png + shap_importance.csv  — SHAP on the chosen LightGBM model
    6. learning_curve.png            — RF train vs CV val MAE

Run after train.py has completed:
    python -m src.models.lap_time.evaluate
"""

import logging
import os
from pathlib import Path

# mlflow 3.x blocks the local file:// store unless opted in (Errors log #13).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import learning_curve

from src.pipeline.features import (
    MODEL_FEATURE_COLUMNS,
    build_features,
    load_encoding_mappings,
)
from src.pipeline.splits import make_splits
from src.pipeline.validate import validate_laps
from src.models.lap_time.train import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    MODELS_DIR,
    PROJECT_ROOT,
    get_X_y,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports" / "lap_time"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = ["Era", "CircuitKey", "TrackStatus"]
LAP_KEYS  = ["Season", "RoundNumber", "Driver", "LapNumber"]


# ---------------------------------------------------------------------------
# Core MAE breakdown helper
# ---------------------------------------------------------------------------

def mae_breakdown(y_true, y_pred, meta: pd.DataFrame) -> dict:
    """y_true, y_pred, meta must be index-aligned. meta needs columns: Era, CircuitKey, TrackStatus."""
    d = meta.reset_index(drop=True).copy()
    d["abs_err"] = (pd.Series(y_true).reset_index(drop=True) - pd.Series(y_pred).reset_index(drop=True)).abs()
    green = d[d["TrackStatus"] == "1"]
    return {
        "all_laps_mae": d["abs_err"].mean(),     "n_all": len(d),
        "green_mae": green["abs_err"].mean(),     "n_green": len(green),
        "per_era": d.groupby("Era")["abs_err"].mean().to_dict(),
        "per_era_green": green.groupby("Era")["abs_err"].mean().to_dict(),
        "per_circuit": d.groupby(["Era", "CircuitKey"])["abs_err"].mean().to_dict(),
    }


# ---------------------------------------------------------------------------
# Data loading — carries TrackStatus (eval-only meta) onto the feature matrix
# ---------------------------------------------------------------------------

def load_eval_data() -> pd.DataFrame:
    """
    Like train.load_data() but keeps TrackStatus aligned with the feature matrix.

    TrackStatus is NOT in FEATURE_COLUMNS, so build_features() drops it. Here we
    re-attach it by merging on lap identity (Season, RoundNumber, Driver, LapNumber)
    after feature building. Era and CircuitKey already survive (both in FEATURE_COLUMNS).
    """
    raw_dir = PROJECT_ROOT / "data" / "raw"
    files = sorted(raw_dir.glob("laps_*_r*.parquet"))  # per-round only — never the _full aggregates
    if not files:
        raise FileNotFoundError(
            f"No per-round parquet files (laps_*_r*.parquet) in {raw_dir}. Run ingest first."
        )

    raw = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    raw = raw.drop_duplicates(subset=LAP_KEYS).reset_index(drop=True)

    meta = raw[LAP_KEYS + ["TrackStatus"]].copy()
    meta["TrackStatus"] = meta["TrackStatus"].astype(str)

    # Encode with the mappings frozen at train time — never rebuild from eval data.
    circuit_map, team_map = load_encoding_mappings()
    feats = build_features(
        validate_laps(raw), circuit_mapping=circuit_map, team_mapping=team_map
    )
    merged = feats.merge(meta, on=LAP_KEYS, how="left")

    n_missing = merged["TrackStatus"].isna().sum()
    if n_missing:
        logger.warning("%d eval laps missing TrackStatus after merge — set to 'NA'", n_missing)
        merged["TrackStatus"] = merged["TrackStatus"].fillna("NA")

    logger.info("Eval data: %d laps with TrackStatus attached", len(merged))
    return merged


# ---------------------------------------------------------------------------
# Run all 3 models × both splits → report tables
# ---------------------------------------------------------------------------

ERA_LABEL = {0: "ground_effect", 1: "active_aero"}


def run_breakdowns(models: dict, splits: dict) -> dict:
    """models: name -> fitted model. splits: split_name -> df. Returns {(model,split): breakdown}."""
    out = {}
    for split_name, df in splits.items():
        if len(df) == 0:
            logger.warning("Split %s empty — skipping", split_name)
            continue
        X, y = get_X_y(df)
        meta = df[META_COLS].copy()
        for model_name, model in models.items():
            y_pred = model.predict(X)
            out[(model_name, split_name)] = mae_breakdown(y.values, y_pred, meta)
    return out


def _tft_comparison_rows() -> list[dict]:
    """Fold the TFT's overall metrics (written by train_tft.py on the GPU box)
    into the comparison table, so model_comparison.csv is the single four-model
    source of truth. Silently skipped if the TFT hasn't been trained yet."""
    tft_path = REPORTS_DIR / "tft_breakdown.csv"
    if not tft_path.exists():
        logger.info("tft_breakdown.csv not found — comparison table is trees-only")
        return []
    tft = pd.read_csv(tft_path)
    rows = []
    for split, grp in tft.groupby("split"):
        overall = grp[grp["scope"] == "overall"]
        eras    = grp[grp["scope"] == "era"].set_index("key")
        eras.index = eras.index.astype(str)   # csv may parse era keys as ints
        if overall.empty:
            continue
        o = overall.iloc[0]
        rows.append({
            "model":        "TFT",
            "split":        split,
            "all_laps_mae": round(float(o["mae_all"]), 4),
            "green_mae":    round(float(o["mae_green"]), 4),
            "n_all":        int(o["laps"]),
            "n_green":      pd.NA,   # train_tft reports green MAE, not green lap count
            "era0_mae":     round(float(eras.loc["0", "mae_all"]), 4) if "0" in eras.index else float("nan"),
            "era1_mae":     round(float(eras.loc["1", "mae_all"]), 4) if "1" in eras.index else float("nan"),
        })
    return rows


def write_model_comparison(bd: dict) -> pd.DataFrame:
    rows = []
    for (model, split), b in bd.items():
        rows.append({
            "model":        model,
            "split":        split,
            "all_laps_mae": round(b["all_laps_mae"], 4),
            "green_mae":    round(b["green_mae"], 4),
            "n_all":        b["n_all"],
            "n_green":      b["n_green"],
            "era0_mae":     round(b["per_era"].get(0, float("nan")), 4),
            "era1_mae":     round(b["per_era"].get(1, float("nan")), 4),
        })
    rows.extend(_tft_comparison_rows())
    df = pd.DataFrame(rows).sort_values(["split", "all_laps_mae"]).reset_index(drop=True)
    df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    logger.info("model_comparison.csv\n%s", df.to_string(index=False))
    return df


def write_per_era(bd: dict) -> pd.DataFrame:
    rows = []
    for (model, split), b in bd.items():
        for era, mae in b["per_era"].items():
            rows.append({
                "model":        model,
                "split":        split,
                "Era":          int(era),
                "era_label":    ERA_LABEL.get(int(era), str(era)),
                "all_laps_mae": round(mae, 4),
                "green_mae":    round(b["per_era_green"].get(era, float("nan")), 4),
            })
    df = pd.DataFrame(rows).sort_values(["split", "model", "Era"]).reset_index(drop=True)
    df.to_csv(REPORTS_DIR / "per_era_mae.csv", index=False)
    logger.info("per_era_mae.csv written (%d rows)", len(df))
    return df


def write_greenflag(bd: dict) -> pd.DataFrame:
    rows = []
    for (model, split), b in bd.items():
        rows.append({
            "model":        model,
            "split":        split,
            "all_laps_mae": round(b["all_laps_mae"], 4),
            "green_mae":    round(b["green_mae"], 4),
            "delta":        round(b["all_laps_mae"] - b["green_mae"], 4),
            "n_all":        b["n_all"],
            "n_green":      b["n_green"],
        })
    df = pd.DataFrame(rows).sort_values(["split", "model"]).reset_index(drop=True)
    df.to_csv(REPORTS_DIR / "greenflag_vs_alllaps_mae.csv", index=False)
    logger.info("greenflag_vs_alllaps_mae.csv written (%d rows)", len(df))
    return df


def write_per_circuit(bd: dict) -> pd.DataFrame:
    rows = []
    for (model, split), b in bd.items():
        for (era, circuit), mae in b["per_circuit"].items():
            rows.append({
                "model":      model,
                "split":      split,
                "Era":        int(era),
                "CircuitKey": circuit,
                "mae":        round(mae, 4),
            })
    df = (pd.DataFrame(rows)
          .sort_values(["split", "model", "Era", "CircuitKey"])
          .reset_index(drop=True))
    df.to_csv(REPORTS_DIR / "per_circuit_mae.csv", index=False)
    logger.info("per_circuit_mae.csv written (%d rows)", len(df))
    return df


# ---------------------------------------------------------------------------
# Learning curve (RF — train set, cross-validated)
# ---------------------------------------------------------------------------

def plot_learning_curves(rf_model, train_df: pd.DataFrame) -> None:
    logger.info("Computing learning curves (this may take ~2 min on CPU)...")
    from sklearn.base import clone

    X_train, y_train = get_X_y(train_df)
    train_sizes, train_scores, val_scores = learning_curve(
        clone(rf_model), X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="neg_mean_absolute_error", cv=3, n_jobs=-1, verbose=0,
    )
    train_mae = -train_scores.mean(axis=1)
    val_mae   = -val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mae, "o-", color="#E8002D", label="Train MAE")
    ax.plot(train_sizes, val_mae,   "o-", color="#1565C0", label="Val MAE (CV)")
    ax.set_xlabel("Training set size (laps)")
    ax.set_ylabel("MAE (seconds)")
    ax.set_title("Learning Curve — Random Forest Lap Time Predictor")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = REPORTS_DIR / "learning_curve.png"
    fig.savefig(out_path, dpi=150); plt.close(fig)
    logger.info("Learning curve saved to %s", out_path)
    try:
        mlflow.log_artifact(str(out_path))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# SHAP — chosen LightGBM model
# ---------------------------------------------------------------------------

def plot_shap_summary(model, val_df: pd.DataFrame, max_display: int = 14, sample_n: int = 1000) -> None:
    """
    SHAP summary for the chosen LightGBM model. Sampled for speed.

    Sanity check we care about: TeamEncoded should carry real weight — that
    validates the team-pace feature. Directional SHAP (high fuel -> slower) also
    confirms the model learned physical intuition, not noise.
    """
    logger.info("Computing SHAP values for LightGBM (sample n=%d)...", sample_n)

    X_val, _ = get_X_y(val_df)
    feature_cols = [c for c in MODEL_FEATURE_COLUMNS if c in val_df.columns]

    X_sample = X_val.sample(n=sample_n, random_state=42) if len(X_val) > sample_n else X_val

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X_sample, feature_names=feature_cols,
        max_display=max_display, show=False, plot_size=None,
    )
    plt.title("SHAP Feature Importance — LightGBM Lap Time Predictor")
    plt.tight_layout()
    out_path = REPORTS_DIR / "shap_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("SHAP summary saved to %s", out_path)

    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_shap}) \
        .sort_values("mean_abs_shap", ascending=False)
    logger.info("\nFeature importance (mean |SHAP|):\n%s", shap_df.to_string(index=False))
    shap_df.to_csv(REPORTS_DIR / "shap_importance.csv", index=False)

    try:
        mlflow.log_artifact(str(out_path))
        mlflow.log_artifact(str(REPORTS_DIR / "shap_importance.csv"))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate() -> None:
    """Load trained models and run per-era + green-flag evaluation."""

    br_path   = MODELS_DIR / "bayesian_ridge_lap.joblib"
    rf_path   = MODELS_DIR / "rf_lap.joblib"
    lgbm_path = MODELS_DIR / "lgbm_lap.joblib"

    for p in [br_path, rf_path, lgbm_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run train.py first.")

    br_model   = joblib.load(br_path)    # sklearn Pipeline (encoder+scaler+BayesianRidge)
    rf_model   = joblib.load(rf_path)
    lgbm_model = joblib.load(lgbm_path)  # chosen model
    logger.info("Models loaded from %s", MODELS_DIR)

    features_df = load_eval_data()
    train_df, val_df, test_df = make_splits(features_df)

    models = {"BayesianRidge": br_model, "RandomForest": rf_model, "LightGBM": lgbm_model}
    splits = {"val": val_df, "test": test_df}

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="evaluation"):
        bd = run_breakdowns(models, splits)

        write_model_comparison(bd)
        write_per_era(bd)
        write_greenflag(bd)
        write_per_circuit(bd)
        for fname in ["model_comparison.csv", "per_era_mae.csv",
                      "greenflag_vs_alllaps_mae.csv", "per_circuit_mae.csv"]:
            try:
                mlflow.log_artifact(str(REPORTS_DIR / fname))
            except Exception:
                pass

        if len(val_df) > 0:
            plot_learning_curves(rf_model, train_df)
            plot_shap_summary(lgbm_model, val_df)
        else:
            logger.warning("No val data — skipping learning curve + SHAP.")

    logger.info("Evaluation complete. Reports in %s", REPORTS_DIR)


if __name__ == "__main__":
    evaluate()
