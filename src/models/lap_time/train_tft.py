"""Temporal Fusion Transformer lap-time trainer (Phase 2, A6000).

Headline model. Quantile outputs -> calibrated uncertainty for the sim engine.
Must beat LightGBM green-test MAE (2.14s) on val AND hold across the 2026 era
boundary, or report honestly why not. Saves a CPU-loadable artifact for serving.

Run:
    python -m src.models.lap_time.train_tft
    python -m src.models.lap_time.train_tft --fast   # 2-epoch smoke run

Pins: pytorch-forecasting==1.7.0, lightning==2.6.5, torch==2.11.0+cu128
Contracts (Section 5/6/11): column names are law; feature roles by NAME only.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import mlflow
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

# carry-back loader: raw glob -> validate -> add_stint_id -> build_features
# -> merge Team/Compound/StintID/TrackStatus back on LAP_KEYS (Error log item)
from src.models.lap_time.train_tft_data import load_tft_data, MLFLOW_TRACKING_URI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR   = PROJECT_ROOT / "models"
EXPERIMENT_NAME = "lap_time_predictor"

TRAIN_SEASONS = [2022, 2023, 2024]
VAL_SEASONS   = [2025]
TEST_SEASONS  = [2026]

# Feature roles BY NAME (Section 11 — never positional index)
STATIC_CATEGORICALS       = ["CircuitKey", "Team", "EraStr", "CompoundStr"]
TIME_VARYING_KNOWN_REALS  = ["time_idx", "NormLapNumber", "FuelLoad", "TyreLife"]
TIME_VARYING_UNKNOWN_REALS = ["TrackTemp", "AirTemp"]
TARGET    = "LapTimeSeconds"
GROUP_IDS = ["GroupID"]

QUANTILES           = [0.1, 0.5, 0.9]
MAX_ENCODER_LENGTH  = 12
MIN_ENCODER_LENGTH  = 3
MAX_PREDICTION_LENGTH = 1   # one lap ahead from stint history
MIN_PREDICTION_LENGTH = 1


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add TFT-specific columns.

    Requires columns from load_tft_data() carry-back:
      StintID, Compound, Team, TrackStatus (eval only), Era, + all FEATURE_COLUMNS.
    """
    df = df.copy()
    df = df.sort_values(["Season", "RoundNumber", "Driver", "StintID", "LapNumber"])
    df["GroupID"] = (
        df["Season"].astype(str) + "_" + df["RoundNumber"].astype(str) + "_"
        + df["Driver"].astype(str) + "_" + df["StintID"].astype(str)
    )
    df["time_idx"]    = df.groupby("GroupID").cumcount().astype(int)
    df["EraStr"]      = df["Era"].astype(str)
    df["CompoundStr"] = df["Compound"].astype(str)
    # Impute weather — median fill; carry-back already ran validate so nulls are rare
    for c in ("TrackTemp", "AirTemp"):
        med = df[c].median()
        df[c] = df[c].fillna(med).astype(float)
    return df


def make_datasets(df: pd.DataFrame):
    """Build TimeSeriesDataSet for train / val / test.

    Returns (training, validation, test) — val/test may be None if seasons absent.
    Encoders frozen from training set; NaNLabelEncoder(add_nan=True) handles
    unseen circuits/teams at inference (equivalent to -1 sentinel, Section 6).
    """
    train_df = prepare(df[df["Season"].isin(TRAIN_SEASONS)])
    val_df   = prepare(df[df["Season"].isin(VAL_SEASONS)])
    test_df  = prepare(df[df["Season"].isin(TEST_SEASONS)])

    if train_df.empty:
        raise ValueError("Training split is empty — check parquet files and TRAIN_SEASONS.")

    cat_encoders = {c: NaNLabelEncoder(add_nan=True) for c in STATIC_CATEGORICALS}

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=GROUP_IDS,
        min_encoder_length=MIN_ENCODER_LENGTH,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=MIN_PREDICTION_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=STATIC_CATEGORICALS,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS + [TARGET],
        target_normalizer=GroupNormalizer(groups=GROUP_IDS),
        categorical_encoders=cat_encoders,
        add_relative_time_idx=True,
        add_target_scales=True,
        allow_missing_timesteps=True,
    )

    # pf 1.7.0: from_dataset + predict_mode=True for eval sets (stop_randomization removed)
    validation = (
        TimeSeriesDataSet.from_dataset(training, val_df, predict=True)
        if not val_df.empty else None
    )
    test = (
        TimeSeriesDataSet.from_dataset(training, test_df, predict=True)
        if not test_df.empty else None
    )
    return training, validation, test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def green_mae(model: TemporalFusionTransformer,
              dataloader,
              raw_df: pd.DataFrame) -> dict[str, float]:
    """MAE overall + green-flag only (TrackStatus=='1'). Section 13 rule 4.

    pf 1.7.0: model.predict(dl, return_index=True) returns a pd.DataFrame where
    columns are GroupID, time_idx (from return_index) + prediction columns.
    The median quantile (index 1 of 3 for [0.1,0.5,0.9]) is the point estimate.
    """
    pred_df: pd.DataFrame = model.predict(
        dataloader,
        return_index=True,
        return_decoder_lengths=False,
        mode="quantiles",          # returns all quantiles as columns
    )
    # pred_df columns: group_ids + time_idx + [0, 1, 2] (quantile indices)
    # column 1 = median (0.5 quantile)
    pred_df = pred_df.rename(columns={1: "yhat"})
    pred_df["yhat"] = pred_df["yhat"].astype(float)

    # merge back actual target + TrackStatus
    key_cols = GROUP_IDS + ["time_idx"]
    need_cols = key_cols + [TARGET] + (["TrackStatus"] if "TrackStatus" in raw_df.columns else [])
    j = pred_df.merge(raw_df[need_cols], on=key_cols, how="inner")

    err = (j["yhat"] - j[TARGET]).abs()
    out: dict[str, float] = {"mae_all": float(err.mean())}
    if "TrackStatus" in j.columns:
        g = j["TrackStatus"].astype(str) == "1"
        out["mae_green"] = float(err[g].mean()) if g.any() else float("nan")
    return out


def export_cpu(model: TemporalFusionTransformer, path: Path) -> None:
    """CPU-loadable artifact for serving (Section 14 rule 4)."""
    model = model.to("cpu")
    torch.save({"state_dict": model.state_dict(), "hparams": dict(model.hparams)}, path)
    log.info("CPU artifact -> %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(fast: bool = False) -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_tft_data()   # carry-back: Team/Compound/StintID/TrackStatus present
    training, validation, test = make_datasets(df)

    bs = 256 if fast else 512
    train_dl = training.to_dataloader(train=True,  batch_size=bs,     num_workers=4, pin_memory=True)
    val_dl   = validation.to_dataloader(train=False, batch_size=bs*2, num_workers=4) if validation else None
    test_dl  = test.to_dataloader(train=False,      batch_size=bs*2,  num_workers=4) if test else None

    gpu = torch.cuda.is_available()
    log.info("CUDA=%s  device=%s", gpu, torch.cuda.get_device_name(0) if gpu else "cpu")

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=32,             # right-sized to data, not GPU (Section 14 rule 3)
        attention_head_size=4,
        dropout=0.15,
        hidden_continuous_size=16,
        loss=QuantileLoss(quantiles=QUANTILES),
        log_interval=50,
        optimizer="adamw",
        reduce_on_plateau_patience=4,
    )
    n_params = sum(p.numel() for p in tft.parameters())
    log.info("TFT params: %d", n_params)

    monitor = "val_loss" if val_dl else "train_loss"
    ckpt  = ModelCheckpoint(dirpath=str(MODELS_DIR), filename="tft_lap",
                            monitor=monitor, mode="min", save_top_k=1)
    early = EarlyStopping(monitor=monitor, patience=10, mode="min")

    trainer = pl.Trainer(
        max_epochs=2 if fast else 100,
        accelerator="gpu" if gpu else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[ckpt, early],
        limit_train_batches=5 if fast else 1.0,
        enable_progress_bar=True,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="tft_lap"):
        mlflow.log_params({
            "model": "TFT",
            "hidden_size": 32, "heads": 4, "dropout": 0.15,
            "max_encoder_length": MAX_ENCODER_LENGTH,
            "min_encoder_length": MIN_ENCODER_LENGTH,
            "quantiles": str(QUANTILES),
            "n_params": n_params,
            "pf_version": "1.7.0",
        })

        trainer.fit(tft, train_dl, val_dl)

        best = TemporalFusionTransformer.load_from_checkpoint(ckpt.best_model_path)

        metrics: dict[str, float] = {}
        if val_dl:
            val_raw = prepare(df[df["Season"].isin(VAL_SEASONS)])
            m_val   = green_mae(best, val_dl, val_raw)
            metrics.update({f"val_{k}": v for k, v in m_val.items()})
            log.info("VAL  %s", m_val)
        if test_dl:
            test_raw = prepare(df[df["Season"].isin(TEST_SEASONS)])
            m_test   = green_mae(best, test_dl, test_raw)
            metrics.update({f"test_{k}": v for k, v in m_test.items()})
            log.info("TEST %s", m_test)

        mlflow.log_metrics(metrics)
        export_cpu(best, MODELS_DIR / "tft_lap.pt")
        mlflow.log_artifact(str(MODELS_DIR / "tft_lap.pt"))
        mlflow.log_artifact(ckpt.best_model_path)

    bar = 2.14
    actual = metrics.get("test_mae_green", float("nan"))
    log.info("DONE | LightGBM bar: %.2fs | TFT test_mae_green: %s | beat=%s",
             bar, actual, actual < bar if not np.isnan(actual) else "no test data")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    main(**vars(ap.parse_args()))
