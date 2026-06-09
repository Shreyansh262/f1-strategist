"""
src/models/lap_time/train_transformer.py

TabTransformer lap time predictor — Week 5.

Architecture (from v2 spec):
    Categorical features (CircuitEncoded, CompoundEncoded, Era)
        → Embedding layers (dim=32 each)
        → 2× TransformerBlock (4 heads, dim=32, FFN=64, dropout=0.1)
    Continuous features (TyreLife, FuelLoad, TrackTemp, ...)
        → LayerNorm
    Concatenate → MLP head [concat_dim → 64 → 32 → 1]
    Output: predicted LapTimeSeconds

Key design decisions:
    - NOT autoregressive — each lap predicted independently
    - ~50K parameters — trains in minutes on Colab free GPU
    - MC Dropout — 30 forward passes at inference → mean + std
    - Era embedding — 32-dim learned vector per regulation era
    - Huber loss — robust to safety car / outlier laps
    - Frozen circuit mapping from training set

Training:
    Locally  : python -m src.models.lap_time.train_transformer  (slow, CPU)
    On Colab : use notebooks/03_transformer_training.ipynb      (recommended)

Splits: train=2022-2024, val=2025, test=2026
Save  : models/transformer_lap.pt

Entry point:
    python -m src.models.lap_time.train_transformer
"""

import logging
import time
from pathlib import Path
from typing import Final

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

from src.pipeline.features import (
    MODEL_FEATURE_COLUMNS,
    CAT_FEATURE_INDICES,
    CONT_FEATURE_INDICES,
    CAT_CARDINALITIES,
    build_features,
)
from src.pipeline.splits import assert_no_leakage, make_splits
from src.pipeline.validate import validate_laps
from src.models.lap_time.train import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    PROJECT_ROOT,
    TARGET,
    load_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters — match v2 spec exactly
# ---------------------------------------------------------------------------
EMB_DIM:        Final[int]   = 32
N_HEADS:        Final[int]   = 4
N_TRANSFORMER:  Final[int]   = 2
FFN_DIM:        Final[int]   = 64
DROPOUT:        Final[float] = 0.1
BATCH_SIZE:     Final[int]   = 512
LR:             Final[float] = 1e-3
WEIGHT_DECAY:   Final[float] = 1e-4
MAX_EPOCHS:     Final[int]   = 100
PATIENCE:       Final[int]   = 10
MC_SAMPLES:     Final[int]   = 30     # forward passes for uncertainty estimation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single transformer encoder block: MultiHeadAttention + FFN + residuals."""

    def __init__(self, dim: int, n_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn    = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ff      = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.norm1   = nn.LayerNorm(dim)
        self.norm2   = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim]  where seq_len = number of categorical features
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class TabTransformer(nn.Module):
    """
    TabTransformer for F1 lap time prediction.

    Categorical features (CircuitEncoded, CompoundEncoded, Era) are embedded
    into a shared dim=32 space and processed by transformer blocks.
    Continuous features are layer-normed.
    Both streams are concatenated and passed through an MLP head.

    MC Dropout: call model.train() at inference time + run MC_SAMPLES forward
    passes to get prediction distribution.
    """

    def __init__(
        self,
        cat_cardinalities: list[int],
        n_cont: int,
        emb_dim: int       = EMB_DIM,
        n_heads: int       = N_HEADS,
        n_transformer: int = N_TRANSFORMER,
        ffn_dim: int       = FFN_DIM,
        dropout: float     = DROPOUT,
    ) -> None:
        super().__init__()

        # One embedding per categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, emb_dim)   # +1 for sentinel -1 → mapped to 0
            for card in cat_cardinalities
        ])

        # Transformer blocks operate over the sequence of embedded cat features
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(emb_dim, n_heads, ffn_dim, dropout)
            for _ in range(n_transformer)
        ])

        # Continuous feature normalisation
        self.cont_norm = nn.LayerNorm(n_cont)

        # MLP head: [cat_out + cont] → 64 → 32 → 1
        cat_out_dim = len(cat_cardinalities) * emb_dim
        mlp_in_dim  = cat_out_dim + n_cont
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_cat  : [batch, n_cat]  — integer indices, -1 sentinel shifted to 0
        x_cont : [batch, n_cont] — scaled continuous features

        Returns
        -------
        [batch, 1] — predicted LapTimeSeconds
        """
        # Embed each categorical feature → [batch, n_cat, emb_dim]
        embedded = torch.stack(
            [emb(x_cat[:, i].clamp(min=0)) for i, emb in enumerate(self.embeddings)],
            dim=1,
        )

        # Transformer over categorical embeddings
        transformed = self.transformer_blocks(embedded)    # [batch, n_cat, emb_dim]
        cat_out = transformed.flatten(start_dim=1)         # [batch, n_cat * emb_dim]

        # Normalise continuous features
        cont_out = self.cont_norm(x_cont)                  # [batch, n_cont]

        # Concatenate and predict
        combined = torch.cat([cat_out, cont_out], dim=1)   # [batch, cat_out + n_cont]
        return self.mlp(combined)                          # [batch, 1]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_tensors(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
    """
    Split DataFrame into categorical and continuous tensors.

    Parameters
    ----------
    df          : features DataFrame with MODEL_FEATURE_COLUMNS present
    scaler      : fitted StandardScaler for continuous features
    fit_scaler  : if True, fit scaler on this df (training mode only)

    Returns
    -------
    x_cat, x_cont, y, scaler
    """
    feature_cols = [c for c in MODEL_FEATURE_COLUMNS if c in df.columns]
    cat_cols  = [feature_cols[i] for i in CAT_FEATURE_INDICES]
    cont_cols = [feature_cols[i] for i in CONT_FEATURE_INDICES]

    x_cat_np  = df[cat_cols].values.astype(np.int64)
    x_cont_np = df[cont_cols].values.astype(np.float32)
    y_np      = df[TARGET].values.astype(np.float32)

    if fit_scaler:
        scaler = StandardScaler()
        x_cont_np = scaler.fit_transform(x_cont_np).astype(np.float32)
    else:
        x_cont_np = scaler.transform(x_cont_np).astype(np.float32)

    x_cat  = torch.tensor(x_cat_np,  dtype=torch.long)
    x_cont = torch.tensor(x_cont_np, dtype=torch.float32)
    y      = torch.tensor(y_np,      dtype=torch.float32).unsqueeze(1)

    return x_cat, x_cont, y, scaler


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: TabTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for x_cat, x_cont, y in loader:
        x_cat, x_cont, y = x_cat.to(DEVICE), x_cont.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(x_cat, x_cont)
        loss  = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(
    model: TabTransformer,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Returns (loss, MAE)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for x_cat, x_cont, y in loader:
        x_cat, x_cont, y = x_cat.to(DEVICE), x_cont.to(DEVICE), y.to(DEVICE)
        preds = model(x_cat, x_cont)
        total_loss += criterion(preds, y).item() * len(y)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    loss = total_loss / len(loader.dataset)
    mae  = mean_absolute_error(
        np.concatenate(all_targets), np.concatenate(all_preds)
    )
    return loss, mae


# ---------------------------------------------------------------------------
# MC Dropout inference
# ---------------------------------------------------------------------------

def predict_with_uncertainty(
    model: TabTransformer,
    x_cat: torch.Tensor,
    x_cont: torch.Tensor,
    n_samples: int = MC_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """
    MC Dropout uncertainty estimation.
    Runs n_samples forward passes with dropout enabled.

    Returns
    -------
    mean_preds : [N] predicted lap times
    std_preds  : [N] prediction standard deviations (uncertainty)
    """
    model.train()   # enable dropout
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            p = model(x_cat.to(DEVICE), x_cont.to(DEVICE))
            preds.append(p.cpu().numpy())
    preds_arr = np.stack(preds, axis=0)   # [n_samples, N, 1]
    return preds_arr.mean(axis=0).squeeze(), preds_arr.std(axis=0).squeeze()


# ---------------------------------------------------------------------------
# Per-circuit MAE evaluation
# ---------------------------------------------------------------------------

def evaluate_per_circuit(
    model: TabTransformer,
    val_df: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    rows = []
    for circuit, grp in val_df.groupby("CircuitKey"):
        x_cat, x_cont, y, _ = prepare_tensors(grp, scaler=scaler, fit_scaler=False)
        mean_p, std_p = predict_with_uncertainty(model, x_cat, x_cont)
        mae = mean_absolute_error(y.numpy().squeeze(), mean_p)
        rows.append({
            "CircuitKey":      circuit,
            "n_laps":          len(grp),
            "transformer_mae": round(float(mae), 4),
            "mean_uncertainty": round(float(std_p.mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("transformer_mae", ascending=False)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

TARGET = "LapTimeSeconds"

def train() -> None:
    """Full transformer training pipeline."""
    logger.info("Device: %s", DEVICE)

    # ---- data --------------------------------------------------------------
    features_df = load_data()
    train_df, val_df, test_df = make_splits(
        features_df,
        train_seasons=[2022, 2023, 2024],
        val_seasons=[2025],
        test_seasons=[2026],
    )
    assert_no_leakage(train_df, val_df, test_df)

    if len(train_df) == 0:
        raise ValueError("Training set empty — fetch 2022-2024 data first.")
    if len(val_df) == 0:
        logger.warning("Val set empty — fetch 2025 data.")

    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    # ---- tensors -----------------------------------------------------------
    x_cat_train, x_cont_train, y_train, scaler = prepare_tensors(
        train_df, fit_scaler=True,
    )
    x_cat_val, x_cont_val, y_val, _ = prepare_tensors(
        val_df, scaler=scaler, fit_scaler=False,
    ) if len(val_df) > 0 else (None, None, None, None)

    train_loader = DataLoader(
        TensorDataset(x_cat_train, x_cont_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(x_cat_val, x_cont_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    ) if x_cat_val is not None else None

    n_cont = len(CONT_FEATURE_INDICES)

    # ---- model -------------------------------------------------------------
    model = TabTransformer(
        cat_cardinalities=CAT_CARDINALITIES,
        n_cont=n_cont,
        emb_dim=EMB_DIM,
        n_heads=N_HEADS,
        n_transformer=N_TRANSFORMER,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d", n_params)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS,
    )

    # ---- MLflow ------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="tabtransformer"):
        mlflow.log_params({
            "model_type":      "TabTransformer",
            "emb_dim":         EMB_DIM,
            "n_heads":         N_HEADS,
            "n_transformer":   N_TRANSFORMER,
            "ffn_dim":         FFN_DIM,
            "dropout":         DROPOUT,
            "batch_size":      BATCH_SIZE,
            "lr":              LR,
            "weight_decay":    WEIGHT_DECAY,
            "max_epochs":      MAX_EPOCHS,
            "patience":        PATIENCE,
            "n_params":        n_params,
            "train_seasons":   [2022, 2023, 2024],
            "val_seasons":     [2025],
            "test_seasons":    [2026],
            "n_train_laps":    len(train_df),
            "n_val_laps":      len(val_df),
            "features":        MODEL_FEATURE_COLUMNS,
            "loss":            "HuberLoss(delta=1.0)",
        })

        # ---- training loop -------------------------------------------------
        best_val_mae   = float("inf")
        best_val_loss  = float("inf")
        patience_count = 0
        best_state     = None

        t0 = time.time()
        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()

            if val_loader is not None:
                val_loss, val_mae = evaluate_epoch(model, val_loader, criterion)
            else:
                val_loss, val_mae = float("nan"), float("nan")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "val_mae":    val_mae,
                "lr":         scheduler.get_last_lr()[0],
            }, step=epoch)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d — train loss: %.4f | val loss: %.4f | val MAE: %.4f s",
                    epoch, MAX_EPOCHS, train_loss, val_loss, val_mae,
                )

            # Early stopping
            if val_mae < best_val_mae:
                best_val_mae   = val_mae
                best_val_loss  = val_loss
                best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, PATIENCE)
                    break

        train_time = time.time() - t0
        logger.info("Training complete in %.1f s", train_time)

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(DEVICE)

        # ---- final metrics -------------------------------------------------
        mlflow.log_metrics({
            "best_val_mae":  best_val_mae,
            "best_val_loss": best_val_loss,
            "train_time_s":  train_time,
        })

        # Per-circuit MAE on val
        if len(val_df) > 0:
            circuit_df = evaluate_per_circuit(model, val_df, scaler)
            logger.info("\nPer-circuit MAE (val 2025):\n%s", circuit_df.to_string(index=False))
            circuit_csv = MODELS_DIR.parent / "reports" / "lap_time" / "transformer_per_circuit_mae.csv"
            circuit_csv.parent.mkdir(parents=True, exist_ok=True)
            circuit_df.to_csv(circuit_csv, index=False)
            mlflow.log_artifact(str(circuit_csv))

        # Test set (2026) — only if data exists
        if len(test_df) > 0:
            x_cat_test, x_cont_test, y_test, _ = prepare_tensors(
                test_df, scaler=scaler, fit_scaler=False,
            )
            test_preds, test_std = predict_with_uncertainty(model, x_cat_test, x_cont_test)
            test_mae = mean_absolute_error(y_test.numpy().squeeze(), test_preds)
            mlflow.log_metric("test_mae_2026", test_mae)
            logger.info("Test MAE (2026): %.4f s", test_mae)
        else:
            logger.warning("No 2026 data — skipping test evaluation. Ingest 2026 races first.")

        # ---- save ----------------------------------------------------------
        model_path  = MODELS_DIR / "transformer_lap.pt"
        scaler_path = MODELS_DIR / "scaler_transformer.joblib"

        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": {
                "cat_cardinalities": CAT_CARDINALITIES,
                "n_cont":            n_cont,
                "emb_dim":           EMB_DIM,
                "n_heads":           N_HEADS,
                "n_transformer":     N_TRANSFORMER,
                "ffn_dim":           FFN_DIM,
                "dropout":           DROPOUT,
            },
            "best_val_mae":   best_val_mae,
            "feature_columns": MODEL_FEATURE_COLUMNS,
        }, model_path)

        joblib.dump(scaler, scaler_path)

        logger.info("Saved transformer → %s", model_path)
        logger.info("Saved scaler     → %s", scaler_path)

        mlflow.log_artifact(str(model_path))

        # ---- summary -------------------------------------------------------
        logger.info("=" * 60)
        logger.info("TRANSFORMER TRAINING COMPLETE")
        logger.info("  Best val MAE (2025): %.4f s", best_val_mae)
        if len(test_df) > 0:
            logger.info("  Test MAE (2026):     %.4f s", test_mae)
        logger.info("  Parameters:          %d", n_params)
        logger.info("  Device:              %s", DEVICE)
        logger.info("  Model saved to:      %s", model_path)
        logger.info("=" * 60)
        logger.info("Run: mlflow ui --port 5000  to inspect all runs")


if __name__ == "__main__":
    train()