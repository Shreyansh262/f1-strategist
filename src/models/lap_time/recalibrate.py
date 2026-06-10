"""Phase 3.5 — Conformal recalibration of the TFT quantile bands.

Why this exists
---------------
Measured central coverage of the raw 0.1–0.9 band: 0.756 (val, era 0) and 0.690
(test, era 1) vs 0.80 nominal — the model is overconfident, worse across the
2026 regulation boundary. The Monte Carlo engine samples lap times from these
bands; feeding it overconfident bands makes every strategy probability
overconfident. Split-conformal widening fixes coverage with zero retraining.

Method (split conformal, era-aware)
-----------------------------------
For a calibration set, the nonconformity score of lap i is
    s_i = max(q10_i − y_i, y_i − q90_i)
(positive ⇒ y outside the band). The finite-sample-corrected (1−α) quantile of
the scores, ŝ = Quantile_{⌈(n+1)(1−α)⌉/n}(s), widens both edges:
    q10' = q10 − ŝ,   q90' = q90 + ŝ
which guarantees ≥ 1−α marginal coverage on exchangeable data.

Era-awareness — calibration and evaluation NEVER share laps:
    era 0: calibrate on ODD 2025 rounds, report coverage on EVEN 2025 rounds.
    era 1: calibrate on the FIRST 2026 round, report coverage on rounds 2-6.
           (R1 is sacrificed from the test set for calibration — documented.)

Outputs
-------
models/tft_calibration.json                — the shifts the sim engine applies
reports/lap_time/tft_recalibration.csv     — before/after coverage table

Run (needs pf/lightning + models/tft_lap.ckpt):
    python -m src.models.lap_time.recalibrate
"""
from __future__ import annotations

import json
import logging
import math
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from src.models.lap_time.train_tft import (
    QUANTILES, GROUP_IDS, TARGET, prepare, _extract_index,
)
from src.models.lap_time.train_tft_data import load_tft_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports" / "lap_time"
CKPT_PATH    = MODELS_DIR / "tft_lap.ckpt"

KEY_COLS = GROUP_IDS + ["time_idx"]
ALPHA    = 0.20   # nominal miscoverage of the 0.1–0.9 band


# ---------------------------------------------------------------------------
# Quantile predictions joined to actuals
# ---------------------------------------------------------------------------

def quantile_frame(
    model: TemporalFusionTransformer,
    dataset_params: dict,
    season_df: pd.DataFrame,
    green_only: bool = True,
) -> pd.DataFrame:
    """Per-lap quantile predictions merged to actuals for one data subset.

    Rebuilds the dataset with the model's OWN fitted dataset parameters
    (from_parameters) so categorical encodings/normalisation match training
    exactly. predict=False — one window per decodable lap (Errors log #16).

    green_only=True restricts BOTH calibration and evaluation to green-flag
    laps: the sim engine samples green-racing pace from these bands; SC laps
    are +20-40s outliers that would blow the conformal shift up to a useless
    band width (observed: 8.4s shift calibrating on the SC-heavy 2026 R1).
    """
    p = prepare(season_df)
    ds = TimeSeriesDataSet.from_parameters(dataset_params, p, predict=False)
    dl = ds.to_dataloader(train=False, batch_size=1024, num_workers=0)

    arr, idx = _extract_index(model.predict(dl, mode="quantiles", return_index=True))
    if arr.ndim == 3:
        arr = arr[:, 0, :]
    idx = idx.copy()
    for i, q in enumerate(QUANTILES):
        idx[f"q{q}"] = arr[:, i]
    meta = [c for c in [TARGET, "TrackStatus"] if c in p.columns]
    j = idx.merge(p[KEY_COLS + meta], on=KEY_COLS, how="inner")
    if green_only and "TrackStatus" in j.columns:
        j = j[j["TrackStatus"].astype(str) == "1"]
    return j


# ---------------------------------------------------------------------------
# Conformal math
# ---------------------------------------------------------------------------

def conformal_shift(qf: pd.DataFrame, alpha: float = ALPHA) -> float:
    """Finite-sample-corrected split-conformal widening (seconds)."""
    lo, hi = f"q{QUANTILES[0]}", f"q{QUANTILES[-1]}"
    s = np.maximum(qf[lo] - qf[TARGET], qf[TARGET] - qf[hi]).to_numpy()
    n = len(s)
    if n == 0:
        raise ValueError("Empty calibration set")
    k = math.ceil((n + 1) * (1 - alpha))
    k = min(k, n)                      # clamp for tiny calibration sets
    return float(np.sort(s)[k - 1])


def coverage(qf: pd.DataFrame, shift: float = 0.0) -> dict[str, float]:
    """Central-band coverage with the band widened by `shift` on each side."""
    lo, hi = f"q{QUANTILES[0]}", f"q{QUANTILES[-1]}"
    inside = (qf[TARGET] >= qf[lo] - shift) & (qf[TARGET] <= qf[hi] + shift)
    return {
        "central_coverage": float(inside.mean()),
        "n_laps": int(len(qf)),
        "mean_band_width_s": float((qf[hi] - qf[lo] + 2 * shift).mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"{CKPT_PATH} not found — train the TFT first (or download the "
            "checkpoint from the Kaggle run output; models/ is gitignored)."
        )

    log.info("Loading checkpoint %s (CPU)", CKPT_PATH)
    # The checkpoint was trained on GPU; its pickled torchmetrics objects (loss,
    # logging_metrics) remember device=cuda and crash `.to()` on a CPU-only box.
    # Overriding them with fresh CPU instances via load_from_checkpoint kwargs
    # sidesteps that; the monkeypatch fallback pins Metric.device to cpu.
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(
            str(CKPT_PATH), map_location="cpu",
            loss=QuantileLoss(quantiles=QUANTILES),
            logging_metrics=torch.nn.ModuleList(),
        )
    except AssertionError:
        import torchmetrics
        torchmetrics.Metric.device = property(lambda self: torch.device("cpu"))
        model = TemporalFusionTransformer.load_from_checkpoint(
            str(CKPT_PATH), map_location="cpu",
        )
    model.eval()
    dataset_params = model.dataset_parameters
    if dataset_params is None:
        raise RuntimeError(
            "Checkpoint has no dataset_parameters — cannot rebuild the exact "
            "training encoders. Re-export the checkpoint with pf>=1.7."
        )

    df = load_tft_data()

    # --- era 0: calibrate odd 2025 rounds, evaluate even 2025 rounds --------
    val = df[df["Season"] == 2025]
    calib0 = val[val["RoundNumber"] % 2 == 1]
    eval0  = val[val["RoundNumber"] % 2 == 0]

    # --- era 1: calibrate first 2026 round, evaluate the rest ---------------
    test = df[df["Season"] == 2026]
    r1 = int(test["RoundNumber"].min())
    calib1 = test[test["RoundNumber"] == r1]
    eval1  = test[test["RoundNumber"] != r1]

    results: dict[str, dict] = {}
    shifts: dict[str, float] = {}
    rows = []
    for era, calib_df, eval_df, calib_desc in [
        (0, calib0, eval0, "2025 odd rounds"),
        (1, calib1, eval1, f"2026 round {r1} only"),
    ]:
        log.info("Era %d: calibrating on %s (%d laps)...", era, calib_desc, len(calib_df))
        qf_cal  = quantile_frame(model, dataset_params, calib_df)
        qf_eval = quantile_frame(model, dataset_params, eval_df)

        shift = conformal_shift(qf_cal)
        before = coverage(qf_eval, 0.0)
        after  = coverage(qf_eval, shift)
        shifts[f"era{era}_shift_s"] = round(shift, 4)
        results[f"era{era}"] = {"before": before, "after": after}

        log.info(
            "Era %d | shift=%.3fs | eval coverage %.3f -> %.3f (nominal %.2f, n=%d)",
            era, shift, before["central_coverage"], after["central_coverage"],
            1 - ALPHA, after["n_laps"],
        )
        for label, cov, s in [("raw", before, 0.0), ("recalibrated", after, shift)]:
            rows.append({
                "era": era, "calibrated_on": calib_desc, "bands": label,
                "shift_s": round(s, 4), **{k: round(v, 4) for k, v in cov.items()},
            })

    # --- save ----------------------------------------------------------------
    out = {
        "method": "split_conformal",
        "scope": "green_flag_laps_only",
        "alpha": ALPHA,
        "quantiles": QUANTILES,
        "calibrated_on": {
            "era0": "2025 odd rounds (eval: even rounds)",
            "era1": f"2026 round {r1} (eval: remaining 2026 rounds — R{r1} is "
                    "excluded from test coverage reporting)",
        },
        "date": date.today().isoformat(),
        **shifts,
    }
    cal_path = MODELS_DIR / "tft_calibration.json"
    with open(cal_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Calibration shifts -> %s", cal_path)

    rep = pd.DataFrame(rows)
    rep_path = REPORTS_DIR / "tft_recalibration.csv"
    rep.to_csv(rep_path, index=False)
    log.info("Recalibration report -> %s\n%s", rep_path, rep.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
