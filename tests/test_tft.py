"""TFT smoke tests — structure, no GPU/full-train needed (Phase 2).

Run: pytest tests/test_tft.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.lap_time.train_tft import (
    prepare, make_datasets, STATIC_CATEGORICALS,
    TIME_VARYING_KNOWN_REALS, TARGET, MAX_ENCODER_LENGTH,
)


def _synthetic(n_groups=20, lap_len=15) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for g in range(n_groups):
        season = [2022, 2023, 2024, 2025, 2026][g % 5]
        for lap in range(1, lap_len + 1):
            rows.append({
                "Season": season, "RoundNumber": 1, "Driver": f"D{g%4}",
                "StintID": 1, "LapNumber": lap, "Compound": "MEDIUM",
                "Era": int(season >= 2026), "CircuitKey": "Monaco",
                "Team": f"T{g%3}", "TyreLife": lap,
                "NormLapNumber": lap / lap_len, "FuelLoad": 110 - lap * 1.5,
                "TrackTemp": 40.0, "AirTemp": 25.0, "TrackStatus": "1",
                "LapTimeSeconds": 80 + 0.05 * lap + rng.normal(0, 0.2),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def df():
    return _synthetic()


def test_prepare_time_idx_contiguous(df):
    p = prepare(df)
    for _, grp in p.groupby("GroupID"):
        ti = grp["time_idx"].to_numpy()
        assert (ti == np.arange(len(ti))).all()


def test_prepare_string_cats(df):
    p = prepare(df)
    assert p["EraStr"].dtype == object and p["CompoundStr"].dtype == object


def test_prepare_no_null_weather(df):
    d = df.copy()
    d.loc[0, "TrackTemp"] = np.nan
    p = prepare(d)
    assert p["TrackTemp"].notna().all()


def test_datasets_build_and_roles(df):
    training, val, test = make_datasets(df)
    assert set(STATIC_CATEGORICALS).issubset(training.static_categoricals)
    assert TARGET in training.reals or TARGET == training.target
    assert training.max_encoder_length == MAX_ENCODER_LENGTH


def test_dataloader_forward_shape(df):
    training, _, _ = make_datasets(df)
    dl = training.to_dataloader(train=True, batch_size=8, num_workers=0)
    x, y = next(iter(dl))
    assert x["encoder_cont"].ndim == 3        # [batch, time, features]
    assert x["encoder_cont"].shape[0] <= 8


def test_unseen_category_handled(df):
    """add_nan encoders must not crash on a circuit unseen in train (-1 semantics)."""
    training, _, _ = make_datasets(df)
    novel = prepare(_synthetic(n_groups=4))
    novel["CircuitKey"] = "NeverSeenTrack"
    from pytorch_forecasting import TimeSeriesDataSet
    ds = TimeSeriesDataSet.from_dataset(training, novel, predict=False, stop_randomization=True)
    dl = ds.to_dataloader(train=False, batch_size=4, num_workers=0)
    next(iter(dl))  # no raise