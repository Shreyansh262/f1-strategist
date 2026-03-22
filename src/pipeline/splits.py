"""
src/pipeline/splits.py

Season-aware temporal train / validation / test splits.

Split design:
    Train : 2021 + 2022  (model learns from two full seasons)
    Val   : 2023          (hyperparameter tuning, early stopping, threshold selection)
    Test  : 2024          (blind evaluation — touch this ONCE at final evaluation)

Why seasons, not random rows?
    F1 lap times are non-stationary across seasons: regulation changes (e.g. 2022
    ground-effect cars), tyre compound reformulations, and circuit modifications
    mean that shuffling laps across years would massively overestimate generalisation.
    A model that "works" on shuffled data but fails on 2024 races is useless.

Interview answer: "Temporal splits by season mirror how the model would be deployed —
trained on historical data, evaluated on future races it has never seen."

Usage:
    from src.pipeline.splits import make_splits
    train_df, val_df, test_df = make_splits(features_df)
"""

import logging
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

TRAIN_SEASONS: Final[list[int]] = [2021, 2022]
VAL_SEASONS: Final[list[int]] = [2023]
TEST_SEASONS: Final[list[int]] = [2024]


def make_splits(
    df: pd.DataFrame,
    train_seasons: list[int] = TRAIN_SEASONS,
    val_seasons: list[int] = VAL_SEASONS,
    test_seasons: list[int] = TEST_SEASONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a features DataFrame into train / val / test by season.

    Parameters
    ----------
    df : pd.DataFrame
        Features DataFrame from build_features(). Must contain a 'Season' column.
    train_seasons : list[int]
        Seasons to include in the training set. Default: [2021, 2022].
    val_seasons : list[int]
        Seasons to include in the validation set. Default: [2023].
    test_seasons : list[int]
        Seasons to include in the test set. Default: [2024].

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)

    Raises
    ------
    ValueError
        If any split is empty — this means the data doesn't cover that season yet.
        Callers should handle this gracefully (e.g. skip val/test when building
        the pipeline before 2024 data is available).
    """
    if "Season" not in df.columns:
        raise ValueError("DataFrame must contain a 'Season' column")

    train_df = df[df["Season"].isin(train_seasons)].copy()
    val_df   = df[df["Season"].isin(val_seasons)].copy()
    test_df  = df[df["Season"].isin(test_seasons)].copy()

    _log_split("train", train_df, train_seasons)
    _log_split("val",   val_df,   val_seasons)
    _log_split("test",  test_df,  test_seasons)

    # Warn if a split is empty but don't raise — the user may not have
    # fetched all seasons yet during development
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(split) == 0:
            logger.warning(
                "Split '%s' is empty — have you fetched data for seasons %s?",
                name,
                {"train": train_seasons, "val": val_seasons, "test": test_seasons}[name],
            )

    return train_df, val_df, test_df


def _log_split(name: str, df: pd.DataFrame, seasons: list[int]) -> None:
    """Log split summary statistics."""
    if len(df) == 0:
        logger.info("%-5s (%s): 0 rows", name, seasons)
        return

    circuits = df["CircuitKey"].nunique() if "CircuitKey" in df.columns else "?"
    drivers  = df["Driver"].nunique()    if "Driver"     in df.columns else "?"
    logger.info(
        "%-5s (%s): %6d laps | %2s circuits | %2s drivers",
        name, seasons, len(df), circuits, drivers,
    )


def assert_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Assert that train / val / test seasons do not overlap.
    Call this in CI to catch accidental leakage.

    Raises
    ------
    AssertionError if any season appears in more than one split.
    """
    train_s = set(train_df["Season"].unique())
    val_s   = set(val_df["Season"].unique())
    test_s  = set(test_df["Season"].unique())

    assert train_s.isdisjoint(val_s),  \
        f"Season overlap between train and val: {train_s & val_s}"
    assert train_s.isdisjoint(test_s), \
        f"Season overlap between train and test: {train_s & test_s}"
    assert val_s.isdisjoint(test_s),   \
        f"Season overlap between val and test: {val_s & test_s}"

    logger.info("Leakage check passed: no season overlap between splits")