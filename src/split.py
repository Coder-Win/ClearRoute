"""Audit-defined chronological partitions, never a random fallback.

2016-2020: train; 2021: validation; 2022: calibration; 2023: test.
Validation stays reserved. All duration consumers use the same base train IDs.
"""
import pandas as pd


def create_chronological_split(
    dataframe: pd.DataFrame,
    year_column: str,
    train_years: list[int],
    validation_years: list[int],
    calibration_years: list[int],
    test_years: list[int],
) -> pd.Series:
    split = pd.Series(
        "unassigned",
        index=dataframe.index,
        dtype="object",
    )

    split.loc[dataframe[year_column].isin(train_years)] = "train"
    split.loc[dataframe[year_column].isin(validation_years)] = "validation"
    split.loc[dataframe[year_column].isin(calibration_years)] = "calibration"
    split.loc[dataframe[year_column].isin(test_years)] = "test"

    return split


def chronological_indices(dataframe):
    """Return train/calibration/test indices; fail on stale or ambiguous data."""
    if 'split' not in dataframe.columns:
        raise ValueError("Missing split metadata. Rerun phase0_audit, spatiotemporal_features and reliability_fusion2.")
    if not dataframe.index.is_unique or 'ID' not in dataframe or not dataframe['ID'].is_unique or dataframe['ID'].isna().any():
        raise ValueError("Chronological split requires unique indices and non-null, unique incident IDs.")
    allowed = {'train', 'validation', 'calibration', 'test'}
    if dataframe['split'].isna().any() or not dataframe['split'].isin(allowed).all():
        raise ValueError("Unassigned/invalid split labels. Check event years in the Phase 0 audit.")
    indices = tuple(dataframe.index[dataframe['split'].eq(label)] for label in ('train', 'calibration', 'test'))
    if any(len(index) == 0 for index in indices):
        raise ValueError("Train, calibration and test must all be non-empty after filtering/merging.")
    return indices
