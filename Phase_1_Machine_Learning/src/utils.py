"""
Data I/O & Target Quarantine Utilities
Supports transparent multi-part Parquet loading and leakage prevention assertions.
"""

from pathlib import Path
import pandas as pd
from typing import Tuple, List

TARGET_COLUMNS = ["ccme_values", "ccme_wqi", "ccme_wqi_code"]


def load_parquet_dataset(path: str) -> pd.DataFrame:
    """
    Load a Parquet file or a transparent multi-part directory of .parquet shards.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet path does not exist: {path}")
    return pd.read_parquet(p)


def separate_features_and_targets(
    df: pd.DataFrame,
    feature_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Strictly quarantine target variables from feature matrix X.
    Returns:
        X: Feature matrix guaranteed to have 0 target leakage
        y_reg: Continuous CCME values (0-100)
        y_cls: Discrete WQI class code (0-4)
    """
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"Expected target column '{col}' missing from DataFrame.")

    y_reg = df["ccme_values"].copy()
    y_cls = df["ccme_wqi_code"].copy()

    if feature_cols is not None:
        # Check no target column is in feature list
        for t in TARGET_COLUMNS:
            assert t not in feature_cols, f"CRITICAL LEAKAGE: Target '{t}' found in feature list!"
        X = df[feature_cols].copy()
    else:
        # Drop all target columns and non-feature metadata
        drop_cols = [c for c in TARGET_COLUMNS if c in df.columns]
        X = df.drop(columns=drop_cols)

    return X, y_reg, y_cls


# Convenient alias
quarantine_target = separate_features_and_targets

