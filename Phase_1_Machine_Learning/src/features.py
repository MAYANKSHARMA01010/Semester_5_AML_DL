"""
Domain Feature Engineering & Outlier Filtering
Enforces aquatic chemistry bounds and constructs geochemical transformations.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

# Domain bounds based on aquatic chemistry standards
PHYSICAL_BOUNDS = {
    "ph": (3.0, 12.0),
    "temperature": (0.0, 45.0),
    "temperature_c": (0.0, 45.0),
    "dissolved_oxygen": (0.0, 25.0),
    "do_mg_l": (0.0, 25.0),
    "conductivity": (0.0, np.inf),
    "bod5": (0.0, np.inf),
    "bod_mg_l": (0.0, np.inf),
    "total_oxidised_nitrogen": (0.0, np.inf),
    "ammoniacal_nitrogen": (0.0, np.inf),
    "ammonia_mg_l": (0.0, np.inf),
    "orthophosphate": (0.0, np.inf),
    "orthophosphate_mg_l": (0.0, np.inf),
    "nitrogen_mg_l": (0.0, np.inf),
    "nitrate_mg_l": (0.0, np.inf),
}

SKEWED_NUTRIENTS = [
    "orthophosphate",
    "ammoniacal_nitrogen",
    "total_oxidised_nitrogen",
    "bod5",
]


def filter_aquatic_physics(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove rows violating aquatic chemistry physical constraints.
    Returns (cleaned_df, rows_removed).
    """
    initial_rows = len(df)
    valid_mask = pd.Series(True, index=df.index)

    for col, (lower, upper) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            valid_mask &= (df[col] >= lower) & (df[col] <= upper)

    cleaned = df[valid_mask].copy()
    rows_removed = initial_rows - len(cleaned)
    return cleaned, rows_removed


def compute_log1p_features(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    """
    Apply y = log(1 + x) transformation to tame extreme right-skewed nutrient distributions.
    """
    df_out = df.copy()
    target_cols = cols if cols is not None else SKEWED_NUTRIENTS
    for col in target_cols:
        if col in df_out.columns:
            # Ensure non-negative before log1p
            safe_vals = np.clip(df_out[col], 0, None)
            df_out[f"log1p_{col}"] = np.log1p(safe_vals)
    return df_out


def compute_stoichiometry_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct chemical stoichiometry ratios:
    - N:P molar/mass ratio
    - DO saturation percentage (Benson & Krause approximation)
    """
    df_out = df.copy()

    # N:P Ratio
    if "total_oxidised_nitrogen" in df_out.columns and "orthophosphate" in df_out.columns:
        p_safe = np.clip(df_out["orthophosphate"], 1e-4, None)
        df_out["ratio_n_to_p"] = df_out["total_oxidised_nitrogen"] / p_safe
        df_out["log1p_ratio_n_to_p"] = np.log1p(np.clip(df_out["ratio_n_to_p"], 0, None))

    # Theoretical DO Saturation at atmospheric pressure (Benson & Krause 1984 simplified)
    if "temperature" in df_out.columns and "dissolved_oxygen" in df_out.columns:
        t_c = np.clip(df_out["temperature"], 0.0, 45.0)
        # Empirical solubility curve approximation: DO_sat ~ 14.652 - 0.41022*T + 0.007991*T^2 - 0.000077774*T^3
        do_sat_theoretical = 14.652 - 0.41022 * t_c + 0.007991 * (t_c ** 2) - 0.000077774 * (t_c ** 3)
        do_sat_theoretical = np.clip(do_sat_theoretical, 5.0, 15.0)
        df_out["do_saturation_pct"] = (df_out["dissolved_oxygen"] / do_sat_theoretical) * 100.0

    return df_out


def compute_station_lags(
    df: pd.DataFrame,
    station_col: str = "Area",
    date_col: str = "Date",
    lag_cols: List[str] = None
) -> pd.DataFrame:
    """
    Compute chronological shift-1 historical lags per station without future lookahead.
    Missing historical values (first station observation) are filled with global historical medians.
    """
    if lag_cols is None:
        lag_cols = ["dissolved_oxygen", "orthophosphate", "temperature"]

    df_out = df.copy()
    if date_col in df_out.columns:
        df_out = df_out.sort_values(by=[station_col, date_col])

    for col in lag_cols:
        if col in df_out.columns:
            # Shift 1 within station group strictly enforces zero lookahead leakage
            lag_series = df_out.groupby(station_col)[col].shift(1)
            median_val = df_out[col].median()
            df_out[f"hist_{col}_lag1"] = lag_series.fillna(median_val)

    return df_out
