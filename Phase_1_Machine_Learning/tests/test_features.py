"""
Unit tests for domain physical bounds, log1p normalization, and station lags.
"""

import pytest
import numpy as np
import pandas as pd
from src.features import (
    filter_aquatic_physics,
    compute_log1p_features,
    compute_stoichiometry_ratios,
    compute_station_lags,
)


def test_filter_aquatic_physics():
    # Construct synthetic DataFrame with valid and invalid values
    df_raw = pd.DataFrame({
        "ph": [7.5, 2.5, 13.0, 8.0, 7.0],           # 2.5 and 13.0 violate [3.0, 12.0]
        "temperature": [15.0, 20.0, -5.0, 50.0, 18.0], # -5.0 and 50.0 violate [0.0, 45.0]
        "dissolved_oxygen": [9.0, 8.5, -1.0, 30.0, 10.0], # -1.0 and 30.0 violate [0.0, 25.0]
        "orthophosphate": [0.05, 0.1, -0.02, 0.08, 0.04],  # -0.02 violates non-negativity
    })

    cleaned, removed = filter_aquatic_physics(df_raw)

    # Row 0 (all valid) and Row 4 (all valid) should survive
    assert len(cleaned) == 2, f"Expected 2 valid rows, got {len(cleaned)}"
    assert removed == 3, f"Expected 3 removed rows, got {removed}"
    assert (cleaned["ph"] >= 3.0).all() and (cleaned["ph"] <= 12.0).all()
    assert (cleaned["temperature"] >= 0.0).all() and (cleaned["temperature"] <= 45.0).all()
    assert (cleaned["dissolved_oxygen"] >= 0.0).all() and (cleaned["dissolved_oxygen"] <= 25.0).all()
    assert (cleaned["orthophosphate"] >= 0.0).all()


def test_compute_log1p_features():
    df = pd.DataFrame({
        "orthophosphate": [0.0, 1.0, 10.0, 100.0],
        "bod5": [2.0, 5.0, 20.0, 50.0],
    })

    df_log = compute_log1p_features(df, cols=["orthophosphate", "bod5"])

    assert "log1p_orthophosphate" in df_log.columns
    assert "log1p_bod5" in df_log.columns
    assert np.isclose(df_log["log1p_orthophosphate"].iloc[0], 0.0)
    assert np.isclose(df_log["log1p_orthophosphate"].iloc[1], np.log1p(1.0))
    assert not df_log["log1p_orthophosphate"].isna().any()
    assert not np.isinf(df_log["log1p_orthophosphate"]).any()


def test_compute_stoichiometry_ratios():
    df = pd.DataFrame({
        "total_oxidised_nitrogen": [2.0, 5.0, 10.0],
        "orthophosphate": [0.1, 0.5, 1.0],
        "temperature": [10.0, 20.0, 25.0],
        "dissolved_oxygen": [8.0, 7.5, 6.0],
    })

    df_out = compute_stoichiometry_ratios(df)

    assert "ratio_n_to_p" in df_out.columns
    assert "log1p_ratio_n_to_p" in df_out.columns
    assert "do_saturation_pct" in df_out.columns
    assert (df_out["ratio_n_to_p"] > 0).all()
    assert (df_out["do_saturation_pct"] > 0).all()


def test_station_lags_no_leakage():
    # 2 stations, 3 dates each
    df = pd.DataFrame({
        "Area": ["Station_A", "Station_A", "Station_A", "Station_B", "Station_B", "Station_B"],
        "Date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-01", "2020-01-02", "2020-01-03"],
        "dissolved_oxygen": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
        "orthophosphate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "temperature": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
    })

    df_lags = compute_station_lags(df, station_col="Area", date_col="Date")

    # For Station_A:
    # Row 1 (2020-01-02) lag should be Row 0 value (10.0)
    # Row 2 (2020-01-03) lag should be Row 1 value (9.0)
    assert df_lags.loc[1, "hist_dissolved_oxygen_lag1"] == 10.0
    assert df_lags.loc[2, "hist_dissolved_oxygen_lag1"] == 9.0

    # For Station_B:
    # Row 4 (2020-01-02) lag should be Row 3 value (7.0), NOT Station_A value!
    assert df_lags.loc[4, "hist_dissolved_oxygen_lag1"] == 7.0
    assert df_lags.loc[5, "hist_dissolved_oxygen_lag1"] == 6.0
