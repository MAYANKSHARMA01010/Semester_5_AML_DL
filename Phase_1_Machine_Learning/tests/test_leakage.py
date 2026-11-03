"""
Unit tests for data leakage prevention and target quarantine.
"""

from pathlib import Path
import json
import pytest
import pandas as pd
from src.utils import separate_features_and_targets, TARGET_COLUMNS


def test_target_quarantine_function():
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0],
        "feature_2": [10.0, 20.0, 30.0],
        "ccme_values": [85.0, 92.0, 78.0],
        "ccme_wqi": ["Good", "Excellent", "Fair"],
        "ccme_wqi_code": [3, 4, 2],
    })

    X, y_reg, y_cls = separate_features_and_targets(df)

    # Ensure none of the target columns exist in X
    for col in TARGET_COLUMNS:
        assert col not in X.columns, f"Target column '{col}' leaked into feature matrix X!"

    assert len(X.columns) == 2
    assert list(y_reg) == [85.0, 92.0, 78.0]
    assert list(y_cls) == [3, 4, 2]


def test_metadata_feature_list_leakage_free():
    root = Path(__file__).resolve().parents[1]
    metadata_file = root / "data" / "GeneratedOutput" / "02.Preprocessing&FeatureSelection" / "preprocessing_metadata.json"

    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        features = metadata.get("input_features_ordered", [])

        assert len(features) == 38, f"Expected 38 engineered features, got {len(features)}"
        for target in TARGET_COLUMNS:
            assert target not in features, f"CRITICAL LEAKAGE: Target '{target}' is listed in official feature set!"
