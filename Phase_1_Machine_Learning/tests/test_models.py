"""
Unit tests for Phase 1 Machine Learning Models and Serialization.
Tests MeanBaselineRegressor, tree model instantiation, and loading.
"""

import pytest
import numpy as np
from pathlib import Path
from src.models import (
    MeanBaselineRegressor,
    create_random_forest_regressor,
    create_xgboost_regressor,
    load_tree_model,
)


def test_mean_baseline_regressor():
    X = np.random.randn(100, 5)
    y = np.array([80.0] * 50 + [90.0] * 50)  # Mean is 85.0

    model = MeanBaselineRegressor()
    model.fit(X, y)

    preds = model.predict(X[:10])
    assert len(preds) == 10
    assert np.allclose(preds, 85.0)


def test_create_tree_models():
    rf = create_random_forest_regressor(n_estimators=10, max_depth=5)
    assert rf.n_estimators == 10
    assert rf.max_depth == 5
    assert rf.max_features == "sqrt"

    xgb = create_xgboost_regressor(n_estimators=20, max_depth=4)
    assert xgb.n_estimators == 20
    assert xgb.max_depth == 4
    assert xgb.learning_rate == 0.05


def test_load_serialized_tree_models():
    root = Path(__file__).resolve().parents[1]
    rf_path = root / "data" / "GeneratedOutput" / "03.ModelTraining" / "random_forest_model.joblib"
    xgb_path = root / "data" / "GeneratedOutput" / "03.ModelTraining" / "xgboost_model.json"

    X_dummy = np.random.randn(5, 38)

    if rf_path.exists():
        rf = load_tree_model(str(rf_path))
        preds_rf = rf.predict(X_dummy)
        assert len(preds_rf) == 5

    if xgb_path.exists():
        xgb = load_tree_model(str(xgb_path))
        preds_xgb = xgb.predict(X_dummy)
        assert len(preds_xgb) == 5
