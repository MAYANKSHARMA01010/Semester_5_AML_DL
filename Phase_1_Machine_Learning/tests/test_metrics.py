"""
Unit tests for regression metrics, classification metrics, and calibration.
"""

import pytest
import numpy as np
from src.metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
    compute_brier_score,
    compute_expected_calibration_error,
)


def test_compute_regression_metrics():
    y_true = np.array([80.0, 85.0, 90.0, 95.0, 100.0])
    y_pred = np.array([80.0, 85.0, 90.0, 95.0, 100.0])

    res = compute_regression_metrics(y_true, y_pred)
    assert np.isclose(res["mae"], 0.0)
    assert np.isclose(res["rmse"], 0.0)
    assert np.isclose(res["r2"], 1.0)

    # Offset by 1
    y_pred_off = y_true + 1.0
    res_off = compute_regression_metrics(y_true, y_pred_off)
    assert np.isclose(res_off["mae"], 1.0)
    assert np.isclose(res_off["rmse"], 1.0)


def test_compute_classification_metrics():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])

    res = compute_classification_metrics(y_true, y_pred)
    assert np.isclose(res["accuracy"], 1.0)
    assert np.isclose(res["macro_f1"], 1.0)


def test_brier_score_and_ece():
    # 3 samples, 5 classes
    y_true = np.array([0, 2, 4])
    # Perfect probability matrix
    y_probs_perfect = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])

    brier_perfect = compute_brier_score(y_true, y_probs_perfect, num_classes=5)
    assert np.isclose(brier_perfect, 0.0), f"Perfect calibration should have Brier score 0, got {brier_perfect}"

    ece, accs, confs, counts = compute_expected_calibration_error(y_true, y_probs_perfect, n_bins=5)
    assert np.isclose(ece, 0.0), f"Perfect calibration should have ECE 0, got {ece}"
