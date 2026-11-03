"""
Evaluation Metrics & Calibration Calculations
Computes regression metrics (MAE, RMSE, R²), classification metrics (F1, Accuracy),
and probability calibration diagnostics (Brier Score, Expected Calibration Error).
"""

import numpy as np
from typing import Dict, Tuple, Any
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    brier_score_loss,
)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and R² for continuous CCME index predictions."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute Accuracy, Macro F1, and Weighted F1 for discrete WQI tiers."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def compute_brier_score(y_true: np.ndarray, y_probs: np.ndarray, num_classes: int = 5) -> float:
    """
    Compute multi-class Brier score.
    Lower score indicates better calibrated probability predictions (0 = perfect calibration).
    """
    n_samples = len(y_true)
    one_hot = np.zeros((n_samples, num_classes))
    for i, label in enumerate(y_true):
        one_hot[i, int(label)] = 1.0

    brier = float(np.mean(np.sum((y_probs - one_hot) ** 2, axis=1)))
    return brier


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE) for multi-class classification.
    Returns:
        ece: Scalar expected calibration error
        bin_accuracies: Accuracy in each confidence bin
        bin_confidences: Average confidence in each bin
        bin_counts: Number of samples in each bin
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            bin_accuracies.append(float(accuracy_in_bin))
            bin_confidences.append(float(avg_confidence_in_bin))
            bin_counts.append(int(np.sum(in_bin)))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(float((bin_boundaries[i] + bin_boundaries[i + 1]) / 2.0))
            bin_counts.append(0)

    return float(ece), np.array(bin_accuracies), np.array(bin_confidences), np.array(bin_counts)
