"""
Phase 2 Deep Learning — Task 4: Out-of-Fold Prediction Generator
================================================================
Generates leakage-free out-of-fold (OOF) predictions for Phase 1 tree models
(Random Forest and XGBoost) using KFold cross-validation on the training set.

Strict Design Principles:
1. Feature Separation: Tree models receive their canonical 38-feature representation.
2. Leakage Prevention: KFold splitting is performed strictly on the training partition.
   Validation and test observations are never exposed to fold training.
3. Verification: Exact 1-to-1 assignment assertions ensure no missing or duplicate predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Phase_1_Machine_Learning.src.models import (
    create_random_forest_regressor,
    create_xgboost_regressor,
)


def get_phase1_model_defaults(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Return exact hyperparameter configurations matching Phase 1 trained models.
    Verified against Phase 1 training notebook and serialized artifacts:
      - Random Forest: 30 trees, max_depth=14, min_samples_leaf=5, max_features=1.0
      - XGBoost: 150 trees, max_depth=6, lr=0.08, subsample=0.8, colsample=0.8, hist
    """
    return {
        'rf': {
            'n_estimators': 30,
            'max_depth': 14,
            'min_samples_leaf': 5,
            'max_features': 1.0,
            'random_state': random_state,
            'n_jobs': -1,
        },
        'xgb': {
            'n_estimators': 150,
            'max_depth': 6,
            'learning_rate': 0.08,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'random_state': random_state,
            'n_jobs': -1,
        },
    }


def generate_oof_predictions(
    X_train_tree: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    model_type: str = 'rf',
    n_splits: int = 5,
    random_state: int = 42,
    **model_kwargs: Any,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Generate out-of-fold predictions using KFold cross-validation on training data.

    Args:
        X_train_tree: (N, 38) numpy array or DataFrame containing the Phase 1 tree features.
        y_train: (N,) target array (ccme_values).
        model_type: 'rf' for Random Forest or 'xgb' for XGBoost.
        n_splits: Number of cross-validation folds (default: 5).
        random_state: Reproducibility seed.
        **model_kwargs: Hyperparameter overrides forwarded to the model constructor.

    Returns:
        oof_preds: (N,) array of out-of-fold predictions.
        fold_metrics: List of metric dicts for each fold (mae, rmse, r2).
    """
    if model_type not in ('rf', 'xgb'):
        raise ValueError(f"model_type must be 'rf' or 'xgb', got '{model_type}'")

    if isinstance(X_train_tree, pd.DataFrame):
        X_np = X_train_tree.to_numpy(dtype=np.float32)
    else:
        X_np = np.asarray(X_train_tree, dtype=np.float32)

    y_np = np.asarray(y_train, dtype=np.float64).flatten()

    n_samples = len(y_np)
    if len(X_np) != n_samples:
        raise ValueError(
            f"Row count mismatch between features ({len(X_np)}) and target ({n_samples})"
        )

    if np.isnan(X_np).any() or np.isnan(y_np).any():
        raise ValueError("Input data contains NaN values before OOF generation.")

    defaults = get_phase1_model_defaults(random_state=random_state)[model_type]
    defaults.update(model_kwargs)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_preds = np.zeros(n_samples, dtype=np.float64)
    assigned_counts = np.zeros(n_samples, dtype=np.int32)
    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_np)):
        X_fold_train, X_fold_val = X_np[train_idx], X_np[val_idx]
        y_fold_train, y_fold_val = y_np[train_idx], y_np[val_idx]

        if model_type == 'rf':
            model = RandomForestRegressor(**defaults)
        else:
            model = XGBRegressor(**defaults)

        model.fit(X_fold_train, y_fold_train)
        fold_preds = model.predict(X_fold_val)

        oof_preds[val_idx] = fold_preds
        assigned_counts[val_idx] += 1

        mae = float(mean_absolute_error(y_fold_val, fold_preds))
        mse = float(mean_squared_error(y_fold_val, fold_preds))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_fold_val, fold_preds))

        fold_metrics.append({
            'fold': fold_idx + 1,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'val_samples': len(val_idx),
        })

        print(
            f"  [{model_type.upper()} Fold {fold_idx + 1}/{n_splits}] "
            f"Val Samples: {len(val_idx):,}  |  MAE: {mae:.4f}  RMSE: {rmse:.4f}  R²: {r2:.4f}"
        )

    if not np.all(assigned_counts == 1):
        missing = int(np.sum(assigned_counts == 0))
        duplicates = int(np.sum(assigned_counts > 1))
        raise AssertionError(
            f"OOF assignment failure: {missing} missing, {duplicates} duplicates out of {n_samples} rows."
        )

    if np.isnan(oof_preds).any() or np.isinf(oof_preds).any():
        raise AssertionError("OOF predictions contain NaN or Inf values.")

    return oof_preds, fold_metrics


def generate_ensemble_oof(
    X_train_tree: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    n_splits: int = 5,
    random_state: int = 42,
    rf_kwargs: Optional[Dict[str, Any]] = None,
    xgb_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[Dict[str, float]]]]:
    """
    Generate out-of-fold predictions for both Random Forest and XGBoost.

    Args:
        X_train_tree: (N, 38) Phase 1 tree feature matrix.
        y_train: (N,) target array.
        n_splits: Number of cross-validation splits.
        random_state: Reproducibility seed.
        rf_kwargs: Optional hyperparameter overrides for Random Forest.
        xgb_kwargs: Optional hyperparameter overrides for XGBoost.

    Returns:
        oof_rf: (N,) RF out-of-fold predictions.
        oof_xgb: (N,) XGBoost out-of-fold predictions.
        fold_metrics_dict: Dictionary containing fold metrics for 'rf' and 'xgb'.
    """
    rf_kwargs = rf_kwargs or {}
    xgb_kwargs = xgb_kwargs or {}

    print("=" * 60)
    print(f"Generating Random Forest OOF predictions ({n_splits}-fold CV) ...")
    print("=" * 60)
    oof_rf, rf_metrics = generate_oof_predictions(
        X_train_tree=X_train_tree,
        y_train=y_train,
        model_type='rf',
        n_splits=n_splits,
        random_state=random_state,
        **rf_kwargs,
    )

    print()
    print("=" * 60)
    print(f"Generating XGBoost OOF predictions ({n_splits}-fold CV) ...")
    print("=" * 60)
    oof_xgb, xgb_metrics = generate_oof_predictions(
        X_train_tree=X_train_tree,
        y_train=y_train,
        model_type='xgb',
        n_splits=n_splits,
        random_state=random_state,
        **xgb_kwargs,
    )

    return oof_rf, oof_xgb, {'rf': rf_metrics, 'xgb': xgb_metrics}


def compute_residuals(
    y_true: Union[np.ndarray, pd.Series],
    base_pred: Union[np.ndarray, pd.Series],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute signed residuals: residual = y_true - base_pred.

    Args:
        y_true: Ground-truth target values.
        base_pred: Base model predictions.

    Returns:
        residuals: (N,) signed residual values.
        stats: Dictionary of diagnostic statistics.
    """
    y_arr = np.asarray(y_true, dtype=np.float64).flatten()
    pred_arr = np.asarray(base_pred, dtype=np.float64).flatten()

    if len(y_arr) != len(pred_arr):
        raise ValueError(f"Shape mismatch: y_true={len(y_arr)}, base_pred={len(pred_arr)}")

    residuals = y_arr - pred_arr

    stats = {
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'median': float(np.median(residuals)),
        'mae': float(np.mean(np.abs(residuals))),
        'rmse': float(np.sqrt(np.mean(residuals ** 2))),
        'max_abs': float(np.max(np.abs(residuals))),
        'pct_within_1': float(100.0 * np.mean(np.abs(residuals) <= 1.0)),
        'pct_within_5': float(100.0 * np.mean(np.abs(residuals) <= 5.0)),
    }
    return residuals, stats


def save_oof_artifacts(
    oof_rf: np.ndarray,
    oof_xgb: np.ndarray,
    residuals_rf: np.ndarray,
    residuals_xgb: np.ndarray,
    y_train: np.ndarray,
    save_dir: Union[str, Path],
    filename: str = 'oof_predictions.parquet',
) -> Path:
    """
    Persist OOF predictions and residuals to a parquet file.
    Includes both individual model residuals and combined base tree predictions.
    """
    save_path = Path(save_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_arr = np.asarray(y_train, dtype=np.float64).flatten()
    rf_arr = np.asarray(oof_rf, dtype=np.float64).flatten()
    xgb_arr = np.asarray(oof_xgb, dtype=np.float64).flatten()
    base_arr = 0.5 * (rf_arr + xgb_arr)

    df_oof = pd.DataFrame({
        'y_true': y_arr,
        'oof_rf': rf_arr,
        'oof_xgb': xgb_arr,
        'oof_base_tree': base_arr,
        'residual_rf': np.asarray(residuals_rf, dtype=np.float64).flatten(),
        'residual_xgb': np.asarray(residuals_xgb, dtype=np.float64).flatten(),
        'residual_base_tree': y_arr - base_arr,
    })

    df_oof.to_parquet(save_path, index=False)
    print(f"✓ OOF artifacts saved → {save_path} ({len(df_oof):,} rows)")
    return save_path


def load_oof_artifacts(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load cached OOF predictions from a parquet file or directory containing oof_predictions.parquet."""
    p = Path(file_path)
    if p.is_dir():
        p = p / 'oof_predictions.parquet'
    if not p.exists():
        raise FileNotFoundError(f"OOF predictions file not found at: {p}")
    df = pd.read_parquet(p)
    required_cols = {'y_true', 'oof_rf', 'oof_xgb'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"OOF file missing required columns: {required_cols - set(df.columns)}")
    return df
