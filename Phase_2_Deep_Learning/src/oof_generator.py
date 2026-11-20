"""
Phase 4 — Out-of-Fold Prediction Generator
===========================================
Generates out-of-fold predictions for RF and XGBoost using KFold cross-validation
on the training set, ensuring no data leakage into the residual MLP training.

Functions
---------
generate_oof_predictions  — KFold OOF predictions for a single model type.
generate_ensemble_oof     — OOF predictions for both RF and XGBoost.
compute_residuals         — Signed residuals + diagnostic statistics.
save_oof_artifacts        — Persist OOF predictions to parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Add project root to sys.path so Phase 1 src is importable ─────────────────
_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]  # Phase_2_Deep_Learning/src → repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Phase_1_Machine_Learning.src.models import (
    create_random_forest_regressor,
    create_xgboost_regressor,
)


# ══════════════════════════════════════════════════════════════════════════════
# OOF Prediction Generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_oof_predictions(
    X_train_np  : np.ndarray,
    y_train_np  : np.ndarray,
    model_type  : str = 'rf',
    n_splits    : int = 5,
    random_state: int = 42,
    **model_kwargs: Any,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Generate out-of-fold predictions using KFold cross-validation.

    Trains a fresh model on each fold's training subset and predicts on the
    held-out validation subset, assembling full training-set OOF predictions
    with zero data leakage.

    Args
    ----
    X_train_np   : (N, 53) numpy feature matrix for the training split.
    y_train_np   : (N,) numpy target array.
    model_type   : 'rf' for RandomForest or 'xgb' for XGBoost.
    n_splits     : Number of KFold splits (default 5).
    random_state : Reproducibility seed.
    **model_kwargs: Additional keyword arguments forwarded to the model factory.

    Returns
    -------
    oof_preds    : (N,) numpy array of out-of-fold predictions.
    fold_metrics : List of dicts with keys 'fold', 'mae', 'rmse', 'r2' per fold.
    """
    if model_type not in ('rf', 'xgb'):
        raise ValueError(f"model_type must be 'rf' or 'xgb', got '{model_type}'")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_preds    = np.zeros(len(y_train_np), dtype=np.float64)
    fold_metrics : List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_np)):
        X_fold_train, X_fold_val = X_train_np[train_idx], X_train_np[val_idx]
        y_fold_train, y_fold_val = y_train_np[train_idx], y_train_np[val_idx]

        # Instantiate a fresh model for this fold
        if model_type == 'rf':
            model = create_random_forest_regressor(random_state=random_state, **model_kwargs)
        else:
            model = create_xgboost_regressor(random_state=random_state, **model_kwargs)

        model.fit(X_fold_train, y_fold_train)
        fold_preds = model.predict(X_fold_val)
        oof_preds[val_idx] = fold_preds

        # Compute per-fold metrics
        mae  = float(mean_absolute_error(y_fold_val, fold_preds))
        mse  = float(mean_squared_error(y_fold_val, fold_preds))
        rmse = float(np.sqrt(mse))
        r2   = float(r2_score(y_fold_val, fold_preds))

        fold_metrics.append({
            'fold': fold_idx + 1,
            'mae' : mae,
            'rmse': rmse,
            'r2'  : r2,
        })

        print(
            f"  [{model_type.upper()} Fold {fold_idx + 1}/{n_splits}] "
            f"MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}"
        )

    return oof_preds, fold_metrics


def generate_ensemble_oof(
    X_train_np  : np.ndarray,
    y_train_np  : np.ndarray,
    n_splits    : int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[Dict[str, float]]]]:
    """
    Generate out-of-fold predictions for both Random Forest and XGBoost.

    Args
    ----
    X_train_np   : (N, 53) numpy feature matrix for the training split.
    y_train_np   : (N,) numpy target array.
    n_splits     : Number of KFold splits (default 5).
    random_state : Reproducibility seed.

    Returns
    -------
    oof_rf           : (N,) RF out-of-fold predictions.
    oof_xgb          : (N,) XGBoost out-of-fold predictions.
    fold_metrics_dict: Dict with keys 'rf' and 'xgb', each a list of per-fold metric dicts.
    """
    print("=" * 60)
    print("Generating RF out-of-fold predictions …")
    print("=" * 60)
    oof_rf, rf_metrics = generate_oof_predictions(
        X_train_np, y_train_np,
        model_type='rf', n_splits=n_splits, random_state=random_state,
    )

    print()
    print("=" * 60)
    print("Generating XGBoost out-of-fold predictions …")
    print("=" * 60)
    oof_xgb, xgb_metrics = generate_oof_predictions(
        X_train_np, y_train_np,
        model_type='xgb', n_splits=n_splits, random_state=random_state,
    )

    fold_metrics_dict = {
        'rf' : rf_metrics,
        'xgb': xgb_metrics,
    }
    return oof_rf, oof_xgb, fold_metrics_dict


# ══════════════════════════════════════════════════════════════════════════════
# Residual Computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_residuals(
    y_true_np   : np.ndarray,
    base_pred_np: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute signed residuals: residuals = y_true − base_pred.

    Args
    ----
    y_true_np    : (N,) ground-truth CCME WQI values.
    base_pred_np : (N,) base model (RF/XGB) predictions.

    Returns
    -------
    residuals : (N,) signed residual array.
    stats     : Dict with diagnostic statistics:
                  mean, std, max_abs, pct_within_5
    """
    y_true_np    = np.asarray(y_true_np,    dtype=np.float64).flatten()
    base_pred_np = np.asarray(base_pred_np, dtype=np.float64).flatten()

    residuals = y_true_np - base_pred_np

    stats = {
        'mean'        : float(np.mean(residuals)),
        'std'         : float(np.std(residuals)),
        'max_abs'     : float(np.max(np.abs(residuals))),
        'pct_within_5': float(100.0 * np.mean(np.abs(residuals) <= 5.0)),
    }

    return residuals, stats


# ══════════════════════════════════════════════════════════════════════════════
# OOF Artifact Persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_oof_artifacts(
    oof_rf       : np.ndarray,
    oof_xgb      : np.ndarray,
    residuals_rf : np.ndarray,
    residuals_xgb: np.ndarray,
    y_train      : np.ndarray,
    save_dir     : Path,
) -> Path:
    """
    Persist OOF predictions and residuals to a parquet file.

    Columns: y_true, oof_rf, oof_xgb, residual_rf, residual_xgb.

    Args
    ----
    oof_rf        : (N,) RF out-of-fold predictions.
    oof_xgb       : (N,) XGBoost out-of-fold predictions.
    residuals_rf  : (N,) RF residuals (y_true − oof_rf).
    residuals_xgb : (N,) XGB residuals (y_true − oof_xgb).
    y_train       : (N,) ground-truth training targets.
    save_dir      : Directory where `oof_predictions.parquet` will be saved.

    Returns
    -------
    Path to saved parquet file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df_oof = pd.DataFrame({
        'y_true'      : np.asarray(y_train).flatten(),
        'oof_rf'      : np.asarray(oof_rf).flatten(),
        'oof_xgb'     : np.asarray(oof_xgb).flatten(),
        'residual_rf' : np.asarray(residuals_rf).flatten(),
        'residual_xgb': np.asarray(residuals_xgb).flatten(),
    })

    out_path = save_dir / 'oof_predictions.parquet'
    df_oof.to_parquet(out_path, index=False)
    print(f'✓ OOF artifacts saved → {out_path}  ({len(df_oof):,} rows)')
    return out_path
