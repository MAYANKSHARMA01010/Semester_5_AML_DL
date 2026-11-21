"""
Phase 2 Task 6 Unit Tests — Explainability and Prediction Error Analysis
========================================================================
Tests for Integrated Gradients completeness, reference baseline validity,
residual sign conventions, and difficult cases integrity.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Phase_2_Deep_Learning.src.dataset import (
    load_feature_config,
    load_transformers,
    prepare_tensors,
    get_default_paths,
)
from Phase_2_Deep_Learning.src.dl_models import WaterQualityMLP, load_model_checkpoint
from Phase_2_Deep_Learning.src.hybrid_models import load_hybrid_checkpoint


def test_modal_baseline_construction():
    """Verify modal baseline has 0.0 numerical values and valid one-hot categorical sums."""
    paths = get_default_paths()
    cfg = load_feature_config(paths['feature_config'])
    scaler, encoder = load_transformers(paths['artifacts_dir'])
    test_df = pd.read_parquet(paths['data_dir'] / 'test.parquet').head(10)

    dummy_row = test_df.iloc[0:1].copy()
    for col in cfg['numerical_features']:
        dummy_row[col] = scaler.mean_[cfg['numerical_features'].index(col)]
    dummy_row['country_code'] = test_df['country_code'].mode()[0]
    dummy_row['waterbody_type_code'] = test_df['waterbody_type_code'].mode()[0]

    _, _, base_tensor, _ = prepare_tensors(
        dummy_row, scaler, encoder,
        cfg['numerical_features'], cfg['categorical_features'], cfg['target_variable']
    )

    num_dim = len(cfg['numerical_features'])
    cat_dim = len(cfg['one_hot_feature_names'])

    assert base_tensor.shape == (1, 53)
    # Check numerical elements are 0.0
    assert torch.allclose(base_tensor[0, :num_dim], torch.zeros(num_dim), atol=1e-5)
    # Check categorical elements represent valid one-hot categories
    country_sum = base_tensor[0, 36:41].sum().item()
    wb_sum = base_tensor[0, 41:53].sum().item()
    assert country_sum == pytest.approx(1.0, abs=1e-5)
    assert wb_sum == pytest.approx(1.0, abs=1e-5)


def test_integrated_gradients_completeness():
    """Verify sum of Integrated Gradients equals F(x) - F(baseline) within numerical tolerance."""
    paths = get_default_paths()
    cfg = load_feature_config(paths['feature_config'])
    scaler, encoder = load_transformers(paths['artifacts_dir'])
    test_df = pd.read_parquet(paths['data_dir'] / 'test.parquet').head(20)

    _, _, x_tensor, _ = prepare_tensors(
        test_df, scaler, encoder,
        cfg['numerical_features'], cfg['categorical_features'], cfg['target_variable']
    )

    model, _ = load_model_checkpoint(paths['phase2_dir'] / 'models' / 'mlp_best_checkpoint.pt', device=torch.device('cpu'))
    model.eval()

    dummy_row = test_df.iloc[0:1].copy()
    for col in cfg['numerical_features']:
        dummy_row[col] = scaler.mean_[cfg['numerical_features'].index(col)]
    dummy_row['country_code'] = test_df['country_code'].mode()[0]
    dummy_row['waterbody_type_code'] = test_df['waterbody_type_code'].mode()[0]

    _, _, base_tensor, _ = prepare_tensors(
        dummy_row, scaler, encoder,
        cfg['numerical_features'], cfg['categorical_features'], cfg['target_variable']
    )

    diff = x_tensor - base_tensor
    steps = 50
    alphas = torch.linspace(0.0, 1.0, steps)
    total_grads = torch.zeros_like(x_tensor)
    for alpha in alphas:
        x_step = (base_tensor + alpha * diff).detach().requires_grad_(True)
        out = model(x_step)
        grad = torch.autograd.grad(out.sum(), x_step)[0]
        total_grads += grad
    ig = diff * (total_grads / steps)

    with torch.no_grad():
        f_x = model(x_tensor).numpy().flatten()
        f_base = model(base_tensor).numpy().flatten()

    target_delta = f_x - f_base
    sum_attributions = ig.sum(dim=1).detach().numpy()
    errors = np.abs(sum_attributions - target_delta)

    # Mean completeness error should be small on the continuous output scale [0, 100]
    assert np.mean(errors) < 1.0
    assert np.all(np.isfinite(sum_attributions))


def test_residual_sign_convention():
    """Verify that Residual = Prediction - Actual, with positive = overprediction, negative = underprediction."""
    y_actual = np.array([80.0, 50.0])
    y_pred = np.array([85.0, 45.0])

    residuals = y_pred - y_actual
    assert residuals[0] == 5.0  # Overprediction
    assert residuals[1] == -5.0  # Underprediction


def test_difficult_cases_direction_consistency():
    """Verify that saved difficult cases overpredictions have positive residuals and underpredictions have negative residuals."""
    output_dir = REPO_ROOT / 'Phase_2_Deep_Learning' / 'reports' / 'task_6_outputs'
    over_csv = output_dir / 'difficult_cases_overpredictions.csv'
    under_csv = output_dir / 'difficult_cases_underpredictions.csv'

    assert over_csv.exists(), "Missing difficult_cases_overpredictions.csv"
    assert under_csv.exists(), "Missing difficult_cases_underpredictions.csv"

    df_over = pd.read_csv(over_csv)
    df_under = pd.read_csv(under_csv)

    assert len(df_over) > 0
    assert len(df_under) > 0

    # Overpredictions: Residual = Pred - Actual must be strictly positive
    assert (df_over['MLP_Residual'] > 0).all(), "All overprediction residuals must be positive"
    assert (df_over['Direction'] == 'Overpredicted').all()

    # Underpredictions: Residual = Pred - Actual must be strictly negative
    assert (df_under['MLP_Residual'] < 0).all(), "All underprediction residuals must be negative"
    assert (df_under['Direction'] == 'Underpredicted').all()


def test_residual_hybrid_prediction_and_gradient():
    """Verify Residual Hybrid computes valid residual correction r and supports backward gradients."""
    paths = get_default_paths()
    cfg = load_feature_config(paths['feature_config'])
    scaler, encoder = load_transformers(paths['artifacts_dir'])
    test_df = pd.read_parquet(paths['data_dir'] / 'test.parquet').head(10)

    _, _, x_tensor, _ = prepare_tensors(
        test_df, scaler, encoder,
        cfg['numerical_features'], cfg['categorical_features'], cfg['target_variable']
    )
    base_pred = torch.full((len(test_df), 1), 75.0)

    hybrid_model, _, _ = load_hybrid_checkpoint(paths['phase2_dir'] / 'models' / 'hybrid_ensemble_checkpoint.pt', device=torch.device('cpu'))
    hybrid_model.eval()

    inp = torch.cat([x_tensor, base_pred], dim=-1).requires_grad_(True)
    r = hybrid_model.residual_net(inp)
    assert r.shape == (len(test_df), 1)
    assert torch.all(torch.isfinite(r))

    grad = torch.autograd.grad(r.sum(), inp)[0]
    assert grad.shape == (len(test_df), 54)
    assert torch.all(torch.isfinite(grad))
