"""
Phase 2 Task 5 Unit Tests — Evaluation and Ablation Suite
========================================================
Tests for ModelEvaluator, FeatureAblationEngine, SubgroupEvaluator,
and ComputationalProfiler in Phase_2_Deep_Learning/src/evaluation.py.
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

# ── Ensure project root is on sys.path ────────────────────────────────────────
_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Phase_2_Deep_Learning.src.evaluation import (
    FEATURE_GROUPS,
    ModelEvaluator,
    FeatureAblationEngine,
    SubgroupEvaluator,
    ComputationalProfiler,
)
from Phase_2_Deep_Learning.src.training import compute_regression_metrics


# ══════════════════════════════════════════════════════════════════════════════
# 1. Feature Groups Integrity
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_groups_integrity():
    """Verify that all 7 official feature groups exist and have non-empty lists."""
    expected_groups = [
        "base_sensors",
        "stoichiometry",
        "log1p_normalized",
        "historical_memory",
        "environmental_stress",
        "contextual_categorical",
        "temporal_cyclical",
    ]
    for g in expected_groups:
        assert g in FEATURE_GROUPS, f"Missing expected feature group: {g}"
        assert len(FEATURE_GROUPS[g]) > 0, f"Feature group {g} is empty"

    # Verify key geochemical features are assigned to correct groups
    assert "orthophosphate_mg_l" in FEATURE_GROUPS["base_sensors"]
    assert "np_ratio" in FEATURE_GROUPS["stoichiometry"]
    assert "log1p_orthophosphate" in FEATURE_GROUPS["log1p_normalized"]
    assert "hist_do_lag1" in FEATURE_GROUPS["historical_memory"]
    assert "exceedance_counter" in FEATURE_GROUPS["environmental_stress"]
    assert "country_code" in FEATURE_GROUPS["contextual_categorical"]
    assert "year" in FEATURE_GROUPS["temporal_cyclical"]


# ══════════════════════════════════════════════════════════════════════════════
# 2. ModelEvaluator Initialization
# ══════════════════════════════════════════════════════════════════════════════

def test_model_evaluator_initialization():
    """Verify ModelEvaluator loads feature config and fitted scalers cleanly."""
    evaluator = ModelEvaluator(device=torch.device('cpu'))
    assert len(evaluator.num_features) == 36
    assert len(evaluator.cat_features) == 2
    assert len(evaluator.all_tree_features) == 38
    assert evaluator.target_col == "ccme_values"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Model Loading
# ══════════════════════════════════════════════════════════════════════════════

def test_model_evaluator_load_all():
    """Verify all 4 trained models (RF, XGBoost, MLP, Hybrid) load from disk."""
    evaluator = ModelEvaluator(device=torch.device('cpu'))
    evaluator.load_all_models()

    assert evaluator.rf_model is not None, "Failed to load Random Forest"
    assert evaluator.xgb_model is not None, "Failed to load XGBoost"
    assert evaluator.mlp_model is not None, "Failed to load MLP"
    assert evaluator.weighted_ensemble is not None, "Failed to create WeightedEnsemble"
    assert evaluator.hybrid_model is not None, "Failed to load HybridEnsemble"


# ══════════════════════════════════════════════════════════════════════════════
# 4. Metric Computation Consistency
# ══════════════════════════════════════════════════════════════════════════════

def test_regression_metric_calculations():
    """Verify MAE, RMSE, R² calculations match exact mathematical definitions."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 19.0, 32.0, 39.0])

    metrics = compute_regression_metrics(y_true, y_pred)

    expected_mae = (2.0 + 1.0 + 2.0 + 1.0) / 4.0  # 1.5
    expected_mse = (4.0 + 1.0 + 4.0 + 1.0) / 4.0  # 2.5
    expected_rmse = np.sqrt(expected_mse)

    assert metrics["mae"] == pytest.approx(expected_mae, abs=1e-5)
    assert metrics["rmse"] == pytest.approx(expected_rmse, abs=1e-5)
    assert metrics["r2"] > 0.95  # High correlation


# ══════════════════════════════════════════════════════════════════════════════
# 5. MC Dropout BatchNorm Safeguard
# ══════════════════════════════════════════════════════════════════════════════

def test_mc_dropout_batchnorm_preservation():
    """Verify MC dropout keeps BatchNorm in eval mode and returns finite predictions."""
    import torch.nn as nn
    from Phase_2_Deep_Learning.src.dl_models import mc_dropout_predictions

    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 8)
            self.bn = nn.BatchNorm1d(8)
            self.drop = nn.Dropout(0.5)
            self.fc2 = nn.Linear(8, 1)

        def forward(self, x):
            assert not self.bn.training, "BatchNorm must stay in eval mode during MC sampling!"
            x = self.fc1(x)
            x = self.bn(x)
            x = self.drop(x)
            return self.fc2(x)

    dummy = DummyNet()
    x = torch.randn(5, 10)
    mean_p, std_p = mc_dropout_predictions(dummy, x, n_samples=10, device=torch.device('cpu'))

    assert mean_p.shape == (5,)
    assert std_p.shape == (5,)
    assert np.all(np.isfinite(mean_p))
    assert np.all(np.isfinite(std_p))
    assert not dummy.training, "Model should be in eval mode after sampling"
    assert not dummy.bn.training, "BatchNorm should remain in eval mode"
    assert not dummy.drop.training, "Dropout should be set to eval mode after sampling"


# ══════════════════════════════════════════════════════════════════════════════
# 6. Prediction Diversity and Model Separation
# ══════════════════════════════════════════════════════════════════════════════

def test_model_evaluator_predict_all_models():
    """Verify predict_all returns all 6 distinct model predictions with no duplicates."""
    evaluator = ModelEvaluator(device=torch.device('cpu'))
    evaluator.load_all_models()

    test_path = REPO_ROOT / 'Phase_1_Machine_Learning' / 'data' / 'GeneratedOutput' / '02.Preprocessing&FeatureSelection' / 'test.parquet'
    test_df = pd.read_parquet(test_path).head(100)

    preds = evaluator.predict_all(test_df)
    expected_models = [
        'Random Forest',
        'XGBoost',
        'MLP',
        'Base Tree Ensemble',
        'Weighted Ensemble',
        'Residual Hybrid',
    ]
    for m in expected_models:
        assert m in preds, f"Missing prediction for model {m}"
        assert len(preds[m]) == 100
        assert np.all(np.isfinite(preds[m]))

    # Verify Residual Hybrid is genuinely distinct from Weighted Ensemble
    diff_hybrid_weighted = np.abs(preds['Residual Hybrid'] - preds['Weighted Ensemble'])
    assert np.max(diff_hybrid_weighted) > 1e-3, "Residual Hybrid must not duplicate Weighted Ensemble"

    # Verify Residual Hybrid is distinct from Base Tree Ensemble
    diff_hybrid_base = np.abs(preds['Residual Hybrid'] - preds['Base Tree Ensemble'])
    assert np.max(diff_hybrid_base) > 1e-4, "Residual Hybrid must not duplicate Base Tree Ensemble"

    # Evaluate predictions
    metrics_df = evaluator.evaluate_predictions(preds)
    assert len(metrics_df) == 6
    assert 'MAE' in metrics_df.columns
    assert 'RMSE' in metrics_df.columns
    assert 'R2' in metrics_df.columns
    assert 'MedAE' in metrics_df.columns
    assert 'MaxAE' in metrics_df.columns
    assert 'ExplainedVariance' in metrics_df.columns


# ══════════════════════════════════════════════════════════════════════════════
# 5. Subgroup Evaluator Safeguards
# ══════════════════════════════════════════════════════════════════════════════

def test_subgroup_evaluator_min_samples():
    """Verify subgroup evaluation flags categories below min_samples threshold."""
    df_mock = pd.DataFrame({
        "country": ["A"] * 150 + ["B"] * 20,
    })
    preds_mock = {
        "y_true": np.ones(170) * 50.0,
        "MockModel": np.ones(170) * 51.0,
    }

    res = SubgroupEvaluator.evaluate_by_group(
        df=df_mock,
        preds_dict=preds_mock,
        group_col="country",
        min_samples=100,
    )

    assert len(res) == 2
    row_a = res[res["Category"] == "A"].iloc[0]
    row_b = res[res["Category"] == "B"].iloc[0]

    assert row_a["Reliable_Sample"] is True or row_a["Reliable_Sample"] == True
    assert row_b["Reliable_Sample"] is False or row_b["Reliable_Sample"] == False
    assert row_a["Sample_Count"] == 150
    assert row_b["Sample_Count"] == 20
