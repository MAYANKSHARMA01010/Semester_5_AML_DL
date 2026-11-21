"""
Phase 4 Unit Tests — Hybrid Model Components
Tests for HybridConfig, ResidualMLPNet, WeightedEnsemble, OOF generation,
residual computation, and ResidualDataset.

Run with:
    pytest Phase_2_Deep_Learning/tests/test_hybrid.py -v
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# ── Add repo root to sys.path ─────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Phase_2_Deep_Learning.src.hybrid_models import (
    HybridConfig,
    ResidualMLPNet,
    HybridEnsemble,
    WeightedEnsemble,
    save_hybrid_checkpoint,
    load_hybrid_checkpoint,
)
from Phase_2_Deep_Learning.src.oof_generator import compute_residuals, generate_oof_predictions
from Phase_2_Deep_Learning.src.residual_trainer import ResidualDataset


# ══════════════════════════════════════════════════════════════════════════════
# 1. HybridConfig defaults
# ══════════════════════════════════════════════════════════════════════════════

def test_hybrid_config_defaults():
    """Verify that HybridConfig instantiates with correct default field values."""
    cfg = HybridConfig()

    assert cfg.input_dim       == 53
    assert cfg.hidden_dims     == [256, 128, 64]
    assert cfg.dropout_rate    == pytest.approx(0.2, abs=1e-9)
    assert cfg.use_batch_norm  is True
    assert cfg.rf_weight       == pytest.approx(0.4, abs=1e-9)
    assert cfg.xgb_weight      == pytest.approx(0.4, abs=1e-9)
    assert cfg.mlp_weight      == pytest.approx(0.2, abs=1e-9)
    assert cfg.residual_weight == pytest.approx(1.0, abs=1e-9)
    assert cfg.name            == 'Hybrid_RF_XGB_ResidualMLP'

    # Derived property
    assert cfg.residual_input_dim == 54

    # Round-trip serialisation
    d   = cfg.to_dict()
    cfg2 = HybridConfig.from_dict(d)
    assert cfg2.input_dim    == cfg.input_dim
    assert cfg2.hidden_dims  == cfg.hidden_dims
    assert cfg2.name         == cfg.name


# ══════════════════════════════════════════════════════════════════════════════
# 2. ResidualMLPNet — forward pass output shape
# ══════════════════════════════════════════════════════════════════════════════

def test_residual_mlp_forward():
    """ResidualMLPNet(input_dim=53) should produce shape (8, 1) for batch of 8."""
    cfg   = HybridConfig(input_dim=53)
    model = ResidualMLPNet(cfg)
    model.eval()

    x = torch.randn(8, 54)  # 53 features + 1 base pred
    with torch.no_grad():
        out = model(x)

    assert out.shape == (8, 1), f"Expected (8, 1), got {out.shape}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. ResidualMLPNet — no NaN in output
# ══════════════════════════════════════════════════════════════════════════════

def test_residual_mlp_no_nan():
    """Output must contain no NaN values for random 54-dim input."""
    cfg   = HybridConfig(input_dim=53)
    model = ResidualMLPNet(cfg)
    model.eval()

    x = torch.randn(32, 54)
    with torch.no_grad():
        out = model(x)

    assert not torch.isnan(out).any(), "NaN detected in ResidualMLPNet output"
    assert not torch.isinf(out).any(), "Inf detected in ResidualMLPNet output"


# ══════════════════════════════════════════════════════════════════════════════
# 4. WeightedEnsemble — raises ValueError if weights don't sum to 1
# ══════════════════════════════════════════════════════════════════════════════

def test_weighted_ensemble_sum_to_one():
    """WeightedEnsemble must raise ValueError when weights don't sum to 1 within 1e-5."""
    with pytest.raises(ValueError, match="sum to 1"):
        WeightedEnsemble(rf_weight=0.5, xgb_weight=0.5, mlp_weight=0.1)

    # These should NOT raise (valid weights)
    we1 = WeightedEnsemble(rf_weight=0.4,  xgb_weight=0.4,  mlp_weight=0.2)
    assert we1 is not None

    we2 = WeightedEnsemble(rf_weight=1.0,  xgb_weight=0.0,  mlp_weight=0.0)
    assert we2 is not None

    # At the tolerance boundary — should raise
    with pytest.raises(ValueError):
        WeightedEnsemble(rf_weight=0.4, xgb_weight=0.4, mlp_weight=0.2 + 1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# 5. WeightedEnsemble — predict output shape and range
# ══════════════════════════════════════════════════════════════════════════════

def test_weighted_ensemble_predict():
    """WeightedEnsemble.predict() must return shape (100,) with reasonable values."""
    we = WeightedEnsemble(rf_weight=0.4, xgb_weight=0.4, mlp_weight=0.2)

    rng      = np.random.default_rng(0)
    rf_preds  = rng.uniform(0, 100, size=100)
    xgb_preds = rng.uniform(0, 100, size=100)
    mlp_preds = rng.uniform(0, 100, size=100)

    out = we.predict(rf_preds, xgb_preds, mlp_preds)

    assert out.shape == (100,), f"Expected shape (100,), got {out.shape}"
    # Weighted average of values in [0,100] must also be in [0,100]
    assert float(out.min()) >= 0.0, "Output below 0"
    assert float(out.max()) <= 100.0, "Output above 100"

    # Verify arithmetic correctness on a small case
    rf_s  = np.array([10.0])
    xgb_s = np.array([20.0])
    mlp_s = np.array([30.0])
    expected = 0.4 * 10 + 0.4 * 20 + 0.2 * 30  # = 18.0
    result   = we.predict(rf_s, xgb_s, mlp_s)
    assert result[0] == pytest.approx(expected, abs=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# 6. compute_residuals — values and stats keys
# ══════════════════════════════════════════════════════════════════════════════

def test_compute_residuals():
    """Residuals must equal y_true - base_pred; stats dict must have required keys."""
    rng        = np.random.default_rng(1)
    y_true     = rng.uniform(0, 100, size=200)
    base_pred  = rng.uniform(0, 100, size=200)

    residuals, stats = compute_residuals(y_true, base_pred)

    # Shape
    assert residuals.shape == (200,)

    # Values: residuals = y_true - base_pred
    np.testing.assert_allclose(residuals, y_true - base_pred, rtol=1e-6)

    # Stats keys
    for key in ('mean', 'std', 'max_abs', 'pct_within_5'):
        assert key in stats, f"Missing stats key: '{key}'"

    assert stats['max_abs'] >= 0
    assert 0.0 <= stats['pct_within_5'] <= 100.0


# ══════════════════════════════════════════════════════════════════════════════
# 7. ResidualDataset — length and item shapes
# ══════════════════════════════════════════════════════════════════════════════

def test_residual_dataset():
    """ResidualDataset must return correct length and item shapes (54,) and (1,)."""
    N = 50
    X_feat   = torch.randn(N, 53)
    base_p   = torch.randn(N, 1)
    residuals= torch.randn(N, 1)

    ds = ResidualDataset(X_feat, base_p, residuals)

    assert len(ds) == N, f"Expected length {N}, got {len(ds)}"

    x_in, res = ds[0]
    assert x_in.shape == (54,), f"Expected x_in shape (54,), got {x_in.shape}"
    assert res.shape  == (1,),  f"Expected residual shape (1,), got {res.shape}"

    # Verify concatenation: last element of x_in should equal base_preds[0]
    assert x_in[-1].item() == pytest.approx(base_p[0].item(), abs=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# 8. OOF shape matches input
# ══════════════════════════════════════════════════════════════════════════════

def test_oof_shape_matches_input():
    """generate_oof_predictions must return an array of the same length as input."""
    rng     = np.random.default_rng(42)
    N       = 100
    n_feats = 53

    X = rng.standard_normal((N, n_feats)).astype(np.float32)
    y = rng.uniform(0, 100, size=N).astype(np.float32)

    # Use RF (lighter than XGB for unit tests)
    oof_preds, fold_metrics = generate_oof_predictions(
        X, y,
        model_type   = 'rf',
        n_splits     = 5,
        random_state = 0,
        n_estimators = 10,   # small for speed
        max_depth    = 3,
        n_jobs       = 1,
    )

    assert oof_preds.shape == (N,), f"Expected OOF shape ({N},), got {oof_preds.shape}"
    assert len(fold_metrics) == 5, f"Expected 5 fold metrics, got {len(fold_metrics)}"

    for m in fold_metrics:
        assert 'mae'  in m
        assert 'rmse' in m
        assert 'r2'   in m
        assert m['mae'] >= 0


# ══════════════════════════════════════════════════════════════════════════════
# 9. save_hybrid_checkpoint / load_hybrid_checkpoint round-trip
# ══════════════════════════════════════════════════════════════════════════════

def test_save_load_hybrid_checkpoint():
    """Save and reload HybridEnsemble; verify config matches and predictions are identical."""
    cfg   = HybridConfig(input_dim=53, hidden_dims=[64, 32], dropout_rate=0.1)
    model = HybridEnsemble(cfg)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / 'test_hybrid.pt'

        save_hybrid_checkpoint(
            model    = model,
            config   = cfg,
            path     = ckpt_path,
            metadata = {'test_mae': 1.23},
        )

        # Load back
        loaded_model, loaded_cfg, ckpt_dict = load_hybrid_checkpoint(
            ckpt_path, device=torch.device('cpu')
        )

    # Config must match
    assert loaded_cfg.input_dim    == cfg.input_dim
    assert loaded_cfg.hidden_dims  == cfg.hidden_dims
    assert loaded_cfg.dropout_rate == pytest.approx(cfg.dropout_rate, abs=1e-9)
    assert loaded_cfg.name         == cfg.name

    # Metadata preserved
    assert 'test_mae' in ckpt_dict

    # Predictions must be identical (same weights loaded)
    rng        = np.random.default_rng(99)
    X_np       = rng.standard_normal((16, 53)).astype(np.float32)
    base_np    = rng.uniform(0, 100, 16).astype(np.float32)

    with torch.no_grad():
        preds_orig   = model.predict_numpy(X_np, base_np, device=torch.device('cpu'))
        preds_loaded = loaded_model.predict_numpy(X_np, base_np, device=torch.device('cpu'))

    np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5, atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# 10. Integration Contract Tests: 38-feature Tree vs 53-feature DL
# ══════════════════════════════════════════════════════════════════════════════

def test_phase1_tree_models_require_38_features():
    """Verify that actual serialized Phase 1 tree models expect exactly 38 features."""
    from Phase_1_Machine_Learning.src.models import load_tree_model
    from Phase_2_Deep_Learning.src.dataset import get_default_paths

    paths = get_default_paths()
    p1_models = paths['phase1_dir'] / 'data' / 'GeneratedOutput' / '03.ModelTraining'
    rf_path = p1_models / 'random_forest_model.joblib'
    xgb_path = p1_models / 'xgboost_model.json'

    if not rf_path.exists() or not xgb_path.exists():
        pytest.skip("Phase 1 model artifacts not found; skipping integration contract test.")

    rf = load_tree_model(str(rf_path), model_type='rf')
    xgb = load_tree_model(str(xgb_path), model_type='xgb')

    assert getattr(rf, 'n_features_in_', None) == 38
    assert getattr(xgb, 'n_features_in_', None) == 38

    # 38-feature prediction must succeed
    x_38 = np.ones((5, 38), dtype=np.float32)
    rf_out = rf.predict(x_38)
    xgb_out = xgb.predict(x_38)
    assert len(rf_out) == 5
    assert len(xgb_out) == 5

    # Passing 53-feature DL matrix to RF MUST raise an error
    x_53 = np.ones((5, 53), dtype=np.float32)
    with pytest.raises(ValueError):
        rf.predict(x_53)


def test_residual_mlp_requires_54_features():
    """Verify that ResidualMLPNet requires exactly 54 inputs (53 features + 1 base pred)."""
    cfg = HybridConfig(input_dim=53)
    net = ResidualMLPNet(cfg)
    net.eval()

    # Valid 54-dim input succeeds
    valid_x = torch.randn(4, 54)
    with torch.no_grad():
        out = net(valid_x)
    assert out.shape == (4, 1)

    # 53-dim or 38-dim input must raise ValueError
    with pytest.raises(ValueError, match="dimension mismatch"):
        net(torch.randn(4, 53))

    with pytest.raises(ValueError, match="dimension mismatch"):
        net(torch.randn(4, 38))


def test_hybrid_ensemble_batched_predict():
    """Verify that HybridEnsemble.predict_numpy handles batched inference without error."""
    cfg = HybridConfig(input_dim=53)
    model = HybridEnsemble(cfg)
    model.eval()

    n_samples = 500
    x_features = np.random.randn(n_samples, 53).astype(np.float32)
    base_pred = np.random.uniform(40, 100, size=n_samples).astype(np.float32)

    # Test with batch_size smaller than total samples
    preds = model.predict_numpy(x_features, base_pred, batch_size=128)
    assert len(preds) == n_samples
    assert not np.isnan(preds).any()
    assert not np.isinf(preds).any()


def test_oof_assertions_prevent_leakage_and_alignment():
    """Verify that generate_oof_predictions validates alignment and rejects NaNs."""
    X = np.ones((20, 38), dtype=np.float32)
    y = np.ones(20, dtype=np.float64)

    # Row mismatch should raise ValueError
    with pytest.raises(ValueError, match="Row count mismatch"):
        generate_oof_predictions(X[:10], y, model_type='rf', n_splits=2)

    # NaN in inputs should raise ValueError
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        generate_oof_predictions(X_nan, y, model_type='rf', n_splits=2)
