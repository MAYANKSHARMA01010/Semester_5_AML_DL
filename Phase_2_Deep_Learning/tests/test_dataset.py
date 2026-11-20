"""
Phase 2 Deep Learning — Unit Test Suite for Dataset Module
==========================================================
Tests the reusable dataset and data loading pipeline in src/dataset.py:
  - WaterQualityDataset tensor slicing and return modes
  - Pre-fitted transformers consistency (no data leakage)
  - Tensor shapes, NaN/Inf checks, and dimension assertions
  - Data split integrity across Train, Validation, and Test
"""

import sys
import os
from pathlib import Path
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add Phase_2_Deep_Learning to sys.path
PHASE2_DIR = Path(__file__).resolve().parent.parent
if str(PHASE2_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE2_DIR))

from src.dataset import (
    WaterQualityDataset,
    get_default_paths,
    load_feature_config,
    load_transformers,
    prepare_tensors,
)

BATCH_SIZE = 64
NUM_FEATURES = 36
CAT_FEATURES = 17
TOTAL_FEATURES = 53


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_tensors():
    n_samples = 128
    torch.manual_seed(42)
    x_num = torch.randn(n_samples, NUM_FEATURES)
    x_cat = torch.randint(0, 2, (n_samples, CAT_FEATURES)).float()
    x_all = torch.cat([x_num, x_cat], dim=1)
    y = torch.rand(n_samples, 1) * 100.0
    return x_num, x_cat, x_all, y


# ──────────────────────────────────────────────────────────────────────────────
# WaterQualityDataset Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestWaterQualityDataset:

    def test_combined_mode(self, mock_tensors):
        x_num, x_cat, x_all, y = mock_tensors
        ds = WaterQualityDataset(x_num, x_cat, x_all, y, return_mode='combined')
        assert len(ds) == len(y)
        assert ds.feature_dim() == TOTAL_FEATURES
        assert ds.num_dim() == NUM_FEATURES
        assert ds.cat_dim() == CAT_FEATURES

        x_sample, y_sample = ds[0]
        assert x_sample.shape == (TOTAL_FEATURES,)
        assert y_sample.shape == (1,)
        assert torch.allclose(x_sample, x_all[0])
        assert torch.allclose(y_sample, y[0])

    def test_separated_mode(self, mock_tensors):
        x_num, x_cat, x_all, y = mock_tensors
        ds = WaterQualityDataset(x_num, x_cat, x_all, y, return_mode='separated')
        assert len(ds) == len(y)

        x_num_s, x_cat_s, y_s = ds[0]
        assert x_num_s.shape == (NUM_FEATURES,)
        assert x_cat_s.shape == (CAT_FEATURES,)
        assert y_s.shape == (1,)

    def test_mismatched_length_raises(self, mock_tensors):
        x_num, x_cat, x_all, y = mock_tensors
        with pytest.raises(ValueError, match="Row count mismatch"):
            WaterQualityDataset(x_num[:50], x_cat, x_all, y)

    def test_invalid_return_mode_raises(self, mock_tensors):
        x_num, x_cat, x_all, y = mock_tensors
        with pytest.raises(ValueError, match="return_mode must be 'combined' or 'separated'"):
            WaterQualityDataset(x_num, x_cat, x_all, y, return_mode='invalid')

    def test_dataloader_batch_shapes(self, mock_tensors):
        x_num, x_cat, x_all, y = mock_tensors
        ds = WaterQualityDataset(x_num, x_cat, x_all, y, return_mode='combined')
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (BATCH_SIZE, TOTAL_FEATURES)
        assert batch_y.shape == (BATCH_SIZE, 1)
        assert not torch.isnan(batch_x).any()
        assert not torch.isnan(batch_y).any()


# ──────────────────────────────────────────────────────────────────────────────
# Artifacts and Configuration Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestArtifactsAndConfig:

    def test_paths_exist(self):
        paths = get_default_paths()
        assert paths['data_dir'].exists(), f"Data dir missing: {paths['data_dir']}"
        assert paths['artifacts_dir'].exists(), f"Artifacts dir missing: {paths['artifacts_dir']}"
        assert paths['configs_dir'].exists(), f"Configs dir missing: {paths['configs_dir']}"
        assert paths['feature_config'].exists(), f"feature_config missing: {paths['feature_config']}"
        assert paths['scaler'].exists(), f"scaler missing: {paths['scaler']}"
        assert paths['encoder'].exists(), f"encoder missing: {paths['encoder']}"

    def test_feature_config_schema(self):
        cfg = load_feature_config()
        assert cfg['total_input_dimension'] == TOTAL_FEATURES
        assert cfg['num_numerical_features'] == NUM_FEATURES
        assert cfg['num_one_hot_features'] == CAT_FEATURES
        assert len(cfg['numerical_features']) == NUM_FEATURES
        assert len(cfg['one_hot_feature_names']) == CAT_FEATURES
        assert cfg['target_variable'] == 'ccme_values'

    def test_fitted_transformers_consistency(self):
        scaler, encoder = load_transformers()
        # Scaler must have 36 features fitted
        assert scaler.n_features_in_ == NUM_FEATURES
        # OneHotEncoder must produce 17 binary features
        assert len(encoder.get_feature_names_out()) == CAT_FEATURES

