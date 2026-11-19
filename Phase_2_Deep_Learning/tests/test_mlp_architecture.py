"""
Phase 2 Deep Learning — Unit Test Suite for MLP Architecture
=============================================================
Run:  pytest Phase_2_Deep_Learning/tests/test_mlp_architecture.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataclasses import fields

from src.dl_models import (
    MLPConfig,
    MLPBlock,
    WaterQualityMLP,
    ResidualBlock,
    ResidualMLP,
    build_model,
    save_model_checkpoint,
    load_model_checkpoint,
    mc_dropout_predictions,
    count_params_per_layer,
    _get_activation,
)

DEVICE = torch.device('cpu')
INPUT_DIM = 53
BATCH = 64


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_config():
    return MLPConfig(
        input_dim=INPUT_DIM,
        hidden_dims=[512, 256, 128, 64],
        dropout_rate=0.3,
        use_batch_norm=True,
        activation='relu',
        name='MLP_Base',
    )


@pytest.fixture
def base_model(base_config):
    model = WaterQualityMLP(base_config).to(DEVICE)
    model.eval()
    return model


@pytest.fixture
def dummy_batch():
    torch.manual_seed(0)
    return torch.randn(BATCH, INPUT_DIM)


@pytest.fixture
def tmp_ckpt(tmp_path):
    return tmp_path / 'test_ckpt.pt'


# ── MLPConfig Tests ───────────────────────────────────────────────────────────

class TestMLPConfig:

    def test_default_instantiation(self):
        cfg = MLPConfig()
        assert cfg.input_dim == 53
        assert cfg.hidden_dims == [512, 256, 128, 64]
        assert cfg.dropout_rate == 0.30
        assert cfg.use_batch_norm is True
        assert cfg.activation == 'relu'
        assert cfg.output_dim == 1

    def test_to_dict_round_trip(self, base_config):
        d = base_config.to_dict()
        cfg2 = MLPConfig.from_dict(d)
        assert cfg2.input_dim    == base_config.input_dim
        assert cfg2.hidden_dims  == base_config.hidden_dims
        assert cfg2.dropout_rate == base_config.dropout_rate
        assert cfg2.name         == base_config.name

    def test_n_layers_property(self, base_config):
        assert base_config.n_layers == len(base_config.hidden_dims)


# ── Activation Helper Tests ───────────────────────────────────────────────────

class TestActivationHelper:

    @pytest.mark.parametrize('act', ['relu', 'leakyrelu', 'gelu'])
    def test_valid_activations(self, act):
        module = _get_activation(act)
        assert isinstance(module, nn.Module)

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="Unsupported activation"):
            _get_activation('sigmoid')


# ── MLPBlock Tests ────────────────────────────────────────────────────────────

class TestMLPBlock:

    def test_output_shape_with_bn(self):
        block = MLPBlock(53, 256, dropout_rate=0.3, use_batch_norm=True)
        block.eval()
        x = torch.randn(BATCH, 53)
        out = block(x)
        assert out.shape == (BATCH, 256)

    def test_output_shape_no_bn(self):
        block = MLPBlock(53, 128, dropout_rate=0.0, use_batch_norm=False)
        block.eval()
        x = torch.randn(BATCH, 53)
        out = block(x)
        assert out.shape == (BATCH, 128)

    def test_no_nan_in_output(self):
        block = MLPBlock(53, 256, dropout_rate=0.3, use_batch_norm=True)
        block.eval()
        x = torch.randn(BATCH, 53)
        assert not torch.isnan(block(x)).any()

    @pytest.mark.parametrize('act', ['relu', 'leakyrelu', 'gelu'])
    def test_activation_variants(self, act):
        block = MLPBlock(53, 64, dropout_rate=0.0, use_batch_norm=False, activation=act)
        block.eval()
        out = block(torch.randn(16, 53))
        assert out.shape == (16, 64)


# ── WaterQualityMLP Tests ─────────────────────────────────────────────────────

class TestWaterQualityMLP:

    def test_output_shape(self, base_model, dummy_batch):
        with torch.no_grad():
            out = base_model(dummy_batch)
        assert out.shape == (BATCH, 1)

    def test_no_nan_at_init(self, base_model, dummy_batch):
        with torch.no_grad():
            out = base_model(dummy_batch)
        assert not torch.isnan(out).any(), 'NaN in forward pass at initialisation'

    def test_no_inf_at_init(self, base_model, dummy_batch):
        with torch.no_grad():
            out = base_model(dummy_batch)
        assert not torch.isinf(out).any(), 'Inf in forward pass at initialisation'

    def test_input_dim_guard_raises(self, base_model):
        """forward() must raise ValueError on wrong input dimension."""
        wrong_x = torch.randn(BATCH, INPUT_DIM + 5)
        with pytest.raises(ValueError, match='Input dimension mismatch'):
            base_model(wrong_x)

    def test_predict_clipped_range(self, base_model, dummy_batch):
        """predict_clipped() must always return values in [0, 100]."""
        out = base_model.predict_clipped(dummy_batch)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 100.0

    def test_trainable_params_positive(self, base_model):
        p = base_model.count_parameters()
        assert p['trainable'] > 0
        assert p['total'] >= p['trainable']

    def test_per_layer_param_table_string(self, base_model):
        table = base_model.per_layer_param_table()
        assert isinstance(table, str)
        assert 'TOTAL' in table
        assert 'hidden_layers' in table

    def test_kaiming_init_bias_zero(self, base_model):
        """All Linear biases must be initialised to zero."""
        for name, param in base_model.named_parameters():
            if 'bias' in name and 'batch_norm' not in name:
                assert param.abs().max().item() == 0.0, \
                    f'Non-zero bias at init for {name}'

    def test_invalid_config_empty_hidden(self):
        cfg = MLPConfig(hidden_dims=[])
        with pytest.raises(ValueError, match='hidden_dims must be a non-empty list'):
            WaterQualityMLP(cfg)

    def test_invalid_config_bad_dropout(self):
        cfg = MLPConfig(dropout_rate=1.5)
        with pytest.raises(ValueError, match='dropout_rate must be in'):
            WaterQualityMLP(cfg)

    @pytest.mark.parametrize('config', [
        MLPConfig(input_dim=53, hidden_dims=[128, 128], dropout_rate=0.0,
                  use_batch_norm=False, name='MLP_Tiny'),
        MLPConfig(input_dim=53, hidden_dims=[1024, 512, 256], dropout_rate=0.3,
                  use_batch_norm=True, name='MLP_Wide'),
        MLPConfig(input_dim=53, hidden_dims=[512]*4+[256,128], dropout_rate=0.3,
                  use_batch_norm=True, name='MLP_Deep'),
    ])
    def test_all_variants_forward(self, config):
        m = WaterQualityMLP(config).eval()
        x = torch.randn(32, 53)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (32, 1)
        assert not torch.isnan(out).any()


# ── ResidualMLP Tests ─────────────────────────────────────────────────────────

class TestResidualMLP:

    @pytest.fixture
    def resnet(self):
        return ResidualMLP(input_dim=53, hidden_dim=128, num_blocks=3,
                           dropout_rate=0.2).eval()

    def test_output_shape(self, resnet):
        x = torch.randn(BATCH, 53)
        with torch.no_grad():
            out = resnet(x)
        assert out.shape == (BATCH, 1)

    def test_no_nan(self, resnet):
        x = torch.randn(BATCH, 53)
        with torch.no_grad():
            out = resnet(x)
        assert not torch.isnan(out).any()

    def test_input_dim_guard(self, resnet):
        with pytest.raises(ValueError, match='Input dimension mismatch'):
            resnet(torch.randn(16, 10))

    def test_residual_block_identity_shortcut(self):
        block = ResidualBlock(128, 128, dropout_rate=0.0)
        block.eval()
        x   = torch.randn(32, 128)
        out = block(x)
        assert out.shape == (32, 128)

    def test_residual_block_projection_shortcut(self):
        block = ResidualBlock(53, 128, dropout_rate=0.0)
        block.eval()
        x   = torch.randn(32, 53)
        out = block(x)
        assert out.shape == (32, 128)


# ── Factory & Checkpoint Tests ────────────────────────────────────────────────

class TestCheckpointing:

    def test_build_model_reproducible(self, base_config):
        m1 = build_model(base_config, seed=42)
        m2 = build_model(base_config, seed=42)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2), 'build_model not reproducible with same seed'

    def test_save_load_round_trip(self, base_model, base_config, tmp_ckpt, dummy_batch):
        save_model_checkpoint(base_model, tmp_ckpt, config=base_config,
                               extra_meta={'epoch': 10, 'val_mae': 0.5})
        assert tmp_ckpt.exists()

        reloaded, ckpt_dict = load_model_checkpoint(tmp_ckpt)
        reloaded.eval()

        # Loaded checkpoint must reproduce identical predictions
        with torch.no_grad():
            orig_out   = base_model(dummy_batch)
            reload_out = reloaded(dummy_batch)
        assert torch.allclose(orig_out, reload_out, atol=1e-5), \
            'Reloaded model predictions differ from original'

        assert ckpt_dict['epoch']   == 10
        assert ckpt_dict['val_mae'] == 0.5

    def test_checkpoint_contains_config(self, base_model, base_config, tmp_ckpt):
        save_model_checkpoint(base_model, tmp_ckpt, config=base_config)
        ckpt = torch.load(tmp_ckpt, map_location='cpu', weights_only=False)
        assert 'architecture_config' in ckpt
        assert ckpt['architecture_config']['name'] == 'MLP_Base'


# ── MC Dropout Tests ──────────────────────────────────────────────────────────

class TestMCDropout:

    def test_output_shape(self, base_model, dummy_batch):
        mean_p, std_p = mc_dropout_predictions(base_model, dummy_batch, n_samples=10)
        assert mean_p.shape == (BATCH,)
        assert std_p.shape  == (BATCH,)

    def test_std_positive_with_dropout(self, base_model, dummy_batch):
        """With dropout active, predictions should vary across samples."""
        mean_p, std_p = mc_dropout_predictions(base_model, dummy_batch, n_samples=20)
        assert std_p.mean() > 0.0, 'MC Dropout produced zero variance — check dropout rate'

    def test_model_returns_to_eval(self, base_model, dummy_batch):
        mc_dropout_predictions(base_model, dummy_batch, n_samples=5)
        assert not base_model.training, 'mc_dropout_predictions must restore eval mode'


# ── count_params_per_layer Tests ──────────────────────────────────────────────

class TestParamTable:

    def test_returns_string(self, base_model):
        table = count_params_per_layer(base_model)
        assert isinstance(table, str)

    def test_contains_total(self, base_model):
        table = count_params_per_layer(base_model)
        assert 'TOTAL' in table

    def test_total_matches_count_parameters(self, base_model):
        expected = base_model.count_parameters()['total']
        table    = count_params_per_layer(base_model)
        # Extract last numeric from TOTAL row
        total_row = [line for line in table.split('\n') if 'TOTAL' in line][0]
        # Parse the rightmost integer
        import re
        nums = re.findall(r'[\d,]+', total_row)
        total_from_table = int(nums[-1].replace(',', ''))
        assert total_from_table == expected
