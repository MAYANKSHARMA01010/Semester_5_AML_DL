"""
Phase 2 Deep Learning — Unit Test Suite for Training Engine (Stages 2 & 3)
========================================================================
Tests the training, optimization, early stopping, checkpointing, and monitoring features:
  - Device selection and fallback logic
  - Loss function factories (MSE, L1, SmoothL1)
  - Optimizer factories (Adam, AdamW, SGD with momentum)
  - Learning rate schedulers (ReduceLROnPlateau, CosineAnnealingLR) and step logic
  - EarlyStopping patience, min_delta, best score tracking, and weight restoration
  - Checkpoint save and load round-trip + backward compatibility with Task 2 load_model_checkpoint
  - Training and validation loop execution and sample-weighted loss aggregation
  - Training monitoring and Overfitting analysis diagnostics
  - MLPTrainer coordination with early stopping, scheduler, and history logging
"""

import math
import os
import sys
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add Phase_2_Deep_Learning to sys.path
PHASE2_DIR = Path(__file__).resolve().parent.parent
if str(PHASE2_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE2_DIR))

from src.dl_models import MLPConfig, build_model, load_model_checkpoint
from src.training import (
    get_device,
    get_loss_function,
    get_optimizer,
    get_scheduler,
    step_scheduler,
    compute_regression_metrics,
    EarlyStopping,
    save_training_checkpoint,
    load_training_checkpoint,
    train_one_epoch,
    validate,
    analyze_overfitting,
    format_history_table,
    MLPTrainer,
)

INPUT_DIM = 53
BATCH_SIZE = 32
DEVICE = torch.device('cpu')


@pytest.fixture
def dummy_config():
    return MLPConfig(
        input_dim=INPUT_DIM,
        hidden_dims=[64, 32],
        dropout_rate=0.0,
        use_batch_norm=True,
        name='MLP_Test',
    )


@pytest.fixture
def dummy_model(dummy_config):
    return build_model(dummy_config, device=DEVICE, seed=42)


@pytest.fixture
def dummy_loaders():
    torch.manual_seed(42)
    n_train = 100  # Intentionally not a multiple of BATCH_SIZE (32) to test incomplete batches
    n_val = 50

    x_train = torch.randn(n_train, INPUT_DIM)
    y_train = torch.rand(n_train, 1) * 80.0 + 10.0

    x_val = torch.randn(n_val, INPUT_DIM)
    y_val = torch.rand(n_val, 1) * 80.0 + 10.0

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# 1. Device Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDevice:

    def test_cpu_device(self):
        dev = get_device('cpu')
        assert dev.type == 'cpu'

    def test_auto_device(self):
        dev = get_device('auto')
        assert isinstance(dev, torch.device)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Loss Function Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestLossFunctions:

    @pytest.mark.parametrize('name,expected_cls', [
        ('mse', nn.MSELoss),
        ('mseloss', nn.MSELoss),
        ('l1', nn.L1Loss),
        ('l1loss', nn.L1Loss),
        ('mae', nn.L1Loss),
        ('smoothl1', nn.SmoothL1Loss),
        ('smoothl1loss', nn.SmoothL1Loss),
        ('huber', nn.SmoothL1Loss),
    ])
    def test_supported_losses(self, name, expected_cls):
        crit = get_loss_function(name)
        assert isinstance(crit, expected_cls)

    def test_smoothl1_beta(self):
        crit = get_loss_function('smoothl1', beta=0.5)
        assert isinstance(crit, nn.SmoothL1Loss)
        assert crit.beta == 0.5

    def test_unsupported_loss_raises(self):
        with pytest.raises(ValueError, match="Unsupported loss function"):
            get_loss_function('cross_entropy')


# ──────────────────────────────────────────────────────────────────────────────
# 3. Optimizer Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestOptimizers:

    def test_adam(self, dummy_model):
        opt = get_optimizer('adam', dummy_model, lr=1e-3, weight_decay=1e-4)
        assert isinstance(opt, torch.optim.Adam)
        assert opt.param_groups[0]['lr'] == 1e-3
        assert opt.param_groups[0]['weight_decay'] == 1e-4

    def test_adamw(self, dummy_model):
        opt = get_optimizer('adamw', dummy_model, lr=5e-4, weight_decay=1e-2)
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.param_groups[0]['lr'] == 5e-4
        assert opt.param_groups[0]['weight_decay'] == 1e-2

    def test_sgd_momentum(self, dummy_model):
        opt = get_optimizer('sgd', dummy_model, lr=1e-2, momentum=0.9)
        assert isinstance(opt, torch.optim.SGD)
        assert opt.param_groups[0]['momentum'] == 0.9

    def test_unsupported_optimizer_raises(self, dummy_model):
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            get_optimizer('rmsprop', dummy_model)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Learning Rate Scheduler Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSchedulers:

    def test_plateau_scheduler(self, dummy_model):
        opt = get_optimizer('adam', dummy_model, lr=1e-3)
        sched = get_scheduler('plateau', opt, factor=0.5, patience=2)
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert sched.factor == 0.5
        assert sched.patience == 2

        # Step with metric
        step_scheduler(sched, metric_val=10.0)

    def test_cosine_scheduler(self, dummy_model):
        opt = get_optimizer('adam', dummy_model, lr=1e-3)
        sched = get_scheduler('cosine', opt, T_max=10, eta_min=1e-5)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert sched.T_max == 10
        assert sched.eta_min == 1e-5

        # Step optimizer then step scheduler
        opt.step()
        step_scheduler(sched)

    def test_plateau_missing_metric_raises(self, dummy_model):
        opt = get_optimizer('adam', dummy_model)
        sched = get_scheduler('plateau', opt)
        with pytest.raises(ValueError, match="ReduceLROnPlateau requires metric_val"):
            step_scheduler(sched, metric_val=None)

    def test_unsupported_scheduler_raises(self, dummy_model):
        opt = get_optimizer('adam', dummy_model)
        with pytest.raises(ValueError, match="Unsupported scheduler"):
            get_scheduler('exponential', opt)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Early Stopping Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestEarlyStopping:

    def test_patience_exhaustion_triggers_stop(self, dummy_model):
        es = EarlyStopping(patience=3, min_delta=0.01, mode='min')

        assert not es(10.0, dummy_model, epoch=1)
        assert es.counter == 0
        assert not es(10.0, dummy_model, epoch=2)
        assert es.counter == 1
        assert not es(10.0, dummy_model, epoch=3)
        assert es.counter == 2
        assert es(10.0, dummy_model, epoch=4)  # 3rd consecutive non-improvement -> triggers stop
        assert es.early_stop is True

    def test_improvement_resets_counter(self, dummy_model):
        es = EarlyStopping(patience=2, min_delta=0.1, mode='min')

        es(10.0, dummy_model, epoch=1)
        assert es.counter == 0
        es(9.95, dummy_model, epoch=2)  # delta = 0.05 < 0.1 -> not enough improvement
        assert es.counter == 1
        es(9.50, dummy_model, epoch=3)  # delta = 0.50 >= 0.1 -> significant improvement
        assert es.counter == 0
        assert es.best_score == 9.50
        assert not es.early_stop

    def test_restore_best_weights(self, dummy_model):
        es = EarlyStopping(patience=2, mode='min')

        # Epoch 1: set best weights
        es(5.0, dummy_model, epoch=1)
        best_w = dummy_model.output_layer.weight.clone()

        # Modify weights in subsequent training
        with torch.no_grad():
            dummy_model.output_layer.weight.add_(10.0)

        assert not torch.allclose(dummy_model.output_layer.weight, best_w)

        # Restore weights
        es.restore_best_weights(dummy_model)
        assert torch.allclose(dummy_model.output_layer.weight, best_w)

    def test_checkpoint_file_saving(self, dummy_model, tmp_path):
        ckpt_file = tmp_path / 'best_model.pt'
        es = EarlyStopping(patience=2, checkpoint_path=ckpt_file)

        es(15.0, dummy_model, epoch=1)
        assert ckpt_file.exists()

        # Modify weights
        with torch.no_grad():
            dummy_model.output_layer.weight.fill_(99.0)

        # Restore from file
        es.best_state_dict = None  # Force disk restore
        es.restore_best_weights(dummy_model)
        assert not torch.allclose(dummy_model.output_layer.weight, torch.full_like(dummy_model.output_layer.weight, 99.0))


# ──────────────────────────────────────────────────────────────────────────────
# 6. Checkpoint Management Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestCheckpointManagement:

    def test_save_and_load_training_checkpoint(self, dummy_model, dummy_config, tmp_path):
        ckpt_path = tmp_path / 'train_ckpt.pt'
        opt = get_optimizer('adam', dummy_model, lr=2e-3)
        sched = get_scheduler('cosine', opt, T_max=10)

        history = {'train_loss': [10.0, 8.0], 'val_loss': [11.0, 9.0]}

        save_training_checkpoint(
            path=ckpt_path,
            model=dummy_model,
            optimizer=opt,
            epoch=2,
            scheduler=sched,
            config=dummy_config,
            best_val_loss=9.0,
            history=history,
        )
        assert ckpt_path.exists()

        # Create new model & optimizer to restore into
        new_model = build_model(dummy_config, device=DEVICE, seed=99)
        new_opt = get_optimizer('adam', new_model, lr=1e-1)
        new_sched = get_scheduler('cosine', new_opt, T_max=10)

        loaded_meta = load_training_checkpoint(
            ckpt_path, new_model, optimizer=new_opt, scheduler=new_sched, device=DEVICE
        )

        assert loaded_meta['epoch'] == 2
        assert loaded_meta['best_val_loss'] == 9.0
        assert loaded_meta['history'] == history

        # Model weights must match
        for p1, p2 in zip(dummy_model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

        # Optimizer learning rate must match
        assert new_opt.param_groups[0]['lr'] == opt.param_groups[0]['lr']

    def test_task2_backward_compatibility(self, dummy_model, dummy_config, tmp_path):
        """Verify that Task 2 load_model_checkpoint() can directly load this training checkpoint."""
        ckpt_path = tmp_path / 'compat_ckpt.pt'
        opt = get_optimizer('adam', dummy_model)

        save_training_checkpoint(
            path=ckpt_path,
            model=dummy_model,
            optimizer=opt,
            epoch=5,
            config=dummy_config,
        )

        # Use Task 2 loader from src.dl_models
        t2_model, t2_meta = load_model_checkpoint(ckpt_path, device=DEVICE)
        assert isinstance(t2_model, nn.Module)
        assert t2_meta['epoch'] == 5
        assert t2_meta['architecture_config']['name'] == 'MLP_Test'


# ──────────────────────────────────────────────────────────────────────────────
# 7. Metrics & Monitoring Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMetricsAndMonitoring:

    def test_exact_metric_values(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 33.0, 39.0])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert math.isclose(metrics['mae'], 2.0, rel_tol=1e-5)
        assert math.isclose(metrics['rmse'], math.sqrt(4.5), rel_tol=1e-5)
        assert metrics['r2'] > 0.9

    def test_analyze_overfitting_healthy(self):
        history = {
            'train_loss': [20.0, 15.0, 10.0, 8.0, 7.0],
            'val_loss':   [21.0, 16.0, 10.5, 8.2, 7.2],
            'train_mae':  [4.0, 3.5, 3.0, 2.5, 2.2],
            'val_mae':    [4.1, 3.6, 3.1, 2.6, 2.3],
        }
        res = analyze_overfitting(history)
        assert res['best_epoch'] == 5
        assert res['overfitting_risk'] == 'Low'
        assert not res['is_overfitting']
        assert math.isclose(res['generalization_gap_mae'], 0.1, abs_tol=1e-4)

    def test_analyze_overfitting_severe(self):
        history = {
            'train_loss': [15.0, 10.0, 5.0, 2.0, 1.0],
            'val_loss':   [16.0, 11.0, 12.0, 18.0, 25.0],  # Severe divergence after epoch 2
            'train_mae':  [3.5, 2.8, 1.8, 1.1, 0.8],
            'val_mae':    [3.7, 3.0, 3.2, 4.0, 5.0],
        }
        res = analyze_overfitting(history)
        assert res['best_epoch'] == 2
        assert res['overfitting_risk'] == 'High'
        assert res['is_overfitting'] is True

    def test_format_history_table(self):
        history = {
            'train_loss': [10.0, 8.0],
            'val_loss': [11.0, 9.0],
            'train_mae': [3.0, 2.5],
            'val_mae': [3.2, 2.7],
            'val_rmse': [3.5, 3.0],
            'val_r2': [0.8, 0.85],
            'lr': [1e-3, 1e-3],
            'epoch_duration': [1.2, 1.1],
        }
        table = format_history_table(history)
        assert isinstance(table, str)
        assert 'Epoch' in table
        assert 'Gap (MAE)' in table
        assert 'Time(s)' in table


# ──────────────────────────────────────────────────────────────────────────────
# 8. MLPTrainer Integration Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMLPTrainer:

    def test_trainer_fit_with_scheduler_and_early_stopping(self, dummy_model, dummy_loaders, tmp_path):
        train_loader, val_loader = dummy_loaders
        ckpt_path = tmp_path / 'best_trainer.pt'

        optimizer = get_optimizer('adam', dummy_model, lr=1e-3)
        scheduler = get_scheduler('plateau', optimizer, patience=1, factor=0.5)
        early_stopping = EarlyStopping(patience=2, min_delta=0.01)
        criterion = get_loss_function('mse')

        trainer = MLPTrainer(
            model=dummy_model,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            clip_grad_norm=1.0,
        )

        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=3,
            scheduler=scheduler,
            early_stopping=early_stopping,
            checkpoint_path=ckpt_path,
            verbose=False,
        )

        # Check all 9 monitored metric lists
        assert len(history['train_loss']) == 3
        assert len(history['train_mae']) == 3
        assert len(history['val_loss']) == 3
        assert len(history['val_mae']) == 3
        assert len(history['val_rmse']) == 3
        assert len(history['val_r2']) == 3
        assert len(history['lr']) == 3
        assert len(history['epoch_duration']) == 3
        assert len(history['generalization_gap']) == 3

        # Checkpoint should have been created
        assert ckpt_path.exists()
