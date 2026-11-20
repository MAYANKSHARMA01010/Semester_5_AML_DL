"""
Phase 2 Deep Learning — Reusable MLP Training & Validation Engine
=================================================================
Provides modular, device-agnostic training and evaluation routines for PyTorch MLP models:
  - Training loop: batch forward, loss calculation, backward, optimizer step, gradient clipping
  - Validation loop: no_grad inference, batch aggregation, comprehensive regression metrics
  - Loss functions: MSELoss, L1Loss (MAE), SmoothL1Loss (Huber)
  - Optimizers: Adam, AdamW, SGD with momentum
  - Learning rate schedulers: ReduceLROnPlateau, CosineAnnealingLR
  - Early stopping: configurable patience, delta, best state caching and weight restoration
  - Checkpoint management: model, optimizer, scheduler, epoch, and metric history
  - Training monitoring & Overfitting analysis: generalization gap and metrics logging
  - Hardware acceleration: CUDA > Apple Silicon MPS > CPU fallback
  - Compatible with WaterQualityMLP and ResidualMLP from src.dl_models
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ──────────────────────────────────────────────────────────────────────────────
# 1. Device Compatibility
# ──────────────────────────────────────────────────────────────────────────────

def get_device(preference: Optional[str] = None) -> torch.device:
    """
    Select best available compute device: CUDA > Apple Silicon MPS > CPU fallback.

    Args:
        preference: Optional string: 'cuda', 'mps', 'cpu', or 'auto' (default).

    Returns:
        torch.device
    """
    if preference is not None and preference.lower() != 'auto':
        req = preference.lower()
        if req == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif req == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        elif req == 'cpu':
            return torch.device('cpu')
        # If preferred device is unavailable, fall through with fallback

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ──────────────────────────────────────────────────────────────────────────────
# 2. Loss Functions Factory
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_LOSSES = {
    'mse': nn.MSELoss,
    'mseloss': nn.MSELoss,
    'l1': nn.L1Loss,
    'l1loss': nn.L1Loss,
    'mae': nn.L1Loss,
    'smoothl1': nn.SmoothL1Loss,
    'smoothl1loss': nn.SmoothL1Loss,
    'huber': nn.SmoothL1Loss,
}


def get_loss_function(name: str = 'mse', **kwargs: Any) -> nn.Module:
    """
    Construct a loss function by name.

    Args:
        name: Name of loss: 'mse' / 'mseloss', 'l1' / 'mae', 'smoothl1' / 'huber'.
        **kwargs: Arguments passed to the loss constructor (e.g. beta=1.0 for SmoothL1Loss).

    Returns:
        nn.Module loss function.
    """
    key = name.lower().replace('_', '').replace(' ', '')
    if key not in SUPPORTED_LOSSES:
        raise ValueError(
            f"Unsupported loss function: '{name}'. "
            f"Supported options: {sorted(list(set(SUPPORTED_LOSSES.keys())))}"
        )

    loss_cls = SUPPORTED_LOSSES[key]
    if key in ('smoothl1', 'smoothl1loss', 'huber'):
        beta = kwargs.get('beta', 1.0)
        return nn.SmoothL1Loss(beta=beta)
    clean_kwargs = {k: v for k, v in kwargs.items() if k != 'beta'}
    return loss_cls(**clean_kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Optimizers Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_optimizer(
    name: str,
    params_or_model: Union[nn.Module, Any],
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """
    Construct a PyTorch optimizer for model parameters.

    Args:
        name: Optimizer name: 'adam', 'adamw', or 'sgd'.
        params_or_model: nn.Module or iterable of parameters.
        lr: Learning rate.
        weight_decay: L2 regularization / weight decay coefficient.
        **kwargs: Additional optimizer arguments (e.g. momentum=0.9 for SGD).

    Returns:
        torch.optim.Optimizer
    """
    key = name.lower().strip()
    params = params_or_model.parameters() if isinstance(params_or_model, nn.Module) else params_or_model

    if key == 'adam':
        betas = kwargs.get('betas', (0.9, 0.999))
        eps = kwargs.get('eps', 1e-8)
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    elif key == 'adamw':
        betas = kwargs.get('betas', (0.9, 0.999))
        eps = kwargs.get('eps', 1e-8)
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    elif key == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        nesterov = kwargs.get('nesterov', False)
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
        )

    else:
        raise ValueError(
            f"Unsupported optimizer: '{name}'. Supported options: 'adam', 'adamw', 'sgd'."
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4. Learning Rate Scheduling
# ──────────────────────────────────────────────────────────────────────────────

def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    **kwargs: Any,
) -> Any:
    """
    Factory constructing a learning-rate scheduler.

    Supported schedulers:
      - 'plateau' / 'reducelronplateau': ReduceLROnPlateau (monitors validation metric)
      - 'cosine' / 'cosineannealinglr': CosineAnnealingLR (epoch-based decay)

    Args:
        name: Name of scheduler ('plateau', 'cosine').
        optimizer: Optimizer whose learning rate will be scheduled.
        **kwargs: Additional hyperparameters (e.g. patience, factor, T_max, eta_min).

    Returns:
        PyTorch lr_scheduler instance.
    """
    key = name.lower().replace('_', '').replace('-', '').strip()

    if key in ('plateau', 'reducelronplateau'):
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.5)
        patience = kwargs.get('patience', 3)
        min_lr = kwargs.get('min_lr', 1e-6)
        threshold = kwargs.get('threshold', 1e-4)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr, threshold=threshold
        )

    elif key in ('cosine', 'cosineannealing', 'cosineannealinglr'):
        T_max = kwargs.get('T_max', 20)
        eta_min = kwargs.get('eta_min', 1e-6)
        last_epoch = kwargs.get('last_epoch', -1)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch
        )

    else:
        raise ValueError(
            f"Unsupported scheduler: '{name}'. Supported options: 'plateau', 'cosine'."
        )


def step_scheduler(scheduler: Any, metric_val: Optional[float] = None) -> None:
    """
    Step the scheduler correctly based on its type.

    Args:
        scheduler: lr_scheduler instance or None.
        metric_val: Validation loss or metric value (required for ReduceLROnPlateau).
    """
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if metric_val is None:
            raise ValueError("ReduceLROnPlateau requires metric_val for step()")
        scheduler.step(metric_val)
    else:
        scheduler.step()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Regression Evaluation Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_regression_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute standard regression metrics (MAE, RMSE, R²) for continuous CCME WQI values.
    Consistent with Phase 1 src.metrics.compute_regression_metrics.

    Args:
        y_true: Ground truth target array/tensor.
        y_pred: Predicted target array/tensor.

    Returns:
        Dict with 'mae', 'rmse', 'r2'.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().flatten()
    else:
        y_true = np.asarray(y_true).flatten()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy().flatten()
    else:
        y_pred = np.asarray(y_pred).flatten()

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))

    # Guard against degenerate zero-variance in target
    var = float(np.var(y_true))
    if var < 1e-12:
        r2 = 0.0
    else:
        r2 = float(r2_score(y_true, y_pred))

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. Early Stopping
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Early Stopping handler to prevent overfitting.

    Monitors a validation metric (default: validation loss) and triggers early stop
    if no improvement of at least min_delta is observed for `patience` consecutive epochs.
    Caches the best model weights in memory and restores them on request.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        checkpoint_path: Optional[Union[str, Path]] = None,
        verbose: bool = False,
    ):
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.min_delta = abs(min_delta)
        self.mode = mode
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.verbose = verbose

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.best_epoch: int = 0
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def is_better(self, current: float) -> bool:
        """Check if current metric is strictly better than best_score by min_delta."""
        if self.best_score is None:
            return True
        if self.mode == 'min':
            return current < (self.best_score - self.min_delta)
        else:
            return current > (self.best_score + self.min_delta)

    def __call__(
        self,
        val_metric: float,
        model: nn.Module,
        epoch: int = 0,
    ) -> bool:
        """
        Evaluate early stopping condition.

        Args:
            val_metric: Current epoch validation metric (e.g. val_loss).
            model: PyTorch model being trained.
            epoch: Current epoch number.

        Returns:
            bool: True if early stopping was triggered, else False.
        """
        if self.is_better(val_metric):
            if self.verbose and self.best_score is not None:
                print(
                    f"  [EarlyStopping] Metric improved from {self.best_score:.4f} "
                    f"to {val_metric:.4f}. Resetting patience counter."
                )
            self.best_score = val_metric
            self.best_epoch = epoch
            self.counter = 0
            # Cache best model weights in CPU memory
            self.best_state_dict = copy.deepcopy(
                {k: v.cpu() for k, v in model.state_dict().items()}
            )
            # Save checkpoint if path specified
            if self.checkpoint_path is not None:
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'best_val_metric': val_metric,
                        'best_epoch': epoch,
                    },
                    self.checkpoint_path,
                )
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs "
                    f"(Best: {self.best_score:.4f})."
                )
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore the best cached state_dict into the model."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            if self.verbose:
                print(f"✓ Restored model weights from best epoch {self.best_epoch} (Score: {self.best_score:.4f})")
        elif self.checkpoint_path is not None and self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            if self.verbose:
                print(f"✓ Restored model weights from file: {self.checkpoint_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Checkpoint Management
# ──────────────────────────────────────────────────────────────────────────────

def save_training_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    scheduler: Optional[Any] = None,
    config: Optional[Any] = None,
    best_val_loss: Optional[float] = None,
    history: Optional[Dict[str, List[float]]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save an extended training checkpoint for resuming training.

    Maintains 100% backward compatibility with Task 2 load_model_checkpoint() by
    embedding 'model_state_dict' and 'architecture_config'.

    Args:
        path: Path where checkpoint will be saved.
        model: Trained model instance.
        optimizer: Optimizer instance.
        epoch: Current completed epoch count.
        scheduler: Optional learning rate scheduler.
        config: Optional MLPConfig (dataclass or dict).
        best_val_loss: Lowest validation loss achieved so far.
        history: Dictionary containing training metric histories.
        extra_meta: Additional metadata dictionary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'history': history or {},
        'pytorch_version': torch.__version__,
        'timestamp': time.time(),
    }

    if scheduler is not None and hasattr(scheduler, 'state_dict'):
        payload['scheduler_state_dict'] = scheduler.state_dict()

    if config is not None:
        if hasattr(config, 'to_dict'):
            payload['architecture_config'] = config.to_dict()
        elif isinstance(config, dict):
            payload['architecture_config'] = config

    if extra_meta:
        payload.update(extra_meta)

    torch.save(payload, path)
    print(f"✓ Checkpoint saved → {path}")


def load_training_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load training state from a checkpoint to resume training.

    Args:
        path: Path to checkpoint .pt file.
        model: Model module to receive loaded weights.
        optimizer: Optional optimizer to restore state dict.
        scheduler: Optional scheduler to restore state dict.
        device: Target compute device.

    Returns:
        Checkpoint dictionary with 'epoch', 'best_val_loss', 'history', etc.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    device = device or torch.device('cpu')
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    print(f"✓ Restored training checkpoint from {path} (Epoch: {ckpt.get('epoch', 0)})")
    return ckpt


# ──────────────────────────────────────────────────────────────────────────────
# 8. Training Loop (Single Epoch)
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    """
    Execute one training epoch over the given dataloader.

    Properly handles batch weighting for incomplete final batches.
    Calculates both train loss and train MAE on the fly.

    Args:
        model: PyTorch model (e.g. WaterQualityMLP).
        dataloader: Training DataLoader yielding (x, y) batches.
        criterion: Loss function (e.g. MSELoss).
        optimizer: PyTorch optimizer (e.g. Adam).
        device: Target compute device.
        clip_grad_norm: Optional max gradient norm threshold for clipping.

    Returns:
        Dict containing 'loss', 'train_loss', 'train_mae', and 'samples'.
    """
    model.train()
    running_loss = 0.0
    running_abs_err = 0.0
    total_samples = 0

    for batch in dataloader:
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            # Separated mode: (x_num, x_cat, y) -> concatenate for standard MLP
            x_num, x_cat, y = batch
            x = torch.cat([x_num, x_cat], dim=-1)
        else:
            raise ValueError(f"Unexpected batch tuple length: {len(batch)}")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_size = x.size(0)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)

        if torch.isnan(loss):
            raise FloatingPointError("NaN loss encountered during training step.")

        loss.backward()

        if clip_grad_norm is not None and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

        optimizer.step()

        running_loss += loss.item() * batch_size
        running_abs_err += torch.sum(torch.abs(preds.detach() - y)).item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_mae = running_abs_err / total_samples if total_samples > 0 else 0.0

    return {
        'loss': epoch_loss,
        'train_loss': epoch_loss,
        'train_mae': epoch_mae,
        'samples': total_samples,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 9. Validation Loop
# ──────────────────────────────────────────────────────────────────────────────

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the model on a validation or test DataLoader.

    Ensures no gradients are tracked (torch.no_grad), evaluates in eval mode,
    and aggregates exact MAE, RMSE, and R² metrics across the full split.

    Args:
        model: PyTorch model.
        dataloader: Validation or Test DataLoader.
        criterion: Loss function.
        device: Target compute device.

    Returns:
        Dict containing 'val_loss', 'val_mae', 'val_rmse', 'val_r2', 'samples'.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                x, y = batch
            elif len(batch) == 3:
                x_num, x_cat, y = batch
                x = torch.cat([x_num, x_cat], dim=-1)
            else:
                raise ValueError(f"Unexpected batch tuple length: {len(batch)}")

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            batch_size = x.size(0)

            preds = model(x)
            loss = criterion(preds, y)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(preds.detach().cpu())
            all_targets.append(y.detach().cpu())

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0

    y_pred_arr = torch.cat(all_preds, dim=0).numpy().flatten()
    y_true_arr = torch.cat(all_targets, dim=0).numpy().flatten()

    metrics = compute_regression_metrics(y_true_arr, y_pred_arr)

    return {
        'val_loss': epoch_loss,
        'val_mae': metrics['mae'],
        'val_rmse': metrics['rmse'],
        'val_r2': metrics['r2'],
        'samples': total_samples,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 10. Overfitting Analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_overfitting(history: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Analyze training and validation metrics to quantify generalization gap,
    overfitting risk, and early stopping behavior.

    Args:
        history: Training history dictionary with 'train_loss', 'val_loss', 'train_mae', 'val_mae'.

    Returns:
        Dict with generalization gap statistics and qualitative diagnosis.
    """
    train_losses = history.get('train_loss', [])
    val_losses = history.get('val_loss', [])
    train_maes = history.get('train_mae', [])
    val_maes = history.get('val_mae', [])

    if not train_losses or not val_losses:
        return {'status': 'Empty history'}

    best_epoch_idx = int(np.argmin(val_losses))
    best_epoch = best_epoch_idx + 1
    best_val_loss = float(val_losses[best_epoch_idx])
    best_val_mae = float(val_maes[best_epoch_idx]) if val_maes else None

    final_train_loss = float(train_losses[-1])
    final_val_loss = float(val_losses[-1])
    loss_gap = final_val_loss - final_train_loss

    final_train_mae = float(train_maes[-1]) if train_maes else None
    final_val_mae = float(val_maes[-1]) if val_maes else None
    mae_gap = (final_val_mae - final_train_mae) if (final_val_mae is not None and final_train_mae is not None) else None

    # Check divergence after best epoch
    loss_divergence = final_val_loss - best_val_loss
    is_overfitting = loss_divergence > (0.05 * best_val_loss)

    if loss_divergence > (0.20 * best_val_loss):
        risk = 'High'
    elif loss_divergence > (0.05 * best_val_loss):
        risk = 'Moderate'
    else:
        risk = 'Low'

    return {
        'total_epochs_trained': len(train_losses),
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_mae': best_val_mae,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'generalization_gap_loss': loss_gap,
        'final_train_mae': final_train_mae,
        'final_val_mae': final_val_mae,
        'generalization_gap_mae': mae_gap,
        'post_best_divergence': loss_divergence,
        'is_overfitting': is_overfitting,
        'overfitting_risk': risk,
    }


def format_history_table(history: Dict[str, List[float]]) -> str:
    """Return a formatted ASCII table of the training progress history."""
    n_epochs = len(history.get('train_loss', []))
    if n_epochs == 0:
        return "No training history recorded."

    headers = [
        f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Train MAE':<10} "
        f"{'Val MAE':<10} {'Val RMSE':<10} {'Val R²':<10} {'Gap (MAE)':<10} {'LR':<10} {'Time(s)':<8}"
    ]
    lines = ['─' * 104]
    lines.extend(headers)
    lines.append('─' * 104)

    for ep in range(n_epochs):
        tr_l = history['train_loss'][ep]
        va_l = history['val_loss'][ep]
        tr_m = history.get('train_mae', [0.0] * n_epochs)[ep]
        va_m = history.get('val_mae', [0.0] * n_epochs)[ep]
        va_r = history.get('val_rmse', [0.0] * n_epochs)[ep]
        va_r2 = history.get('val_r2', [0.0] * n_epochs)[ep]
        gap_m = va_m - tr_m
        lr = history.get('lr', [0.0] * n_epochs)[ep]
        dur = history.get('epoch_duration', [0.0] * n_epochs)[ep]

        lines.append(
            f"{ep+1:<6} {tr_l:<12.4f} {va_l:<12.4f} {tr_m:<10.4f} "
            f"{va_m:<10.4f} {va_r:<10.4f} {va_r2:<10.4f} {gap_m:<10.4f} {lr:<10.2e} {dur:<8.2f}"
        )
    lines.append('─' * 104)
    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 11. Reusable MLPTrainer Class
# ──────────────────────────────────────────────────────────────────────────────

class MLPTrainer:
    """
    High-level trainer coordinating training loops, validation loops, metric logging,
    learning rate scheduling, early stopping, and checkpointing for WaterQualityMLP models.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        clip_grad_norm: Optional[float] = None,
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.clip_grad_norm = clip_grad_norm

        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_mae': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_r2': [],
            'lr': [],
            'epoch_duration': [],
            'generalization_gap': [],
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch and return training metrics dictionary."""
        return train_one_epoch(
            model=self.model,
            dataloader=train_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            clip_grad_norm=self.clip_grad_norm,
        )

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run evaluation on dataloader and return dictionary of metrics."""
        return validate(
            model=self.model,
            dataloader=dataloader,
            criterion=self.criterion,
            device=self.device,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        scheduler: Optional[Any] = None,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config: Optional[Any] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Fit the model for a specified number of epochs.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Total number of epochs to train.
            scheduler: Optional learning rate scheduler.
            early_stopping: Optional EarlyStopping callback instance.
            checkpoint_path: Optional file path to save the best checkpoint.
            config: Optional MLPConfig or dictionary for checkpoint metadata.
            verbose: Whether to print progress each epoch.

        Returns:
            Dictionary with training and validation history lists.
        """
        if verbose:
            print(f"Starting training on device: {self.device}")
            print(f"Epochs: {epochs} | Optimizer: {type(self.optimizer).__name__} | Loss: {type(self.criterion).__name__}")
            print(
                f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Train MAE':<10} "
                f"{'Val MAE':<10} {'Val RMSE':<10} {'Val R²':<10} {'LR':<10} {'Time(s)':<8}"
            )
            print("─" * 92)

        best_val_loss = float('inf')

        for epoch in range(1, epochs + 1):
            t_start = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            t_elapsed = time.time() - t_start

            train_loss = train_metrics['train_loss']
            train_mae = train_metrics['train_mae']
            val_loss = val_metrics['val_loss']
            val_mae = val_metrics['val_mae']
            val_rmse = val_metrics['val_rmse']
            val_r2 = val_metrics['val_r2']
            current_lr = self.optimizer.param_groups[0]['lr']
            gap = val_mae - train_mae

            self.history['train_loss'].append(train_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['val_rmse'].append(val_rmse)
            self.history['val_r2'].append(val_r2)
            self.history['lr'].append(current_lr)
            self.history['epoch_duration'].append(t_elapsed)
            self.history['generalization_gap'].append(gap)

            if verbose:
                print(
                    f"{epoch:<6} {train_loss:<12.4f} {val_loss:<12.4f} {train_mae:<10.4f} "
                    f"{val_mae:<10.4f} {val_rmse:<10.4f} {val_r2:<10.4f} {current_lr:<10.2e} {t_elapsed:<8.2f}"
                )

            # Best model checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if checkpoint_path is not None:
                    save_training_checkpoint(
                        path=checkpoint_path,
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        scheduler=scheduler,
                        config=config,
                        best_val_loss=best_val_loss,
                        history=self.history,
                    )

            # Step scheduler
            if scheduler is not None:
                step_scheduler(scheduler, metric_val=val_loss)

            # Early stopping check
            if early_stopping is not None:
                early_stopping(val_loss, self.model, epoch=epoch)
                if early_stopping.early_stop:
                    if verbose:
                        print(f"\n[EarlyStopping] Stopped early at epoch {epoch}.")
                    early_stopping.restore_best_weights(self.model)
                    break

        return self.history
