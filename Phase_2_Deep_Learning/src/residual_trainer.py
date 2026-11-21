"""
Phase 2 Deep Learning — Task 4: Residual MLP Training Engine
=============================================================
Trains the ResidualMLPNet to correct systematic errors of tree ensemble base models.
Input: [53-dim features, base_prediction] (54-dim) → predicts signed residual correction.

Classes
-------
ResidualDataset            — PyTorch Dataset combining features, base predictions, and residuals.

Functions
---------
build_residual_dataloaders — Construct train and validation DataLoaders.
train_residual_mlp         — Training loop with Huber loss, ReduceLROnPlateau, and EarlyStopping.
evaluate_residual_model    — Compute evaluation metrics on DataLoader.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .hybrid_models import ResidualMLPNet, HybridConfig, save_hybrid_checkpoint
from .training import EarlyStopping, get_device, compute_regression_metrics


class ResidualDataset(Dataset):
    """
    PyTorch Dataset for residual MLP training.

    Pairs the concatenated [feature_vector (53), base_prediction (1)] input
    with the signed residual target (y_true - base_pred).
    """

    def __init__(
        self,
        X_features: torch.Tensor,
        base_preds: torch.Tensor,
        residuals: torch.Tensor,
    ):
        n = len(X_features)
        if len(base_preds) != n or len(residuals) != n:
            raise ValueError(
                f"Row count mismatch in ResidualDataset: "
                f"X_features={n}, base_preds={len(base_preds)}, residuals={len(residuals)}"
            )

        if X_features.shape[1] != 53:
            raise ValueError(
                f"Expected 53 input features for ResidualDataset, got {X_features.shape[1]}"
            )

        self.X_features = X_features.float()
        self.base_preds = base_preds.float().reshape(-1, 1)
        self.residuals = residuals.float().reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.residuals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_in: (54,) concatenated [features(53), base_pred(1)].
            residual: (1,) signed residual target.
        """
        x_feat = self.X_features[idx]
        base_p = self.base_preds[idx]
        x_in = torch.cat([x_feat, base_p], dim=0)
        return x_in, self.residuals[idx]


def build_residual_dataloaders(
    X_train: torch.Tensor,
    base_train: torch.Tensor,
    res_train: torch.Tensor,
    X_val: torch.Tensor,
    base_val: torch.Tensor,
    res_val: torch.Tensor,
    batch_size: int = 1024,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation DataLoaders for residual MLP training.
    """
    train_ds = ResidualDataset(X_train, base_train, res_train)
    val_ds = ResidualDataset(X_val, base_val, res_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader


def train_residual_mlp(
    model: ResidualMLPNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    device: Optional[torch.device] = None,
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[HybridConfig] = None,
) -> Dict[str, Any]:
    """
    Train ResidualMLPNet using AdamW + SmoothL1Loss (Huber) + ReduceLROnPlateau + EarlyStopping.

    Args:
        model: ResidualMLPNet instance to train.
        train_loader: DataLoader yielding (x_54dim, residual) batches.
        val_loader: Validation DataLoader.
        n_epochs: Maximum epochs.
        lr: Learning rate for AdamW.
        patience: EarlyStopping patience.
        device: Compute device.
        save_path: Destination path for best checkpoint.
        config: HybridConfig for metadata serialization.

    Returns:
        history dict containing loss curves, best_val_mae, and best_epoch.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    cfg = config or getattr(model, 'config', HybridConfig())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6
    )

    early_stopper = EarlyStopping(
        patience=patience,
        mode='min',
        min_delta=0.0,
        verbose=True,
    )

    history: Dict[str, Any] = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'lr': [],
    }

    best_val_mae = float('inf')
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"Training ResidualMLPNet — {n_epochs} max epochs, patience={patience}")
    print(f"Device: {device}  |  LR: {lr}  |  Loss: Huber (beta=1.0)")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # ── Train Phase ──
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for x_in, res_target in train_loader:
            x_in = x_in.to(device, non_blocking=True)
            res_target = res_target.to(device, non_blocking=True)
            bs = x_in.size(0)

            optimizer.zero_grad()
            pred = model(x_in)
            loss = criterion(pred, res_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += loss.item() * bs
            train_n += bs

        train_loss = train_loss_sum / max(train_n, 1)

        # ── Validation Phase ──
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        with torch.no_grad():
            for x_in, res_target in val_loader:
                x_in = x_in.to(device, non_blocking=True)
                res_target = res_target.to(device, non_blocking=True)
                bs = x_in.size(0)

                pred = model(x_in)
                val_loss = criterion(pred, res_target)

                val_loss_sum += val_loss.item() * bs
                val_n += bs
                all_preds.append(pred.detach().cpu().numpy().flatten())
                all_targets.append(res_target.detach().cpu().numpy().flatten())

        val_loss_epoch = val_loss_sum / max(val_n, 1)
        val_preds_arr = np.concatenate(all_preds)
        val_true_arr = np.concatenate(all_targets)
        val_metrics = compute_regression_metrics(val_true_arr, val_preds_arr)
        val_mae = val_metrics['mae']

        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss_epoch)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:>3}/{n_epochs}  |  "
            f"Train Loss: {train_loss:.5f}  Val Loss: {val_loss_epoch:.5f}  "
            f"Val MAE: {val_mae:.4f}  LR: {current_lr:.2e}  [{elapsed:.1f}s]"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch

        scheduler.step(val_loss_epoch)

        # Correct operator precedence when saving intermediate checkpoint
        is_best_loss = (early_stopper.best_score is None) or early_stopper.is_better(val_loss_epoch)
        if save_path is not None and is_best_loss:
            save_p = Path(save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'hybrid_config': cfg.to_dict(),
                    'epoch': epoch,
                    'val_loss': float(val_loss_epoch),
                    'val_mae': float(val_mae),
                    'pytorch_version': torch.__version__,
                    'seed': 42,
                },
                save_p,
            )

        should_stop = early_stopper(val_loss_epoch, model, epoch)
        if should_stop:
            print(f"\n⚡ Early stopping triggered at epoch {epoch} (best epoch: {early_stopper.best_epoch})")
            early_stopper.restore_best_weights(model)
            break

    # Ensure best weights are restored and persisted
    if early_stopper.best_epoch is not None:
        early_stopper.restore_best_weights(model)

    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'hybrid_config': cfg.to_dict(),
                'epoch': best_epoch,
                'val_loss': float(min(history['val_loss'])),
                'val_mae': float(best_val_mae),
                'pytorch_version': torch.__version__,
                'seed': 42,
            },
            save_p,
        )
        print(f"✓ Best residual model checkpoint persisted → {save_p}")

    history['best_val_mae'] = best_val_mae
    history['best_epoch'] = best_epoch
    print(f"\n✓ Training complete. Best Val MAE: {best_val_mae:.4f} @ epoch {best_epoch}")

    return history


def evaluate_residual_model(
    model: ResidualMLPNet,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate ResidualMLPNet on a DataLoader and compute regression metrics.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    criterion = nn.SmoothL1Loss(beta=1.0)
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_in, res_target in data_loader:
            x_in = x_in.to(device)
            res_target = res_target.to(device)
            bs = x_in.size(0)

            pred = model(x_in)
            loss = criterion(pred, res_target)
            total_loss += loss.item() * bs
            total_samples += bs

            all_preds.append(pred.cpu().numpy().flatten())
            all_targets.append(res_target.cpu().numpy().flatten())

    preds_arr = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)
    metrics = compute_regression_metrics(targets_arr, preds_arr)
    metrics['loss'] = total_loss / max(total_samples, 1)

    return metrics
