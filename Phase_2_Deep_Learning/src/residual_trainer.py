"""
Phase 4 — Residual MLP Training Engine
=======================================
Trains the ResidualMLPNet to correct systematic errors of tree ensemble base models.
Input: [53-dim features, base_prediction] → predicts residual correction.

Classes
-------
ResidualDataset          — PyTorch Dataset combining features, base preds, and residuals.

Functions
---------
build_residual_dataloaders — Build train/val DataLoaders for residual training.
train_residual_mlp         — Full training loop with EarlyStopping and LR scheduling.
evaluate_residual_model    — Evaluate residual model (MAE, RMSE, R²) on a DataLoader.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── Ensure project root is on sys.path ────────────────────────────────────────
_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .hybrid_models import ResidualMLPNet, HybridConfig, save_hybrid_checkpoint
from .training import EarlyStopping, get_device, compute_regression_metrics


# ══════════════════════════════════════════════════════════════════════════════
# ResidualDataset
# ══════════════════════════════════════════════════════════════════════════════

class ResidualDataset(Dataset):
    """
    PyTorch Dataset for residual MLP training.

    Each item pairs the concatenated [feature_vector, base_prediction] input
    with the signed residual target (y_true − base_pred).

    Args
    ----
    X_features : Tensor (N, 53) — scaled + encoded feature vectors.
    base_preds : Tensor (N, 1)  — base tree model predictions.
    residuals  : Tensor (N, 1)  — signed residual targets.
    """

    def __init__(
        self,
        X_features: torch.Tensor,
        base_preds: torch.Tensor,
        residuals : torch.Tensor,
    ):
        n = len(X_features)
        if len(base_preds) != n or len(residuals) != n:
            raise ValueError(
                f"Row count mismatch: X_features={n}, "
                f"base_preds={len(base_preds)}, residuals={len(residuals)}"
            )

        # Ensure correct shapes
        self.X_features = X_features.float()                              # (N, 53)
        self.base_preds = base_preds.float().reshape(-1, 1)               # (N, 1)
        self.residuals  = residuals.float().reshape(-1, 1)                # (N, 1)

    def __len__(self) -> int:
        return len(self.residuals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x_in     : (54,) concatenated [features(53), base_pred(1)].
        residual : (1,) signed residual target.
        """
        x_feat   = self.X_features[idx]          # (53,)
        base_p   = self.base_preds[idx]           # (1,)
        x_in     = torch.cat([x_feat, base_p], dim=0)   # (54,)
        return x_in, self.residuals[idx]


# ══════════════════════════════════════════════════════════════════════════════
# DataLoader Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_residual_dataloaders(
    X_train    : torch.Tensor,
    base_train : torch.Tensor,
    res_train  : torch.Tensor,
    X_val      : torch.Tensor,
    base_val   : torch.Tensor,
    res_val    : torch.Tensor,
    batch_size : int = 1024,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation DataLoaders for residual MLP training.

    Args
    ----
    X_train    : (N_train, 53) feature tensor.
    base_train : (N_train,) or (N_train, 1) base predictions on train set.
    res_train  : (N_train,) or (N_train, 1) residuals on train set.
    X_val      : (N_val, 53) feature tensor.
    base_val   : (N_val,) or (N_val, 1) base predictions on val set.
    res_val    : (N_val,) or (N_val, 1) residuals on val set.
    batch_size : Mini-batch size (default 1024).
    num_workers: DataLoader worker processes (default 0 for macOS stability).

    Returns
    -------
    (train_loader, val_loader)
    """
    train_ds = ResidualDataset(X_train, base_train, res_train)
    val_ds   = ResidualDataset(X_val,   base_val,   res_val)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = False,
    )
    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train_residual_mlp(
    model       : ResidualMLPNet,
    train_loader: DataLoader,
    val_loader  : DataLoader,
    n_epochs    : int = 50,
    lr          : float = 1e-3,
    patience    : int = 10,
    device      : Optional[torch.device] = None,
    save_path   : Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Train ResidualMLPNet using AdamW + SmoothL1Loss + ReduceLROnPlateau + EarlyStopping.

    Args
    ----
    model        : ResidualMLPNet instance to train.
    train_loader : DataLoader yielding (x_54dim, residual) batches.
    val_loader   : Validation DataLoader.
    n_epochs     : Maximum training epochs (default 50).
    lr           : Initial learning rate for AdamW (default 1e-3).
    patience     : EarlyStopping patience in epochs (default 10).
    device       : Compute device (auto-detected if None).
    save_path    : Optional path to save best checkpoint (.pt file).

    Returns
    -------
    history : Dict with keys:
        'train_loss' (List[float]) — per-epoch Huber loss on train set.
        'val_loss'   (List[float]) — per-epoch Huber loss on val set.
        'best_val_mae' (float)     — best validation MAE achieved.
        'best_epoch'   (int)       — epoch at which best val MAE occurred.
    """
    if device is None:
        device = get_device()

    model = model.to(device)

    # Optimizer: AdamW with L2 weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Loss: SmoothL1 (Huber) — robust to outlier residuals
    criterion = nn.SmoothL1Loss(beta=1.0)

    # Scheduler: reduce LR on val_loss plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6,
    )

    # Early stopping (monitors val_loss)
    early_stopper = EarlyStopping(
        patience  = patience,
        mode      = 'min',
        min_delta = 0.0,
        verbose   = True,
    )

    history: Dict[str, Any] = {
        'train_loss' : [],
        'val_loss'   : [],
        'val_mae'    : [],
        'lr'         : [],
    }

    best_val_mae = float('inf')
    best_epoch   = 0

    print(f"\n{'='*60}")
    print(f"Training ResidualMLPNet — {n_epochs} max epochs, patience={patience}")
    print(f"Device: {device}  |  LR: {lr}  |  Optimizer: AdamW")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_n        = 0

        for x_in, res_target in train_loader:
            x_in       = x_in.to(device, non_blocking=True)
            res_target = res_target.to(device, non_blocking=True)
            bs         = x_in.size(0)

            optimizer.zero_grad()
            pred = model(x_in)
            loss = criterion(pred, res_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += loss.item() * bs
            train_n        += bs

        train_loss = train_loss_sum / max(train_n, 1)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_n        = 0
        all_preds    : List[np.ndarray] = []
        all_targets  : List[np.ndarray] = []

        with torch.no_grad():
            for x_in, res_target in val_loader:
                x_in       = x_in.to(device, non_blocking=True)
                res_target = res_target.to(device, non_blocking=True)
                bs         = x_in.size(0)

                pred     = model(x_in)
                val_loss = criterion(pred, res_target)

                val_loss_sum += val_loss.item() * bs
                val_n        += bs
                all_preds.append(pred.detach().cpu().numpy().flatten())
                all_targets.append(res_target.detach().cpu().numpy().flatten())

        val_loss_epoch = val_loss_sum / max(val_n, 1)
        val_preds_arr  = np.concatenate(all_preds)
        val_true_arr   = np.concatenate(all_targets)
        val_metrics    = compute_regression_metrics(val_true_arr, val_preds_arr)
        val_mae        = val_metrics['mae']

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

        # Track best val MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch   = epoch

        # LR Scheduler step
        scheduler.step(val_loss_epoch)

        # Save best checkpoint
        if save_path is not None and early_stopper.best_score is None or (
            early_stopper.best_score is not None and
            early_stopper.is_better(val_loss_epoch)
        ):
            if save_path is not None:
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'epoch'           : epoch,
                        'val_loss'        : val_loss_epoch,
                        'val_mae'         : val_mae,
                    },
                    save_path,
                )

        # Early stopping check
        should_stop = early_stopper(val_loss_epoch, model, epoch)
        if should_stop:
            print(f"\n⚡ Early stopping triggered at epoch {epoch} (best epoch: {early_stopper.best_epoch})")
            early_stopper.restore_best_weights(model)
            break

    history['best_val_mae'] = best_val_mae
    history['best_epoch']   = best_epoch

    print(f"\n✓ Training complete. Best Val MAE: {best_val_mae:.4f} @ epoch {best_epoch}")
    if save_path:
        print(f"  Checkpoint: {save_path}")

    return history


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_residual_model(
    model : ResidualMLPNet,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate the trained ResidualMLPNet on a DataLoader.

    Args
    ----
    model  : Trained ResidualMLPNet (or any nn.Module with matching I/O).
    loader : DataLoader yielding (x_54dim, residual) batches.
    device : Compute device (auto-detected if None).

    Returns
    -------
    Dict with keys 'mae', 'rmse', 'r2' — computed on the residual targets.
    """
    if device is None:
        device = get_device()

    model.eval()
    all_preds   : List[np.ndarray] = []
    all_targets : List[np.ndarray] = []

    with torch.no_grad():
        for x_in, res_target in loader:
            x_in       = x_in.to(device, non_blocking=True)
            res_target = res_target.to(device, non_blocking=True)
            pred       = model(x_in)
            all_preds.append(pred.detach().cpu().numpy().flatten())
            all_targets.append(res_target.detach().cpu().numpy().flatten())

    preds_arr   = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)

    return compute_regression_metrics(targets_arr, preds_arr)
