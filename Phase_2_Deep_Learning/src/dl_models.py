"""
Phase 2 Deep Learning — Shared Model Definitions
=================================================
Importable by all Phase 2 notebooks (02 architecture, 03 training,
04 hybrid, 05 evaluation, 06 explainability) without copy-paste.

Classes
-------
MLPConfig          — Hyperparameter dataclass for WaterQualityMLP.
MLPBlock           — Single hidden block: Linear → [BN] → ReLU → [DO].
WaterQualityMLP    — Primary MLP regressor with Kaiming init.
ResidualBlock      — Tabular residual block with optional projection shortcut.
ResidualMLP        — ResNet-style MLP with skip connections.

Functions
---------
build_model               — Factory: MLPConfig → WaterQualityMLP on device.
save_model_checkpoint     — Save state_dict + config + metadata.
load_model_checkpoint     — Restore model from checkpoint.
mc_dropout_predictions    — Monte Carlo Dropout uncertainty estimation.
count_params_per_layer    — Per-layer parameter breakdown table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# Configuration Dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MLPConfig:
    """
    Hyperparameter configuration for WaterQualityMLP.

    Attributes
    ----------
    input_dim     : Total input feature dimension (default 53 = 36 num + 17 OHE).
    hidden_dims   : List of hidden layer widths defining the funnel topology.
    dropout_rate  : Dropout probability applied after each ReLU (0 = disabled).
    use_batch_norm: Whether to apply BatchNorm1d between Linear and ReLU.
    activation    : Activation function name — 'relu', 'leakyrelu', or 'gelu'.
    output_dim    : Output dimension (1 for CCME WQI regression).
    name          : Human-readable experiment identifier.
    """

    input_dim     : int        = 53
    hidden_dims   : List[int]  = field(default_factory=lambda: [512, 256, 128, 64])
    dropout_rate  : float      = 0.30
    use_batch_norm: bool       = True
    activation    : str        = 'relu'       # 'relu' | 'leakyrelu' | 'gelu'
    output_dim    : int        = 1
    name          : str        = 'MLP_Base'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MLPConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def n_layers(self) -> int:
        return len(self.hidden_dims)


# ══════════════════════════════════════════════════════════════════════════════
# Activation Helper
# ══════════════════════════════════════════════════════════════════════════════

def _get_activation(name: str) -> nn.Module:
    """Return an activation module by name string."""
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    elif name == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation: '{name}'. Choose relu/leakyrelu/gelu.")


# ══════════════════════════════════════════════════════════════════════════════
# Building Block
# ══════════════════════════════════════════════════════════════════════════════

class MLPBlock(nn.Module):
    """
    Single MLP hidden-layer block.

    Forward order:  Linear  →  [BatchNorm1d]  →  Activation  →  [Dropout]

    Pre-activation BatchNorm (BN before activation) is preferred for regression:
    - Keeps gradient magnitudes bounded through deep stacks.
    - Prevents dead neurons by normalizing pre-activation logits.

    Args
    ----
    in_features    : Input dimension.
    out_features   : Output dimension.
    dropout_rate   : Dropout probability (0.0 = no dropout).
    use_batch_norm : Whether to include BatchNorm1d.
    activation     : Activation function name string.
    """

    def __init__(
        self,
        in_features   : int,
        out_features  : int,
        dropout_rate  : float = 0.30,
        use_batch_norm: bool  = True,
        activation    : str   = 'relu',
    ):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_features, out_features)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(_get_activation(activation))
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(p=dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ══════════════════════════════════════════════════════════════════════════════
# Primary MLP
# ══════════════════════════════════════════════════════════════════════════════

class WaterQualityMLP(nn.Module):
    """
    Fully-connected MLP for CCME WQI regression.

    Architecture (default MLP_Base):
        Input(53) → Block(53→512) → Block(512→256) → Block(256→128)
                  → Block(128→64) → Linear(64→1)

    Design choices
    --------------
    - Funnel topology: decreasing widths for progressive feature compression.
    - BatchNorm1d (pre-activation): stabilises 1.97M-row gradient updates.
    - Dropout: stochastic regularisation preventing confident memorisation.
    - Raw linear output: no sigmoid/tanh — covers full real line.
    - Kaiming He init: recommended for ReLU family activations.
    - Input validation: raises ValueError on dimension mismatch.

    Args
    ----
    config : MLPConfig dataclass carrying all hyperparameters.
    """

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config

        # ── Validate config ───────────────────────────────────────────────────
        if config.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {config.input_dim}")
        if not config.hidden_dims:
            raise ValueError("hidden_dims must be a non-empty list")
        if not (0.0 <= config.dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1), got {config.dropout_rate}")

        # ── Hidden blocks ─────────────────────────────────────────────────────
        dims = [config.input_dim] + config.hidden_dims
        self.hidden_layers = nn.ModuleList([
            MLPBlock(
                in_features    = dims[i],
                out_features   = dims[i + 1],
                dropout_rate   = config.dropout_rate,
                use_batch_norm = config.use_batch_norm,
                activation     = config.activation,
            )
            for i in range(len(dims) - 1)
        ])

        # ── Regression output head ────────────────────────────────────────────
        self.output_layer = nn.Linear(config.hidden_dims[-1], config.output_dim)

        # ── Kaiming He weight initialisation ─────────────────────────────────
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming He normal init for linear layers; ones/zeros for BN."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with runtime input-dimension guard.

        Args
        ----
        x : Tensor of shape (batch_size, input_dim).

        Returns
        -------
        Tensor of shape (batch_size, output_dim).
        """
        # ── Runtime shape guard ───────────────────────────────────────────────
        if x.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.config.input_dim}, "
                f"got {x.shape[-1]}. Check that the scaler/encoder from "
                f"notebook 01 was applied before passing data to this model."
            )

        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def predict_clipped(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference-time forward pass with CCME WQI domain clipping to [0, 100].
        Does NOT modify loss computation during training.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x).clamp(0.0, 100.0)

    def count_parameters(self) -> Dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

    def per_layer_param_table(self) -> str:
        """
        Return a human-readable per-layer parameter breakdown table.
        Useful for viva explanations and architecture reports.
        """
        rows = [f"{'Layer':<35} {'Type':<18} {'Shape':<25} {'Params':>10}"]
        rows.append('─' * 90)
        grand_total = 0
        for name, param in self.named_parameters():
            shape = str(tuple(param.shape))
            n     = param.numel()
            grand_total += n
            rows.append(f"  {name:<33} {'Tensor':<18} {shape:<25} {n:>10,}")
        rows.append('─' * 90)
        rows.append(f"  {'TOTAL':<33} {'':<18} {'':<25} {grand_total:>10,}")
        return '\n'.join(rows)

    def architecture_summary(self) -> str:
        dims = [self.config.input_dim] + self.config.hidden_dims + [self.config.output_dim]
        flow = ' → '.join(str(d) for d in dims)
        p    = self.count_parameters()
        return (
            f"Architecture  : {self.config.name}\n"
            f"Layer flow    : {flow}\n"
            f"Activation    : {self.config.activation}\n"
            f"BatchNorm     : {self.config.use_batch_norm}\n"
            f"Dropout       : {self.config.dropout_rate}\n"
            f"Parameters    : {p['trainable']:,} trainable / {p['total']:,} total"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Residual MLP
# ══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """
    Tabular residual block with optional projection shortcut.

    Forward:  x → Linear → BN → ReLU → Dropout → Linear → BN → (+shortcut(x)) → ReLU

    When in_features ≠ out_features, a 1×1 Linear projection aligns dimensions.
    """

    def __init__(
        self,
        in_features : int,
        out_features: int,
        dropout_rate: float = 0.20,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                nn.BatchNorm1d(out_features),
            )
            if in_features != out_features
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + self.shortcut(x))


class ResidualMLP(nn.Module):
    """
    MLP with residual skip connections for CCME WQI regression.

    Architecture:
        Input Projection (input_dim → hidden_dim)
            → [ResidualBlock(hidden_dim)] × num_blocks
            → Linear(hidden_dim → 1)

    Args
    ----
    input_dim    : Input feature dimension (53).
    hidden_dim   : Uniform hidden dimension within residual blocks.
    num_blocks   : Number of stacked residual blocks.
    dropout_rate : Dropout within each block.
    """

    def __init__(
        self,
        input_dim   : int   = 53,
        hidden_dim  : int   = 256,
        num_blocks  : int   = 4,
        dropout_rate: float = 0.20,
    ):
        super().__init__()
        self.name       = 'MLP_ResNet'
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.output_head = nn.Linear(hidden_dim, 1)
        nn.init.kaiming_normal_(self.output_head.weight, nonlinearity='relu')
        nn.init.zeros_(self.output_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, got {x.shape[-1]}"
            )
        x = self.input_proj(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.output_head(x)

    def count_parameters(self) -> Dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


# ══════════════════════════════════════════════════════════════════════════════
# Factory & Checkpoint Utilities
# ══════════════════════════════════════════════════════════════════════════════

def build_model(
    config: MLPConfig,
    device: Optional[torch.device] = None,
    seed  : int = SEED,
) -> WaterQualityMLP:
    """
    Factory: construct a reproducible WaterQualityMLP from config.

    Args
    ----
    config : MLPConfig with architecture hyperparameters.
    device : torch.device (defaults to CUDA if available, else CPU).
    seed   : Random seed for weight initialisation reproducibility.

    Returns
    -------
    WaterQualityMLP moved to device, in eval mode.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    model = WaterQualityMLP(config).to(device)
    return model


def save_model_checkpoint(
    model     : nn.Module,
    path      : Path,
    config    : Optional[MLPConfig] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a portable model checkpoint embedding state_dict + config.

    Args
    ----
    model      : Trained PyTorch module.
    path       : Destination .pt file path.
    config     : MLPConfig (stored for reconstruction; optional).
    extra_meta : Additional metadata (epoch, val_mae, val_r2, …).
    """
    payload: Dict[str, Any] = {
        'model_state_dict' : model.state_dict(),
        'pytorch_version'  : torch.__version__,
        'seed'             : SEED,
    }
    if config is not None:
        payload['architecture_config'] = config.to_dict()
    if extra_meta:
        payload.update(extra_meta)
    torch.save(payload, path)
    print(f'✓ Checkpoint saved → {path}')


def load_model_checkpoint(
    path  : Path,
    device: Optional[torch.device] = None,
) -> Tuple['WaterQualityMLP', Dict[str, Any]]:
    """
    Load WaterQualityMLP from a checkpoint.

    Args
    ----
    path   : Path to .pt checkpoint file.
    device : Target device (auto-detected if None).

    Returns
    -------
    (model, checkpoint_dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(path, map_location=device, weights_only=False)
    cfg_d  = ckpt['architecture_config']
    config = MLPConfig.from_dict(cfg_d)
    model  = WaterQualityMLP(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


# ══════════════════════════════════════════════════════════════════════════════
# Analysis Utilities
# ══════════════════════════════════════════════════════════════════════════════

def mc_dropout_predictions(
    model    : nn.Module,
    x        : torch.Tensor,
    n_samples: int = 50,
    device   : Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo Dropout: stochastic forward passes for uncertainty estimation.

    Keeps dropout active while maintaining BatchNorm / LayerNorm in evaluation mode
    to sample different sub-networks without corrupting batch normalization statistics.

    Args
    ----
    model     : Trained model with Dropout layers.
    x         : Input tensor (N, input_dim).
    n_samples : Number of stochastic forward passes.
    device    : Target device.

    Returns
    -------
    mean_preds : (N,) mean prediction across MC samples.
    std_preds  : (N,) predictive std (epistemic uncertainty proxy).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    all_preds: List[np.ndarray] = []
    with torch.no_grad():
        x_dev = x.to(device)
        for _ in range(n_samples):
            p = model(x_dev).cpu().numpy().flatten()
            all_preds.append(p)
    model.eval()

    preds_arr = np.stack(all_preds, axis=0)   # (n_samples, N)
    mean_preds = preds_arr.mean(axis=0)
    std_preds = preds_arr.std(axis=0)

    if not np.all(np.isfinite(mean_preds)) or not np.all(np.isfinite(std_preds)):
        raise ValueError("MC Dropout produced non-finite predictions.")

    return mean_preds, std_preds


def count_params_per_layer(model: nn.Module) -> str:
    """
    Return a formatted per-layer parameter breakdown table.

    Args
    ----
    model : Any PyTorch nn.Module.

    Returns
    -------
    Multi-line string table suitable for print() or Jupyter display.
    """
    rows = [f"{'Layer Name':<40} {'Shape':<25} {'Params':>10} {'Trainable':>10}"]
    rows.append('─' * 90)
    grand_total = grand_trainable = 0

    for name, param in model.named_parameters():
        shape = str(tuple(param.shape))
        n     = param.numel()
        t     = n if param.requires_grad else 0
        grand_total     += n
        grand_trainable += t
        rows.append(
            f"  {name:<38} {shape:<25} {n:>10,} {t:>10,}"
        )

    rows.append('─' * 90)
    rows.append(
        f"  {'TOTAL':<38} {'':<25} {grand_total:>10,} {grand_trainable:>10,}"
    )
    return '\n'.join(rows)
