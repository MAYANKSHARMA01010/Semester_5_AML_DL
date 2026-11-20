"""
Phase 4 — Hybrid ML + DL Model Components
==========================================
HybridConfig      — Dataclass for all hyperparameters.
ResidualMLPNet    — Lightweight MLP that predicts residuals (y - base_pred).
HybridEnsemble   — Wrapper combining RF/XGB base predictions + ResidualMLPNet correction.
WeightedEnsemble — Simple weighted average of RF, XGB and MLP base predictions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .dl_models import MLPBlock, MLPConfig

SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# Configuration Dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HybridConfig:
    """
    Hyperparameter configuration for the Hybrid ML + DL ensemble.

    Attributes
    ----------
    input_dim       : Base feature dimension (53 = 36 numerical + 17 OHE).
    hidden_dims     : Hidden layer widths for ResidualMLPNet.
    dropout_rate    : Dropout probability applied after each hidden block.
    use_batch_norm  : Whether to apply BatchNorm1d in hidden blocks.
    rf_weight       : Weight for Random Forest predictions in WeightedEnsemble.
    xgb_weight      : Weight for XGBoost predictions in WeightedEnsemble.
    mlp_weight      : Weight for base MLP predictions in WeightedEnsemble.
    residual_weight : Scale factor applied to residual correction output.
    name            : Human-readable experiment identifier.
    """

    input_dim      : int        = 53
    hidden_dims    : List[int]  = field(default_factory=lambda: [256, 128, 64])
    dropout_rate   : float      = 0.2
    use_batch_norm : bool       = True
    rf_weight      : float      = 0.4
    xgb_weight     : float      = 0.4
    mlp_weight     : float      = 0.2
    residual_weight: float      = 1.0
    name           : str        = 'Hybrid_RF_XGB_ResidualMLP'

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to plain dict (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HybridConfig':
        """Reconstruct HybridConfig from a dict (e.g., loaded from JSON)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def residual_input_dim(self) -> int:
        """Input dimension for ResidualMLPNet: features (53) + 1 base prediction = 54."""
        return self.input_dim + 1


# ══════════════════════════════════════════════════════════════════════════════
# ResidualMLPNet — lightweight residual corrector
# ══════════════════════════════════════════════════════════════════════════════

class ResidualMLPNet(nn.Module):
    """
    Lightweight MLP that predicts the signed residual correction
    (y_true − base_pred) from concatenated [features, base_prediction].

    Architecture (default HybridConfig):
        Input(54) → Block(54→256) → Block(256→128) → Block(128→64) → Linear(64→1)

    The model intentionally uses a lighter topology (3 blocks vs 4 in WaterQualityMLP)
    and lower dropout (0.2) to avoid over-correcting the base tree predictions.

    Args
    ----
    config : HybridConfig carrying architecture hyperparameters.
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config

        # Input dimension: 53 features + 1 base prediction scalar
        actual_input_dim = config.input_dim + 1

        # Build hidden blocks using reusable MLPBlock from dl_models
        dims = [actual_input_dim] + config.hidden_dims
        self.hidden_layers = nn.ModuleList([
            MLPBlock(
                in_features    = dims[i],
                out_features   = dims[i + 1],
                dropout_rate   = config.dropout_rate,
                use_batch_norm = config.use_batch_norm,
                activation     = 'relu',
            )
            for i in range(len(dims) - 1)
        ])

        # Output head: predicts signed residual (single scalar)
        self.output_layer = nn.Linear(config.hidden_dims[-1], 1)

        # Kaiming He initialisation for all linear layers
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming He normal init for linear layers; ones/zeros for BatchNorm."""
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
        Forward pass.

        Args
        ----
        x : Tensor of shape (batch_size, input_dim + 1) where the last element
            is the base model prediction scalar.

        Returns
        -------
        Tensor of shape (batch_size, 1) — predicted signed residual correction.
        """
        expected_dim = self.config.input_dim + 1
        if x.shape[-1] != expected_dim:
            raise ValueError(
                f"ResidualMLPNet input dimension mismatch: "
                f"expected {expected_dim} (53 features + 1 base pred), got {x.shape[-1]}"
            )
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def count_parameters(self) -> Dict[str, int]:
        """Return total and trainable parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

    def architecture_summary(self) -> str:
        """Human-readable architecture summary string."""
        actual_in = self.config.input_dim + 1
        dims = [actual_in] + self.config.hidden_dims + [1]
        flow = ' → '.join(str(d) for d in dims)
        p = self.count_parameters()
        return (
            f"Architecture  : {self.config.name} / ResidualMLPNet\n"
            f"Layer flow    : {flow}\n"
            f"BatchNorm     : {self.config.use_batch_norm}\n"
            f"Dropout       : {self.config.dropout_rate}\n"
            f"Parameters    : {p['trainable']:,} trainable / {p['total']:,} total"
        )


# ══════════════════════════════════════════════════════════════════════════════
# HybridEnsemble — tree base + residual correction wrapper
# ══════════════════════════════════════════════════════════════════════════════

class HybridEnsemble(nn.Module):
    """
    Wraps ResidualMLPNet to provide a corrected prediction on top of tree-model
    base predictions.

    Usage
    -----
    ensemble = HybridEnsemble(config)
    corrected = ensemble.predict_numpy(X_features_np, base_pred_np)
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.residual_net = ResidualMLPNet(config)

    def forward(
        self,
        x_features : torch.Tensor,
        base_pred  : torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: concatenate features + base_pred and predict residual correction.

        Args
        ----
        x_features : Tensor (batch, 53) — scaled + encoded feature vector.
        base_pred  : Tensor (batch, 1)  — base model prediction.

        Returns
        -------
        Tensor (batch, 1) — corrected prediction (base_pred + residual).
        """
        if base_pred.dim() == 1:
            base_pred = base_pred.unsqueeze(1)
        x_in = torch.cat([x_features, base_pred], dim=1)
        residual = self.residual_net(x_in)
        return base_pred + self.config.residual_weight * residual

    @torch.no_grad()
    def predict_numpy(
        self,
        x_features_np : np.ndarray,
        base_pred_np  : np.ndarray,
        device        : Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Convenience method: accepts numpy arrays, returns numpy array.

        Args
        ----
        x_features_np : (N, 53) float32 numpy array.
        base_pred_np  : (N,) or (N, 1) float32 numpy array.
        device        : torch.device (auto-detected if None).

        Returns
        -------
        corrected_preds : (N,) numpy array of corrected predictions.
        """
        if device is None:
            device = next(self.residual_net.parameters()).device

        self.eval()
        x_t = torch.from_numpy(x_features_np.astype(np.float32)).to(device)
        b_t = torch.from_numpy(base_pred_np.astype(np.float32)).to(device)
        if b_t.dim() == 1:
            b_t = b_t.unsqueeze(1)

        preds = self.forward(x_t, b_t)
        return preds.cpu().numpy().flatten()


# ══════════════════════════════════════════════════════════════════════════════
# WeightedEnsemble — simple linear combination of three base models
# ══════════════════════════════════════════════════════════════════════════════

class WeightedEnsemble:
    """
    Simple weighted average of RF, XGBoost, and base MLP predictions.

    Not an nn.Module — purely numpy-based inference.

    Parameters
    ----------
    rf_weight  : float — weight for Random Forest predictions.
    xgb_weight : float — weight for XGBoost predictions.
    mlp_weight : float — weight for base MLP predictions.

    Raises
    ------
    ValueError : if the three weights do not sum to 1.0 within tolerance 1e-5.
    """

    def __init__(
        self,
        rf_weight : float = 0.4,
        xgb_weight: float = 0.4,
        mlp_weight: float = 0.2,
    ):
        total = rf_weight + xgb_weight + mlp_weight
        if abs(total - 1.0) > 1e-5:
            raise ValueError(
                f"WeightedEnsemble weights must sum to 1.0, got {total:.6f} "
                f"(rf={rf_weight}, xgb={xgb_weight}, mlp={mlp_weight}). "
                f"Adjust weights so they sum to exactly 1."
            )
        self.rf_weight  = rf_weight
        self.xgb_weight = xgb_weight
        self.mlp_weight = mlp_weight

    def predict(
        self,
        rf_preds : np.ndarray,
        xgb_preds: np.ndarray,
        mlp_preds: np.ndarray,
    ) -> np.ndarray:
        """
        Compute weighted average of three prediction arrays.

        Args
        ----
        rf_preds  : (N,) Random Forest predictions.
        xgb_preds : (N,) XGBoost predictions.
        mlp_preds : (N,) base MLP predictions.

        Returns
        -------
        (N,) weighted ensemble predictions.
        """
        rf_preds  = np.asarray(rf_preds,  dtype=np.float64).flatten()
        xgb_preds = np.asarray(xgb_preds, dtype=np.float64).flatten()
        mlp_preds = np.asarray(mlp_preds, dtype=np.float64).flatten()

        return (
            self.rf_weight  * rf_preds
            + self.xgb_weight * xgb_preds
            + self.mlp_weight * mlp_preds
        )

    def to_dict(self) -> Dict[str, float]:
        """Serialize weights to plain dict."""
        return {
            'rf_weight' : self.rf_weight,
            'xgb_weight': self.xgb_weight,
            'mlp_weight': self.mlp_weight,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'WeightedEnsemble':
        """Reconstruct WeightedEnsemble from a dict."""
        return cls(
            rf_weight  = d['rf_weight'],
            xgb_weight = d['xgb_weight'],
            mlp_weight = d['mlp_weight'],
        )

    def __repr__(self) -> str:
        return (
            f"WeightedEnsemble(rf={self.rf_weight}, "
            f"xgb={self.xgb_weight}, mlp={self.mlp_weight})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint Utilities
# ══════════════════════════════════════════════════════════════════════════════

def save_hybrid_checkpoint(
    model   : nn.Module,
    config  : HybridConfig,
    path    : Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a HybridEnsemble or ResidualMLPNet checkpoint with embedded config.

    Args
    ----
    model    : Trained nn.Module (HybridEnsemble or ResidualMLPNet).
    config   : HybridConfig used to construct the model.
    path     : Destination .pt file path.
    metadata : Optional extra metadata dict (e.g. val_mae, best_epoch).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        'model_state_dict': model.state_dict(),
        'hybrid_config'   : config.to_dict(),
        'pytorch_version' : torch.__version__,
        'seed'            : SEED,
    }
    if metadata:
        payload.update(metadata)

    torch.save(payload, path)
    print(f'✓ Hybrid checkpoint saved → {path}')


def load_hybrid_checkpoint(
    path  : Path,
    device: Optional[torch.device] = None,
) -> Tuple[HybridEnsemble, HybridConfig, Dict[str, Any]]:
    """
    Load a HybridEnsemble model from a checkpoint.

    Args
    ----
    path   : Path to .pt checkpoint file.
    device : Target device (auto-detected if None).

    Returns
    -------
    (model, config, checkpoint_dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt   = torch.load(path, map_location=device, weights_only=False)
    config = HybridConfig.from_dict(ckpt['hybrid_config'])
    model  = HybridEnsemble(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f'✓ Hybrid checkpoint loaded from {path}')
    return model, config, ckpt
