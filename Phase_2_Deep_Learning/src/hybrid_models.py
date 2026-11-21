"""
Phase 2 Deep Learning — Task 4: Hybrid ML + DL Model Components
================================================================
Defines hybrid model architectures combining Phase 1 tree base predictions
with Phase 2 deep learning residual corrections.

Components
----------
HybridConfig      — Dataclass holding hyperparameter configurations.
ResidualMLPNet    — PyTorch neural network predicting signed residual errors.
HybridEnsemble    — Module integrating tree base predictions + ResidualMLPNet corrections.
WeightedEnsemble  — Linear combination of RF, XGBoost, and base MLP predictions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .dl_models import MLPBlock

SEED = 42


@dataclass
class HybridConfig:
    """
    Hyperparameter configuration for the Hybrid ML + DL ensemble.

    Attributes:
        input_dim       : DL feature dimension (53 = 36 numerical + 17 OHE).
        hidden_dims     : Hidden layer dimensions for ResidualMLPNet.
        dropout_rate    : Dropout probability in hidden blocks.
        use_batch_norm  : Whether to use BatchNorm1d in hidden blocks.
        rf_weight       : RF weight in WeightedEnsemble.
        xgb_weight      : XGBoost weight in WeightedEnsemble.
        mlp_weight      : Base MLP weight in WeightedEnsemble.
        residual_weight : Scaling factor applied to predicted residual correction.
        name            : Identifier for the model architecture.
    """
    input_dim: int = 53
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    rf_weight: float = 0.4
    xgb_weight: float = 0.4
    mlp_weight: float = 0.2
    residual_weight: float = 1.0
    name: str = 'Hybrid_RF_XGB_ResidualMLP'

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HybridConfig':
        """Reconstruct HybridConfig from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def residual_input_dim(self) -> int:
        """Total input dimension for ResidualMLPNet: 53 DL features + 1 base tree prediction = 54."""
        return self.input_dim + 1


class ResidualMLPNet(nn.Module):
    """
    Lightweight Deep Neural Network that predicts signed residual errors
    (residual = y_true - base_pred) from input features + base prediction.

    Input dimension: 53 scaled/encoded features + 1 base prediction = 54 inputs.
    Output: 1 scalar representing predicted residual correction.
    """

    def __init__(self, config: Optional[HybridConfig] = None):
        super().__init__()
        self.config = config or HybridConfig()

        actual_input_dim = self.config.input_dim + 1
        dims = [actual_input_dim] + self.config.hidden_dims

        self.hidden_layers = nn.ModuleList([
            MLPBlock(
                in_features=dims[i],
                out_features=dims[i + 1],
                dropout_rate=self.config.dropout_rate,
                use_batch_norm=self.config.use_batch_norm,
                activation='relu',
            )
            for i in range(len(dims) - 1)
        ])

        self.output_layer = nn.Linear(self.config.hidden_dims[-1], 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming normal initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, 54).
        Returns:
            Tensor of shape (batch_size, 1) representing predicted residual.
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
        """Return parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

    def architecture_summary(self) -> str:
        """Return formatted architecture summary string."""
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


class HybridEnsemble(nn.Module):
    """
    Integrates tree base predictions with ResidualMLPNet predictions:
        Hybrid Prediction = base_pred + residual_weight * predicted_residual
    """

    def __init__(self, config: Optional[HybridConfig] = None):
        super().__init__()
        self.config = config or HybridConfig()
        self.residual_net = ResidualMLPNet(self.config)

    def forward(
        self,
        x_features: torch.Tensor,
        base_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x_features: Tensor of shape (batch, 53).
            base_pred: Tensor of shape (batch, 1) or (batch,).
        Returns:
            Tensor of shape (batch, 1) with corrected predictions.
        """
        if x_features.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.config.input_dim}, got {x_features.shape[-1]}"
            )
        if base_pred.dim() == 1:
            base_pred = base_pred.unsqueeze(1)

        x_in = torch.cat([x_features, base_pred], dim=1)
        residual = self.residual_net(x_in)
        return base_pred + self.config.residual_weight * residual

    @torch.no_grad()
    def predict_numpy(
        self,
        x_features_np: np.ndarray,
        base_pred_np: np.ndarray,
        device: Optional[torch.device] = None,
        batch_size: int = 16384,
    ) -> np.ndarray:
        """
        Execute inference on numpy arrays with batched execution for memory safety.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        n_samples = len(x_features_np)
        if len(base_pred_np) != n_samples:
            raise ValueError(
                f"Row mismatch between features ({n_samples}) and base_pred ({len(base_pred_np)})"
            )

        if n_samples <= batch_size:
            x_t = torch.from_numpy(np.asarray(x_features_np, dtype=np.float32)).to(device)
            b_t = torch.from_numpy(np.asarray(base_pred_np, dtype=np.float32)).to(device)
            out = self.forward(x_t, b_t)
            return out.cpu().numpy().flatten()

        all_preds: List[np.ndarray] = []
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            x_batch = torch.from_numpy(
                np.asarray(x_features_np[i:end_idx], dtype=np.float32)
            ).to(device)
            b_batch = torch.from_numpy(
                np.asarray(base_pred_np[i:end_idx], dtype=np.float32)
            ).to(device)
            out_batch = self.forward(x_batch, b_batch)
            all_preds.append(out_batch.cpu().numpy().flatten())

        return np.concatenate(all_preds)


class WeightedEnsemble:
    """
    Computes a weighted average of Random Forest, XGBoost, and MLP predictions:
        y_ens = w_rf * y_rf + w_xgb * y_xgb + w_mlp * y_mlp
    """

    def __init__(
        self,
        rf_weight: float = 0.4,
        xgb_weight: float = 0.4,
        mlp_weight: float = 0.2,
    ):
        total = rf_weight + xgb_weight + mlp_weight
        if abs(total - 1.0) > 1e-5:
            raise ValueError(
                f"WeightedEnsemble weights must sum to 1.0, got {total:.6f} "
                f"(rf={rf_weight}, xgb={xgb_weight}, mlp={mlp_weight})"
            )
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        self.mlp_weight = mlp_weight

    def predict(
        self,
        rf_preds: np.ndarray,
        xgb_preds: np.ndarray,
        mlp_preds: np.ndarray,
    ) -> np.ndarray:
        """Compute weighted linear combination of three predictions."""
        rf_p = np.asarray(rf_preds, dtype=np.float64).flatten()
        xgb_p = np.asarray(xgb_preds, dtype=np.float64).flatten()
        mlp_p = np.asarray(mlp_preds, dtype=np.float64).flatten()

        if not (len(rf_p) == len(xgb_p) == len(mlp_p)):
            raise ValueError(
                f"Length mismatch: rf={len(rf_p)}, xgb={len(xgb_p)}, mlp={len(mlp_p)}"
            )

        return (
            self.rf_weight * rf_p
            + self.xgb_weight * xgb_p
            + self.mlp_weight * mlp_p
        )

    def to_dict(self) -> Dict[str, float]:
        """Serialize weights to dict."""
        return {
            'rf_weight': self.rf_weight,
            'xgb_weight': self.xgb_weight,
            'mlp_weight': self.mlp_weight,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'WeightedEnsemble':
        """Reconstruct from dict."""
        return cls(
            rf_weight=d['rf_weight'],
            xgb_weight=d['xgb_weight'],
            mlp_weight=d['mlp_weight'],
        )

    def __repr__(self) -> str:
        return (
            f"WeightedEnsemble(rf={self.rf_weight}, "
            f"xgb={self.xgb_weight}, mlp={self.mlp_weight})"
        )


def save_hybrid_checkpoint(
    model: nn.Module,
    config: HybridConfig,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save HybridEnsemble or ResidualMLPNet checkpoint with embedded config and metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        'model_state_dict': model.state_dict(),
        'hybrid_config': config.to_dict(),
        'pytorch_version': torch.__version__,
        'seed': SEED,
    }
    if metadata:
        payload.update(metadata)

    torch.save(payload, path)
    print(f"✓ Hybrid checkpoint saved → {path}")


def load_hybrid_checkpoint(
    path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[HybridEnsemble, HybridConfig, Dict[str, Any]]:
    """
    Load a HybridEnsemble from a checkpoint file.
    Supports checkpoints saved directly from ResidualMLPNet or HybridEnsemble.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = HybridConfig.from_dict(ckpt.get('hybrid_config', HybridConfig().to_dict()))
    model = HybridEnsemble(config).to(device)

    state_dict = ckpt['model_state_dict']
    if any(k.startswith('residual_net.') for k in state_dict.keys()):
        model.load_state_dict(state_dict)
    else:
        model.residual_net.load_state_dict(state_dict)

    model.eval()
    print(f"✓ Hybrid checkpoint loaded from {path}")
    return model, config, ckpt
