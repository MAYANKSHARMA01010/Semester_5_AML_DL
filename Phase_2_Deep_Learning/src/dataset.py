"""
Phase 2 Deep Learning — Reusable Dataset and Data Loading Module
================================================================
Provides standardized PyTorch Dataset and DataLoader pipeline for CCME WQI regression.
Preserves the Task 1 feature engineering and preprocessing transformations:
  - 36 numerical features standardized with fitted StandardScaler
  - 2 categorical features encoded to 17 one-hot binary features via fitted OneHotEncoder
  - 53 total input dimension
  - Target: ccme_values (continuous continuous CCME Water Quality Index in [0, 100])

All transformations use pre-fitted artifacts from Task 1. NO refitting on validation/test data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
# Path Resolution Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    """Resolve the project root directory (Semester_5_AML_DL)."""
    current = Path.cwd()
    for candidate in [current, current.parent, current.parent.parent]:
        if (candidate / 'Phase_1_Machine_Learning').exists() and (candidate / 'Phase_2_Deep_Learning').exists():
            return candidate.resolve()
    # Fallback to current working directory
    return current.resolve()


def get_default_paths() -> Dict[str, Path]:
    """Return dictionary of canonical paths for data, artifacts, and configs."""
    root = get_project_root()
    phase1_dir = root / 'Phase_1_Machine_Learning'
    phase2_dir = root / 'Phase_2_Deep_Learning'

    return {
        'root': root,
        'phase1_dir': phase1_dir,
        'phase2_dir': phase2_dir,
        'data_dir': phase1_dir / 'data' / 'GeneratedOutput' / '02.Preprocessing&FeatureSelection',
        'artifacts_dir': phase2_dir / 'artifacts',
        'configs_dir': phase2_dir / 'configs',
        'feature_config': phase2_dir / 'configs' / 'feature_config.json',
        'scaler': phase2_dir / 'artifacts' / 'numerical_scaler.joblib',
        'encoder': phase2_dir / 'artifacts' / 'categorical_encoder.joblib',
        'sample_batch': phase2_dir / 'artifacts' / 'sample_batch.pt',
    }


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class WaterQualityDataset(Dataset):
    """
    PyTorch Dataset for the CCME WQI regression task.

    Stores pre-computed float32 tensors for high-performance memory-mapped / in-memory __getitem__.
    Supports two return modes:
      - 'combined'  : (X_all, y)           <- default, used by MLP architectures
      - 'separated' : (X_num, X_cat, y)    <- for embedding-aware or multimodal architectures
    """

    def __init__(
        self,
        x_numerical: torch.Tensor,
        x_categorical: torch.Tensor,
        x_combined: torch.Tensor,
        y: torch.Tensor,
        return_mode: str = 'combined',
    ):
        if len(x_numerical) != len(y) or len(x_categorical) != len(y) or len(x_combined) != len(y):
            raise ValueError(
                f"Row count mismatch in dataset tensors: "
                f"num={len(x_numerical)}, cat={len(x_categorical)}, all={len(x_combined)}, y={len(y)}"
            )

        if return_mode not in ('combined', 'separated'):
            raise ValueError(f"return_mode must be 'combined' or 'separated', got '{return_mode}'")

        self.x_num = x_numerical.float()
        self.x_cat = x_categorical.float()
        self.x_all = x_combined.float()
        self.y = y.float()
        self.return_mode = return_mode

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.return_mode == 'separated':
            return self.x_num[idx], self.x_cat[idx], self.y[idx]
        return self.x_all[idx], self.y[idx]

    def feature_dim(self) -> int:
        """Total input dimension (default: 53)."""
        return self.x_all.shape[1]

    def num_dim(self) -> int:
        """Numerical feature dimension (default: 36)."""
        return self.x_num.shape[1]

    def cat_dim(self) -> int:
        """One-hot categorical feature dimension (default: 17)."""
        return self.x_cat.shape[1]


# ──────────────────────────────────────────────────────────────────────────────
# Data Transformation and Loading Utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_feature_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the feature_config.json containing feature names and dimensions."""
    if config_path is None:
        config_path = get_default_paths()['feature_config']
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_transformers(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[StandardScaler, OneHotEncoder]:
    """
    Load pre-fitted StandardScaler and OneHotEncoder artifacts from Task 1.
    Strictly read-only; no refitting.
    """
    paths = get_default_paths()
    base_dir = artifacts_dir or paths['artifacts_dir']
    scaler_path = base_dir / 'numerical_scaler.joblib'
    encoder_path = base_dir / 'categorical_encoder.joblib'

    if not scaler_path.exists():
        raise FileNotFoundError(f"Fitted numerical scaler not found at: {scaler_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Fitted categorical encoder not found at: {encoder_path}")

    scaler: StandardScaler = joblib.load(scaler_path)
    encoder: OneHotEncoder = joblib.load(encoder_path)
    return scaler, encoder


def prepare_tensors(
    df: pd.DataFrame,
    scaler: StandardScaler,
    encoder: OneHotEncoder,
    numerical_features: List[str],
    categorical_features: List[str],
    target_column: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transform a pandas DataFrame into standardized PyTorch tensors using pre-fitted transformers.

    Returns:
        x_num_tensor: Shape (N, 36)
        x_cat_tensor: Shape (N, 17)
        x_all_tensor: Shape (N, 53)
        y_tensor    : Shape (N, 1)
    """
    # 1. Numerical features: standardize with pre-fitted scaler
    x_num = scaler.transform(df[numerical_features].astype(np.float32))

    # 2. Categorical features: encode with pre-fitted one-hot encoder
    x_cat = encoder.transform(df[categorical_features])
    if hasattr(x_cat, 'toarray'):
        x_cat = x_cat.toarray()

    # 3. Combine numerical + categorical
    x_all = np.hstack([x_num, x_cat]).astype(np.float32)

    # 4. Target variable: CCME WQI continuous values (N, 1)
    y = df[target_column].values.astype(np.float32).reshape(-1, 1)

    return (
        torch.from_numpy(x_num.astype(np.float32)),
        torch.from_numpy(x_cat.astype(np.float32)),
        torch.from_numpy(x_all),
        torch.from_numpy(y),
    )


def load_raw_splits(
    data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw Phase 1 parquet splits (train, validation, test).
    Preserves exact partitions and split assignments.
    """
    base_dir = data_dir or get_default_paths()['data_dir']
    train_path = base_dir / 'train.parquet'
    val_path = base_dir / 'val.parquet'
    test_path = base_dir / 'test.parquet'

    if not train_path.exists():
        raise FileNotFoundError(f"Training split not found at: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation split not found at: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found at: {test_path}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    return train_df, val_df, test_df


def build_datasets(
    data_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    configs_dir: Optional[Path] = None,
    return_mode: str = 'combined',
) -> Tuple[WaterQualityDataset, WaterQualityDataset, WaterQualityDataset, Dict[str, Any]]:
    """
    Build WaterQualityDataset instances for train, val, and test splits.
    Guarantees zero data leakage by strictly applying pre-fitted transformers.
    """
    cfg = load_feature_config(
        (configs_dir / 'feature_config.json') if configs_dir else None
    )
    scaler, encoder = load_transformers(artifacts_dir)

    numerical_cols = cfg['numerical_features']
    categorical_cols = cfg['categorical_features']
    target_col = cfg['target_variable']

    train_df, val_df, test_df = load_raw_splits(data_dir)

    x_train_num, x_train_cat, x_train_all, y_train = prepare_tensors(
        train_df, scaler, encoder, numerical_cols, categorical_cols, target_col
    )
    x_val_num, x_val_cat, x_val_all, y_val = prepare_tensors(
        val_df, scaler, encoder, numerical_cols, categorical_cols, target_col
    )
    x_test_num, x_test_cat, x_test_all, y_test = prepare_tensors(
        test_df, scaler, encoder, numerical_cols, categorical_cols, target_col
    )

    train_dataset = WaterQualityDataset(x_train_num, x_train_cat, x_train_all, y_train, return_mode=return_mode)
    val_dataset = WaterQualityDataset(x_val_num, x_val_cat, x_val_all, y_val, return_mode=return_mode)
    test_dataset = WaterQualityDataset(x_test_num, x_test_cat, x_test_all, y_test, return_mode=return_mode)

    return train_dataset, val_dataset, test_dataset, cfg


def get_dataloaders(
    batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    return_mode: str = 'combined',
    data_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    configs_dir: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Construct ready-to-train PyTorch DataLoaders for Task 3.

    Args:
        batch_size: Batch size (default 1024 from feature_config.json).
        num_workers: Number of workers for data loading (default 0 for macOS stability).
        pin_memory: Whether to pin memory (defaults to True if CUDA is available).
        return_mode: 'combined' for MLP (x_all, y) or 'separated' for (x_num, x_cat, y).
        data_dir: Custom data directory if non-default.
        artifacts_dir: Custom artifacts directory if non-default.
        configs_dir: Custom configs directory if non-default.

    Returns:
        (train_loader, val_loader, test_loader, feature_config)
    """
    train_dataset, val_dataset, test_dataset, cfg = build_datasets(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        configs_dir=configs_dir,
        return_mode=return_mode,
    )

    bs = batch_size if batch_size is not None else cfg.get('batch_size', 1024)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, cfg

