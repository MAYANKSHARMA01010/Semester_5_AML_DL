"""
Phase 2 Task 5 — Dropout and Regularization Experiments Runner
==============================================================
Runs controlled, reproducible sweeps over:
  1. Dropout rates: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] (fixed weight_decay=1e-4)
  2. Weight decay:  [0.0, 1e-5, 1e-4, 1e-3, 1e-2] (fixed dropout=0.3)

Records training loss, validation loss, validation MAE, RMSE, R²,
generalization gap, and overfitting diagnostics.

Outputs saved to:
  Phase_2_Deep_Learning/reports/evaluation_results/dropout_sweep_results.csv
  Phase_2_Deep_Learning/reports/evaluation_results/dropout_sweep_results.json
  Phase_2_Deep_Learning/reports/evaluation_results/weight_decay_results.csv
  Phase_2_Deep_Learning/reports/evaluation_results/weight_decay_results.json
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Ensure project root is on sys.path ────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Phase_2_Deep_Learning.src.dataset import (
    load_feature_config,
    load_transformers,
    prepare_tensors,
    load_raw_splits,
)
from Phase_2_Deep_Learning.src.dl_models import WaterQualityMLP, MLPConfig, build_model
from Phase_2_Deep_Learning.src.training import (
    get_device,
    compute_regression_metrics,
    train_one_epoch,
    validate,
)

def train_and_eval_mlp(
    config: MLPConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    weight_decay: float = 1e-4,
    lr: float = 1e-3,
    epochs: int = 3,
) -> Dict[str, Any]:
    """Train an MLP variant for a fixed number of epochs and evaluate performance."""
    model = build_model(config, device=device, seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    t0 = time.time()
    for ep in range(epochs):
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)

    val_m = validate(model, val_loader, criterion, device)
    elapsed = time.time() - t0

    gap_mae = val_m['val_mae'] - train_m['train_mae']
    gap_loss = val_m['val_loss'] - train_m['train_loss']

    return {
        'Config_Name': config.name,
        'Dropout_Rate': config.dropout_rate,
        'Weight_Decay': weight_decay,
        'Train_Loss': float(train_m['train_loss']),
        'Train_MAE': float(train_m['train_mae']),
        'Val_Loss': float(val_m['val_loss']),
        'Val_MAE': float(val_m['val_mae']),
        'Val_RMSE': float(val_m['val_rmse']),
        'Val_R2': float(val_m['val_r2']),
        'Generalization_Gap_MAE': float(gap_mae),
        'Generalization_Gap_Loss': float(gap_loss),
        'Duration_sec': float(elapsed),
    }


def run_regularization_sweeps() -> None:
    print("=" * 75)
    print("  PHASE 2 TASK 5: DROPOUT & WEIGHT DECAY REGULARIZATION EXPERIMENTS")
    print("=" * 75)

    device = get_device()
    print(f"Compute Device: {device}")

    # 1. Load data
    data_dir = REPO_ROOT / 'Phase_1_Machine_Learning' / 'data' / 'GeneratedOutput' / '02.Preprocessing&FeatureSelection'
    configs_dir = REPO_ROOT / 'Phase_2_Deep_Learning' / 'configs'
    artifacts_dir = REPO_ROOT / 'Phase_2_Deep_Learning' / 'artifacts'

    cfg = load_feature_config(configs_dir / 'feature_config.json')
    scaler, encoder = load_transformers(artifacts_dir)
    num_cols = cfg['numerical_features']
    cat_cols = cfg['categorical_features']
    target_col = cfg['target_variable']

    print("\nLoading train and validation splits...")
    train_df = pd.read_parquet(data_dir / 'train.parquet')
    val_df = pd.read_parquet(data_dir / 'val.parquet')

    # Subsample 50k train, 25k val for fast controlled regularization sweeping
    train_sub = train_df.sample(n=50000, random_state=42)
    val_sub = val_df.sample(n=25000, random_state=42)

    _, _, x_train_all, y_train = prepare_tensors(train_sub, scaler, encoder, num_cols, cat_cols, target_col)
    _, _, x_val_all, y_val = prepare_tensors(val_sub, scaler, encoder, num_cols, cat_cols, target_col)

    train_ds = TensorDataset(x_train_all, y_train)
    val_ds = TensorDataset(x_val_all, y_val)

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    results_dir = REPO_ROOT / 'Phase_2_Deep_Learning' / 'reports' / 'evaluation_results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Experiment 1: Dropout Sweep ───────────────────────────────────────────
    print("\n" + "-" * 75)
    print("  Experiment 1: Dropout Sweep [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]")
    print("-" * 75)

    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    dropout_results = []

    for dr in dropout_rates:
        conf = MLPConfig(
            input_dim=53,
            hidden_dims=[512, 256, 128, 64],
            dropout_rate=dr,
            use_batch_norm=True,
            name=f"MLP_Dropout_{dr:.1f}",
        )
        res = train_and_eval_mlp(conf, train_loader, val_loader, device, weight_decay=1e-4, epochs=3)
        dropout_results.append(res)
        print(f"  Dropout {dr:.1f} → Val MAE: {res['Val_MAE']:.4f} | Val RMSE: {res['Val_RMSE']:.4f} | R²: {res['Val_R2']:.4f} | Gap: {res['Generalization_Gap_MAE']:+.4f}")

    df_dropout = pd.DataFrame(dropout_results)
    df_dropout.to_csv(results_dir / 'dropout_sweep_results.csv', index=False)
    with open(results_dir / 'dropout_sweep_results.json', 'w') as f:
        json.dump(dropout_results, f, indent=2)
    print(f"✓ Saved dropout sweep results → {results_dir / 'dropout_sweep_results.csv'}")

    # ── Experiment 2: Weight Decay Sweep ──────────────────────────────────────
    print("\n" + "-" * 75)
    print("  Experiment 2: Weight Decay Sweep [0.0, 1e-5, 1e-4, 1e-3, 1e-2]")
    print("-" * 75)

    weight_decays = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
    wd_results = []

    for wd in weight_decays:
        conf = MLPConfig(
            input_dim=53,
            hidden_dims=[512, 256, 128, 64],
            dropout_rate=0.3,
            use_batch_norm=True,
            name=f"MLP_WD_{wd:.0e}",
        )
        res = train_and_eval_mlp(conf, train_loader, val_loader, device, weight_decay=wd, epochs=3)
        wd_results.append(res)
        print(f"  Weight Decay {wd:.0e} → Val MAE: {res['Val_MAE']:.4f} | Val RMSE: {res['Val_RMSE']:.4f} | R²: {res['Val_R2']:.4f} | Gap: {res['Generalization_Gap_MAE']:+.4f}")

    df_wd = pd.DataFrame(wd_results)
    df_wd.to_csv(results_dir / 'weight_decay_results.csv', index=False)
    with open(results_dir / 'weight_decay_results.json', 'w') as f:
        json.dump(wd_results, f, indent=2)
    print(f"✓ Saved weight decay results → {results_dir / 'weight_decay_results.csv'}")

    # ── Experiment 3: Monte Carlo Dropout Uncertainty Estimation ──────────────
    print("\n" + "-" * 75)
    print("  Experiment 3: Monte Carlo Dropout Epistemic Uncertainty (Trained Checkpoint)")
    print("-" * 75)
    from Phase_2_Deep_Learning.src.dl_models import load_model_checkpoint, mc_dropout_predictions
    trained_mlp, _ = load_model_checkpoint(REPO_ROOT / 'Phase_2_Deep_Learning' / 'models' / 'mlp_best_checkpoint.pt', device=device)

    sample_x = x_val_all[:1000].to(device)
    sample_y = y_val[:1000].numpy().flatten()
    mc_means, mc_stds = mc_dropout_predictions(trained_mlp, sample_x, n_samples=30, device=device)

    mc_m = compute_regression_metrics(sample_y, mc_means)
    mc_summary = {
        'N_Samples': 1000,
        'N_MC_Passes': 30,
        'MC_Mean_MAE': mc_m['mae'],
        'MC_Mean_RMSE': mc_m['rmse'],
        'MC_Mean_R2': mc_m['r2'],
        'Mean_Epistemic_Uncertainty_Std': float(np.mean(mc_stds)),
        'Median_Epistemic_Uncertainty_Std': float(np.median(mc_stds)),
        'Min_Epistemic_Uncertainty_Std': float(np.min(mc_stds)),
        'Max_Epistemic_Uncertainty_Std': float(np.max(mc_stds)),
    }
    with open(results_dir / 'mc_dropout_uncertainty.json', 'w') as f:
        json.dump(mc_summary, f, indent=2)
    print(f"  MC Dropout Mean MAE: {mc_m['mae']:.4f} | Mean Uncertainty (Std): {mc_summary['Mean_Epistemic_Uncertainty_Std']:.4f}")
    print(f"✓ Saved MC dropout uncertainty summary → {results_dir / 'mc_dropout_uncertainty.json'}")

    print("\n" + "=" * 75)
    print("  REGULARIZATION EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 75)

if __name__ == '__main__':
    run_regularization_sweeps()
