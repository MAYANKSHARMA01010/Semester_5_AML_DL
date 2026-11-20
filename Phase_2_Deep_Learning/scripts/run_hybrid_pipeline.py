"""
Phase 4 Hybrid Pipeline Runner
===============================
Executes the complete Hybrid ML + DL pipeline:
1. Load Phase 1 tree models (RF, XGBoost)
2. Generate out-of-fold predictions on training set
3. Compute and analyze residuals
4. Train ResidualMLP on OOF residuals
5. Evaluate full hybrid ensemble on val + test sets
6. Save all artifacts and metrics

Usage
-----
    cd /path/to/Semester_5_AML_DL
    python Phase_2_Deep_Learning/scripts/run_hybrid_pipeline.py

    # Optional flags:
    python Phase_2_Deep_Learning/scripts/run_hybrid_pipeline.py --n_splits 5 --n_epochs 50 --patience 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Resolve repo root and add to sys.path ─────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = _SCRIPT_DIR.parents[1]   # scripts/ → Phase_2_Deep_Learning/ → repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Imports from project modules ──────────────────────────────────────────────
from Phase_2_Deep_Learning.src.dataset import (
    load_feature_config, load_transformers, prepare_tensors, load_raw_splits
)
from Phase_2_Deep_Learning.src.dl_models import load_model_checkpoint
from Phase_2_Deep_Learning.src.training import get_device, compute_regression_metrics
from Phase_2_Deep_Learning.src.hybrid_models import (
    HybridConfig, ResidualMLPNet, HybridEnsemble, WeightedEnsemble,
    save_hybrid_checkpoint,
)
from Phase_2_Deep_Learning.src.oof_generator import (
    generate_ensemble_oof, compute_residuals, save_oof_artifacts
)
from Phase_2_Deep_Learning.src.residual_trainer import (
    build_residual_dataloaders, train_residual_mlp, evaluate_residual_model
)
from Phase_1_Machine_Learning.src.models import load_tree_model


# ══════════════════════════════════════════════════════════════════════════════
# Path Constants
# ══════════════════════════════════════════════════════════════════════════════

PHASE2_DIR    = REPO_ROOT / 'Phase_2_Deep_Learning'
ARTIFACTS_DIR = PHASE2_DIR / 'artifacts'
CONFIGS_DIR   = PHASE2_DIR / 'configs'
MODELS_DIR    = PHASE2_DIR / 'models'
DATA_DIR      = REPO_ROOT / 'data' / 'GeneratedOutput' / '02.Preprocessing&FeatureSelection'
MODEL_TRAIN_DIR = REPO_ROOT / 'data' / 'GeneratedOutput' / '03.ModelTraining'
OUTPUT_DIR    = REPO_ROOT / 'data' / 'GeneratedOutput' / '04.HybridModel'

RF_MODEL_PATH  = MODEL_TRAIN_DIR / 'random_forest_model.joblib'
XGB_MODEL_PATH = MODEL_TRAIN_DIR / 'xgboost_model.json'
MLP_CKPT_PATH  = MODELS_DIR / 'mlp_best_checkpoint.pt'


# ══════════════════════════════════════════════════════════════════════════════
# Banner Helper
# ══════════════════════════════════════════════════════════════════════════════

def _banner(title: str, width: int = 70) -> None:
    print('\n' + '═' * width)
    print(f'  {title}')
    print('═' * width)


def _section(title: str, width: int = 60) -> None:
    print(f'\n{"─"*width}')
    print(f'  {title}')
    print(f'{"─"*width}')


# ══════════════════════════════════════════════════════════════════════════════
# Metrics Printer
# ══════════════════════════════════════════════════════════════════════════════

def _print_metrics_table(results: list[dict]) -> None:
    """Print a formatted comparison table of all models."""
    header = f"{'Model':<35} {'Split':<6} {'MAE':>8} {'RMSE':>8} {'R²':>8}"
    print('\n' + '='*70)
    print('  COMPREHENSIVE MODEL COMPARISON')
    print('='*70)
    print(header)
    print('-'*70)
    for r in results:
        print(
            f"  {r['model']:<33} {r['split']:<6} "
            f"{r['mae']:>8.4f} {r['rmse']:>8.4f} {r['r2']:>8.4f}"
        )
    print('='*70)


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    n_splits : int = 5,
    n_epochs : int = 50,
    patience : int = 10,
    lr       : float = 1e-3,
    batch_size: int = 1024,
) -> None:
    """Execute the full Phase 4 Hybrid pipeline end-to-end."""
    t_start = time.time()

    _banner('Phase 4 — Hybrid ML + DL Pipeline Runner')
    print(f'  REPO ROOT : {REPO_ROOT}')
    print(f'  Device    : {get_device()}')
    print(f'  OOF splits: {n_splits}  |  Epochs: {n_epochs}  |  Patience: {patience}')

    # ── 0. Create output directories ──────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load tree models ───────────────────────────────────────────────────
    _section('Step 1 — Load Phase 1 Tree Models')
    rf_model  = load_tree_model(str(RF_MODEL_PATH),  model_type='rf')
    xgb_model = load_tree_model(str(XGB_MODEL_PATH), model_type='xgb')
    print(f'  ✓ RF  model : {type(rf_model).__name__}')
    print(f'  ✓ XGB model : {type(xgb_model).__name__}')

    # ── Load feature config + transformers ────────────────────────────────────
    feature_cfg = load_feature_config(CONFIGS_DIR / 'feature_config.json')
    scaler, encoder = load_transformers(ARTIFACTS_DIR)
    NUM_FEATURES = feature_cfg['numerical_features']
    CAT_FEATURES = feature_cfg['categorical_features']
    TARGET_COL   = feature_cfg['target_variable']
    print(f'  Features    : {len(NUM_FEATURES)} numerical + {len(CAT_FEATURES)} categorical = 53 total')

    # ── 2. Load train/val/test data ───────────────────────────────────────────
    _section('Step 2 — Load Data Splits')
    train_df, val_df, test_df = load_raw_splits(DATA_DIR)
    print(f'  Train : {len(train_df):,} rows')
    print(f'  Val   : {len(val_df):,} rows')
    print(f'  Test  : {len(test_df):,} rows')

    # Prepare tensors
    x_train_num, x_train_cat, x_train_all, y_train_t = prepare_tensors(
        train_df, scaler, encoder, NUM_FEATURES, CAT_FEATURES, TARGET_COL
    )
    x_val_num, x_val_cat, x_val_all, y_val_t = prepare_tensors(
        val_df, scaler, encoder, NUM_FEATURES, CAT_FEATURES, TARGET_COL
    )
    x_test_num, x_test_cat, x_test_all, y_test_t = prepare_tensors(
        test_df, scaler, encoder, NUM_FEATURES, CAT_FEATURES, TARGET_COL
    )

    X_train_np = x_train_all.numpy()
    y_train_np = y_train_t.numpy().flatten()
    X_val_np   = x_val_all.numpy()
    y_val_np   = y_val_t.numpy().flatten()
    X_test_np  = x_test_all.numpy()
    y_test_np  = y_test_t.numpy().flatten()

    # ── 3. Generate OOF predictions ───────────────────────────────────────────
    _section('Step 3 — Generate OOF Predictions (5-fold KFold)')
    oof_rf, oof_xgb, fold_metrics_dict = generate_ensemble_oof(
        X_train_np, y_train_np, n_splits=n_splits, random_state=42
    )

    # ── 4. Compute residuals ──────────────────────────────────────────────────
    _section('Step 4 — Compute OOF Residuals')
    base_train_pred   = (oof_rf + oof_xgb) / 2.0
    residuals_avg, _  = compute_residuals(y_train_np, base_train_pred)
    residuals_rf,  rs_rf  = compute_residuals(y_train_np, oof_rf)
    residuals_xgb, rs_xgb = compute_residuals(y_train_np, oof_xgb)

    print(f'  RF  residuals — mean: {rs_rf["mean"]:+.4f}  std: {rs_rf["std"]:.4f}  pct_within_5: {rs_rf["pct_within_5"]:.1f}%')
    print(f'  XGB residuals — mean: {rs_xgb["mean"]:+.4f}  std: {rs_xgb["std"]:.4f}  pct_within_5: {rs_xgb["pct_within_5"]:.1f}%')

    save_oof_artifacts(oof_rf, oof_xgb, residuals_rf, residuals_xgb, y_train_np, OUTPUT_DIR)

    # Validation residuals
    rf_val_pred   = rf_model.predict(X_val_np)
    xgb_val_pred  = xgb_model.predict(X_val_np)
    base_val_pred = (rf_val_pred + xgb_val_pred) / 2.0
    residuals_val, _ = compute_residuals(y_val_np, base_val_pred)

    # ── 5. Train ResidualMLP ──────────────────────────────────────────────────
    _section('Step 5 — Train ResidualMLPNet')

    DEVICE = get_device()
    hybrid_cfg   = HybridConfig()
    residual_net = ResidualMLPNet(hybrid_cfg).to(DEVICE)
    print(f'  {residual_net.architecture_summary()}')

    base_train_t = torch.from_numpy(base_train_pred.astype(np.float32))
    res_train_t  = torch.from_numpy(residuals_avg.astype(np.float32))
    base_val_t   = torch.from_numpy(base_val_pred.astype(np.float32))
    res_val_t    = torch.from_numpy(residuals_val.astype(np.float32))

    train_loader, val_loader = build_residual_dataloaders(
        x_train_all, base_train_t, res_train_t,
        x_val_all,   base_val_t,   res_val_t,
        batch_size=batch_size,
    )

    RESIDUAL_CKPT = MODELS_DIR / 'residual_mlp_checkpoint.pt'
    history = train_residual_mlp(
        model        = residual_net,
        train_loader = train_loader,
        val_loader   = val_loader,
        n_epochs     = n_epochs,
        lr           = lr,
        patience     = patience,
        device       = DEVICE,
        save_path    = RESIDUAL_CKPT,
    )
    print(f"\n  Best Val MAE: {history['best_val_mae']:.4f} @ epoch {history['best_epoch']}")

    # ── 6. Compute final predictions ──────────────────────────────────────────
    _section('Step 6 — Compute Final Hybrid Predictions')

    # Load base MLP if available
    mlp_preds_val  = np.zeros(len(y_val_np))
    mlp_preds_test = np.zeros(len(y_test_np))
    if MLP_CKPT_PATH.exists():
        base_mlp, _ = load_model_checkpoint(MLP_CKPT_PATH, device=DEVICE)
        base_mlp.eval()
        with torch.no_grad():
            mlp_preds_val  = base_mlp(x_val_all.to(DEVICE)).cpu().numpy().flatten()
            mlp_preds_test = base_mlp(x_test_all.to(DEVICE)).cpu().numpy().flatten()
        print('  ✓ Base MLP loaded for weighted ensemble')
    else:
        print(f'  ⚠ MLP checkpoint not found at {MLP_CKPT_PATH}; MLP weight set to 0')
        hybrid_cfg.mlp_weight  = 0.0
        hybrid_cfg.rf_weight   = 0.5
        hybrid_cfg.xgb_weight  = 0.5

    # RF + XGB base predictions
    rf_val_full  = rf_model.predict(X_val_np)
    xgb_val_full = xgb_model.predict(X_val_np)
    rf_test      = rf_model.predict(X_test_np)
    xgb_test     = xgb_model.predict(X_test_np)

    # Weighted Ensemble
    weighted_ens  = WeightedEnsemble(
        rf_weight  = hybrid_cfg.rf_weight,
        xgb_weight = hybrid_cfg.xgb_weight,
        mlp_weight = hybrid_cfg.mlp_weight,
    )
    we_val  = weighted_ens.predict(rf_val_full,  xgb_val_full,  mlp_preds_val)
    we_test = weighted_ens.predict(rf_test,       xgb_test,       mlp_preds_test)

    # Hybrid (RF+XGB+ResidualMLP)
    hybrid_model = HybridEnsemble(hybrid_cfg).to(DEVICE)
    hybrid_model.residual_net.load_state_dict(residual_net.state_dict())
    hybrid_model.eval()

    base_val_avg  = (rf_val_full + xgb_val_full) / 2.0
    base_test_avg = (rf_test     + xgb_test)     / 2.0

    hybrid_val  = hybrid_model.predict_numpy(X_val_np,  base_val_avg,  device=DEVICE)
    hybrid_test = hybrid_model.predict_numpy(X_test_np, base_test_avg, device=DEVICE)

    print('  ✓ All predictions computed')

    # ── 7. Metrics comparison ─────────────────────────────────────────────────
    _section('Step 7 — Comprehensive Metrics Comparison')

    all_results = []
    model_preds = {
        'RF'                            : (rf_val_full,  rf_test),
        'XGBoost'                       : (xgb_val_full, xgb_test),
        'MLP (base)'                    : (mlp_preds_val, mlp_preds_test),
        'Weighted Ensemble'             : (we_val,        we_test),
        'Hybrid (RF+XGB+ResidualMLP)'   : (hybrid_val,    hybrid_test),
    }

    for model_name, (val_preds, test_preds) in model_preds.items():
        for split, y_true, preds in [('Val', y_val_np, val_preds), ('Test', y_test_np, test_preds)]:
            m = compute_regression_metrics(y_true, preds)
            all_results.append({
                'model': model_name, 'split': split,
                'mae': m['mae'], 'rmse': m['rmse'], 'r2': m['r2'],
            })

    _print_metrics_table(all_results)

    # ── 8. Save artifacts ─────────────────────────────────────────────────────
    _section('Step 8 — Save Artifacts')

    # Save metrics CSV
    metrics_df = pd.DataFrame(all_results)
    metrics_path = OUTPUT_DIR / 'hybrid_metrics_comparison.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f'  ✓ Metrics CSV       → {metrics_path}')

    # Save test predictions parquet
    test_preds_df = pd.DataFrame({
        'y_true'              : y_test_np,
        'rf_pred'             : rf_test,
        'xgb_pred'            : xgb_test,
        'mlp_pred'            : mlp_preds_test,
        'weighted_ensemble'   : we_test,
        'hybrid_residual_mlp' : hybrid_test,
    })
    preds_path = OUTPUT_DIR / 'hybrid_test_predictions.parquet'
    test_preds_df.to_parquet(preds_path, index=False)
    print(f'  ✓ Test predictions  → {preds_path}  ({len(test_preds_df):,} rows)')

    # Save hybrid config JSON
    cfg_path = OUTPUT_DIR / 'hybrid_config.json'
    with open(cfg_path, 'w') as f:
        json.dump(hybrid_cfg.to_dict(), f, indent=2)
    print(f'  ✓ Hybrid config     → {cfg_path}')

    # Save weighted ensemble weights JSON
    we_path = OUTPUT_DIR / 'weighted_ensemble_weights.json'
    with open(we_path, 'w') as f:
        json.dump(weighted_ens.to_dict(), f, indent=2)
    print(f'  ✓ Ensemble weights  → {we_path}')

    # Save full hybrid checkpoint
    test_m = compute_regression_metrics(y_test_np, hybrid_test)
    save_hybrid_checkpoint(
        model    = hybrid_model,
        config   = hybrid_cfg,
        path     = MODELS_DIR / 'hybrid_ensemble_checkpoint.pt',
        metadata = {
            'best_val_mae': float(history['best_val_mae']),
            'best_epoch'  : history['best_epoch'],
            'test_mae'    : test_m['mae'],
            'test_rmse'   : test_m['rmse'],
            'test_r2'     : test_m['r2'],
        },
    )

    # Save OOF fold metrics JSON
    oof_meta_path = OUTPUT_DIR / 'oof_fold_metrics.json'
    with open(oof_meta_path, 'w') as f:
        json.dump(fold_metrics_dict, f, indent=2)
    print(f'  ✓ OOF fold metrics  → {oof_meta_path}')

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    best_test = metrics_df[metrics_df['split'] == 'Test'].loc[
        metrics_df[metrics_df['split'] == 'Test']['mae'].idxmin()
    ]

    _banner('Phase 4 Complete ✓')
    print(f'  Total runtime : {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'  Best model    : {best_test["model"]}')
    print(f'    Test MAE    : {best_test["mae"]:.4f}')
    print(f'    Test RMSE   : {best_test["rmse"]:.4f}')
    print(f'    Test R²     : {best_test["r2"]:.4f}')
    print(f'\n  Artifacts in  : {OUTPUT_DIR}')
    print('═' * 70)


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Phase 4 Hybrid ML + DL Pipeline Runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--n_splits',   type=int,   default=5,    help='KFold splits for OOF generation')
    parser.add_argument('--n_epochs',   type=int,   default=50,   help='Max training epochs for ResidualMLP')
    parser.add_argument('--patience',   type=int,   default=10,   help='EarlyStopping patience (epochs)')
    parser.add_argument('--lr',         type=float, default=1e-3, help='AdamW initial learning rate')
    parser.add_argument('--batch_size', type=int,   default=1024, help='Mini-batch size for DataLoaders')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run_pipeline(
        n_splits  = args.n_splits,
        n_epochs  = args.n_epochs,
        patience  = args.patience,
        lr        = args.lr,
        batch_size= args.batch_size,
    )
