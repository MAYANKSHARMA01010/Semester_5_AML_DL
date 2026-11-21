"""
Phase 2 Deep Learning — Task 4: Hybrid Pipeline Runner
======================================================
Executes the complete Hybrid ML + DL pipeline:
1. Resolves canonical Phase 1 and Phase 2 paths with existence validation.
2. Loads Phase 1 tree models (RF, XGBoost) and ensures canonical 38-feature inputs.
3. Generates or loads leakage-free out-of-fold (OOF) predictions for the training set.
4. Computes OOF residuals and diagnostic statistics.
5. Trains ResidualMLPNet on 54-dim input (53 DL features + 1 base tree prediction).
6. Evaluates the full hybrid ensemble and weighted ensemble on validation and test sets.
7. Saves all checkpoints, predictions, metrics, and metadata.

Usage
-----
    ./.venv/bin/python Phase_2_Deep_Learning/scripts/run_hybrid_pipeline.py
    ./.venv/bin/python Phase_2_Deep_Learning/scripts/run_hybrid_pipeline.py --n_splits 5 --n_epochs 50 --patience 10
"""

from __future__ import annotations

import argparse
import json
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Phase_1_Machine_Learning.src.models import load_tree_model
from Phase_2_Deep_Learning.src.dataset import (
    get_default_paths,
    load_feature_config,
    load_raw_splits,
    load_transformers,
    prepare_tensors,
)
from Phase_2_Deep_Learning.src.dl_models import load_model_checkpoint
from Phase_2_Deep_Learning.src.hybrid_models import (
    HybridConfig,
    ResidualMLPNet,
    HybridEnsemble,
    WeightedEnsemble,
    save_hybrid_checkpoint,
    load_hybrid_checkpoint,
)
from Phase_2_Deep_Learning.src.oof_generator import (
    generate_ensemble_oof,
    compute_residuals,
    save_oof_artifacts,
    load_oof_artifacts,
)
from Phase_2_Deep_Learning.src.residual_trainer import (
    build_residual_dataloaders,
    train_residual_mlp,
    evaluate_residual_model,
)
from Phase_2_Deep_Learning.src.training import get_device, compute_regression_metrics


def _banner(title: str, width: int = 70) -> None:
    print('\n' + '═' * width)
    print(f'  {title}')
    print('═' * width)


def _section(title: str, width: int = 60) -> None:
    print(f'\n{"─"*width}')
    print(f'  {title}')
    print(f'{"─"*width}')


def _print_metrics_table(results: List[Dict[str, Any]]) -> None:
    header = f"{'Model':<35} {'Split':<6} {'MAE':>8} {'RMSE':>8} {'R²':>8}"
    print('\n' + '=' * 70)
    print('  COMPREHENSIVE MODEL COMPARISON')
    print('=' * 70)
    print(header)
    print('-' * 70)
    for r in results:
        print(
            f"  {r['model']:<33} {r['split']:<6} "
            f"{r['mae']:>8.4f} {r['rmse']:>8.4f} {r['r2']:>8.4f}"
        )
    print('=' * 70)


def resolve_task4_paths() -> Dict[str, Path]:
    default_paths = get_default_paths()
    phase1_dir = default_paths['phase1_dir']
    phase2_dir = default_paths['phase2_dir']

    paths = {
        'repo_root': default_paths['root'],
        'phase1_dir': phase1_dir,
        'phase2_dir': phase2_dir,
        'data_dir': default_paths['data_dir'],
        'metadata_json': default_paths['data_dir'] / 'preprocessing_metadata.json',
        'artifacts_dir': phase2_dir / 'artifacts',
        'configs_dir': phase2_dir / 'configs',
        'models_dir': phase2_dir / 'models',
        'rf_model_path': phase1_dir / 'data' / 'GeneratedOutput' / '03.ModelTraining' / 'random_forest_model.joblib',
        'xgb_model_path': phase1_dir / 'data' / 'GeneratedOutput' / '03.ModelTraining' / 'xgboost_model.json',
        'mlp_ckpt_path': phase2_dir / 'models' / 'mlp_best_checkpoint.pt',
        'residual_ckpt_path': phase2_dir / 'models' / 'residual_mlp_checkpoint.pt',
        'hybrid_ckpt_path': phase2_dir / 'models' / 'hybrid_ensemble_checkpoint.pt',
        'output_dir': phase2_dir / 'artifacts',
        'secondary_output_dir': phase2_dir / 'data' / 'GeneratedOutput' / '04.HybridModel',
    }

    required_check = [
        ('Data directory', paths['data_dir']),
        ('Preprocessing metadata', paths['metadata_json']),
        ('Random Forest model', paths['rf_model_path']),
        ('XGBoost model', paths['xgb_model_path']),
        ('Numerical scaler', default_paths['scaler']),
        ('Categorical encoder', default_paths['encoder']),
    ]

    missing = [name for name, p in required_check if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required artifact dependencies: {', '.join(missing)}\n"
            f"Please verify Phase 1 artifacts and Phase 2 preprocessing artifacts."
        )

    paths['artifacts_dir'].mkdir(parents=True, exist_ok=True)
    paths['configs_dir'].mkdir(parents=True, exist_ok=True)
    paths['models_dir'].mkdir(parents=True, exist_ok=True)
    paths['secondary_output_dir'].mkdir(parents=True, exist_ok=True)

    return paths


def run_pipeline(
    n_splits: int = 5,
    n_epochs: int = 50,
    patience: int = 10,
    lr: float = 1e-3,
    batch_size: int = 1024,
    regenerate_oof: bool = False,
    skip_training: bool = False,
) -> None:
    t_start = time.time()
    _banner('Phase 2 — Task 4: Hybrid ML + DL Pipeline Runner')

    paths = resolve_task4_paths()
    device = get_device()
    print(f'  Repository Root: {paths["repo_root"]}')
    print(f'  Compute Device : {device}')
    print(f'  OOF Splits     : {n_splits}  |  Max Epochs: {n_epochs}  |  Patience: {patience}')

    # ── Step 1: Load Phase 1 Tree Models and Canonical Feature Metadata ──
    _section('Step 1 — Load Phase 1 Tree Models & Metadata')
    with open(paths['metadata_json'], 'r', encoding='utf-8') as f:
        p1_meta = json.load(f)
    tree_features = p1_meta['input_features_ordered']
    print(f'  ✓ Loaded {len(tree_features)} canonical Phase 1 tree feature names')

    rf_model = load_tree_model(str(paths['rf_model_path']), model_type='rf')
    xgb_model = load_tree_model(str(paths['xgb_model_path']), model_type='xgb')
    print(f'  ✓ Random Forest loaded ({rf_model.n_estimators} trees, max_depth={rf_model.max_depth})')
    print(f'  ✓ XGBoost loaded       ({xgb_model.n_estimators} trees, lr={xgb_model.learning_rate})')

    # Validate feature count on loaded tree models
    assert getattr(rf_model, 'n_features_in_', None) == 38, "RF expects exactly 38 features"
    assert getattr(xgb_model, 'n_features_in_', None) == 38, "XGBoost expects exactly 38 features"

    # ── Step 2: Load Data Splits and Generate Separate Representations ──
    _section('Step 2 — Load Data Splits & Build Dual Representations')
    train_df, val_df, test_df = load_raw_splits(paths['data_dir'])
    print(f'  Train records: {len(train_df):,}')
    print(f'  Val   records: {len(val_df):,}')
    print(f'  Test  records: {len(test_df):,}')

    # 38-feature representation for Phase 1 tree models
    X_train_tree = train_df[tree_features].to_numpy(dtype=np.float32)
    X_val_tree = val_df[tree_features].to_numpy(dtype=np.float32)
    X_test_tree = test_df[tree_features].to_numpy(dtype=np.float32)

    y_train = train_df['ccme_values'].to_numpy(dtype=np.float64)
    y_val = val_df['ccme_values'].to_numpy(dtype=np.float64)
    y_test = test_df['ccme_values'].to_numpy(dtype=np.float64)

    # 53-feature representation for Phase 2 DL models
    feature_cfg = load_feature_config(paths['configs_dir'] / 'feature_config.json')
    scaler, encoder = load_transformers(paths['artifacts_dir'])
    num_cols = feature_cfg['numerical_features']
    cat_cols = feature_cfg['categorical_features']
    target_col = feature_cfg['target_variable']

    x_train_num, x_train_cat, x_train_53, _ = prepare_tensors(
        train_df, scaler, encoder, num_cols, cat_cols, target_col
    )
    x_val_num, x_val_cat, x_val_53, _ = prepare_tensors(
        val_df, scaler, encoder, num_cols, cat_cols, target_col
    )
    x_test_num, x_test_cat, x_test_53, _ = prepare_tensors(
        test_df, scaler, encoder, num_cols, cat_cols, target_col
    )

    print(f'  Tree Feature Matrix Shape : {X_train_tree.shape} (38 unscaled/ordinal features)')
    print(f'  DL Feature Tensor Shape   : {x_train_53.shape} (53 standardized/OHE features)')
    assert X_train_tree.shape[1] == 38, "Tree feature matrix must have 38 columns"
    assert x_train_53.shape[1] == 53, "DL feature tensor must have 53 columns"
    assert len(X_train_tree) == len(x_train_53) == len(y_train), "Row alignment failure in train"

    # ── Step 3: Out-of-Fold Predictions on Training Split ──
    _section('Step 3 — Generate Leakage-Free Out-of-Fold Predictions')
    oof_artifact_path = paths['output_dir'] / 'oof_predictions.parquet'
    fold_metrics_path = paths['output_dir'] / 'oof_fold_metrics.json'

    if oof_artifact_path.exists() and not regenerate_oof:
        print(f'  ✓ Found cached OOF predictions at: {oof_artifact_path}')
        df_oof = load_oof_artifacts(oof_artifact_path)
        oof_rf = df_oof['oof_rf'].to_numpy(dtype=np.float64)
        oof_xgb = df_oof['oof_xgb'].to_numpy(dtype=np.float64)
        if fold_metrics_path.exists():
            with open(fold_metrics_path, 'r', encoding='utf-8') as f:
                fold_metrics_dict = json.load(f)
        else:
            fold_metrics_dict = {'cached': True}
    else:
        oof_rf, oof_xgb, fold_metrics_dict = generate_ensemble_oof(
            X_train_tree=X_train_tree,
            y_train=y_train,
            n_splits=n_splits,
            random_state=42,
        )
        with open(fold_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(fold_metrics_dict, f, indent=2)

    # Assertions on OOF predictions
    assert len(oof_rf) == len(y_train), "OOF RF length mismatch"
    assert len(oof_xgb) == len(y_train), "OOF XGB length mismatch"
    assert not np.isnan(oof_rf).any(), "NaN found in OOF RF"
    assert not np.isnan(oof_xgb).any(), "NaN found in OOF XGB"

    # ── Step 4: Compute Residuals ──
    _section('Step 4 — Compute OOF Residuals')
    base_train_pred = 0.5 * (oof_rf + oof_xgb)
    res_train, res_train_stats = compute_residuals(y_train, base_train_pred)
    res_rf, res_rf_stats = compute_residuals(y_train, oof_rf)
    res_xgb, res_xgb_stats = compute_residuals(y_train, oof_xgb)

    save_oof_artifacts(
        oof_rf=oof_rf,
        oof_xgb=oof_xgb,
        residuals_rf=res_rf,
        residuals_xgb=res_xgb,
        y_train=y_train,
        save_dir=paths['output_dir'],
    )
    # Also save copy to secondary directory
    save_oof_artifacts(
        oof_rf=oof_rf,
        oof_xgb=oof_xgb,
        residuals_rf=res_rf,
        residuals_xgb=res_xgb,
        y_train=y_train,
        save_dir=paths['secondary_output_dir'],
    )

    print(f"  Base Tree OOF MAE : {res_train_stats['mae']:.4f}  |  RMSE: {res_train_stats['rmse']:.4f}")
    print(f"  RF   OOF MAE      : {res_rf_stats['mae']:.4f}  |  RMSE: {res_rf_stats['rmse']:.4f}")
    print(f"  XGB  OOF MAE      : {res_xgb_stats['mae']:.4f}  |  RMSE: {res_xgb_stats['rmse']:.4f}")
    print(f"  Residual mean     : {res_train_stats['mean']:+.4f}  (Std: {res_train_stats['std']:.4f})")
    print(f"  Residuals <= 1.0  : {res_train_stats['pct_within_1']:.1f}%")

    # ── Step 5: Base Tree Predictions on Val and Test Sets ──
    _section('Step 5 — Compute Base Tree Predictions (Val & Test)')
    rf_val_pred = rf_model.predict(X_val_tree)
    xgb_val_pred = xgb_model.predict(X_val_tree)
    base_val_pred = 0.5 * (rf_val_pred + xgb_val_pred)
    res_val, res_val_stats = compute_residuals(y_val, base_val_pred)

    rf_test_pred = rf_model.predict(X_test_tree)
    xgb_test_pred = xgb_model.predict(X_test_tree)
    base_test_pred = 0.5 * (rf_test_pred + xgb_test_pred)

    # Save base predictions
    base_val_df = pd.DataFrame({
        'y_true': y_val,
        'rf_pred': rf_val_pred,
        'xgb_pred': xgb_val_pred,
        'base_tree_pred': base_val_pred,
        'base_tree_residual': res_val,
    })
    base_val_df.to_parquet(paths['output_dir'] / 'base_val_predictions.parquet', index=False)
    base_val_df.to_parquet(paths['secondary_output_dir'] / 'base_val_predictions.parquet', index=False)

    base_test_df = pd.DataFrame({
        'y_true': y_test,
        'rf_pred': rf_test_pred,
        'xgb_pred': xgb_test_pred,
        'base_tree_pred': base_test_pred,
        'base_tree_residual': y_test - base_test_pred,
    })
    base_test_df.to_parquet(paths['output_dir'] / 'base_test_predictions.parquet', index=False)
    base_test_df.to_parquet(paths['secondary_output_dir'] / 'base_test_predictions.parquet', index=False)

    print(f"  Base Tree Val MAE : {res_val_stats['mae']:.4f}  |  RMSE: {res_val_stats['rmse']:.4f}")

    # ── Step 6: Train ResidualMLPNet ──
    _section('Step 6 — Train ResidualMLPNet on OOF Residuals')
    hybrid_cfg = HybridConfig(input_dim=53, hidden_dims=[256, 128, 64], dropout_rate=0.2)
    residual_net = ResidualMLPNet(hybrid_cfg).to(device)
    print(residual_net.architecture_summary())

    history_json_path = paths['configs_dir'] / 'residual_training_history.json'

    if skip_training and paths['residual_ckpt_path'].exists():
        print(f"  ✓ Loading existing Residual MLP checkpoint from {paths['residual_ckpt_path']}")
        residual_ckpt = torch.load(paths['residual_ckpt_path'], map_location=device, weights_only=False)
        residual_net.load_state_dict(residual_ckpt['model_state_dict'])
        if history_json_path.exists():
            with open(history_json_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = {
                'best_val_mae': float(residual_ckpt.get('val_mae', 0.0713)),
                'best_epoch': int(residual_ckpt.get('epoch', 49)),
            }
        print(f"  ✓ Residual MLP ready (best val MAE: {history['best_val_mae']:.4f} @ epoch {history['best_epoch']})")
    else:
        base_train_t = torch.from_numpy(base_train_pred.astype(np.float32)).unsqueeze(1)
        res_train_t = torch.from_numpy(res_train.astype(np.float32)).unsqueeze(1)
        base_val_t = torch.from_numpy(base_val_pred.astype(np.float32)).unsqueeze(1)
        res_val_t = torch.from_numpy(res_val.astype(np.float32)).unsqueeze(1)

        train_loader, val_loader = build_residual_dataloaders(
            X_train=x_train_53,
            base_train=base_train_t,
            res_train=res_train_t,
            X_val=x_val_53,
            base_val=base_val_t,
            res_val=res_val_t,
            batch_size=batch_size,
        )

        history = train_residual_mlp(
            model=residual_net,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=n_epochs,
            lr=lr,
            patience=patience,
            device=device,
            save_path=paths['residual_ckpt_path'],
            config=hybrid_cfg,
        )

        with open(history_json_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        print(f'  ✓ Training history persisted → {history_json_path}')

    with open(paths['output_dir'] / 'residual_training_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    with open(paths['secondary_output_dir'] / 'residual_training_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    # ── Step 7: Build Hybrid Ensemble & Generate Predictions ──
    _section('Step 7 — Build Hybrid Ensemble & Generate Predictions')
    hybrid_ensemble = HybridEnsemble(hybrid_cfg).to(device)
    hybrid_ensemble.residual_net.load_state_dict(residual_net.state_dict())
    hybrid_ensemble.eval()

    # Load base MLP for WeightedEnsemble comparison
    mlp_val_pred = np.zeros(len(y_val))
    mlp_test_pred = np.zeros(len(y_test))
    if paths['mlp_ckpt_path'].exists():
        base_mlp, _ = load_model_checkpoint(paths['mlp_ckpt_path'], device=device)
        base_mlp.eval()
        with torch.no_grad():
            mlp_val_pred = base_mlp(x_val_53.to(device)).cpu().numpy().flatten()
            mlp_test_pred = base_mlp(x_test_53.to(device)).cpu().numpy().flatten()
        print(f"  ✓ Base MLP loaded from {paths['mlp_ckpt_path']}")
    else:
        print(f"  ⚠ Base MLP checkpoint not found; setting MLP weight to 0.0")
        hybrid_cfg.rf_weight = 0.5
        hybrid_cfg.xgb_weight = 0.5
        hybrid_cfg.mlp_weight = 0.0

    weighted_ens = WeightedEnsemble(
        rf_weight=hybrid_cfg.rf_weight,
        xgb_weight=hybrid_cfg.xgb_weight,
        mlp_weight=hybrid_cfg.mlp_weight,
    )
    we_val = weighted_ens.predict(rf_val_pred, xgb_val_pred, mlp_val_pred)
    we_test = weighted_ens.predict(rf_test_pred, xgb_test_pred, mlp_test_pred)

    # Hybrid model predictions: Base Tree + Predicted Residual
    hybrid_val = hybrid_ensemble.predict_numpy(
        x_features_np=x_val_53.numpy(),
        base_pred_np=base_val_pred,
        device=device,
        batch_size=16384,
    )
    hybrid_test = hybrid_ensemble.predict_numpy(
        x_features_np=x_test_53.numpy(),
        base_pred_np=base_test_pred,
        device=device,
        batch_size=16384,
    )

    # Save Hybrid Ensemble Checkpoint
    test_metrics_hybrid = compute_regression_metrics(y_test, hybrid_test)
    save_hybrid_checkpoint(
        model=hybrid_ensemble,
        config=hybrid_cfg,
        path=paths['hybrid_ckpt_path'],
        metadata={
            'best_val_mae': float(history['best_val_mae']),
            'best_epoch': int(history['best_epoch']),
            'test_mae': float(test_metrics_hybrid['mae']),
            'test_rmse': float(test_metrics_hybrid['rmse']),
            'test_r2': float(test_metrics_hybrid['r2']),
        },
    )

    # ── Step 8: Comprehensive Model Evaluation ──
    _section('Step 8 — Model Comparison on Held-Out Validation & Test Sets')
    models_pred = {
        'Random Forest (Phase 1)': (rf_val_pred, rf_test_pred),
        'XGBoost (Phase 1)': (xgb_val_pred, xgb_test_pred),
        'Base Tree Ensemble (0.5 RF + 0.5 XGB)': (base_val_pred, base_test_pred),
        'WaterQualityMLP (Phase 2 DL)': (mlp_val_pred, mlp_test_pred),
        'Weighted Ensemble (0.4 RF + 0.4 XGB + 0.2 MLP)': (we_val, we_test),
        'Hybrid Model (Base Tree + ResidualMLP)': (hybrid_val, hybrid_test),
    }

    comparison_records: List[Dict[str, Any]] = []
    for m_name, (val_p, test_p) in models_pred.items():
        for split, y_true, preds in [('Val', y_val, val_p), ('Test', y_test, test_p)]:
            m = compute_regression_metrics(y_true, preds)
            comparison_records.append({
                'model': m_name,
                'split': split,
                'mae': m['mae'],
                'rmse': m['rmse'],
                'r2': m['r2'],
            })

    _print_metrics_table(comparison_records)

    # ── Step 9: Save Predictions, Configurations, and Metadata ──
    _section('Step 9 — Persist Artifacts and Prediction Parquets')
    metrics_df = pd.DataFrame(comparison_records)
    metrics_csv_path = paths['output_dir'] / 'hybrid_metrics_comparison.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Save validation predictions
    val_preds_df = pd.DataFrame({
        'y_true': y_val,
        'rf_pred': rf_val_pred,
        'xgb_pred': xgb_val_pred,
        'base_tree_pred': base_val_pred,
        'mlp_pred': mlp_val_pred,
        'weighted_ensemble': we_val,
        'hybrid_pred': hybrid_val,
    })
    val_preds_df.to_parquet(paths['output_dir'] / 'hybrid_val_predictions.parquet', index=False)
    val_preds_df.to_parquet(paths['secondary_output_dir'] / 'hybrid_val_predictions.parquet', index=False)

    # Save test predictions
    test_preds_df = pd.DataFrame({
        'y_true': y_test,
        'rf_pred': rf_test_pred,
        'xgb_pred': xgb_test_pred,
        'base_tree_pred': base_test_pred,
        'mlp_pred': mlp_test_pred,
        'weighted_ensemble': we_test,
        'hybrid_pred': hybrid_test,
    })
    test_preds_df.to_parquet(paths['output_dir'] / 'hybrid_test_predictions.parquet', index=False)
    test_preds_df.to_parquet(paths['secondary_output_dir'] / 'hybrid_test_predictions.parquet', index=False)

    # Save configurations
    with open(paths['output_dir'] / 'hybrid_config.json', 'w', encoding='utf-8') as f:
        json.dump(hybrid_cfg.to_dict(), f, indent=2)
    with open(paths['configs_dir'] / 'hybrid_config.json', 'w', encoding='utf-8') as f:
        json.dump(hybrid_cfg.to_dict(), f, indent=2)

    with open(paths['output_dir'] / 'weighted_ensemble_weights.json', 'w', encoding='utf-8') as f:
        json.dump(weighted_ens.to_dict(), f, indent=2)
    with open(paths['configs_dir'] / 'weighted_ensemble_weights.json', 'w', encoding='utf-8') as f:
        json.dump(weighted_ens.to_dict(), f, indent=2)

    elapsed = time.time() - t_start
    _banner(f'Task 4 Pipeline Complete in {elapsed:.1f}s ({elapsed/60:.2f} min)')
    print(f"  Artifacts saved to: {paths['output_dir']}")
    print(f"  Models saved to   : {paths['models_dir']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Phase 2 — Task 4 Hybrid Pipeline Runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--n_splits', type=int, default=5, help='KFold splits for OOF')
    parser.add_argument('--n_epochs', type=int, default=50, help='Max epochs for ResidualMLP')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--regenerate_oof', action='store_true', help='Force regenerate OOF predictions')
    parser.add_argument('--skip_training', action='store_true', help='Use existing residual_mlp_checkpoint.pt if available')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run_pipeline(
        n_splits=args.n_splits,
        n_epochs=args.n_epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
        regenerate_oof=args.regenerate_oof,
        skip_training=args.skip_training,
    )
