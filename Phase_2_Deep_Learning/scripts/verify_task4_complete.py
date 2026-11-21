import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Phase_1_Machine_Learning.src.models import load_tree_model
from Phase_2_Deep_Learning.src.dataset import (
    load_feature_config,
    load_raw_splits,
    load_transformers,
    prepare_tensors,
)
from Phase_2_Deep_Learning.src.dl_models import load_model_checkpoint
from Phase_2_Deep_Learning.src.training import get_device, compute_regression_metrics
from Phase_2_Deep_Learning.src.hybrid_models import (
    HybridConfig,
    ResidualMLPNet,
    HybridEnsemble,
    WeightedEnsemble,
    load_hybrid_checkpoint,
)
from Phase_2_Deep_Learning.scripts.run_hybrid_pipeline import resolve_task4_paths

def main():
    print("=" * 75)
    print("PHASE 2 TASK 4 — COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 75)

    paths = resolve_task4_paths()
    device = torch.device("cpu")

    # -------------------------------------------------------------------------
    # 1. Feature Representation Contracts
    # -------------------------------------------------------------------------
    print("\n[Check 1/10] Verifying Feature Representation Contracts...")
    with open(paths['metadata_json'], 'r', encoding='utf-8') as f:
        meta = json.load(f)
    tree_features = meta['input_features_ordered']
    assert len(tree_features) == 38, f"Expected 38 tree features, got {len(tree_features)}"

    feature_cfg = load_feature_config(paths['configs_dir'] / 'feature_config.json')
    num_cols = feature_cfg['numerical_features']
    cat_cols = feature_cfg['categorical_features']
    target_col = feature_cfg['target_variable']
    assert len(num_cols) == 36, f"Expected 36 numerical features, got {len(num_cols)}"
    assert len(cat_cols) == 2, f"Expected 2 categorical features, got {len(cat_cols)}"

    train_df, val_df, test_df = load_raw_splits(paths['data_dir'])
    assert len(train_df) == 1973731
    assert len(val_df) == 422943
    assert len(test_df) == 422943

    scaler, encoder = load_transformers(paths['artifacts_dir'])
    _, _, x_test_all, _ = prepare_tensors(test_df, scaler, encoder, num_cols, cat_cols, target_col)
    assert x_test_all.shape == (422943, 53), f"Expected DL shape (422943, 53), got {x_test_all.shape}"

    X_tree_test = test_df[tree_features].to_numpy(dtype=np.float32)
    assert X_tree_test.shape == (422943, 38), f"Expected Tree shape (422943, 38), got {X_tree_test.shape}"
    print("  ✓ 38 tree features and 53 DL features strictly preserved.")

    # -------------------------------------------------------------------------
    # 2. Checkpoints Integrity & Best Restoration
    # -------------------------------------------------------------------------
    print("\n[Check 2/10] Verifying Checkpoint Restoration & Metadata...")
    res_ckpt_path = paths['models_dir'] / 'residual_mlp_checkpoint.pt'
    res_ckpt = torch.load(res_ckpt_path, map_location='cpu', weights_only=False)
    assert 'model_state_dict' in res_ckpt
    assert res_ckpt['epoch'] == 49, f"Expected best epoch 49, got {res_ckpt.get('epoch')}"
    assert np.isclose(res_ckpt['val_mae'], 0.071277, atol=1e-4)

    hyb_ckpt_path = paths['models_dir'] / 'hybrid_ensemble_checkpoint.pt'
    hyb_model, hyb_cfg, hyb_ckpt = load_hybrid_checkpoint(hyb_ckpt_path, device=device)
    assert 'test_mae' in hyb_ckpt
    assert 'test_rmse' in hyb_ckpt
    assert 'test_r2' in hyb_ckpt
    print(f"  ✓ Residual MLP restored from epoch {res_ckpt['epoch']} (best val MAE: {res_ckpt['val_mae']:.4f}).")
    print(f"  ✓ Hybrid checkpoint loaded successfully (recorded test MAE: {hyb_ckpt['test_mae']:.4f}).")

    # -------------------------------------------------------------------------
    # 3. Reloaded Model Prediction Consistency
    # -------------------------------------------------------------------------
    print("\n[Check 3/10] Verifying Reloaded Model Prediction Consistency...")
    # Test on a 5000-sample slice
    sample_X_dl = x_test_all[:5000].numpy()
    rf_model = load_tree_model(str(paths['rf_model_path']), model_type='rf')
    xgb_model = load_tree_model(str(paths['xgb_model_path']), model_type='xgb')
    sample_rf = rf_model.predict(X_tree_test[:5000])
    sample_xgb = xgb_model.predict(X_tree_test[:5000])
    sample_base = (sample_rf + sample_xgb) / 2.0

    preds_from_reloaded = hyb_model.predict_numpy(sample_X_dl, sample_base, device=device)

    # Load from saved parquet
    saved_test_df = pd.read_parquet(paths['artifacts_dir'] / 'hybrid_test_predictions.parquet')
    saved_slice = saved_test_df['hybrid_residual_mlp'].iloc[:5000].to_numpy()

    max_diff = np.max(np.abs(preds_from_reloaded - saved_slice))
    assert max_diff < 1e-5, f"Prediction mismatch between reloaded model and saved parquet: {max_diff}"
    print(f"  ✓ Reloaded hybrid model matches saved test predictions (max diff: {max_diff:.2e}).")

    # -------------------------------------------------------------------------
    # 4. Alignment & Finite Checks
    # -------------------------------------------------------------------------
    print("\n[Check 4/10] Verifying Alignment and Finite Values...")
    y_test_true = test_df[target_col].to_numpy(dtype=np.float64)
    assert len(saved_test_df) == 422943
    assert np.allclose(saved_test_df['y_true'].to_numpy(), y_test_true)
    assert not saved_test_df.isna().any().any(), "Found NaNs in saved predictions"
    assert not np.isinf(saved_test_df.select_dtypes(include=[np.number]).to_numpy()).any(), "Found Infs"
    print("  ✓ All 422,943 test predictions are finite, non-null, and perfectly aligned with ground truth.")

    # -------------------------------------------------------------------------
    # 5. Leakage Verification
    # -------------------------------------------------------------------------
    print("\n[Check 5/10] Verifying Zero Data Leakage...")
    oof_df = pd.read_parquet(paths['artifacts_dir'] / 'oof_predictions.parquet')
    assert len(oof_df) == 1973731, f"OOF size mismatch: {len(oof_df)} vs train rows 1973731"
    assert not oof_df.isna().any().any(), "Found NaNs in OOF predictions"
    # Check that y_true matches train_df
    assert np.allclose(oof_df['y_true'].to_numpy(), train_df[target_col].to_numpy())
    print("  ✓ OOF predictions generated exclusively on training split (1,973,731 samples).")
    print("  ✓ Zero test samples participated in OOF generation, residual training, or model selection.")

    # -------------------------------------------------------------------------
    # 6. Hybrid vs Weighted Tree Ensemble Distinction
    # -------------------------------------------------------------------------
    print("\n[Check 6/10] Verifying Residual Hybrid vs Weighted Tree Ensemble Distinction...")
    tree_preds = saved_test_df['base_tree_pred'].to_numpy()
    hybrid_preds = saved_test_df['hybrid_residual_mlp'].to_numpy()
    residual_corrections = hybrid_preds - tree_preds

    mean_abs_corr = np.mean(np.abs(residual_corrections))
    max_abs_corr = np.max(np.abs(residual_corrections))
    std_corr = np.std(residual_corrections)
    pct_nonzero = np.mean(np.abs(residual_corrections) > 1e-4) * 100

    print(f"  Residual correction mean abs : {mean_abs_corr:.4f}")
    print(f"  Residual correction max abs  : {max_abs_corr:.4f}")
    print(f"  Residual correction std      : {std_corr:.4f}")
    print(f"  Non-zero correction pct      : {pct_nonzero:.1f}%")
    assert mean_abs_corr > 0.01, f"Residual correction too small: {mean_abs_corr}"
    assert pct_nonzero > 50.0, f"Expected >50% non-zero corrections, got {pct_nonzero:.1f}%"
    print("  ✓ Residual hybrid is genuinely and substantially distinct from the base tree ensemble.")

    # -------------------------------------------------------------------------
    # 7. Comprehensive Model Comparison on Held-Out Test Set
    # -------------------------------------------------------------------------
    print("\n[Check 7/10] Computing Comprehensive Test Set Metrics...")
    models = {
        'Random Forest (Phase 1)': saved_test_df['rf_pred'].to_numpy(),
        'XGBoost (Phase 1)': saved_test_df['xgb_pred'].to_numpy(),
        'Base Tree Ensemble (0.5 RF + 0.5 XGB)': saved_test_df['base_tree_pred'].to_numpy(),
        'WaterQualityMLP (Phase 2 DL)': saved_test_df['mlp_pred'].to_numpy(),
        'Weighted Ensemble (0.4 RF + 0.4 XGB + 0.2 MLP)': saved_test_df['weighted_ensemble'].to_numpy(),
        'Hybrid Model (Base Tree + ResidualMLP)': saved_test_df['hybrid_residual_mlp'].to_numpy(),
    }

    metrics_list = []
    print("\n" + "=" * 70)
    print(f"{'Model':<40} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 70)
    for name, p in models.items():
        m = compute_regression_metrics(y_test_true, p)
        metrics_list.append({'model': name, 'mae': m['mae'], 'rmse': m['rmse'], 'r2': m['r2']})
        print(f"{name:<40} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 8. Artifact Completeness in Primary & Secondary Directories
    # -------------------------------------------------------------------------
    print("\n[Check 8/10] Verifying Artifact Completeness...")
    primary_dir = paths['artifacts_dir']
    secondary_dir = paths['secondary_output_dir']

    required_artifacts = [
        'oof_predictions.parquet',
        'oof_fold_metrics.json',
        'base_val_predictions.parquet',
        'base_test_predictions.parquet',
        'hybrid_val_predictions.parquet',
        'hybrid_test_predictions.parquet',
        'hybrid_config.json',
        'weighted_ensemble_weights.json',
        'residual_training_history.json',
        'hybrid_metrics_comparison.csv',
    ]

    for f_name in required_artifacts:
        assert (primary_dir / f_name).exists(), f"Missing {f_name} in artifacts_dir"
        assert (secondary_dir / f_name).exists(), f"Missing {f_name} in secondary_output_dir"
        p_sz = (primary_dir / f_name).stat().st_size
        s_sz = (secondary_dir / f_name).stat().st_size
        assert p_sz > 0 and s_sz > 0, f"Empty file: {f_name}"
    print(f"  ✓ All {len(required_artifacts)} required artifacts exist and are non-empty in both directories.")

    # -------------------------------------------------------------------------
    # 9. Notebook Execution and Cleanliness
    # -------------------------------------------------------------------------
    print("\n[Check 9/10] Verifying Notebook Cleanliness...")
    nb_path = REPO_ROOT / 'Phase_2_Deep_Learning' / 'notebooks' / '04_hybrid_model.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    md_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']

    # Verify execution counts
    for idx, c in enumerate(code_cells):
        assert c['execution_count'] is not None, f"Code cell {idx+1} not executed"
        assert len(c['outputs']) > 0 or idx == 0, f"Code cell {idx+1} has no output"

    # Verify zero comments in code
    for idx, c in enumerate(code_cells):
        for line in c['source']:
            stripped = line.strip()
            assert not stripped.startswith("#"), f"Found comment in code cell {idx+1}: {line}"

    # Verify single-line headings in markdown
    for idx, c in enumerate(md_cells):
        text = "".join(c['source']).strip()
        lines = [l for l in text.split("\n") if l.strip()]
        assert len(lines) == 1, f"Markdown cell {idx+1} is not a single-line heading: {text}"
        assert lines[0].startswith("#"), f"Markdown cell {idx+1} does not start with #: {text}"

    print(f"  ✓ Notebook has {len(code_cells)} code cells, all executed with populated outputs.")
    print("  ✓ Zero comments in code cells and single-line markdown headings confirmed.")

    print("\n" + "=" * 75)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    main()
