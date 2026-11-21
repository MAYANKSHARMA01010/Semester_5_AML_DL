import json
from pathlib import Path

notebook_path = Path("Phase_2_Deep_Learning/notebooks/04_hybrid_model.ipynb")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# 04 — Hybrid Model Development and Ensemble Evaluation"]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 1 — Setup and Imports"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "import json\n",
            "import warnings\n",
            "from pathlib import Path\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import torch\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "current_dir = Path().resolve()\n",
            "REPO_ROOT = current_dir\n",
            "for p in [current_dir, current_dir.parent] + list(current_dir.parents):\n",
            "    if (p / 'Phase_1_Machine_Learning').exists() and (p / 'Phase_2_Deep_Learning').exists():\n",
            "        REPO_ROOT = p\n",
            "        break\n",
            "\n",
            "if str(REPO_ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(REPO_ROOT))\n",
            "\n",
            "PHASE2_SRC = str(REPO_ROOT / 'Phase_2_Deep_Learning')\n",
            "if PHASE2_SRC not in sys.path:\n",
            "    sys.path.insert(0, PHASE2_SRC)\n",
            "\n",
            "from Phase_1_Machine_Learning.src.models import load_tree_model\n",
            "from Phase_2_Deep_Learning.src.dataset import (\n",
            "    get_default_paths,\n",
            "    load_feature_config,\n",
            "    load_raw_splits,\n",
            "    load_transformers,\n",
            "    prepare_tensors,\n",
            ")\n",
            "from Phase_2_Deep_Learning.src.dl_models import load_model_checkpoint\n",
            "from Phase_2_Deep_Learning.src.training import get_device, compute_regression_metrics\n",
            "from Phase_2_Deep_Learning.src.hybrid_models import (\n",
            "    HybridConfig,\n",
            "    ResidualMLPNet,\n",
            "    HybridEnsemble,\n",
            "    WeightedEnsemble,\n",
            "    save_hybrid_checkpoint,\n",
            "    load_hybrid_checkpoint,\n",
            ")\n",
            "from Phase_2_Deep_Learning.src.oof_generator import (\n",
            "    generate_ensemble_oof,\n",
            "    compute_residuals,\n",
            "    save_oof_artifacts,\n",
            "    load_oof_artifacts,\n",
            ")\n",
            "from Phase_2_Deep_Learning.src.residual_trainer import (\n",
            "    ResidualDataset,\n",
            "    build_residual_dataloaders,\n",
            "    train_residual_mlp,\n",
            "    evaluate_residual_model,\n",
            ")\n",
            "from Phase_2_Deep_Learning.scripts.run_hybrid_pipeline import resolve_task4_paths\n",
            "\n",
            "print(f\"REPO_ROOT: {REPO_ROOT}\")\n",
            "print(f\"PyTorch: {torch.__version__}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "paths = resolve_task4_paths()\n",
            "DEVICE = get_device()\n",
            "print(f\"Device: {DEVICE}\")\n",
            "print(f\"RF model exists: {paths['rf_model_path'].exists()}\")\n",
            "print(f\"XGB model exists: {paths['xgb_model_path'].exists()}\")\n",
            "print(f\"MLP checkpoint exists: {paths['mlp_ckpt_path'].exists()}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "feature_cfg = load_feature_config(paths['configs_dir'] / 'feature_config.json')\n",
            "scaler, encoder = load_transformers(paths['artifacts_dir'])\n",
            "\n",
            "NUM_FEATURES = feature_cfg['numerical_features']\n",
            "CAT_FEATURES = feature_cfg['categorical_features']\n",
            "TARGET_COL = feature_cfg['target_variable']\n",
            "\n",
            "with open(paths['metadata_json'], 'r', encoding='utf-8') as f:\n",
            "    p1_meta = json.load(f)\n",
            "tree_features = p1_meta['input_features_ordered']\n",
            "\n",
            "print(f\"Tree features: {len(tree_features)}\")\n",
            "print(f\"DL numerical features: {len(NUM_FEATURES)}\")\n",
            "print(f\"DL categorical features: {len(CAT_FEATURES)}\")\n",
            "print(f\"Target: {TARGET_COL}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "train_df, val_df, test_df = load_raw_splits(paths['data_dir'])\n",
            "print(f\"Train: {len(train_df):,} rows | Val: {len(val_df):,} rows | Test: {len(test_df):,} rows\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "rf_model = load_tree_model(str(paths['rf_model_path']), model_type='rf')\n",
            "xgb_model = load_tree_model(str(paths['xgb_model_path']), model_type='xgb')\n",
            "print(f\"RF model loaded: {type(rf_model).__name__} ({rf_model.n_estimators} trees)\")\n",
            "print(f\"XGB model loaded: {type(xgb_model).__name__} ({xgb_model.n_estimators} trees)\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "X_tree_train = train_df[tree_features].to_numpy(dtype=np.float32)\n",
            "X_tree_val = val_df[tree_features].to_numpy(dtype=np.float32)\n",
            "X_tree_test = test_df[tree_features].to_numpy(dtype=np.float32)\n",
            "\n",
            "y_train_np = train_df[TARGET_COL].to_numpy(dtype=np.float64)\n",
            "y_val_np = val_df[TARGET_COL].to_numpy(dtype=np.float64)\n",
            "y_test_np = test_df[TARGET_COL].to_numpy(dtype=np.float64)\n",
            "\n",
            "_, _, x_train_all, _ = prepare_tensors(train_df, scaler, encoder, NUM_FEATURES, CAT_FEATURES, TARGET_COL)\n",
            "_, _, x_val_all, _ = prepare_tensors(val_df, scaler, encoder, NUM_FEATURES, CAT_FEATURES, TARGET_COL)\n",
            "_, _, x_test_all, _ = prepare_tensors(test_df, scaler, encoder, NUM_FEATURES, CAT_FEATURES, TARGET_COL)\n",
            "\n",
            "print(f\"Tree Train: {X_tree_train.shape} | DL Train: {x_train_all.shape} | Targets: {y_train_np.shape}\")\n",
            "print(f\"Tree Val:   {X_tree_val.shape}   | DL Val:   {x_val_all.shape}   | Targets: {y_val_np.shape}\")\n",
            "print(f\"Tree Test:  {X_tree_test.shape}  | DL Test:  {x_test_all.shape}  | Targets: {y_test_np.shape}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 2 — Out-of-Fold Predictions"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "oof_file = paths['artifacts_dir'] / 'oof_predictions.parquet'\n",
            "if oof_file.exists():\n",
            "    cached_df = load_oof_artifacts(oof_file)\n",
            "    oof_rf = cached_df['oof_rf'].to_numpy(dtype=np.float64)\n",
            "    oof_xgb = cached_df['oof_xgb'].to_numpy(dtype=np.float64)\n",
            "    residuals_rf = cached_df['residual_rf'].to_numpy(dtype=np.float64)\n",
            "    residuals_xgb = cached_df['residual_xgb'].to_numpy(dtype=np.float64)\n",
            "    fold_metrics_path = paths['artifacts_dir'] / 'oof_fold_metrics.json'\n",
            "    if fold_metrics_path.exists():\n",
            "        with open(fold_metrics_path, 'r', encoding='utf-8') as f:\n",
            "            fold_metrics_dict = json.load(f)\n",
            "    else:\n",
            "        fold_metrics_dict = {'rf': [], 'xgb': []}\n",
            "    print(f\"Loaded cached OOF predictions: RF {oof_rf.shape}, XGB {oof_xgb.shape}\")\n",
            "else:\n",
            "    oof_rf, oof_xgb, fold_metrics_dict = generate_ensemble_oof(\n",
            "        X_tree_train, y_train_np, n_splits=5, random_state=42\n",
            "    )\n",
            "    residuals_rf, _ = compute_residuals(y_train_np, oof_rf)\n",
            "    residuals_xgb, _ = compute_residuals(y_train_np, oof_xgb)\n",
            "    save_oof_artifacts(oof_rf, oof_xgb, residuals_rf, residuals_xgb, y_train_np, paths['artifacts_dir'])\n",
            "    save_oof_artifacts(oof_rf, oof_xgb, residuals_rf, residuals_xgb, y_train_np, paths['secondary_output_dir'])\n",
            "    print(f\"Generated OOF predictions: RF {oof_rf.shape}, XGB {oof_xgb.shape}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if fold_metrics_dict.get('rf') and fold_metrics_dict.get('xgb'):\n",
            "    print(f\"{'Model':<6} {'Fold':<6} {'MAE':>8} {'RMSE':>8} {'R2':>8}\")\n",
            "    print('=' * 45)\n",
            "    for model_name in ['rf', 'xgb']:\n",
            "        folds = fold_metrics_dict[model_name]\n",
            "        maes, rmses, r2s = [], [], []\n",
            "        for m in folds:\n",
            "            print(f\"{model_name.upper():<6} {m['fold']:<6} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f}\")\n",
            "            maes.append(m['mae'])\n",
            "            rmses.append(m['rmse'])\n",
            "            r2s.append(m['r2'])\n",
            "        print('-' * 45)\n",
            "        print(f\"{'':6} {'mean':<6} {np.mean(maes):>8.4f} {np.mean(rmses):>8.4f} {np.mean(r2s):>8.4f}\")\n",
            "        print(f\"{'':6} {'std':<6} {np.std(maes):>8.4f} {np.std(rmses):>8.4f} {np.std(r2s):>8.4f}\")\n",
            "        print('=' * 45)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "residuals_rf, stats_rf = compute_residuals(y_train_np, oof_rf)\n",
            "residuals_xgb, stats_xgb = compute_residuals(y_train_np, oof_xgb)\n",
            "print('RF residual stats:', stats_rf)\n",
            "print('XGB residual stats:', stats_xgb)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "for ax, residuals, name, color in zip(\n",
            "    axes, [residuals_rf, residuals_xgb], ['RF', 'XGB'], ['#2196F3', '#FF9800']\n",
            "):\n",
            "    ax.hist(residuals, bins=60, color=color, alpha=0.7, edgecolor='white')\n",
            "    ax.axvline(0, color='black', lw=1.5, ls='--', label='Zero')\n",
            "    ax.axvline(np.mean(residuals), color='red', lw=1.5, ls='-', label=f'Mean={np.mean(residuals):.2f}')\n",
            "    ax.set_title(f'{name} OOF Residual Distribution', fontsize=13, fontweight='bold')\n",
            "    ax.set_xlabel('Residual (y_true - y_pred)')\n",
            "    ax.set_ylabel('Count')\n",
            "    ax.legend(fontsize=10)\n",
            "    stats = stats_rf if name == 'RF' else stats_xgb\n",
            "    ax.text(\n",
            "        0.98, 0.95,\n",
            "        f\"std={stats['std']:.2f}\\nmax_abs={stats['max_abs']:.2f}\\n%within5={stats['pct_within_5']:.1f}%\",\n",
            "        transform=ax.transAxes, ha='right', va='top', fontsize=9,\n",
            "        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)\n",
            "    )\n",
            "\n",
            "plt.suptitle('OOF Residual Distributions — RF vs XGBoost', fontsize=14, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.savefig(paths['artifacts_dir'] / 'oof_residual_distributions.png', dpi=150, bbox_inches='tight')\n",
            "plt.savefig(paths['secondary_output_dir'] / 'oof_residual_distributions.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 3 — Residual MLP Architecture"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "hybrid_cfg = HybridConfig(\n",
            "    input_dim=53,\n",
            "    hidden_dims=[256, 128, 64],\n",
            "    dropout_rate=0.2,\n",
            "    use_batch_norm=True,\n",
            "    rf_weight=0.4,\n",
            "    xgb_weight=0.4,\n",
            "    mlp_weight=0.2,\n",
            ")\n",
            "\n",
            "residual_net = ResidualMLPNet(hybrid_cfg).to(DEVICE)\n",
            "print(residual_net.architecture_summary())"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "dummy_in = torch.zeros(4, 54).to(DEVICE)\n",
            "dummy_out = residual_net(dummy_in)\n",
            "print(f\"Input shape: {dummy_in.shape}\")\n",
            "print(f\"Output shape: {dummy_out.shape}\")\n",
            "assert dummy_out.shape == (4, 1), f\"Expected (4,1), got {dummy_out.shape}\""
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 4 — Residual MLP Training"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "base_train_pred = (oof_rf + oof_xgb) / 2.0\n",
            "residuals_avg = y_train_np - base_train_pred\n",
            "\n",
            "rf_val_pred = rf_model.predict(X_tree_val)\n",
            "xgb_val_pred = xgb_model.predict(X_tree_val)\n",
            "base_val_pred = (rf_val_pred + xgb_val_pred) / 2.0\n",
            "residuals_val = y_val_np - base_val_pred\n",
            "\n",
            "print(f\"Train base preds: {base_train_pred.shape}\")\n",
            "print(f\"Train residuals: mean={residuals_avg.mean():.4f}, std={residuals_avg.std():.4f}\")\n",
            "print(f\"Val residuals: mean={residuals_val.mean():.4f}, std={residuals_val.std():.4f}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "base_train_t = torch.from_numpy(base_train_pred.astype(np.float32))\n",
            "res_train_t = torch.from_numpy(residuals_avg.astype(np.float32))\n",
            "base_val_t = torch.from_numpy(base_val_pred.astype(np.float32))\n",
            "res_val_t = torch.from_numpy(residuals_val.astype(np.float32))\n",
            "\n",
            "train_res_loader, val_res_loader = build_residual_dataloaders(\n",
            "    x_train_all, base_train_t, res_train_t,\n",
            "    x_val_all, base_val_t, res_val_t,\n",
            "    batch_size=1024,\n",
            ")\n",
            "print(f\"Train loader batches: {len(train_res_loader)}\")\n",
            "print(f\"Val loader batches: {len(val_res_loader)}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "RESIDUAL_CKPT_PATH = paths['models_dir'] / 'residual_mlp_checkpoint.pt'\n",
            "\n",
            "if RESIDUAL_CKPT_PATH.exists() and (paths['artifacts_dir'] / 'residual_training_history.json').exists():\n",
            "    with open(paths['artifacts_dir'] / 'residual_training_history.json', 'r', encoding='utf-8') as f:\n",
            "        res_history = json.load(f)\n",
            "    ckpt = torch.load(RESIDUAL_CKPT_PATH, map_location=DEVICE, weights_only=False)\n",
            "    residual_net.load_state_dict(ckpt['model_state_dict'])\n",
            "    print(f\"Loaded existing Residual MLP checkpoint (best val MAE: {res_history['best_val_mae']:.4f} @ epoch {res_history['best_epoch']})\")\n",
            "else:\n",
            "    res_history = train_residual_mlp(\n",
            "        model=residual_net,\n",
            "        train_loader=train_res_loader,\n",
            "        val_loader=val_res_loader,\n",
            "        n_epochs=50,\n",
            "        lr=1e-3,\n",
            "        patience=10,\n",
            "        device=DEVICE,\n",
            "        save_path=RESIDUAL_CKPT_PATH,\n",
            "    )\n",
            "    with open(paths['artifacts_dir'] / 'residual_training_history.json', 'w', encoding='utf-8') as f:\n",
            "        json.dump(res_history, f, indent=2)\n",
            "    with open(paths['secondary_output_dir'] / 'residual_training_history.json', 'w', encoding='utf-8') as f:\n",
            "        json.dump(res_history, f, indent=2)\n",
            "    print(f\"Trained Residual MLP: best val MAE {res_history['best_val_mae']:.4f} @ epoch {res_history['best_epoch']}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "epochs = range(1, len(res_history['train_loss']) + 1)\n",
            "\n",
            "axes[0].plot(epochs, res_history['train_loss'], label='Train Huber Loss', color='#2196F3')\n",
            "axes[0].plot(epochs, res_history['val_loss'], label='Val Huber Loss', color='#F44336')\n",
            "axes[0].axvline(res_history['best_epoch'], color='green', ls='--', lw=1.5, label=f\"Best ep={res_history['best_epoch']}\")\n",
            "axes[0].set_title('ResidualMLP — Training vs Validation Huber Loss', fontweight='bold')\n",
            "axes[0].set_xlabel('Epoch')\n",
            "axes[0].set_ylabel('Loss')\n",
            "axes[0].legend()\n",
            "\n",
            "axes[1].plot(epochs, res_history['val_mae'], label='Val MAE', color='#9C27B0')\n",
            "axes[1].axhline(res_history['best_val_mae'], color='green', ls='--', lw=1.5, label=f\"Best MAE={res_history['best_val_mae']:.4f}\")\n",
            "axes[1].set_title('ResidualMLP — Validation MAE', fontweight='bold')\n",
            "axes[1].set_xlabel('Epoch')\n",
            "axes[1].set_ylabel('MAE')\n",
            "axes[1].legend()\n",
            "\n",
            "plt.suptitle('Phase 2 — Residual MLP Training Convergence', fontsize=14, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.savefig(paths['artifacts_dir'] / 'residual_mlp_training_curves.png', dpi=150, bbox_inches='tight')\n",
            "plt.savefig(paths['secondary_output_dir'] / 'residual_mlp_training_curves.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 5 — Hybrid Ensemble Evaluation"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "base_mlp, mlp_ckpt = load_model_checkpoint(paths['mlp_ckpt_path'], device=DEVICE)\n",
            "base_mlp.eval()\n",
            "print(f\"Base MLP loaded: {mlp_ckpt.get('architecture_config', {}).get('name', 'MLP')}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "@torch.no_grad()\n",
            "def get_mlp_preds(model, X_tensor, device, batch_size=16384):\n",
            "    model.eval()\n",
            "    all_preds = []\n",
            "    n = len(X_tensor)\n",
            "    for start in range(0, n, batch_size):\n",
            "        batch = X_tensor[start:start + batch_size].to(device)\n",
            "        p = model(batch).cpu().numpy().flatten()\n",
            "        all_preds.append(p)\n",
            "    return np.concatenate(all_preds, axis=0)\n",
            "\n",
            "rf_val = rf_model.predict(X_tree_val)\n",
            "xgb_val = xgb_model.predict(X_tree_val)\n",
            "mlp_val = get_mlp_preds(base_mlp, x_val_all, DEVICE)\n",
            "\n",
            "rf_test = rf_model.predict(X_tree_test)\n",
            "xgb_test = xgb_model.predict(X_tree_test)\n",
            "mlp_test = get_mlp_preds(base_mlp, x_test_all, DEVICE)\n",
            "\n",
            "base_val_pred_avg = (rf_val + xgb_val) / 2.0\n",
            "base_test_pred_avg = (rf_test + xgb_test) / 2.0\n",
            "\n",
            "print(f\"Val  — RF: {rf_val.shape}, XGB: {xgb_val.shape}, MLP: {mlp_val.shape}\")\n",
            "print(f\"Test — RF: {rf_test.shape}, XGB: {xgb_test.shape}, MLP: {mlp_test.shape}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "weighted_ens = WeightedEnsemble(rf_weight=0.4, xgb_weight=0.4, mlp_weight=0.2)\n",
            "we_val = weighted_ens.predict(rf_val, xgb_val, mlp_val)\n",
            "we_test = weighted_ens.predict(rf_test, xgb_test, mlp_test)\n",
            "print(f\"Weighted Ensemble — Val: {we_val.shape}, Test: {we_test.shape}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "hybrid_model = HybridEnsemble(hybrid_cfg).to(DEVICE)\n",
            "hybrid_model.residual_net.load_state_dict(residual_net.state_dict())\n",
            "hybrid_model.eval()\n",
            "\n",
            "hybrid_val = hybrid_model.predict_numpy(x_val_all.numpy(), base_val_pred_avg, device=DEVICE)\n",
            "hybrid_test = hybrid_model.predict_numpy(x_test_all.numpy(), base_test_pred_avg, device=DEVICE)\n",
            "print(f\"Hybrid Ensemble — Val: {hybrid_val.shape}, Test: {hybrid_test.shape}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def eval_metrics(y_true, y_pred, label, split):\n",
            "    m = compute_regression_metrics(y_true, y_pred)\n",
            "    return {'Model': label, 'Split': split, 'MAE': m['mae'], 'RMSE': m['rmse'], 'R2': m['r2']}\n",
            "\n",
            "rows = []\n",
            "for split, y_true, preds_dict in [\n",
            "    ('Val', y_val_np, {'RF': rf_val, 'XGB': xgb_val, 'Base Tree (RF+XGB)': base_val_pred_avg, 'MLP (base)': mlp_val, 'Weighted Ensemble': we_val, 'Hybrid (RF+XGB+ResidualMLP)': hybrid_val}),\n",
            "    ('Test', y_test_np, {'RF': rf_test, 'XGB': xgb_test, 'Base Tree (RF+XGB)': base_test_pred_avg, 'MLP (base)': mlp_test, 'Weighted Ensemble': we_test, 'Hybrid (RF+XGB+ResidualMLP)': hybrid_test}),\n",
            "]:\n",
            "    for model_name, preds in preds_dict.items():\n",
            "        rows.append(eval_metrics(y_true, preds, model_name, split))\n",
            "\n",
            "metrics_df = pd.DataFrame(rows)\n",
            "metrics_df[['MAE', 'RMSE', 'R2']] = metrics_df[['MAE', 'RMSE', 'R2']].round(4)\n",
            "print(metrics_df.to_string(index=False))"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "test_preds_all = {\n",
            "    'Random Forest': rf_test,\n",
            "    'XGBoost': xgb_test,\n",
            "    'MLP (base)': mlp_test,\n",
            "    'Weighted Ensemble': we_test,\n",
            "    'Hybrid (ResidualMLP)': hybrid_test,\n",
            "}\n",
            "\n",
            "fig, axes = plt.subplots(2, 3, figsize=(18, 12))\n",
            "axes = axes.flatten()\n",
            "colors = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50', '#F44336']\n",
            "\n",
            "for ax, (model_name, preds), color in zip(axes, test_preds_all.items(), colors):\n",
            "    m = compute_regression_metrics(y_test_np, preds)\n",
            "    ax.scatter(y_test_np, preds, alpha=0.3, s=5, color=color, rasterized=True)\n",
            "    min_v, max_v = y_test_np.min(), y_test_np.max()\n",
            "    ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=1.5, label='Perfect')\n",
            "    ax.set_title(f\"{model_name}\\nMAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R2={m['r2']:.4f}\", fontsize=11)\n",
            "    ax.set_xlabel('Actual CCME WQI')\n",
            "    ax.set_ylabel('Predicted CCME WQI')\n",
            "    ax.legend(fontsize=8)\n",
            "\n",
            "axes[-1].set_visible(False)\n",
            "plt.suptitle('Phase 2 — Predicted vs Actual (Test Set)', fontsize=14, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.savefig(paths['artifacts_dir'] / 'hybrid_scatter_plots.png', dpi=150, bbox_inches='tight')\n",
            "plt.savefig(paths['secondary_output_dir'] / 'hybrid_scatter_plots.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n",
            "axes = axes.flatten()\n",
            "\n",
            "for ax, (model_name, preds), color in zip(axes, test_preds_all.items(), colors):\n",
            "    errors = y_test_np - preds\n",
            "    ax.hist(errors, bins=60, color=color, alpha=0.7, edgecolor='white')\n",
            "    ax.axvline(0, color='black', lw=1.5, ls='--')\n",
            "    ax.axvline(np.mean(errors), color='red', lw=1.5, ls='-', label=f'Mean={np.mean(errors):.3f}')\n",
            "    ax.set_title(f\"{model_name}\\nStd={np.std(errors):.3f}\", fontsize=11)\n",
            "    ax.set_xlabel('Prediction Error (y - y_hat)')\n",
            "    ax.set_ylabel('Count')\n",
            "    ax.legend(fontsize=8)\n",
            "\n",
            "axes[-1].set_visible(False)\n",
            "plt.suptitle('Phase 2 — Residual Error Distributions (Test Set)', fontsize=14, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.savefig(paths['artifacts_dir'] / 'hybrid_residual_plots.png', dpi=150, bbox_inches='tight')\n",
            "plt.savefig(paths['secondary_output_dir'] / 'hybrid_residual_plots.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 6 — Model Export and Summary"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "with open(paths['artifacts_dir'] / 'hybrid_config.json', 'w', encoding='utf-8') as f:\n",
            "    json.dump(hybrid_cfg.to_dict(), f, indent=2)\n",
            "with open(paths['secondary_output_dir'] / 'hybrid_config.json', 'w', encoding='utf-8') as f:\n",
            "    json.dump(hybrid_cfg.to_dict(), f, indent=2)\n",
            "\n",
            "with open(paths['artifacts_dir'] / 'weighted_ensemble_weights.json', 'w', encoding='utf-8') as f:\n",
            "    json.dump(weighted_ens.to_dict(), f, indent=2)\n",
            "with open(paths['secondary_output_dir'] / 'weighted_ensemble_weights.json', 'w', encoding='utf-8') as f:\n",
            "    json.dump(weighted_ens.to_dict(), f, indent=2)\n",
            "\n",
            "test_metrics_hybrid = compute_regression_metrics(y_test_np, hybrid_test)\n",
            "save_hybrid_checkpoint(\n",
            "    model=hybrid_model,\n",
            "    config=hybrid_cfg,\n",
            "    path=paths['models_dir'] / 'hybrid_ensemble_checkpoint.pt',\n",
            "    metadata={\n",
            "        'best_val_mae': float(res_history['best_val_mae']),\n",
            "        'test_mae': test_metrics_hybrid['mae'],\n",
            "        'test_rmse': test_metrics_hybrid['rmse'],\n",
            "        'test_r2': test_metrics_hybrid['r2'],\n",
            "    },\n",
            ")\n",
            "\n",
            "test_preds_df = pd.DataFrame({\n",
            "    'y_true': y_test_np,\n",
            "    'rf_pred': rf_test,\n",
            "    'xgb_pred': xgb_test,\n",
            "    'base_tree_pred': base_test_pred_avg,\n",
            "    'mlp_pred': mlp_test,\n",
            "    'weighted_ensemble': we_test,\n",
            "    'hybrid_residual_mlp': hybrid_test,\n",
            "})\n",
            "val_preds_df = pd.DataFrame({\n",
            "    'y_true': y_val_np,\n",
            "    'rf_pred': rf_val,\n",
            "    'xgb_pred': xgb_val,\n",
            "    'base_tree_pred': base_val_pred_avg,\n",
            "    'mlp_pred': mlp_val,\n",
            "    'weighted_ensemble': we_val,\n",
            "    'hybrid_residual_mlp': hybrid_val,\n",
            "})\n",
            "\n",
            "test_preds_df.to_parquet(paths['artifacts_dir'] / 'hybrid_test_predictions.parquet', index=False)\n",
            "test_preds_df.to_parquet(paths['secondary_output_dir'] / 'hybrid_test_predictions.parquet', index=False)\n",
            "val_preds_df.to_parquet(paths['artifacts_dir'] / 'hybrid_val_predictions.parquet', index=False)\n",
            "val_preds_df.to_parquet(paths['secondary_output_dir'] / 'hybrid_val_predictions.parquet', index=False)\n",
            "\n",
            "metrics_df.to_csv(paths['artifacts_dir'] / 'hybrid_metrics_comparison.csv', index=False)\n",
            "metrics_df.to_csv(paths['secondary_output_dir'] / 'hybrid_metrics_comparison.csv', index=False)\n",
            "print('Saved all hybrid model artifacts.')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "test_rows = metrics_df[metrics_df['Split'] == 'Test'].copy()\n",
            "print(test_rows[['Model', 'MAE', 'RMSE', 'R2']].to_string(index=False))\n",
            "best_row = test_rows.loc[test_rows['MAE'].idxmin()]\n",
            "print(f\"Best model by MAE: {best_row['Model']} (MAE={best_row['MAE']:.4f})\")"
        ]
    }
]

nb_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1)

print(f"Created clean notebook at {notebook_path} with {len(cells)} cells.")
