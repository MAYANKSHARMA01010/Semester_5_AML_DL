"""
Script to create and execute 06_explainability_error_analysis.ipynb with:
- Zero comments in code cells
- Single-line Markdown section headings only
- In-process execution with captured outputs and plots
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.makedirs("/tmp/matplotlib_cache", exist_ok=True)

import base64
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parents[1]
NOTEBOOK_PATH = REPO_ROOT / "Phase_2_Deep_Learning" / "notebooks" / "06_explainability_error_analysis.ipynb"


def build_cells():
    cells = []

    # Cell 0
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Section 1 — Setup and Configuration"]
    })

    # Cell 1
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "import os\n",
            "os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'\n",
            "os.environ['OMP_NUM_THREADS'] = '1'\n",
            "os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'\n",
            "os.makedirs('/tmp/matplotlib_cache', exist_ok=True)\n",
            "\n",
            "import sys\n",
            "import json\n",
            "import time\n",
            "from pathlib import Path\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import torch\n",
            "import matplotlib\n",
            "matplotlib.use('Agg')\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "current = Path.cwd()\n",
            "for parent in [current] + list(current.parents):\n",
            "    if (parent / 'Phase_1_Machine_Learning').exists() and (parent / 'Phase_2_Deep_Learning').exists():\n",
            "        REPO_ROOT = parent\n",
            "        break\n",
            "else:\n",
            "    raise FileNotFoundError('Could not find repository root')\n",
            "\n",
            "if str(REPO_ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(REPO_ROOT))\n",
            "\n",
            "device = torch.device('cpu')\n",
            "SEED = 42\n",
            "IG_SAMPLE_SIZE = 1000\n",
            "IG_STEPS = 50\n",
            "\n",
            "output_directory = REPO_ROOT / 'Phase_2_Deep_Learning' / 'reports' / 'task_6_outputs'\n",
            "output_directory.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "sns.set_theme(style='whitegrid', font_scale=1.0)\n",
            "print('Setup completed successfully')\n"
        ]
    })

    # Cell 2
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Section 2 — Load Data and Models"]
    })

    # Cell 3
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "from Phase_2_Deep_Learning.src.dataset import (\n",
            "    load_feature_config,\n",
            "    load_transformers,\n",
            "    prepare_tensors,\n",
            "    get_default_paths,\n",
            ")\n",
            "from Phase_2_Deep_Learning.src.dl_models import WaterQualityMLP, load_model_checkpoint\n",
            "from Phase_2_Deep_Learning.src.evaluation import ModelEvaluator\n",
            "from Phase_2_Deep_Learning.src.hybrid_models import load_hybrid_checkpoint\n",
            "\n",
            "paths = get_default_paths()\n",
            "feature_config = load_feature_config(paths['feature_config'])\n",
            "\n",
            "numerical_features = feature_config['numerical_features']\n",
            "categorical_features = feature_config['categorical_features']\n",
            "one_hot_feature_names = feature_config['one_hot_feature_names']\n",
            "all_feature_names = feature_config['all_feature_names']\n",
            "target_column = feature_config['target_variable']\n",
            "\n",
            "scaler, encoder = load_transformers(paths['artifacts_dir'])\n",
            "\n",
            "test_path = paths['data_dir'] / 'test.parquet'\n",
            "test_df = pd.read_parquet(test_path)\n",
            "\n",
            "evaluator = ModelEvaluator(device=device, paths=paths)\n",
            "evaluator.load_all_models()\n",
            "\n",
            "print(f'Test Records Ingested: {len(test_df):,} rows')\n",
            "print(f'Numerical Dimensions: {len(numerical_features)}')\n",
            "print(f'Categorical Input Dimensions (Encoded): {len(one_hot_feature_names)}')\n",
            "print(f'Total Deep Learning Input Space: {len(all_feature_names)}')\n",
            "print(f'Residual Hybrid Model Loaded: {evaluator.hybrid_model is not None}')\n"
        ]
    })

    # Cell 4
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Section 3 — Compute Predictions on Test Split"]
    })

    # Cell 5
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "t0 = time.time()\n",
            "test_predictions = evaluator.predict_all(test_df, batch_size=16384)\n",
            "pred_elapsed = time.time() - t0\n",
            "\n",
            "y_true = test_predictions['y_true']\n",
            "y_rf = test_predictions['Random Forest']\n",
            "y_hybrid = test_predictions['Residual Hybrid']\n",
            "y_base_tree = test_predictions['Base Tree Ensemble']\n",
            "y_xgb = test_predictions['XGBoost']\n",
            "y_ens = test_predictions['Weighted Ensemble']\n",
            "y_mlp = test_predictions['MLP']\n",
            "\n",
            "print(f'Predictions generated in {pred_elapsed:.2f} seconds ({len(test_df)/pred_elapsed:,.0f} rows/sec)')\n",
            "baseline_metrics = evaluator.evaluate_predictions(test_predictions)\n",
            "display_metrics = baseline_metrics[['Model', 'MAE', 'RMSE', 'R2', 'MedAE', 'MaxAE', 'Samples']].copy()\n",
            "print(display_metrics.to_string(index=False))\n"
        ]
    })

    # Cell 6
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Section 4 — MLP Integrated Gradients Analysis"]
    })

    # Cell 7
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "def compute_integrated_gradients(model, x, baseline, steps=50):\n",
            "    diff = x - baseline\n",
            "    alphas = torch.linspace(0.0, 1.0, steps, device=x.device)\n",
            "    total_grads = torch.zeros_like(x)\n",
            "    for alpha in alphas:\n",
            "        x_step = (baseline + alpha * diff).detach().requires_grad_(True)\n",
            "        out = model(x_step)\n",
            "        grad = torch.autograd.grad(out.sum(), x_step)[0]\n",
            "        total_grads += grad\n",
            "    return diff * (total_grads / steps)\n",
            "\n",
            "np.random.seed(SEED)\n",
            "sample_indices = np.random.choice(len(test_df), size=IG_SAMPLE_SIZE, replace=False)\n",
            "test_sample_df = test_df.iloc[sample_indices].copy().reset_index(drop=True)\n",
            "\n",
            "_, _, x_sample_tensor, _ = evaluator.prepare_inputs(test_sample_df)\n",
            "x_sample_tensor = x_sample_tensor.to(device)\n",
            "\n",
            "dummy_row = test_sample_df.iloc[0:1].copy()\n",
            "for col in numerical_features:\n",
            "    dummy_row[col] = scaler.mean_[numerical_features.index(col)]\n",
            "dummy_row['country_code'] = test_df['country_code'].mode()[0]\n",
            "dummy_row['waterbody_type_code'] = test_df['waterbody_type_code'].mode()[0]\n",
            "\n",
            "_, _, baseline_tensor, _ = prepare_tensors(\n",
            "    dummy_row, scaler, encoder,\n",
            "    numerical_features, categorical_features, target_column\n",
            ")\n",
            "baseline_tensor = baseline_tensor.to(device)\n",
            "\n",
            "evaluator.mlp_model.eval()\n",
            "t0 = time.time()\n",
            "ig_attributions_tensor = compute_integrated_gradients(\n",
            "    evaluator.mlp_model, x_sample_tensor, baseline_tensor, steps=IG_STEPS\n",
            ")\n",
            "ig_elapsed = time.time() - t0\n",
            "\n",
            "ig_values = ig_attributions_tensor.detach().cpu().numpy()\n",
            "\n",
            "with torch.no_grad():\n",
            "    f_x = evaluator.mlp_model(x_sample_tensor).cpu().numpy().flatten()\n",
            "    f_base = evaluator.mlp_model(baseline_tensor).cpu().numpy().flatten()\n",
            "\n",
            "completeness_errors = np.abs(ig_values.sum(axis=1) - (f_x - f_base))\n",
            "mean_completeness = float(completeness_errors.mean())\n",
            "p95_completeness = float(np.percentile(completeness_errors, 95))\n",
            "max_completeness = float(completeness_errors.max())\n",
            "\n",
            "print(f'Integrated Gradients computed for {IG_SAMPLE_SIZE} samples in {ig_elapsed:.2f}s')\n",
            "print(f'Reference baseline prediction F(x\"): {f_base[0]:.4f} CCME units')\n",
            "print(f'Completeness mean error: {mean_completeness:.4f}')\n",
            "print(f'Completeness P95 error: {p95_completeness:.4f}')\n",
            "print(f'Completeness max error: {max_completeness:.4f}')\n",
            "\n",
            "mean_abs_attributions = np.abs(ig_values).mean(axis=0)\n",
            "mean_signed_attributions = ig_values.mean(axis=0)\n",
            "std_signed_attributions = ig_values.std(axis=0)\n",
            "\n",
            "global_attribution_df = pd.DataFrame({\n",
            "    'Feature': all_feature_names,\n",
            "    'Mean_Absolute_Attribution': mean_abs_attributions,\n",
            "    'Mean_Signed_Attribution': mean_signed_attributions,\n",
            "    'Std_Signed_Attribution': std_signed_attributions,\n",
            "}).sort_values('Mean_Absolute_Attribution', ascending=False).reset_index(drop=True)\n",
            "\n",
            "global_attribution_df.to_csv(output_directory / 'mlp_global_feature_attributions.csv', index=False)\n",
            "\n",
            "country_ohe_cols = [c for c in one_hot_feature_names if c.startswith('country_code')]\n",
            "wb_ohe_cols = [c for c in one_hot_feature_names if c.startswith('waterbody_type_code')]\n",
            "\n",
            "aggregated_records = []\n",
            "for num_f in numerical_features:\n",
            "    idx = all_feature_names.index(num_f)\n",
            "    aggregated_records.append({\n",
            "        'Feature': num_f,\n",
            "        'Group_Type': 'Numerical',\n",
            "        'Mean_Absolute_Attribution': float(mean_abs_attributions[idx]),\n",
            "        'Mean_Signed_Attribution': float(mean_signed_attributions[idx]),\n",
            "    })\n",
            "\n",
            "c_indices = [all_feature_names.index(c) for c in country_ohe_cols]\n",
            "aggregated_records.append({\n",
            "    'Feature': 'country_code',\n",
            "    'Group_Type': 'Categorical_Aggregated',\n",
            "    'Mean_Absolute_Attribution': float(np.abs(ig_values[:, c_indices]).sum(axis=1).mean()),\n",
            "    'Mean_Signed_Attribution': float(ig_values[:, c_indices].sum(axis=1).mean()),\n",
            "})\n",
            "\n",
            "wb_indices = [all_feature_names.index(c) for c in wb_ohe_cols]\n",
            "aggregated_records.append({\n",
            "    'Feature': 'waterbody_type_code',\n",
            "    'Group_Type': 'Categorical_Aggregated',\n",
            "    'Mean_Absolute_Attribution': float(np.abs(ig_values[:, wb_indices]).sum(axis=1).mean()),\n",
            "    'Mean_Signed_Attribution': float(ig_values[:, wb_indices].sum(axis=1).mean()),\n",
            "})\n",
            "\n",
            "aggregated_attribution_df = pd.DataFrame(aggregated_records).sort_values(\n",
            "    'Mean_Absolute_Attribution', ascending=False\n",
            ").reset_index(drop=True)\n",
            "\n",
            "aggregated_attribution_df.to_csv(output_directory / 'mlp_aggregated_feature_attributions.csv', index=False)\n",
            "print('Top 10 Global Features (53-dim):')\n",
            "print(global_attribution_df.head(10).to_string(index=False))\n",
            "print('\\nTop 10 Global Features (38-dim Aggregated):')\n",
            "print(aggregated_attribution_df.head(10).to_string(index=False))\n"
        ]
    })

    # Cell 8
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Section 5 — Global Feature Importance Visualizations"]
    })

    # Cell 9
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "fig, ax = plt.subplots(figsize=(10, 8), dpi=150)\n",
            "top_20_features = global_attribution_df.head(20).iloc[::-1]\n",
            "ax.barh(top_20_features['Feature'], top_20_features['Mean_Absolute_Attribution'], color='#2b5c8f')\n",
            "ax.set_title('Top 20 Input Features by Mean Absolute Integrated Gradients Attribution', fontweight='bold')\n",
            "ax.set_xlabel('Mean |Attribution| (CCME WQI units)')\n",
            "plt.tight_layout()\n",
            "plt.savefig(output_directory / '01_mlp_global_feature_importance.png', bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 8), dpi=150)\n",
            "top_15_names = global_attribution_df.head(15)['Feature'].tolist()[::-1]\n",
            "top_15_indices = [all_feature_names.index(f) for f in top_15_names]\n",
            "top_15_distributions = [ig_values[:, idx] for idx in top_15_indices]\n",
            "\n",
            "box = ax.boxplot(top_15_distributions, tick_labels=top_15_names, patch_artist=True, showfliers=False, orientation='horizontal')\n",
            "for patch in box['boxes']:\n",
            "    patch.set_facecolor('#41b6c4')\n",
            "    patch.set_alpha(0.7)\n",
            "ax.axvline(0, color='red', linestyle='--', linewidth=0.8)\n",
            "ax.set_title('Distribution of Signed Integrated Gradients Attributions (Top 15 Features)', fontweight='bold')\n",
            "ax.set_xlabel('Attribution Value (Positive = Raises WQI, Negative = Suppresses WQI)')\n",
            "plt.tight_layout()\n",
            "plt.savefig(output_directory / '02_mlp_signed_feature_attributions.png', bbox_inches='tight')\n",
            "plt.show()\n"
        ]
    })

    # Cell 10
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Section 6 — Individual Prediction Explanations"]
    })

    # Cell 11
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "sample_actuals = test_sample_df[target_column].values\n",
            "sample_preds = f_x\n",
            "sample_residuals = sample_preds - sample_actuals\n",
            "sample_abs_errors = np.abs(sample_residuals)\n",
            "\n",
            "overpred_idx = int(np.argsort(sample_residuals)[-1])\n",
            "underpred_idx = int(np.argsort(sample_residuals)[0])\n",
            "high_qual_idx = int(np.where((sample_actuals > 90) & (sample_abs_errors < 0.5))[0][0])\n",
            "poor_qual_idx = int(np.where((sample_actuals < 55) & (sample_abs_errors < 1.0))[0][0])\n",
            "\n",
            "case_indices = {\n",
            "    'Excellent Quality (Low Error)': high_qual_idx,\n",
            "    'Poor Quality (Low Error)': poor_qual_idx,\n",
            "    'Notable Overprediction (Pred > Actual)': overpred_idx,\n",
            "    'Notable Underprediction (Pred < Actual)': underpred_idx,\n",
            "}\n",
            "\n",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)\n",
            "axes = axes.flatten()\n",
            "case_records = []\n",
            "\n",
            "for plot_idx, (case_name, idx_case) in enumerate(case_indices.items()):\n",
            "    actual_val = float(sample_actuals[idx_case])\n",
            "    pred_val = float(sample_preds[idx_case])\n",
            "    res_val = float(sample_residuals[idx_case])\n",
            "    row_ig = ig_values[idx_case]\n",
            "\n",
            "    top_pos_indices = np.argsort(row_ig)[-5:][::-1]\n",
            "    top_neg_indices = np.argsort(row_ig)[:5]\n",
            "    top_features_indices = list(top_pos_indices) + list(top_neg_indices)\n",
            "    top_features_indices = sorted(list(set(top_features_indices)), key=lambda idx: row_ig[idx])\n",
            "\n",
            "    case_feature_names = [all_feature_names[i] for i in top_features_indices]\n",
            "    case_feature_values = [row_ig[i] for i in top_features_indices]\n",
            "    colors = ['firebrick' if v < 0 else 'forestgreen' for v in case_feature_values]\n",
            "\n",
            "    ax = axes[plot_idx]\n",
            "    ax.barh(case_feature_names, case_feature_values, color=colors)\n",
            "    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)\n",
            "    ax.set_title(f'{case_name}\\nActual: {actual_val:.2f} | Pred: {pred_val:.2f} | Residual: {res_val:+.2f}', fontsize=10, fontweight='bold')\n",
            "    ax.set_xlabel('Attribution Contribution (CCME units)')\n",
            "\n",
            "    case_records.append({\n",
            "        'Case_Type': case_name,\n",
            "        'Actual_WQI': actual_val,\n",
            "        'Predicted_WQI': pred_val,\n",
            "        'Residual_Pred_Minus_Actual': res_val,\n",
            "        'Absolute_Error': abs(res_val),\n",
            "        'Direction': 'Overpredicted' if res_val > 0 else 'Underpredicted',\n",
            "        'Top_Positive_Feature': all_feature_names[top_pos_indices[0]],\n",
            "        'Top_Positive_Attribution': float(row_ig[top_pos_indices[0]]),\n",
            "        'Top_Negative_Feature': all_feature_names[top_neg_indices[0]],\n",
            "        'Top_Negative_Attribution': float(row_ig[top_neg_indices[0]]),\n",
            "    })\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig(output_directory / '03_individual_prediction_explanations.png', bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "individual_cases_df = pd.DataFrame(case_records)\n",
            "individual_cases_df.to_csv(output_directory / 'individual_prediction_explanations.csv', index=False)\n",
            "print(individual_cases_df[['Case_Type', 'Actual_WQI', 'Predicted_WQI', 'Residual_Pred_Minus_Actual', 'Direction']].to_string(index=False))\n"
        ]
    })

    # Cell 12
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Section 7 — Hybrid Model Architecture and Component Analysis"]
    })

    # Cell 13
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "residual_corrections = y_hybrid - y_base_tree\n",
            "base_tree_abs_errors = np.abs(y_base_tree - y_true)\n",
            "hybrid_abs_errors = np.abs(y_hybrid - y_true)\n",
            "hybrid_error_delta = hybrid_abs_errors - base_tree_abs_errors\n",
            "\n",
            "pct_improved = float(np.mean(hybrid_error_delta < 0) * 100)\n",
            "pct_worsened = float(np.mean(hybrid_error_delta > 0) * 100)\n",
            "pct_neutral = float(np.mean(hybrid_error_delta == 0) * 100)\n",
            "\n",
            "print(f'Residual Correction Range: [{np.min(residual_corrections):.4f}, {np.max(residual_corrections):.4f}]')\n",
            "print(f'Residual Correction Mean: {np.mean(residual_corrections):.4f} | Std: {np.std(residual_corrections):.4f}')\n",
            "print(f'Hybrid Reduces Base Tree Error: {pct_improved:.2f}% of test cases')\n",
            "print(f'Base Tree RMSE: {np.sqrt(np.mean((y_base_tree - y_true)**2)):.4f} -> Hybrid RMSE: {np.sqrt(np.mean((y_hybrid - y_true)**2)):.4f}')\n",
            "print(f'Base Tree MaxAE: {np.max(base_tree_abs_errors):.4f} -> Hybrid MaxAE: {np.max(hybrid_abs_errors):.4f}')\n",
            "\n",
            "_, _, x_tensor_full_sample, _ = evaluator.prepare_inputs(test_sample_df)\n",
            "base_pred_sample = torch.tensor(y_base_tree[sample_indices], dtype=torch.float32).unsqueeze(-1)\n",
            "inp_residual = torch.cat([x_tensor_full_sample, base_pred_sample], dim=-1).requires_grad_(True)\n",
            "r_out = evaluator.hybrid_model.residual_net(inp_residual)\n",
            "grad_residual = torch.autograd.grad(r_out.sum(), inp_residual)[0]\n",
            "\n",
            "feature_names_54 = all_feature_names + ['base_tree_pred']\n",
            "mean_abs_res_grads = grad_residual.abs().mean(dim=0).detach().cpu().numpy()\n",
            "top_res_features_idx = np.argsort(mean_abs_res_grads)[::-1][:10]\n",
            "print('\\nTop Features Driving Residual Corrections:')\n",
            "for idx in top_res_features_idx:\n",
            "    print(f'  {feature_names_54[idx]:<25}: {mean_abs_res_grads[idx]:.6f}')\n",
            "\n",
            "hybrid_comp_sample_df = pd.DataFrame({\n",
            "    'Actual_WQI': y_true[:1000],\n",
            "    'Base_Tree_Prediction': y_base_tree[:1000],\n",
            "    'Residual_Correction': residual_corrections[:1000],\n",
            "    'Residual_Hybrid_Prediction': y_hybrid[:1000],\n",
            "    'Base_Tree_Abs_Error': base_tree_abs_errors[:1000],\n",
            "    'Hybrid_Abs_Error': hybrid_abs_errors[:1000],\n",
            "    'Error_Reduction': -hybrid_error_delta[:1000],\n",
            "})\n",
            "hybrid_comp_sample_df.to_csv(output_directory / 'hybrid_component_analysis.csv', index=False)\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)\n",
            "axes[0].hist(residual_corrections, bins=60, color='#008080', edgecolor='black', alpha=0.7)\n",
            "axes[0].set_title('Distribution of Neural Residual Corrections', fontweight='bold')\n",
            "axes[0].set_xlabel('Correction Value r (Hybrid - Base Tree)')\n",
            "axes[0].set_ylabel('Observation Count')\n",
            "\n",
            "sub_plot_idx = np.random.choice(len(y_true), size=10000, replace=False)\n",
            "axes[1].scatter(y_base_tree[sub_plot_idx], y_hybrid[sub_plot_idx], alpha=0.15, s=6, color='#2b5c8f')\n",
            "axes[1].plot([0, 100], [0, 100], 'r--', label='Identity Line')\n",
            "axes[1].set_title('Base Tree vs Residual Hybrid Predictions (10k Sample)', fontweight='bold')\n",
            "axes[1].set_xlabel('Base Tree Prediction')\n",
            "axes[1].set_ylabel('Residual Hybrid Prediction')\n",
            "axes[1].legend()\n",
            "plt.tight_layout()\n",
            "plt.savefig(output_directory / '04_hybrid_component_contributions.png', bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(8, 6), dpi=150)\n",
            "ax.scatter(np.abs(residual_corrections)[sub_plot_idx], -hybrid_error_delta[sub_plot_idx], alpha=0.15, s=8, color='#008080')\n",
            "ax.axhline(0, color='red', linestyle='--', label='Zero Error Change')\n",
            "ax.set_title('Residual Correction Magnitude vs Base Error Reduction', fontweight='bold')\n",
            "ax.set_xlabel('|Residual Correction| (CCME Units)')\n",
            "ax.set_ylabel('Error Reduction: Error(Base) - Error(Hybrid)')\n",
            "ax.legend()\n",
            "plt.tight_layout()\n",
            "plt.savefig(output_directory / '06_correction_vs_base_error.png', bbox_inches='tight')\n",
            "plt.show()\n"
        ]
    })

    # Cell 14
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Section 8 — Overall Prediction Error Distributions"]
    })

    # Cell 15
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "models_dict = {\n",
            "    'Random Forest': y_rf,\n",
            "    'Residual Hybrid': y_hybrid,\n",
            "    'Base Tree Ensemble': y_base_tree,\n",
            "    'XGBoost': y_xgb,\n",
            "    'Weighted Ensemble': y_ens,\n",
            "    'MLP': y_mlp,\n",
            "}\n",
            "\n",
            "error_summary_rows = []\n",
            "model_residuals = {}\n",
            "model_abs_errors = {}\n",
            "\n",
            "for name, y_pred in models_dict.items():\n",
            "    res = y_pred - y_true\n",
            "    ae = np.abs(res)\n",
            "    model_residuals[name] = res\n",
            "    model_abs_errors[name] = ae\n",
            "\n",
            "    error_summary_rows.append({\n",
            "        'Model': name,\n",
            "        'Mean_Residual': float(np.mean(res)),\n",
            "        'Residual_Std': float(np.std(res)),\n",
            "        'Median_AE': float(np.median(ae)),\n",
            "        'MAE': float(np.mean(ae)),\n",
            "        'RMSE': float(np.sqrt(np.mean(res ** 2))),\n",
            "        'P90_AE': float(np.percentile(ae, 90)),\n",
            "        'P95_AE': float(np.percentile(ae, 95)),\n",
            "        'P99_AE': float(np.percentile(ae, 99)),\n",
            "        'Max_AE': float(np.max(ae)),\n",
            "    })\n",
            "\n",
            "error_summary_df = pd.DataFrame(error_summary_rows).sort_values('MAE').reset_index(drop=True)\n",
            "error_summary_df.to_csv(output_directory / 'model_error_summary_comparison.csv', index=False)\n",
            "print('COMPREHENSIVE ERROR DIAGNOSTIC SUMMARY (Residual = Prediction - Actual):')\n",
            "print(error_summary_df[['Model', 'Mean_Residual', 'Median_AE', 'MAE', 'RMSE', 'Max_AE']].to_string(index=False))\n",
            "\n",
            "model_names = list(models_dict.keys())\n",
            "win_matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=float)\n",
            "for m1 in model_names:\n",
            "    for m2 in model_names:\n",
            "        if m1 == m2:\n",
            "            win_matrix.loc[m1, m2] = 50.0\n",
            "        else:\n",
            "            win_matrix.loc[m1, m2] = float(np.mean(model_abs_errors[m1] < model_abs_errors[m2]) * 100)\n",
            "\n",
            "win_matrix.to_csv(output_directory / 'model_comparison_win_rate_matrix.csv')\n",
            "print('\\nPAIRWISE WIN-RATE MATRIX (% Lower Absolute Error):')\n",
            "print(win_matrix.round(2).to_string())\n"
        ]
    })

    # Cell 16
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Section 9 — Error Distribution Visualizations"]
    })

    # Cell 17
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "palette_models = {\n",
            "    'Random Forest': '#2b5c8f',\n",
            "    'Residual Hybrid': '#008080',\n",
            "    'Base Tree Ensemble': '#41b6c4',\n",
            "    'XGBoost': '#d95f02',\n",
            "    'Weighted Ensemble': '#1b9e77',\n",
            "    'MLP': '#7570b3',\n",
            "}\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 5), dpi=150)\n",
            "for name in ['Random Forest', 'Residual Hybrid', 'XGBoost', 'MLP']:\n",
            "    sns.kdeplot(model_residuals[name], ax=ax, label=name, color=palette_models[name], clip=(-10, 10), common_norm=False)\n",
            "ax.set_title('Residual Density Comparison (Prediction - Actual)', fontweight='bold')\n",
            "ax.set_xlabel('Residual (Positive = Overprediction, Negative = Underprediction)')\n",
            "ax.set_ylabel('Density')\n",
            "ax.set_xlim(-6, 6)\n",
            "ax.axvline(0, color='black', linestyle='--', linewidth=0.8)\n",
            "ax.legend()\n",
            "plt.tight_layout()\n",
            "plt.savefig(output_directory / '07_residual_distributions_comparison.png', bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 5), dpi=150)\n",
            "box_names = ['Random Forest', 'Residual Hybrid', 'Base Tree Ensemble', 'XGBoost', 'Weighted Ensemble', 'MLP']\n",
            "box_data = [model_residuals[name] for name in box_names]\n",
            "box = ax.boxplot(box_data, tick_labels=box_names, patch_artist=True, showfliers=False)\n",
            "for patch, name in zip(box['boxes'], box_names):\n",
            "    patch.set_facecolor(palette_models[name])\n",
            "    patch.set_alpha(0.7)\n",
            "ax.axhline(0, color='black', linestyle='--', linewidth=0.8)\n",
            "ax.set_title('Comparative Residual Boxplots Across Tested Architectures', fontweight='bold')\n",
            "ax.set_ylabel('Residual (Prediction - Actual)')\n",
            "ax.tick_params(axis='x', rotation=20)\n",
            "plt.tight_layout()\n",
            "plt.savefig(output_directory / '08_residual_boxplots.png', bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=150)\n",
            "axes = axes.flatten()\n",
            "key_models = ['Random Forest', 'Residual Hybrid', 'XGBoost', 'MLP']\n",
            "for idx, name in enumerate(key_models):\n",
            "    ax = axes[idx]\n",
            "    ax.hexbin(models_dict[name], y_true, gridsize=50, cmap='Blues', mincnt=1, bins='log')\n",
            "    ax.plot([25, 105], [25, 105], 'r--', label='Identity (y=x)')\n",
            "    ax.set_title(f'{name} — Actual vs Predicted', fontweight='bold')\n",
            "    ax.set_xlabel('Predicted CCME WQI')\n",
            "    ax.set_ylabel('Actual CCME WQI')\n",
            "    ax.set_xlim(25, 105)\n",
            "    ax.set_ylim(25, 105)\n",
            "    if idx == 0: ax.legend()\n",
            "plt.tight_layout()\n",
            "plt.savefig(output_directory / '09_actual_vs_predicted.png', bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=150)\n",
            "axes = axes.flatten()\n",
            "for idx, name in enumerate(key_models):\n",
            "    ax = axes[idx]\n",
            "    ax.hexbin(models_dict[name], model_residuals[name], gridsize=50, cmap='Purples', mincnt=1, bins='log')\n",
            "    ax.axhline(0, color='red', linestyle='--')\n",
            "    ax.set_title(f'{name} — Residual vs Predicted', fontweight='bold')\n",
            "    ax.set_xlabel('Predicted CCME WQI')\n",
            "    ax.set_ylabel('Residual (Prediction - Actual)')\n",
            "    ax.set_xlim(25, 105)\n",
            "    ax.set_ylim(-15, 15)\n",
            "plt.tight_layout()\n",
            "plt.savefig(output_directory / '10_residual_vs_predicted.png', bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(9, 5), dpi=150)\n",
            "sorted_errors = {name: np.sort(model_abs_errors[name]) for name in key_models}\n",
            "cdf_eval = np.linspace(0, 1, len(y_true))\n",
            "for name in key_models:\n",
            "    ax.plot(sorted_errors[name], cdf_eval, label=name, color=palette_models[name], linewidth=1.5)\n",
            "ax.set_title('Empirical Cumulative Distribution Function (CDF) of Absolute Error', fontweight='bold')\n",
            "ax.set_xlabel('Absolute Error |Prediction - Actual|')\n",
            "ax.set_ylabel('Cumulative Probability')\n",
            "ax.set_xlim(0, 5)\n",
            "ax.set_ylim(0, 1.02)\n",
            "ax.axvline(0.5, color='gray', linestyle=':', label='0.5 WQI Threshold')\n",
            "ax.legend(loc='lower right')\n",
            "plt.tight_layout()\n",
            "plt.savefig(output_directory / '11_absolute_error_cdf.png', bbox_inches='tight')\n",
            "plt.show()\n"
        ]
    })

    # Cell 18
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Section 10 — Subgroup Error Pattern Diagnostics"]
    })

    # Cell 19
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "wqi_bins = [0, 45, 65, 80, 95, 100]\n",
            "wqi_labels = ['Poor [0-45]', 'Marginal (45-65]', 'Fair (65-80]', 'Good (80-95]', 'Excellent (95-100]']\n",
            "wqi_tiers = pd.cut(test_df[target_column], bins=wqi_bins, labels=wqi_labels, include_lowest=True)\n",
            "\n",
            "tier_records = []\n",
            "for tier in wqi_labels:\n",
            "    mask = (wqi_tiers == tier).values\n",
            "    count = int(np.sum(mask))\n",
            "    if count == 0: continue\n",
            "    for name in models_dict.keys():\n",
            "        res_sub = model_residuals[name][mask]\n",
            "        ae_sub = model_abs_errors[name][mask]\n",
            "        tier_records.append({\n",
            "            'WQI_Tier': tier,\n",
            "            'Model': name,\n",
            "            'Sample_Count': count,\n",
            "            'Mean_Residual': float(np.mean(res_sub)),\n",
            "            'MAE': float(np.mean(ae_sub)),\n",
            "            'RMSE': float(np.sqrt(np.mean(res_sub ** 2))),\n",
            "            'Max_AE': float(np.max(ae_sub)),\n",
            "        })\n",
            "\n",
            "tier_df = pd.DataFrame(tier_records)\n",
            "tier_df.to_csv(output_directory / 'error_diagnostics_by_wqi_tier.csv', index=False)\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(11, 5), dpi=150)\n",
            "tier_plot_df = tier_df[tier_df['Model'].isin(key_models)]\n",
            "sns.barplot(data=tier_plot_df, x='WQI_Tier', y='MAE', hue='Model', palette=palette_models, ax=ax)\n",
            "ax.set_title('Mean Absolute Error Across Regulatory CCME Water Quality Tiers', fontweight='bold')\n",
            "ax.set_ylabel('MAE (CCME WQI units)')\n",
            "ax.set_xlabel('Water Quality Classification Tier')\n",
            "ax.legend(title='Architecture', loc='upper right')\n",
            "plt.tight_layout()\n",
            "plt.savefig(output_directory / '12_error_by_wqi_tier.png', bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "country_diag_df = pd.read_csv(output_directory / 'error_diagnostics_by_country.csv')\n",
            "wb_diag_df = pd.read_csv(output_directory / 'error_diagnostics_by_waterbody.csv')\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)\n",
            "country_plot_df = country_diag_df[country_diag_df['Model'].isin(key_models)]\n",
            "sns.barplot(data=country_plot_df, x='Country', y='MAE', hue='Model', palette=palette_models, ax=axes[0])\n",
            "axes[0].set_title('MAE by Sovereign Jurisdiction', fontweight='bold')\n",
            "axes[0].set_ylabel('MAE (CCME units)')\n",
            "axes[0].tick_params(axis='x', rotation=20)\n",
            "\n",
            "top_wbs = test_df['waterbody_type'].value_counts().head(6).index.tolist()\n",
            "wb_plot_df = wb_diag_df[(wb_diag_df['Waterbody_Type'].isin(top_wbs)) & (wb_diag_df['Model'].isin(key_models))]\n",
            "sns.barplot(data=wb_plot_df, x='Waterbody_Type', y='MAE', hue='Model', palette=palette_models, ax=axes[1])\n",
            "axes[1].set_title('MAE by Dominant Aquatic Habitat', fontweight='bold')\n",
            "axes[1].set_ylabel('MAE (CCME units)')\n",
            "axes[1].tick_params(axis='x', rotation=25)\n",
            "plt.tight_layout()\n",
            "plt.savefig(output_directory / '13_error_by_country_waterbody.png', bbox_inches='tight')\n",
            "plt.show()\n"
        ]
    })

    # Cell 20
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Section 11 — Difficult Case Analysis"]
    })

    # Cell 21
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "meta_cols = ['country', 'waterbody_type', 'ph', 'do_mg_l', 'bod_mg_l', 'ammonia_mg_l', 'orthophosphate_mg_l', 'exceedance_counter']\n",
            "\n",
            "large_errors_df = pd.read_csv(output_directory / 'difficult_cases_largest_errors.csv')\n",
            "overpred_df = pd.read_csv(output_directory / 'difficult_cases_overpredictions.csv')\n",
            "underpred_df = pd.read_csv(output_directory / 'difficult_cases_underpredictions.csv')\n",
            "tradeoff_df = pd.read_csv(output_directory / 'difficult_cases_ensemble_tradeoffs.csv')\n",
            "\n",
            "print('Representative Difficult Cases (Largest Absolute Errors):')\n",
            "print(large_errors_df[['country', 'waterbody_type', 'Actual_WQI', 'MLP_Predicted', 'MLP_Residual', 'MLP_Abs_Error', 'Direction']].head(5).to_string(index=False))\n",
            "\n",
            "print('\\nNotable Overpredictions (Positive Residuals, Pred > Actual):')\n",
            "print(overpred_df[['country', 'waterbody_type', 'Actual_WQI', 'MLP_Predicted', 'MLP_Residual', 'Direction']].head(5).to_string(index=False))\n",
            "\n",
            "print('\\nNotable Underpredictions (Negative Residuals, Pred < Actual):')\n",
            "print(underpred_df[['country', 'waterbody_type', 'Actual_WQI', 'MLP_Predicted', 'MLP_Residual', 'Direction']].head(5).to_string(index=False))\n"
        ]
    })

    # Cell 22
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Section 12 — Summary Artifacts and Final Verification"]
    })

    # Cell 23
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "with open(output_directory / 'explainability_summary.json') as f:\n",
            "    exp_meta = json.load(f)\n",
            "with open(output_directory / 'error_analysis_summary.json') as f:\n",
            "    err_meta = json.load(f)\n",
            "\n",
            "print('Explainability Summary:')\n",
            "for k, v in exp_meta.items():\n",
            "    print(f'  {k}: {v}')\n",
            "\n",
            "print('\\nError Analysis Summary:')\n",
            "for k, v in err_meta.items():\n",
            "    print(f'  {k}: {v}')\n",
            "\n",
            "print('\\nVerification Checks:')\n",
            "print(f'1. Total models evaluated on common test split: {len(models_dict)}')\n",
            "print(f'2. Test observations aligned: {len(y_true):,}')\n",
            "print(f'3. Residual sign convention: Residual = Prediction - Actual')\n",
            "print(f'4. Overpredictions have positive residuals: {(overpred_df[\"MLP_Residual\"] > 0).all()}')\n",
            "print(f'5. Underpredictions have negative residuals: {(underpred_df[\"MLP_Residual\"] < 0).all()}')\n",
            "print(f'6. Residual Hybrid reduces peak tree error: {err_meta[\"hybrid_reduces_peak_error_pct\"]:.2f}%')\n",
            "print('Task 6 analysis and notebook execution completed successfully.')\n"
        ]
    })

    return cells


def execute_and_save():
    cells = build_cells()
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11"
            },
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    ns = {"__name__": "__main__"}
    exec_count = 1

    print("Executing Task 6 notebook cells top-to-bottom...")
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            print(f"Executing code cell {exec_count} (index {idx})...")
            stdout_buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout_buf
            plt.close("all")
            try:
                exec(code, ns)
            except Exception as e:
                sys.stdout = old_stdout
                print(f"Error executing cell {exec_count}: {e}")
                raise e
            finally:
                sys.stdout = old_stdout

            text_output = stdout_buf.getvalue()
            outputs = []
            if text_output:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": text_output.splitlines(keepends=True)
                })

            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=100)
                img_buf.seek(0)
                img_b64 = base64.b64encode(img_buf.read()).decode("utf-8")
                outputs.append({
                    "data": {
                        "image/png": img_b64,
                        "text/plain": "<Figure size 1400x1000 with 4 Axes>"
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
                plt.close(fig)

            cell["outputs"] = outputs
            cell["execution_count"] = exec_count
            exec_count += 1

    print(f"Writing executed notebook to: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"✓ All {exec_count - 1} code cells executed and saved.")


if __name__ == "__main__":
    execute_and_save()
