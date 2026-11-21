"""
Script to create and execute 05_model_evaluation.ipynb with:
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
NOTEBOOK_PATH = REPO_ROOT / "Phase_2_Deep_Learning" / "notebooks" / "05_model_evaluation.ipynb"


def build_cells():
    cells = []

    # Cell 0
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# 05 — Model Evaluation and Ablation Studies"]
    })

    # Cell 1
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 1 — Setup and Thread Safety"]
    })

    # Cell 2
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "import os\n",
            "os.environ[\"OMP_NUM_THREADS\"] = \"1\"\n",
            "os.environ[\"KMP_DUPLICATE_LIB_OK\"] = \"TRUE\"\n",
            "os.environ[\"MPLCONFIGDIR\"] = \"/tmp/matplotlib_cache\"\n",
            "os.makedirs(\"/tmp/matplotlib_cache\", exist_ok=True)\n",
            "\n",
            "import sys\n",
            "import json\n",
            "import time\n",
            "import platform\n",
            "from pathlib import Path\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import torch\n",
            "import matplotlib\n",
            "matplotlib.use(\"Agg\")\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "current = Path.cwd()\n",
            "for parent in [current] + list(current.parents):\n",
            "    if (parent / 'Phase_1_Machine_Learning').exists() and (parent / 'Phase_2_Deep_Learning').exists():\n",
            "        REPO_ROOT = parent\n",
            "        break\n",
            "else:\n",
            "    raise FileNotFoundError(\"Could not find repository root\")\n",
            "\n",
            "if str(REPO_ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(REPO_ROOT))\n",
            "\n",
            "device = torch.device(\"cpu\")\n",
            "sns.set_theme(style=\"whitegrid\", font_scale=1.0)\n"
        ]
    })

    # Cell 3
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 2 — Module Imports and Project Configuration"]
    })

    # Cell 4
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "from Phase_2_Deep_Learning.src.evaluation import (\n",
            "    ModelEvaluator,\n",
            "    FeatureAblationEngine,\n",
            "    SubgroupEvaluator,\n",
            "    ComputationalProfiler,\n",
            "    FEATURE_GROUPS,\n",
            ")\n",
            "from Phase_2_Deep_Learning.src.training import get_device, compute_regression_metrics\n",
            "from Phase_2_Deep_Learning.src.dl_models import WaterQualityMLP, load_model_checkpoint, mc_dropout_predictions\n",
            "from Phase_2_Deep_Learning.src.hybrid_models import HybridEnsemble, WeightedEnsemble\n",
            "\n",
            "print(\"Repository root:\", REPO_ROOT)\n",
            "print(\"Compute device:\", device)\n",
            "print(\"PyTorch version:\", torch.__version__)\n"
        ]
    })

    # Cell 5
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 3 — Quarantined Test Dataset Ingestion"]
    })

    # Cell 6
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "test_path = REPO_ROOT / 'Phase_1_Machine_Learning' / 'data' / 'GeneratedOutput' / '02.Preprocessing&FeatureSelection' / 'test.parquet'\n",
            "val_path = REPO_ROOT / 'Phase_1_Machine_Learning' / 'data' / 'GeneratedOutput' / '02.Preprocessing&FeatureSelection' / 'val.parquet'\n",
            "\n",
            "test_df = pd.read_parquet(test_path)\n",
            "val_df = pd.read_parquet(val_path)\n",
            "\n",
            "print(\"Test split shape:\", test_df.shape)\n",
            "print(\"Validation split shape:\", val_df.shape)\n",
            "print(\"CCME target range in test: [{:.2f}, {:.2f}]\".format(test_df['ccme_values'].min(), test_df['ccme_values'].max()))\n"
        ]
    })

    # Cell 7
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 4 — Model Artifact Restoration"]
    })

    # Cell 8
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "evaluator = ModelEvaluator(device=device)\n",
            "evaluator.load_all_models()\n",
            "\n",
            "print(\"Loaded Random Forest:\", type(evaluator.rf_model).__name__)\n",
            "print(\"Loaded XGBoost:\", type(evaluator.xgb_model).__name__)\n",
            "print(\"Loaded MLP:\", type(evaluator.mlp_model).__name__)\n",
            "print(\"Loaded Weighted Ensemble:\", type(evaluator.weighted_ensemble).__name__)\n",
            "print(\"Loaded Residual Hybrid:\", type(evaluator.hybrid_model).__name__)\n"
        ]
    })

    # Cell 9
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 5 — Unified Baseline Evaluation on Test Set"]
    })

    # Cell 10
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "t0 = time.time()\n",
            "test_preds = evaluator.predict_all(test_df)\n",
            "inference_time = time.time() - t0\n",
            "\n",
            "print(\"Predictions completed in {:.2f} seconds\".format(inference_time))\n",
            "metrics_df = evaluator.evaluate_predictions(test_preds)\n",
            "print(metrics_df.to_string(index=False))\n",
            "\n",
            "res_dir = REPO_ROOT / 'Phase_2_Deep_Learning' / 'reports' / 'evaluation_results'\n",
            "res_dir.mkdir(parents=True, exist_ok=True)\n",
            "metrics_df.to_csv(res_dir / 'baseline_model_comparison.csv', index=False)\n"
        ]
    })

    # Cell 11
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 6 — Model Performance Visualizations"]
    })

    # Cell 12
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)\n",
            "\n",
            "sns.barplot(data=metrics_df, x='Model', y='MAE', hue='Model', ax=axes[0], palette='Blues_r', legend=False)\n",
            "axes[0].set_title('Mean Absolute Error (Lower is Better)', fontweight='bold')\n",
            "axes[0].set_ylabel('MAE (CCME WQI units)')\n",
            "axes[0].tick_params(axis='x', rotation=30)\n",
            "for p in axes[0].patches:\n",
            "    axes[0].annotate(f\"{p.get_height():.3f}\", (p.get_x() + p.get_width() / 2., p.get_height()),\n",
            "                     ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')\n",
            "\n",
            "sns.barplot(data=metrics_df, x='Model', y='RMSE', hue='Model', ax=axes[1], palette='Oranges_r', legend=False)\n",
            "axes[1].set_title('Root Mean Squared Error (Lower is Better)', fontweight='bold')\n",
            "axes[1].set_ylabel('RMSE')\n",
            "axes[1].tick_params(axis='x', rotation=30)\n",
            "for p in axes[1].patches:\n",
            "    axes[1].annotate(f\"{p.get_height():.3f}\", (p.get_x() + p.get_width() / 2., p.get_height()),\n",
            "                     ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')\n",
            "\n",
            "sns.barplot(data=metrics_df, x='Model', y='R2', hue='Model', ax=axes[2], palette='Greens_r', legend=False)\n",
            "axes[2].set_title('R2 Coefficient of Determination', fontweight='bold')\n",
            "axes[2].set_ylabel('R2 Score')\n",
            "axes[2].set_ylim(0.985, 1.0005)\n",
            "axes[2].tick_params(axis='x', rotation=30)\n",
            "for p in axes[2].patches:\n",
            "    axes[2].annotate(f\"{p.get_height():.4f}\", (p.get_x() + p.get_width() / 2., p.get_height()),\n",
            "                     ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    })

    # Cell 13
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 7 — Systematic Feature Group Ablation Studies"]
    })

    # Cell 14
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "ablation_csv = res_dir / 'feature_group_ablation.csv'\n",
            "if ablation_csv.exists():\n",
            "    ablation_df = pd.read_csv(ablation_csv)\n",
            "else:\n",
            "    ablation_engine = FeatureAblationEngine(evaluator)\n",
            "    ablation_df = ablation_engine.run_all_models_ablation(\n",
            "        val_df=val_df,\n",
            "        models=['Random Forest', 'XGBoost', 'MLP'],\n",
            "        n_subsample=50000,\n",
            "        random_state=42,\n",
            "    )\n",
            "    ablation_df.to_csv(ablation_csv, index=False)\n",
            "\n",
            "print(ablation_df[['Target_Model', 'Removed_Group', 'MAE', 'Delta_MAE', 'Delta_R2']].head(12).to_string(index=False))\n"
        ]
    })

    # Cell 15
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 8 — Feature Ablation Visualizations"]
    })

    # Cell 16
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "df_plot_abl = ablation_df[ablation_df['Removed_Group'] != 'None'].copy()\n",
            "df_plot_abl['Group_Label'] = df_plot_abl['Removed_Group'].apply(lambda x: str(x).replace('_', ' ').title())\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(13, 6), dpi=150)\n",
            "palette_abl = {'Random Forest': '#2b5c8f', 'XGBoost': '#d95f02', 'MLP': '#7570b3'}\n",
            "sns.barplot(\n",
            "    data=df_plot_abl,\n",
            "    x='Group_Label',\n",
            "    y='Delta_MAE',\n",
            "    hue='Target_Model',\n",
            "    palette=palette_abl,\n",
            "    ax=ax\n",
            ")\n",
            "ax.set_title('Feature Group Ablation: Performance Degradation (Delta MAE on Validation Set)', fontweight='bold')\n",
            "ax.set_ylabel('Delta MAE (Higher = Greater Information Loss)')\n",
            "ax.set_xlabel('Removed Feature Group')\n",
            "ax.tick_params(axis='x', rotation=25)\n",
            "ax.legend(title='Model Architecture', loc='upper right')\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    })

    # Cell 17
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 9 — Dropout and Regularization Experiments"]
    })

    # Cell 18
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "df_dropout = pd.read_csv(res_dir / 'dropout_sweep_results.csv')\n",
            "df_wd = pd.read_csv(res_dir / 'weight_decay_results.csv')\n",
            "with open(res_dir / 'mc_dropout_uncertainty.json') as f:\n",
            "    mc_uncertainty = json.load(f)\n",
            "\n",
            "print(\"Dropout sweep results:\")\n",
            "print(df_dropout[['Dropout_Rate', 'Train_MAE', 'Val_MAE', 'Val_R2', 'Generalization_Gap_MAE']].to_string(index=False))\n",
            "\n",
            "print(\"\\nWeight decay results:\")\n",
            "print(df_wd[['Weight_Decay', 'Train_MAE', 'Val_MAE', 'Val_R2', 'Generalization_Gap_MAE']].to_string(index=False))\n",
            "\n",
            "print(\"\\nMC Dropout Uncertainty Summary:\")\n",
            "for k, v in mc_uncertainty.items():\n",
            "    print(f\"  {k}: {v}\")\n"
        ]
    })

    # Cell 19
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 10 — Regularization Response Visualizations"]
    })

    # Cell 20
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)\n",
            "\n",
            "axes[0].plot(df_dropout['Dropout_Rate'], df_dropout['Val_MAE'], 'o-', color='#7570b3', linewidth=2, label='Val MAE')\n",
            "axes[0].plot(df_dropout['Dropout_Rate'], df_dropout['Train_MAE'], 's--', color='#2b5c8f', linewidth=1.5, label='Train MAE')\n",
            "axes[0].set_title('MLP Dropout Rate Sweep vs MAE', fontweight='bold')\n",
            "axes[0].set_xlabel('Dropout Probability')\n",
            "axes[0].set_ylabel('MAE (CCME Scale)')\n",
            "axes[0].legend(loc='best')\n",
            "\n",
            "axes[1].plot(range(len(df_wd)), df_wd['Val_MAE'], '^-', color='#d95f02', linewidth=2, label='Val MAE')\n",
            "axes[1].set_xticks(range(len(df_wd)))\n",
            "axes[1].set_xticklabels([f\"{x:.0e}\" if x > 0 else \"0\" for x in df_wd['Weight_Decay']])\n",
            "axes[1].set_title('MLP AdamW Weight Decay vs MAE', fontweight='bold')\n",
            "axes[1].set_xlabel('L2 Regularization Coefficient')\n",
            "axes[1].set_ylabel('MAE (CCME Scale)')\n",
            "axes[1].legend(loc='best')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    })

    # Cell 21
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 11 — Subgroup Stratification Analysis"]
    })

    # Cell 22
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "df_country = pd.read_csv(res_dir / 'country_subgroup_metrics.csv')\n",
            "df_wb = pd.read_csv(res_dir / 'waterbody_subgroup_metrics.csv')\n",
            "df_temp = pd.read_csv(res_dir / 'temporal_subgroup_metrics.csv')\n",
            "\n",
            "c_show = df_country[df_country['Model'].isin(['Random Forest', 'Residual Hybrid', 'XGBoost', 'MLP'])]\n",
            "print(\"Cross-National Performance:\")\n",
            "print(c_show[['Category', 'Model', 'Sample_Count', 'MAE', 'R2']].to_string(index=False))\n",
            "\n",
            "print(\"\\nAquatic Habitat Performance (Top Categories for Residual Hybrid):\")\n",
            "wb_hybrid = df_wb[df_wb['Model'] == 'Residual Hybrid'].sort_values('Sample_Count', ascending=False)\n",
            "print(wb_hybrid[['Category', 'Sample_Count', 'MAE', 'RMSE', 'R2']].head(8).to_string(index=False))\n"
        ]
    })

    # Cell 23
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 12 — Geographic Performance Visualizations"]
    })

    # Cell 24
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "fig, ax = plt.subplots(figsize=(12, 5), dpi=150)\n",
            "palette_geo = {\n",
            "    'Random Forest': '#2b5c8f',\n",
            "    'Residual Hybrid': '#008080',\n",
            "    'XGBoost': '#d95f02',\n",
            "    'MLP': '#7570b3'\n",
            "}\n",
            "sns.barplot(\n",
            "    data=c_show,\n",
            "    x='Category',\n",
            "    y='MAE',\n",
            "    hue='Model',\n",
            "    palette=palette_geo,\n",
            "    ax=ax\n",
            ")\n",
            "ax.set_title('Cross-National Generalization Error (MAE by Jurisdiction)', fontweight='bold')\n",
            "ax.set_ylabel('MAE (CCME WQI Scale)')\n",
            "ax.set_xlabel('Country Jurisdiction')\n",
            "ax.legend(title='Model Architecture', loc='upper right')\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    })

    # Cell 25
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 13 — Computational Benchmarking and Latency Profiles"]
    })

    # Cell 26
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "df_comp = pd.read_csv(res_dir / 'computational_benchmarks.csv')\n",
            "df_scale = pd.read_csv(res_dir / 'throughput_scaling.csv')\n",
            "\n",
            "print(\"Computational Performance:\")\n",
            "print(df_comp[['Model', 'Single_Sample_Latency_ms', 'Throughput_samples_per_sec', 'Training_Time_sec', 'Throughput_relative_to_RF']].to_string(index=False))\n",
            "\n",
            "print(\"\\nBatch Size Scaling:\")\n",
            "print(df_scale[['Batch_Size', 'Throughput_samples_per_sec', 'Latency_per_batch_ms']].to_string(index=False))\n"
        ]
    })

    # Cell 27
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 14 — Latency and Throughput Visualizations"]
    })

    # Cell 28
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)\n",
            "palette_comp = {\n",
            "    'Random Forest': '#2b5c8f',\n",
            "    'Residual Hybrid': '#008080',\n",
            "    'XGBoost': '#d95f02',\n",
            "    'MLP': '#7570b3'\n",
            "}\n",
            "\n",
            "sns.barplot(data=df_comp, x='Model', y='Single_Sample_Latency_ms', hue='Model', palette=palette_comp, ax=axes[0], legend=False)\n",
            "axes[0].set_title('Single-Sample Latency (Lower is Better)', fontweight='bold')\n",
            "axes[0].set_ylabel('Latency (ms per sample)')\n",
            "axes[0].set_yscale('log')\n",
            "axes[0].tick_params(axis='x', rotation=20)\n",
            "\n",
            "sns.barplot(data=df_comp, x='Model', y='Throughput_samples_per_sec', hue='Model', palette=palette_comp, ax=axes[1], legend=False)\n",
            "axes[1].set_title('Batch Processing Throughput (Higher is Better)', fontweight='bold')\n",
            "axes[1].set_ylabel('Throughput (samples / sec)')\n",
            "axes[1].tick_params(axis='x', rotation=20)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    })

    # Cell 29
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Section 15 — Unified Evaluation Summary and Verification"]
    })

    # Cell 30
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "rf_mae = metrics_df.loc[metrics_df['Model'] == 'Random Forest', 'MAE'].values[0]\n",
            "hyb_mae = metrics_df.loc[metrics_df['Model'] == 'Residual Hybrid', 'MAE'].values[0]\n",
            "hyb_rmse = metrics_df.loc[metrics_df['Model'] == 'Residual Hybrid', 'RMSE'].values[0]\n",
            "rf_rmse = metrics_df.loc[metrics_df['Model'] == 'Random Forest', 'RMSE'].values[0]\n",
            "\n",
            "print(\"Verification Checks:\")\n",
            "print(f\"1. Total models evaluated: {len(metrics_df)}\")\n",
            "print(f\"2. Random Forest MAE: {rf_mae:.6f}, RMSE: {rf_rmse:.6f}\")\n",
            "print(f\"3. Residual Hybrid MAE: {hyb_mae:.6f}, RMSE: {hyb_rmse:.6f}\")\n",
            "print(f\"4. RMSE improvement of Residual Hybrid over RF: {(rf_rmse - hyb_rmse) / rf_rmse * 100:.2f}%\")\n",
            "print(f\"5. All model predictions finite and non-null: {all(np.all(np.isfinite(v)) for k, v in test_preds.items())}\")\n",
            "print(\"Task 5 evaluation completed successfully.\")\n"
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

    print("Executing notebook cells top-to-bottom...")
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
                        "text/plain": "<Figure size 1600x500 with 3 Axes>"
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
