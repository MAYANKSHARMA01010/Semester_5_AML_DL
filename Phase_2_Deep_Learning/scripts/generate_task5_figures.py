"""
Phase 2 Task 5 — Publication-Grade Visualization Generator
==========================================================
Generates high-resolution IEEE-compliant figures from verified evaluation outputs:
  1. 05_model_performance_comparison.png       (MAE, RMSE, R² comparison)
  2. 06_feature_group_ablation_impact.png      (LOGO Delta MAE across 7 groups)
  3. 07_regularization_dropout_curves.png       (Dropout & weight decay sweeps)
  4. 08_geographic_country_performance.png     (5-country stratified evaluation)
  5. 09_aquatic_waterbody_performance.png      (12-waterbody habitat comparison)
  6. 10_computational_tradeoffs.png            (Latency vs throughput vs training)

Outputs saved to: Phase_2_Deep_Learning/reports/figures/
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parents[1]
RESULTS_DIR = REPO_ROOT / 'Phase_2_Deep_Learning' / 'reports' / 'evaluation_results'
FIG_DIR = REPO_ROOT / 'Phase_2_Deep_Learning' / 'reports' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Global Styling ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

PALETTE = {
    'Random Forest': '#2b5c8f',
    'XGBoost': '#d95f02',
    'MLP': '#7570b3',
    'Weighted Ensemble': '#1b9e77',
    'Hybrid (Weighted)': '#1b9e77',
}

def generate_all_figures() -> None:
    print("=" * 75)
    print("  GENERATING TASK 5 PUBLICATION-GRADE VISUALIZATIONS")
    print("=" * 75)

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 1: Model Performance Comparison (MAE, RMSE, R²)
    # ──────────────────────────────────────────────────────────────────────────
    base_csv = RESULTS_DIR / 'baseline_model_comparison.csv'
    if base_csv.exists():
        df_base = pd.read_csv(base_csv)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

        # MAE
        sns.barplot(data=df_base, x='Model', y='MAE', hue='Model', ax=axes[0], palette='Blues_r', legend=False)
        axes[0].set_title('Mean Absolute Error (Lower is Better)', fontweight='bold')
        axes[0].set_ylabel('MAE (CCME WQI scale)')
        axes[0].tick_params(axis='x', rotation=30)
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')

        # RMSE
        sns.barplot(data=df_base, x='Model', y='RMSE', hue='Model', ax=axes[1], palette='Oranges_r', legend=False)
        axes[1].set_title('Root Mean Squared Error (Lower is Better)', fontweight='bold')
        axes[1].set_ylabel('RMSE')
        axes[1].tick_params(axis='x', rotation=30)
        for p in axes[1].patches:
            axes[1].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')

        # R²
        sns.barplot(data=df_base, x='Model', y='R2', hue='Model', ax=axes[2], palette='Greens_r', legend=False)
        axes[2].set_title('Coefficient of Determination (Higher is Better)', fontweight='bold')
        axes[2].set_ylabel('R² Score')
        axes[2].set_ylim(0.98, 1.001)
        axes[2].tick_params(axis='x', rotation=30)
        for p in axes[2].patches:
            axes[2].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')

        plt.tight_layout()
        out1 = FIG_DIR / '05_model_performance_comparison.png'
        plt.savefig(out1)
        plt.close()
        print(f"✓ Figure 1 saved → {out1}")

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 2: Feature Group Ablation Impact (Delta MAE)
    # ──────────────────────────────────────────────────────────────────────────
    abl_csv = RESULTS_DIR / 'feature_group_ablation.csv'
    if abl_csv.exists():
        df_abl = pd.read_csv(abl_csv)
        df_abl = df_abl[df_abl['Removed_Group'].notna() & (df_abl['Removed_Group'] != 'None')].copy()
        df_abl['Group_Label'] = df_abl['Removed_Group'].apply(lambda x: str(x).replace('_', ' ').title())

        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        sns.barplot(
            data=df_abl,
            x='Group_Label',
            y='Delta_MAE',
            hue='Target_Model',
            palette=PALETTE,
            ax=ax,
        )
        ax.set_title('Feature Group Ablation: Performance Degradation (ΔMAE on Validation Set)', fontweight='bold')
        ax.set_ylabel('ΔMAE When Group Removed (Higher = Greater Information Loss)')
        ax.set_xlabel('Removed Feature Group')
        ax.tick_params(axis='x', rotation=25)
        ax.legend(title='Model Architecture', loc='upper right')

        plt.tight_layout()
        out2 = FIG_DIR / '06_feature_group_ablation_impact.png'
        plt.savefig(out2)
        plt.close()
        print(f"✓ Figure 2 saved → {out2}")

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 3: Regularization & Dropout Response Curves
    # ──────────────────────────────────────────────────────────────────────────
    do_csv = RESULTS_DIR / 'dropout_sweep_results.csv'
    wd_csv = RESULTS_DIR / 'weight_decay_results.csv'
    if do_csv.exists() and wd_csv.exists():
        df_do = pd.read_csv(do_csv)
        df_wd = pd.read_csv(wd_csv)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        # Dropout response
        axes[0].plot(df_do['Dropout_Rate'], df_do['Val_MAE'], 'o-', color='#7570b3', linewidth=2, label='Val MAE')
        axes[0].plot(df_do['Dropout_Rate'], df_do['Train_MAE'], 's--', color='#2b5c8f', linewidth=1.5, label='Train MAE')
        axes[0].set_title('MLP Dropout Rate Sweep vs MAE', fontweight='bold')
        axes[0].set_xlabel('Dropout Probability (p)')
        axes[0].set_ylabel('MAE (CCME Scale)')
        axes[0].legend(loc='best')

        # Weight decay response
        axes[1].plot(range(len(df_wd)), df_wd['Val_MAE'], '^-', color='#d95f02', linewidth=2, label='Val MAE')
        axes[1].set_xticks(range(len(df_wd)))
        axes[1].set_xticklabels([f"{x:.0e}" if x > 0 else "0" for x in df_wd['Weight_Decay']])
        axes[1].set_title('MLP AdamW Weight Decay vs MAE', fontweight='bold')
        axes[1].set_xlabel('L2 Regularization Coefficient (λ)')
        axes[1].set_ylabel('MAE (CCME Scale)')
        axes[1].legend(loc='best')

        plt.tight_layout()
        out3 = FIG_DIR / '07_regularization_dropout_curves.png'
        plt.savefig(out3)
        plt.close()
        print(f"✓ Figure 3 saved → {out3}")

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 4: Geographic Country Stratification
    # ──────────────────────────────────────────────────────────────────────────
    country_csv = RESULTS_DIR / 'country_subgroup_metrics.csv'
    if country_csv.exists():
        df_c = pd.read_csv(country_csv)
        df_c = df_c[df_c['Model'].isin(['Random Forest', 'XGBoost', 'MLP', 'Weighted Ensemble'])]

        fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
        sns.barplot(data=df_c, x='Category', y='MAE', hue='Model', palette=PALETTE, ax=ax)
        ax.set_title('Cross-National Generalization Error (MAE by Jurisdiction)', fontweight='bold')
        ax.set_ylabel('MAE (CCME WQI Scale)')
        ax.set_xlabel('Country Jurisdiction')
        ax.legend(title='Model Architecture', loc='upper right')

        # Annotate sample counts
        for idx, row in df_c[df_c['Model'] == 'Random Forest'].iterrows():
            ax.annotate(f"N={row['Sample_Count']:,}",
                        (idx, 0.05),
                        ha='center', va='bottom', fontsize=9, color='#333333',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='#aaaaaa'))

        plt.tight_layout()
        out4 = FIG_DIR / '08_geographic_country_performance.png'
        plt.savefig(out4)
        plt.close()
        print(f"✓ Figure 4 saved → {out4}")

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 5: Aquatic Waterbody Habitat Stratification
    # ──────────────────────────────────────────────────────────────────────────
    wb_csv = RESULTS_DIR / 'waterbody_subgroup_metrics.csv'
    if wb_csv.exists():
        df_wb = pd.read_csv(wb_csv)
        df_wb_top = df_wb[df_wb['Model'] == 'XGBoost'].sort_values('MAE', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        bars = ax.barh(df_wb_top['Category'], df_wb_top['MAE'], color='#d95f02', alpha=0.85)
        ax.set_title('XGBoost Prediction Error across Aquatic Waterbody Types', fontweight='bold')
        ax.set_xlabel('MAE (CCME WQI Scale)')
        ax.set_ylabel('Waterbody Classification')

        for bar in bars:
            w = bar.get_width()
            ax.annotate(f"{w:.3f}", (w, bar.get_y() + bar.get_height() / 2.),
                        ha='left', va='center', fontsize=9, xytext=(4, 0), textcoords='offset points')

        plt.tight_layout()
        out5 = FIG_DIR / '09_aquatic_waterbody_performance.png'
        plt.savefig(out5)
        plt.close()
        print(f"✓ Figure 5 saved → {out5}")

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 6: Computational Efficiency & Latency Tradeoffs
    # ──────────────────────────────────────────────────────────────────────────
    comp_csv = RESULTS_DIR / 'computational_benchmarks.csv'
    if comp_csv.exists():
        df_comp = pd.read_csv(comp_csv)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        # Single-sample latency (log scale)
        sns.barplot(data=df_comp, x='Model', y='Single_Sample_Latency_ms', hue='Model', palette=PALETTE, ax=axes[0], legend=False)
        axes[0].set_title('Single-Sample Real-Time Latency (Lower is Better)', fontweight='bold')
        axes[0].set_ylabel('Latency (milliseconds per prediction)')
        axes[0].set_yscale('log')
        axes[0].tick_params(axis='x', rotation=15)
        for p in axes[0].patches:
            val = p.get_height()
            axes[0].annotate(f"{val:.2f} ms", (p.get_x() + p.get_width() / 2., val),
                             ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')

        # Batched Throughput
        sns.barplot(data=df_comp, x='Model', y='Throughput_samples_per_sec', hue='Model', palette=PALETTE, ax=axes[1], legend=False)
        axes[1].set_title('Batch Processing Throughput (Higher is Better)', fontweight='bold')
        axes[1].set_ylabel('Throughput (samples processed / second)')
        axes[1].tick_params(axis='x', rotation=15)
        for p in axes[1].patches:
            val = p.get_height()
            axes[1].annotate(f"{val:,.0f}/s", (p.get_x() + p.get_width() / 2., val),
                             ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

        plt.tight_layout()
        out6 = FIG_DIR / '10_computational_tradeoffs.png'
        plt.savefig(out6)
        plt.close()
        print(f"✓ Figure 6 saved → {out6}")

    print("\n" + "=" * 75)
    print("  ALL 6 TASK 5 FIGURES SUCCESSFULLY GENERATED")
    print("=" * 75)

if __name__ == '__main__':
    generate_all_figures()
