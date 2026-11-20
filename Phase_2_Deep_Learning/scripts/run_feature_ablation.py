"""
Phase 2 Task 5 — Feature Group Ablation Runner
==============================================
Evaluates Leave-One-Group-Out (LOGO) marginal feature contributions across all 7
geochemical and temporal feature groups for Random Forest, XGBoost, and MLP.

Outputs saved to:
  Phase_2_Deep_Learning/reports/evaluation_results/feature_group_ablation.csv
  Phase_2_Deep_Learning/reports/evaluation_results/feature_group_ablation.json
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sys
import time
from pathlib import Path
import pandas as pd

# ── Ensure project root is on sys.path ────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Phase_2_Deep_Learning.src.evaluation import ModelEvaluator, FeatureAblationEngine

def run_feature_ablation(n_subsample: int = 50000) -> pd.DataFrame:
    print("=" * 75)
    print("  PHASE 2 TASK 5: SYSTEMATIC 7-GROUP FEATURE ABLATION STUDY")
    print("=" * 75)

    evaluator = ModelEvaluator()
    evaluator.load_all_models()

    val_path = REPO_ROOT / 'Phase_1_Machine_Learning' / 'data' / 'GeneratedOutput' / '02.Preprocessing&FeatureSelection' / 'val.parquet'
    print(f"\nLoading validation split from: {val_path}")
    val_df = pd.read_parquet(val_path)
    print(f"Loaded {len(val_df):,} validation records.")

    engine = FeatureAblationEngine(evaluator)
    print(f"\nExecuting Leave-One-Group-Out (LOGO) ablation (subsample: {n_subsample:,} rows)...")
    t0 = time.time()
    ablation_df = engine.run_all_models_ablation(
        val_df=val_df,
        models=['Random Forest', 'XGBoost', 'MLP'],
        n_subsample=n_subsample,
        random_state=42,
    )
    print(f"Ablation completed across all models in {time.time() - t0:.2f}s")

    print("\n" + "=" * 75)
    print("  FEATURE ABLATION RESULTS (TOP DEGRADATIONS)")
    print("=" * 75)
    cols_show = ['Target_Model', 'Removed_Group', 'MAE', 'Delta_MAE', 'Delta_R2']
    print(ablation_df[cols_show].to_string(index=False))
    print("=" * 75)

    # Save outputs
    results_dir = REPO_ROOT / 'Phase_2_Deep_Learning' / 'reports' / 'evaluation_results'
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / 'feature_group_ablation.csv'
    ablation_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV ablation results: {csv_path}")

    json_path = results_dir / 'feature_group_ablation.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ablation_df.to_dict(orient='records'), f, indent=2)
    print(f"✓ Saved JSON ablation results: {json_path}")

    return ablation_df

if __name__ == '__main__':
    run_feature_ablation()
