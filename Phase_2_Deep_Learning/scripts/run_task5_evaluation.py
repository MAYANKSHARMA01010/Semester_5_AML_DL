"""
Phase 2 Task 5 — Master Evaluation Script
=========================================
Runs unified baseline evaluation across all available models on the common
test partition (test.parquet, 422,943 records) with zero data leakage.

Saves structured CSV and JSON metric summaries to:
  Phase_2_Deep_Learning/reports/evaluation_results/baseline_model_comparison.csv
  Phase_2_Deep_Learning/reports/evaluation_results/baseline_model_comparison.json
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
import numpy as np

# ── Ensure project root is on sys.path ────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Phase_2_Deep_Learning.src.evaluation import ModelEvaluator

def run_baseline_evaluation() -> pd.DataFrame:
    print("=" * 75)
    print("  PHASE 2 TASK 5: UNIFIED MODEL BASELINE EVALUATION (TEST SET)")
    print("=" * 75)

    evaluator = ModelEvaluator()
    evaluator.load_all_models()

    test_path = REPO_ROOT / 'Phase_1_Machine_Learning' / 'data' / 'GeneratedOutput' / '02.Preprocessing&FeatureSelection' / 'test.parquet'
    print(f"\nLoading test split from: {test_path}")
    t0 = time.time()
    test_df = pd.read_parquet(test_path)
    print(f"Loaded {len(test_df):,} test records in {time.time() - t0:.2f}s")

    print("\nGenerating model predictions on test split...")
    t_pred = time.time()
    preds = evaluator.predict_all(test_df)
    print(f"All model predictions generated in {time.time() - t_pred:.2f}s")

    print("\nCalculating official regression metrics (MAE, RMSE, R²)...")
    metrics_df = evaluator.evaluate_predictions(preds)

    print("\n" + "=" * 75)
    print("  UNIFIED TEST METRICS SUMMARY")
    print("=" * 75)
    print(metrics_df.to_string(index=False))
    print("=" * 75)

    # Save outputs
    results_dir = REPO_ROOT / 'Phase_2_Deep_Learning' / 'reports' / 'evaluation_results'
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / 'baseline_model_comparison.csv'
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV metrics: {csv_path}")

    json_path = results_dir / 'baseline_model_comparison.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_df.to_dict(orient='records'), f, indent=2)
    print(f"✓ Saved JSON metrics: {json_path}")

    return metrics_df

if __name__ == '__main__':
    run_baseline_evaluation()
