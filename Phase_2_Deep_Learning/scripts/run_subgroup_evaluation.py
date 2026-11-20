"""
Phase 2 Task 5 — Subgroup Stratification Evaluation Runner
==========================================================
Evaluates model generalizability across:
  1. Geographic Strata: Country (England, USA, Ireland, China, Canada)
  2. Aquatic Strata: Waterbody Type (River, Effluent, Lake, Estuarine, etc.)
  3. Temporal Eras: Decadal epochs (< 1990, 1990-2005, > 2005)
  4. Seasonal Cycles: Spring, Summer, Autumn, Winter

Outputs saved to:
  Phase_2_Deep_Learning/reports/evaluation_results/country_subgroup_metrics.csv
  Phase_2_Deep_Learning/reports/evaluation_results/country_subgroup_metrics.json
  Phase_2_Deep_Learning/reports/evaluation_results/waterbody_subgroup_metrics.csv
  Phase_2_Deep_Learning/reports/evaluation_results/waterbody_subgroup_metrics.json
  Phase_2_Deep_Learning/reports/evaluation_results/temporal_subgroup_metrics.csv
  Phase_2_Deep_Learning/reports/evaluation_results/temporal_subgroup_metrics.json
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# ── Ensure project root is on sys.path ────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Phase_2_Deep_Learning.src.evaluation import ModelEvaluator, SubgroupEvaluator

def assign_season(month: int) -> str:
    """Map calendar month to meteorological season."""
    if month in (3, 4, 5):
        return 'Spring'
    elif month in (6, 7, 8):
        return 'Summer'
    elif month in (9, 10, 11):
        return 'Autumn'
    else:
        return 'Winter'

def assign_era(year: int) -> str:
    """Map calendar year to decadal telemetry epoch."""
    if year < 1990:
        return 'Pre-1990 (Historical)'
    elif year <= 2005:
        return '1990-2005 (Expansion)'
    else:
        return 'Post-2005 (Modern Telemetry)'

def run_subgroup_evaluation() -> None:
    print("=" * 75)
    print("  PHASE 2 TASK 5: SUBGROUP STRATIFICATION EVALUATION")
    print("=" * 75)

    evaluator = ModelEvaluator()
    evaluator.load_all_models()

    test_path = REPO_ROOT / 'Phase_1_Machine_Learning' / 'data' / 'GeneratedOutput' / '02.Preprocessing&FeatureSelection' / 'test.parquet'
    print(f"\nLoading test split from: {test_path}")
    test_df = pd.read_parquet(test_path)
    print(f"Loaded {len(test_df):,} test records.")

    # Add temporal stratification helper columns
    test_df['season'] = test_df['month'].apply(assign_season)
    test_df['era'] = test_df['year'].apply(assign_era)

    print("\nGenerating model predictions across full test set...")
    t0 = time.time()
    preds = evaluator.predict_all(test_df)
    print(f"Predictions generated in {time.time() - t0:.2f}s")

    results_dir = REPO_ROOT / 'Phase_2_Deep_Learning' / 'reports' / 'evaluation_results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Country Subgroup Evaluation ────────────────────────────────────────
    print("\n" + "-" * 75)
    print("  1. Geographic Stratification (Country)")
    print("-" * 75)
    country_df = SubgroupEvaluator.evaluate_by_group(
        df=test_df,
        preds_dict=preds,
        group_col='country',
        min_samples=100,
    )
    country_show = country_df[country_df['Model'].isin(['Random Forest', 'XGBoost', 'MLP', 'Weighted Ensemble'])]
    print(country_show[['Category', 'Model', 'Sample_Count', 'MAE', 'R2']].to_string(index=False))

    country_df.to_csv(results_dir / 'country_subgroup_metrics.csv', index=False)
    with open(results_dir / 'country_subgroup_metrics.json', 'w') as f:
        json.dump(country_df.to_dict(orient='records'), f, indent=2)
    print(f"✓ Saved country metrics → {results_dir / 'country_subgroup_metrics.csv'}")

    # ── 2. Waterbody Subgroup Evaluation ──────────────────────────────────────
    print("\n" + "-" * 75)
    print("  2. Aquatic Habitat Stratification (Waterbody Type)")
    print("-" * 75)
    wb_df = SubgroupEvaluator.evaluate_by_group(
        df=test_df,
        preds_dict=preds,
        group_col='waterbody_type',
        min_samples=100,
    )
    wb_show = wb_df[wb_df['Model'] == 'XGBoost'].sort_values('Sample_Count', ascending=False)
    print(wb_show[['Category', 'Sample_Count', 'MAE', 'RMSE', 'R2']].to_string(index=False))

    wb_df.to_csv(results_dir / 'waterbody_subgroup_metrics.csv', index=False)
    with open(results_dir / 'waterbody_subgroup_metrics.json', 'w') as f:
        json.dump(wb_df.to_dict(orient='records'), f, indent=2)
    print(f"✓ Saved waterbody metrics → {results_dir / 'waterbody_subgroup_metrics.csv'}")

    # ── 3. Temporal Stratification (Eras & Seasons) ───────────────────────────
    print("\n" + "-" * 75)
    print("  3. Temporal Stratification (Decadal Eras & Meteorological Seasons)")
    print("-" * 75)
    era_df = SubgroupEvaluator.evaluate_by_group(
        df=test_df,
        preds_dict=preds,
        group_col='era',
        min_samples=100,
    )
    season_df = SubgroupEvaluator.evaluate_by_group(
        df=test_df,
        preds_dict=preds,
        group_col='season',
        min_samples=100,
    )
    temporal_df = pd.concat([era_df, season_df], ignore_index=True)
    temp_show = temporal_df[temporal_df['Model'] == 'Random Forest']
    print(temp_show[['Subgroup_Field', 'Category', 'Sample_Count', 'MAE', 'R2']].to_string(index=False))

    temporal_df.to_csv(results_dir / 'temporal_subgroup_metrics.csv', index=False)
    with open(results_dir / 'temporal_subgroup_metrics.json', 'w') as f:
        json.dump(temporal_df.to_dict(orient='records'), f, indent=2)
    print(f"✓ Saved temporal metrics → {results_dir / 'temporal_subgroup_metrics.csv'}")

    print("\n" + "=" * 75)
    print("  SUBGROUP STRATIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 75)

if __name__ == '__main__':
    run_subgroup_evaluation()
