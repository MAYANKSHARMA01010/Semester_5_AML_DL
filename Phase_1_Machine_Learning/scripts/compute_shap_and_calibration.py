#!/usr/bin/env python3
"""
Compute TreeSHAP values for XGBoost Regressor (Phase 1 Machine Learning).
Generates:
1. data/GeneratedOutput/04.ModelEvaluation/shap_summary.png
2. data/GeneratedOutput/04.ModelEvaluation/shap_top_features.json
"""

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import shap

DATA_DIR = REPO_DIR / "data" / "GeneratedOutput"
OUT_DIR = DATA_DIR / "04.ModelEvaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_shap_analysis():
    print("--- 1. Computing TreeSHAP Values for XGBoost Regressor ---")
    xgb_path = DATA_DIR / "03.ModelTraining" / "xgboost_model.json"
    meta_path = DATA_DIR / "02.Preprocessing&FeatureSelection" / "preprocessing_metadata.json"
    val_path = DATA_DIR / "02.Preprocessing&FeatureSelection" / "val.parquet"

    features = json.loads(meta_path.read_text())["input_features_ordered"]
    xgb_model = XGBRegressor()
    xgb_model.load_model(str(xgb_path))

    val_df = pd.read_parquet(val_path)
    val_sample = val_df[features].sample(n=250, random_state=42)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(val_sample)

    fig = plt.figure(figsize=(10, 8), dpi=150)
    shap.summary_plot(shap_values.values, val_sample, max_display=15, show=False)
    plt.title("TreeSHAP Feature Attribution: Top 15 Predictors (Phase 1 XGBoost)", fontsize=13, pad=15)
    plt.tight_layout()
    shap_fig_path = OUT_DIR / "shap_summary.png"
    plt.savefig(shap_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"SHAP summary figure saved to: {shap_fig_path}", flush=True)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:10]
    top_shap_features = {features[i]: float(mean_abs_shap[i]) for i in top_indices}
    print(f"Top SHAP features: {top_shap_features}")

    out_json = OUT_DIR / "shap_top_features.json"
    out_json.write_text(json.dumps(top_shap_features, indent=2))
    print(f"SHAP top features saved to: {out_json}")
    return top_shap_features


def main():
    run_shap_analysis()
    print("\nPhase 1 TreeSHAP attribution analysis completed successfully.")


if __name__ == "__main__":
    main()
