"""
Phase 2 Deep Learning — Task 5: Model Evaluation & Ablation Studies Module
==========================================================================
Provides unified evaluation adapters, feature ablation engines, subgroup
stratification analyzers, and computational benchmarking utilities for:
  - Random Forest Regressor (Phase 1)
  - XGBoost Regressor (Phase 1)
  - WaterQualityMLP (Phase 2 Task 2/3)
  - Hybrid ML + DL Ensemble (Phase 2 Task 4)

All metrics strictly adhere to project definitions: MAE, RMSE, and R².
"""

from __future__ import annotations

import os
# Prevent OpenMP thread runtime conflicts between scikit-learn and XGBoost on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Ensure project root is accessible on sys.path ─────────────────────────────
_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Phase_1_Machine_Learning.src.models import load_tree_model
from Phase_2_Deep_Learning.src.dataset import (
    load_feature_config,
    load_raw_splits,
    load_transformers,
    prepare_tensors,
    get_default_paths,
)
from Phase_2_Deep_Learning.src.dl_models import WaterQualityMLP, load_model_checkpoint
from Phase_2_Deep_Learning.src.hybrid_models import (
    HybridConfig,
    HybridEnsemble,
    WeightedEnsemble,
    ResidualMLPNet,
    load_hybrid_checkpoint,
)
from Phase_2_Deep_Learning.src.training import get_device, compute_regression_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Feature Group Definitions
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_GROUPS: Dict[str, List[str]] = {
    "base_sensors": [
        "ph", "temperature_c", "do_mg_l", "bod_mg_l",
        "ammonia_mg_l", "orthophosphate_mg_l", "nitrogen_mg_l", "nitrate_mg_l"
    ],
    "stoichiometry": [
        "np_ratio", "bod_do_ratio", "do_deficit", "do_pct_sat",
        "ph_neutral_deviation", "nitrate_nitrogen_ratio", "unionized_ammonia"
    ],
    "log1p_normalized": [
        "log1p_ammonia", "log1p_bod", "log1p_orthophosphate", "log1p_nitrogen",
        "log1p_np_ratio", "log1p_bod_do_ratio", "log1p_unionized_ammonia"
    ],
    "historical_memory": [
        "hist_do_lag1", "hist_temp_lag1", "hist_bod_lag1",
        "hist_ortho_lag1", "hist_do_rolling_3", "hist_temp_rolling_3"
    ],
    "environmental_stress": [
        "exceedance_counter"
    ],
    "contextual_categorical": [
        "country_code", "waterbody_type_code"
    ],
    "temporal_cyclical": [
        "year", "month", "quarter", "month_sin", "month_cos",
        "day_of_year_sin", "day_of_year_cos"
    ]
}


# ══════════════════════════════════════════════════════════════════════════════
# Unified Model Evaluator
# ══════════════════════════════════════════════════════════════════════════════

class ModelEvaluator:
    """
    Unified evaluation adapter for all 4 Phase 1 & 2 models:
    RF, XGBoost, WaterQualityMLP, and HybridEnsemble.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        paths: Optional[Dict[str, Path]] = None,
    ):
        self.device = device or get_device()
        self.paths = paths or get_default_paths()
        self.feature_cfg = load_feature_config(self.paths['feature_config'])
        self.scaler, self.encoder = load_transformers(self.paths['artifacts_dir'])
        
        self.num_features = self.feature_cfg['numerical_features']
        self.cat_features = self.feature_cfg['categorical_features']
        
        # Load canonical feature ordering from preprocessing metadata
        meta_p = (
            self.paths['phase1_dir']
            / 'data'
            / 'GeneratedOutput'
            / '02.Preprocessing&FeatureSelection'
            / 'preprocessing_metadata.json'
        )
        if meta_p.exists():
            with open(meta_p, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            self.all_tree_features = meta.get(
                'input_features_ordered', self.num_features + self.cat_features
            )
        else:
            self.all_tree_features = self.num_features + self.cat_features

        self.target_col = self.feature_cfg['target_variable']

        self.rf_model = None
        self.xgb_model = None
        self.mlp_model = None
        self.hybrid_model = None
        self.weighted_ensemble = None

    def load_all_models(
        self,
        rf_path: Optional[Path] = None,
        xgb_path: Optional[Path] = None,
        mlp_path: Optional[Path] = None,
        hybrid_path: Optional[Path] = None,
    ) -> None:
        """Load RF, XGBoost, MLP, and Hybrid Ensemble from saved checkpoints."""
        p1_gen = self.paths['phase1_dir'] / 'data' / 'GeneratedOutput' / '03.ModelTraining'
        p2_models = self.paths['phase2_dir'] / 'models'

        rf_p = rf_path or (p1_gen / 'random_forest_model.joblib')
        xgb_p = xgb_path or (p1_gen / 'xgboost_model.json')
        mlp_p = mlp_path or (p2_models / 'mlp_best_checkpoint.pt')
        hyb_p = hybrid_path or (p2_models / 'hybrid_ensemble_checkpoint.pt')

        if rf_p.exists():
            self.rf_model = load_tree_model(str(rf_p), model_type='rf')
            print(f"✓ RF loaded from {rf_p}")
        else:
            print(f"⚠ RF artifact not found at {rf_p}")

        if xgb_p.exists():
            self.xgb_model = load_tree_model(str(xgb_p), model_type='xgb')
            print(f"✓ XGBoost loaded from {xgb_p}")
        else:
            print(f"⚠ XGBoost artifact not found at {xgb_p}")

        if mlp_p.exists():
            self.mlp_model, _ = load_model_checkpoint(mlp_p, device=self.device)
            self.mlp_model.eval()
            print(f"✓ MLP loaded from {mlp_p}")
        else:
            print(f"⚠ MLP checkpoint not found at {mlp_p}")

        if hyb_p.exists():
            self.hybrid_model, _, _ = load_hybrid_checkpoint(hyb_p, device=self.device)
            self.hybrid_model.eval()
            print(f"✓ Hybrid Ensemble loaded from {hyb_p}")
        else:
            if hybrid_path is not None:
                raise FileNotFoundError(f"Hybrid checkpoint not found at {hyb_p}")
            print(f"⚠ Hybrid Ensemble checkpoint not found at {hyb_p}")

        # Instantiate default WeightedEnsemble
        self.weighted_ensemble = WeightedEnsemble(rf_weight=0.4, xgb_weight=0.4, mlp_weight=0.2)

    def prepare_inputs(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, torch.Tensor, np.ndarray]:
        """
        Prepare inputs for tree models, MLP, and extract ground truth targets.
        
        Returns:
            X_tree: pd.DataFrame shape (N, 38) with feature names preserved
            X_dl_np: np.ndarray shape (N, 53)
            X_dl_tensor: torch.Tensor shape (N, 53) on self.device
            y_true: np.ndarray shape (N,)
        """
        # 1. Tree input (38 unscaled numerical + categorical codes as DataFrame)
        X_tree = df[self.all_tree_features].copy()

        # 2. DL input (53 standardized numerical + OHE binary features)
        x_num, x_cat, x_all, y_t = prepare_tensors(
            df, self.scaler, self.encoder,
            self.num_features, self.cat_features, self.target_col
        )
        X_dl_np = x_all.numpy()
        X_dl_tensor = x_all.to(self.device)
        y_true = y_t.numpy().flatten()

        return X_tree, X_dl_np, X_dl_tensor, y_true

    def predict_all(
        self, df: pd.DataFrame, batch_size: int = 16384
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions for all available models on the given DataFrame.
        Uses batched execution for memory safety on large evaluation sets.
        """
        X_tree, X_dl_np, X_dl_tensor, y_true = self.prepare_inputs(df)
        n_samples = len(df)
        preds: Dict[str, np.ndarray] = {'y_true': y_true}

        # 1. Random Forest
        if self.rf_model is not None:
            rf_preds = []
            for i in range(0, n_samples, batch_size):
                rf_preds.append(self.rf_model.predict(X_tree[i:i+batch_size]))
            preds['Random Forest'] = np.concatenate(rf_preds)

        # 2. XGBoost
        if self.xgb_model is not None:
            xgb_preds = []
            for i in range(0, n_samples, batch_size):
                xgb_preds.append(self.xgb_model.predict(X_tree[i:i+batch_size]))
            preds['XGBoost'] = np.concatenate(xgb_preds)

        # 3. WaterQualityMLP
        if self.mlp_model is not None:
            self.mlp_model.eval()
            mlp_preds = []
            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    batch_x = X_dl_tensor[i:i+batch_size]
                    out = self.mlp_model(batch_x)
                    mlp_preds.append(out.detach().cpu().numpy().flatten())
            preds['MLP'] = np.concatenate(mlp_preds)

        # 4. Base Tree Ensemble (0.5 RF + 0.5 XGBoost)
        if ('Random Forest' in preds and 'XGBoost' in preds):
            preds['Base Tree Ensemble'] = 0.5 * (preds['Random Forest'] + preds['XGBoost'])

        # 5. Weighted Ensemble (RF + XGBoost + MLP: 0.4 RF + 0.4 XGBoost + 0.2 MLP)
        if ('Random Forest' in preds and 'XGBoost' in preds and 'MLP' in preds):
            preds['Weighted Ensemble'] = self.weighted_ensemble.predict(
                preds['Random Forest'], preds['XGBoost'], preds['MLP']
            )

        # 6. Residual Hybrid (Tree Base + Residual MLP Correction)
        if ('Base Tree Ensemble' in preds and self.hybrid_model is not None):
            preds['Residual Hybrid'] = self.hybrid_model.predict_numpy(
                X_dl_np, preds['Base Tree Ensemble'], device=self.device
            )

        return preds

    def evaluate_predictions(
        self, preds_dict: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Calculate regression metrics for each model in predictions dictionary."""
        y_true = preds_dict['y_true']
        rows = []
        for model_name, y_pred in preds_dict.items():
            if model_name == 'y_true':
                continue
            m = compute_regression_metrics(y_true, y_pred)
            med_ae = float(np.median(np.abs(y_true - y_pred)))
            max_ae = float(np.max(np.abs(y_true - y_pred)))
            var_y = float(np.var(y_true))
            evs = float(1.0 - np.var(y_true - y_pred) / var_y) if var_y > 1e-12 else 0.0
            rows.append({
                'Model': model_name,
                'MAE': m['mae'],
                'RMSE': m['rmse'],
                'R2': m['r2'],
                'MedAE': med_ae,
                'MaxAE': max_ae,
                'ExplainedVariance': evs,
                'Samples': len(y_true),
            })
        return pd.DataFrame(rows).sort_values('MAE').reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Feature Ablation Engine
# ══════════════════════════════════════════════════════════════════════════════

class FeatureAblationEngine:
    """
    Executes systematic feature ablation studies across the 7 recognized groups.
    Evaluates Leave-One-Group-Out (LOGO) impact on validation performance.
    """

    def __init__(self, evaluator: ModelEvaluator):
        self.evaluator = evaluator
        self.feature_groups = FEATURE_GROUPS

    def run_leave_one_group_out(
        self,
        val_df: pd.DataFrame,
        model_type: str = 'xgb',
        n_subsample: Optional[int] = 50000,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Evaluate performance degradation when setting each feature group to its mean/baseline.
        Preserves model weights while assessing marginal information contribution.
        """
        if n_subsample and len(val_df) > n_subsample:
            eval_df = val_df.sample(n=n_subsample, random_state=random_state).copy()
        else:
            eval_df = val_df.copy()

        # 1. Full-feature baseline
        base_preds = self.evaluator.predict_all(eval_df)
        if model_type == 'xgb':
            base_name = 'XGBoost'
        elif model_type == 'rf':
            base_name = 'Random Forest'
        else:
            base_name = 'MLP'
        base_m = compute_regression_metrics(base_preds['y_true'], base_preds[base_name])

        results = [{
            'Ablation': 'Full Feature Baseline',
            'Removed_Group': 'None',
            'Features_Removed': 0,
            'MAE': base_m['mae'],
            'RMSE': base_m['rmse'],
            'R2': base_m['r2'],
            'Delta_MAE': 0.0,
            'Delta_RMSE': 0.0,
            'Delta_R2': 0.0,
        }]

        # 2. Iterate each feature group
        for group_name, features in self.feature_groups.items():
            ablated_df = eval_df.copy()
            # Replace ablated features with their column mean (zero-information proxy)
            for f in features:
                if f in ablated_df.columns:
                    ablated_df[f] = ablated_df[f].mean()

            ablated_preds = self.evaluator.predict_all(ablated_df)
            m = compute_regression_metrics(ablated_preds['y_true'], ablated_preds[base_name])

            results.append({
                'Ablation': f"Without {group_name.replace('_', ' ').title()}",
                'Removed_Group': group_name,
                'Features_Removed': len(features),
                'MAE': m['mae'],
                'RMSE': m['rmse'],
                'R2': m['r2'],
                'Delta_MAE': m['mae'] - base_m['mae'],
                'Delta_RMSE': m['rmse'] - base_m['rmse'],
                'Delta_R2': m['r2'] - base_m['r2'],
            })

        df_res = pd.DataFrame(results)
        return df_res.sort_values('Delta_MAE', ascending=False).reset_index(drop=True)

    def run_all_models_ablation(
        self,
        val_df: pd.DataFrame,
        models: Optional[List[str]] = None,
        n_subsample: Optional[int] = 50000,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Run Leave-One-Group-Out ablation for multiple models (RF, XGBoost, MLP)
        and return a unified DataFrame with performance degradation per model.
        """
        target_models = models or ['Random Forest', 'XGBoost', 'MLP']
        all_dfs = []
        for m_name in target_models:
            m_type = 'xgb' if 'xgb' in m_name.lower() else ('rf' if 'forest' in m_name.lower() else 'mlp')
            df_m = self.run_leave_one_group_out(
                val_df=val_df,
                model_type=m_type,
                n_subsample=n_subsample,
                random_state=random_state,
            )
            df_m['Target_Model'] = m_name
            all_dfs.append(df_m)
        return pd.concat(all_dfs, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Subgroup Stratification Evaluator
# ══════════════════════════════════════════════════════════════════════════════

class SubgroupEvaluator:
    """
    Evaluates model performance across geographic (country), aquatic (waterbody),
    and temporal (year/season) strata with sample size safeguards.
    """

    @staticmethod
    def evaluate_by_group(
        df: pd.DataFrame,
        preds_dict: Dict[str, np.ndarray],
        group_col: str,
        min_samples: int = 100,
    ) -> pd.DataFrame:
        """
        Compute MAE, RMSE, R² per category in group_col.
        Groups with samples < min_samples are flagged with low_sample_warning.
        """
        y_true_all = preds_dict['y_true']
        groups = df[group_col].values
        unique_groups = sorted(list(set(groups)))

        results = []
        for g in unique_groups:
            mask = (groups == g)
            count = int(np.sum(mask))
            if count == 0:
                continue

            y_true_sub = y_true_all[mask]
            for model_name, y_pred_all in preds_dict.items():
                if model_name == 'y_true':
                    continue
                y_pred_sub = y_pred_all[mask]
                m = compute_regression_metrics(y_true_sub, y_pred_sub)
                results.append({
                    'Subgroup_Field': group_col,
                    'Category': str(g),
                    'Model': model_name,
                    'Sample_Count': count,
                    'MAE': m['mae'],
                    'RMSE': m['rmse'],
                    'R2': m['r2'],
                    'Reliable_Sample': count >= min_samples,
                })

        return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# Computational Profiler
# ══════════════════════════════════════════════════════════════════════════════

class ComputationalProfiler:
    """
    Profiles inference latency, batched throughput, and hardware benchmarks.
    """

    @staticmethod
    def benchmark_inference(
        evaluator: ModelEvaluator,
        df_sample: pd.DataFrame,
        n_repeats: int = 50,
    ) -> pd.DataFrame:
        """
        Measure single-sample latency (batch_size=1) and throughput (samples/sec).
        """
        X_tree, X_dl_np, X_dl_tensor, _ = evaluator.prepare_inputs(df_sample)
        n_samples = len(df_sample)
        results = []

        # 1. Random Forest
        if evaluator.rf_model is not None:
            # Warmup
            evaluator.rf_model.predict(X_tree[:10])
            # Single sample latency
            t0 = time.perf_counter()
            for _ in range(n_repeats):
                _ = evaluator.rf_model.predict(X_tree[0:1])
            single_lat_ms = ((time.perf_counter() - t0) / n_repeats) * 1000.0

            # Batch throughput
            t0 = time.perf_counter()
            _ = evaluator.rf_model.predict(X_tree)
            batch_time = time.perf_counter() - t0
            throughput = n_samples / max(batch_time, 1e-6)

            results.append({
                'Model': 'Random Forest',
                'Single_Sample_Latency_ms': single_lat_ms,
                'Batch_Time_sec': batch_time,
                'Throughput_samples_per_sec': throughput,
            })

        # 2. XGBoost
        if evaluator.xgb_model is not None:
            evaluator.xgb_model.predict(X_tree[:10])
            t0 = time.perf_counter()
            for _ in range(n_repeats):
                _ = evaluator.xgb_model.predict(X_tree[0:1])
            single_lat_ms = ((time.perf_counter() - t0) / n_repeats) * 1000.0

            t0 = time.perf_counter()
            _ = evaluator.xgb_model.predict(X_tree)
            batch_time = time.perf_counter() - t0
            throughput = n_samples / max(batch_time, 1e-6)

            results.append({
                'Model': 'XGBoost',
                'Single_Sample_Latency_ms': single_lat_ms,
                'Batch_Time_sec': batch_time,
                'Throughput_samples_per_sec': throughput,
            })

        # 3. MLP
        if evaluator.mlp_model is not None:
            evaluator.mlp_model.eval()
            single_tensor = X_dl_tensor[0:1]
            with torch.no_grad():
                _ = evaluator.mlp_model(single_tensor)
                t0 = time.perf_counter()
                for _ in range(n_repeats):
                    _ = evaluator.mlp_model(single_tensor)
                single_lat_ms = ((time.perf_counter() - t0) / n_repeats) * 1000.0

                t0 = time.perf_counter()
                _ = evaluator.mlp_model(X_dl_tensor)
                batch_time = time.perf_counter() - t0
                throughput = n_samples / max(batch_time, 1e-6)

            results.append({
                'Model': 'MLP',
                'Single_Sample_Latency_ms': single_lat_ms,
                'Batch_Time_sec': batch_time,
                'Throughput_samples_per_sec': throughput,
            })

        # 4. Residual Hybrid
        if evaluator.hybrid_model is not None and evaluator.rf_model is not None and evaluator.xgb_model is not None:
            evaluator.hybrid_model.eval()
            b_10 = 0.5 * (evaluator.rf_model.predict(X_tree[:10]) + evaluator.xgb_model.predict(X_tree[:10]))
            evaluator.hybrid_model.predict_numpy(X_dl_np[:10], b_10, device=evaluator.device)

            t0 = time.perf_counter()
            for _ in range(n_repeats):
                b1 = 0.5 * (evaluator.rf_model.predict(X_tree[0:1]) + evaluator.xgb_model.predict(X_tree[0:1]))
                _ = evaluator.hybrid_model.predict_numpy(X_dl_np[0:1], b1, device=evaluator.device)
            single_lat_ms = ((time.perf_counter() - t0) / n_repeats) * 1000.0

            t0 = time.perf_counter()
            b_all = 0.5 * (evaluator.rf_model.predict(X_tree) + evaluator.xgb_model.predict(X_tree))
            _ = evaluator.hybrid_model.predict_numpy(X_dl_np, b_all, device=evaluator.device)
            batch_time = time.perf_counter() - t0
            throughput = n_samples / max(batch_time, 1e-6)

            results.append({
                'Model': 'Residual Hybrid',
                'Single_Sample_Latency_ms': single_lat_ms,
                'Batch_Time_sec': batch_time,
                'Throughput_samples_per_sec': throughput,
            })

        return pd.DataFrame(results)
