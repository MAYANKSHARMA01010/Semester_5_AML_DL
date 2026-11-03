"""
Phase 1 Machine Learning Models and Evaluation Wrappers
Baseline heuristic, Random Forest, and XGBoost regressor wrappers.
"""

from typing import Any, Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib


class MeanBaselineRegressor(BaseEstimator, RegressorMixin):
    """
    Baseline Heuristic predicting the empirical training mean of the target.
    Serves as the benchmark floor required by course evaluation guidelines.
    """

    def __init__(self):
        self.mean_: Optional[float] = None

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        if self.mean_ is None:
            raise ValueError("MeanBaselineRegressor is not fitted yet.")
        n_samples = len(X)
        return np.full(shape=(n_samples,), fill_value=self.mean_, dtype=np.float64)


def create_random_forest_regressor(
    n_estimators: int = 100,
    max_depth: int = 20,
    min_samples_split: int = 10,
    min_samples_leaf: int = 4,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Instantiate Phase 1 Random Forest Regressor with regularized parameters."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=n_jobs,
    )


def create_xgboost_regressor(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 8,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    n_jobs: int = -1,
) -> XGBRegressor:
    """Instantiate Phase 1 XGBoost Regressor with regularized second-order Taylor gradients."""
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
    )


def load_tree_model(model_path: str, model_type: str = "auto") -> Any:
    """Load serialized tree model from joblib or json."""
    p = str(model_path)
    if p.endswith(".joblib") or model_type == "rf":
        return joblib.load(p)
    elif p.endswith(".json") or model_type == "xgb":
        model = XGBRegressor()
        model.load_model(p)
        return model
    else:
        raise ValueError(f"Unsupported model file format: {model_path}")
