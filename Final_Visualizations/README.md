# Centralized Project Visualizations Directory

This directory contains the final, high-resolution figures and visualizations generated across **Phase 1 (Machine Learning)** and **Phase 2 (Deep Learning)** of the project:  
**Water Quality Prediction via Gradient Boosting and Deep Neural Networks**.

All figures are organized into categorized subdirectories to facilitate report generation, presentation slides (PPT), and academic viva defenses.

---

## Directory Overview

| Subfolder | Image Count | Primary Focus | Originating Tasks |
| :--- | :---: | :--- | :--- |
| [`EDA/`](EDA/) | 13 | Exploratory Data Analysis, chemistry distributions, correlations | Phase 1 Task 1 & 2 |
| [`Phase_1/`](Phase_1/) | 6 | Baseline, Random Forest, XGBoost, and TreeSHAP evaluations | Phase 1 Task 3 & 4 |
| [`Phase_2/`](Phase_2/) | 5 | MLP training convergence, 6-model comparison, feature ablation, latency | Phase 2 Task 3 & 5 |
| [`Model_Comparison/`](Model_Comparison/) | 3 | Unified 6-model benchmarks, actual vs predicted, error CDF curves | Phase 2 Task 5 & 6 |
| [`Explainability/`](Explainability/) | 3 | Integrated Gradients global/signed attributions, waterfall local cases | Phase 2 Task 6 |
| [`Error_Analysis/`](Error_Analysis/) | 5 | Residual distributions, heteroskedasticity, tier and habitat errors | Phase 2 Task 6 |
| [`Hybrid_Model/`](Hybrid_Model/) | 6 | Residual hybrid mechanics, error correlation, component decomposition | Phase 2 Task 4, 5, 6 |
| **Total** | **41** | **Comprehensive Visual Evidence** | |

---

## Detailed Visualizations Catalog

### 1. Exploratory Data Analysis (`EDA/`)
Visualizations establishing empirical ground-truth characteristics across 2,819,617 cleaned complete-case observations:

- `01_country_distribution.png`: Geographic distribution of monitoring stations and sample counts across England, USA, Ireland, China, and Canada.
- `02_waterbody_distribution.png`: Waterbody classification distribution (Rivers, Estuaries, Lakes, Canals, etc.).
- `03_temporal_coverage.png`: Annual monitoring record volume and temporal span (1940–2023) across jurisdictions.
- `04_chemical_boxplots.png`: Boxplots displaying extreme positive skewness and physical ranges for all 8 base chemical parameters.
- `05_chemical_histograms.png`: Histograms with KDE curves showing non-Gaussian long-tailed distributions of water chemistry.
- `06_country_median_heatmap.png`: Heatmap comparing median pollutant concentrations across international jurisdictions.
- `07_do_temperature_saturation.png`: Benson-Krause theoretical dissolved oxygen saturation curve vs observed field telemetry.
- `08_correlation_heatmaps.png`: Pearson and Spearman correlation matrices highlighting non-linear feature inter-relationships.
- `09_np_ratio.png`: Redfield stoichiometric N:P ratio distribution identifying freshwater phosphorus limitation.
- `10_seasonal_patterns.png`: Monthly seasonal variation curves for water temperature, DO, and nutrients.
- `11_target_distribution.png`: Bimodal distribution of the ground-truth CCME Water Quality Index ($y \in [0, 100]$).
- `12_ccme_by_country_waterbody.png`: CCME WQI index distribution grouped by country and aquatic ecosystem type.
- `13_pairplot_key_features.png`: Pairwise scatter matrix of primary nutrients and CCME score.

### 2. Phase 1 Machine Learning Models (`Phase_1/`)
Visualizations supporting Phase 1 tree baseline evaluations:

- `14_MODEL_comparison_mae_rmse_r2.png`: Phase 1 validation benchmark comparing Mean Baseline, Random Forest, and XGBoost.
- `15_MODEL_feature_importance_rf_xgb.png`: Comparative Gini feature importance between Random Forest and XGBoost.
- `16_MODEL_shap_summary_beeswarm.png`: TreeSHAP summary beeswarm plot demonstrating global feature impact on CCME index.
- `feature_importance.png`: Standalone high-resolution Gini feature importance bar chart.
- `model_comparison.png`: Comparative evaluation bar plot across MAE, RMSE, and $R^2$.
- `shap_summary.png`: Standalone TreeSHAP global attribution summary.

### 3. Phase 2 Deep Learning Components (`Phase_2/`)
Visualizations from neural network development and comprehensive benchmarking:

- `03_training_convergence_curves.png`: Training vs validation loss/MAE curves over epochs for WaterQualityMLP.
- `05_model_performance_comparison.png`: Comprehensive test set benchmark across all 6 predictive paradigms.
- `06_feature_group_ablation_impact.png`: Leave-one-group-out (LOGO) ablation impact ($\Delta\text{MAE}$, $\Delta R^2$) across 7 domain feature groups.
- `07_regularization_dropout_curves.png`: Dropout rate sweep ($p \in [0.0, 0.5]$) showing generalization error response.
- `10_computational_tradeoffs.png`: Latency vs accuracy trade-off scatter plot across single-sample and batch regimes.

### 4. Cross-Model Benchmarking (`Model_Comparison/`)
Visualizations comparing all 6 models on the common held-out test split ($N = 422,943$):

- `model_performance_comparison.png`: Unified 6-model benchmark (MAE, RMSE, $R^2$, MaxAE).
- `actual_vs_predicted.png`: Scatter plots of actual vs predicted CCME scores with identity line for all models.
- `absolute_error_cdf.png`: Cumulative Distribution Function of absolute errors highlighting 95th/99th percentile error compression.

### 5. Explainability & Feature Attribution (`Explainability/`)
Visualizations explaining deep neural network representations using axiomatic Integrated Gradients:

- `mlp_global_feature_importance.png`: Path-integral feature attributions for WaterQualityMLP aggregated to 38 domain features.
- `mlp_signed_feature_attributions.png`: Signed attribution bars demonstrating positive vs negative directional influence.
- `individual_prediction_explanations.png`: Waterfall local explanation breakdowns for pristine vs polluted monitoring observations.

### 6. Error & Residual Analysis (`Error_Analysis/`)
Diagnostics under the standard sign convention ($\text{Residual} = \text{Prediction} - \text{Actual}$):

- `residual_distributions.png`: Kernel density estimation of prediction errors showing zero-bias centering.
- `residual_boxplots.png`: Horizontal boxplots comparing error spreads, interquartile ranges, and outlier densities.
- `residual_vs_predicted.png`: Heteroskedasticity diagnostic plots showing residual spreads across predicted WQI values.
- `error_by_wqi_tier.png`: Mean and P95 absolute errors stratified across CCME regulatory water quality tiers (Poor to Excellent).
- `error_by_country_waterbody.png`: Error diagnostics stratified across geographic jurisdictions and aquatic habitat types.

### 7. Hybrid Model Mechanics (`Hybrid_Model/`)
Visualizations demonstrating the multi-paradigm tree + residual neural network synergy:

- `hybrid_residual_plots.png`: Residual distributions and Q-Q plots for ResidualMLPNet corrections.
- `hybrid_scatter_plots.png`: Actual vs predicted plots for base tree vs hybrid model.
- `hybrid_component_contributions.png`: Component decomposition showing base tree prediction vs residual correction delta.
- `correction_vs_base_error.png`: Scatter showing correlation ($r = 0.526$) between base tree error and residual neural network correction.
- `geographic_country_performance.png`: Geographic country-level performance comparison across model paradigms.
- `aquatic_waterbody_performance.png`: Aquatic habitat performance comparison across model paradigms.
