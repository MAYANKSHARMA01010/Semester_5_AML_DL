# Project Presentation Graphs & Visual Assets Guide

This directory consolidates all high-resolution figures and evaluation charts from across the **Water Quality Prediction (Semester 5 AML)** project. 

These figures are numbered sequentially and categorized so an AI agent (such as Claude) or human presenter can directly incorporate them into PowerPoint (`.pptx`), LaTeX Beamer, or HTML presentation slides.

---

## 📊 Presentation Slide Mapping Table

| File Name | Presentation Slide Topic | Key Findings & Speaking Points for Slides |
| :--- | :--- | :--- |
| **`01_EDA_country_distribution.png`** | **Slide: Multi-National Data Provenance** | • 2,827,977 raw records across 5 sovereign nations.<br>• Geographic breakdown: England (75.3%), USA (14.3%), Ireland (8.3%), China (1.6%), Canada (0.1%).<br>• Authentic regulatory telemetry from USGS, UK Environment Agency, EPA Ireland, China MEE, ECCC. |
| **`02_EDA_waterbody_distribution.png`** | **Slide: Ecological Waterbody Types** | • Classification by aquatic ecology: Rivers (dominant), Lakes, Streams, Estuaries, and Effluents.<br>• Demonstrates cross-ecosystem generalizability. |
| **`03_EDA_temporal_coverage_1940_2023.png`** | **Slide: 83-Year Historical Scope** | • Telemetry span from historical baselines (1940) to contemporary telemetry (2023).<br>• >95% concentrated in the modern regulatory monitoring era (2000–2023). |
| **`04_EDA_chemical_boxplots_skewness.png`** | **Slide: Nutrient Distribution & Heavy Skewness** | • Nutrient concentrations (orthophosphate, ammonia, BOD) span up to 4 orders of magnitude ($\gamma_1 > 14$).<br>• Justifies applying non-linear $\log(1+x)$ transformations to stabilize model training. |
| **`05_EDA_chemical_histograms.png`** | **Slide: Sensor Telemetry Histograms** | • Normal bell curves for physical probes (pH, Temperature, Dissolved Oxygen).<br>• Heavy right-tailed exponential distributions for nutrient pollutants. |
| **`06_EDA_country_median_heatmap.png`** | **Slide: Cross-Country Chemical Baselines** | • Heatmap comparing median chemical baselines across nations.<br>• Shows consistent thermodynamic physical laws despite differing geographic regimes. |
| **`07_EDA_do_temperature_saturation.png`** | **Slide: Benson-Krause Thermodynamic Dissolved Oxygen** | • Highlights the non-linear thermodynamic solubility boundary between temperature and oxygen.<br>• Demonstrates the engineering of `do_deficit` and `% DO Saturation` from Henry's Law. |
| **`08_EDA_correlation_heatmaps.png`** | **Slide: Feature Correlation Analysis** | • Pearson & Spearman correlation matrices against the CCME WQI target.<br>• Confirms orthophosphate as the strongest negative driver ($\rho = -0.78$) and dissolved oxygen as positive. |
| **`09_EDA_np_stoichiometric_ratio.png`** | **Slide: Aquatic Stoichiometry (Redfield Ratio)** | • Visualizes nitrogen-to-phosphorus ($N:P$) ratio distributions.<br>• Highlights biological nutrient limitation (phosphorus-limited freshwater systems). |
| **`10_EDA_seasonal_patterns.png`** | **Slide: Temporal & Seasonal Harmonic Dynamics** | • Illustrates summer vs. winter seasonal fluctuations in temperature, oxygen, and runoff.<br>• Justifies circular trigonometric harmonic encodings ($\sin/\cos$). |
| **`11_EDA_target_wqi_distribution.png`** | **Slide: Ground Truth Target (CCME WQI)** | • Distribution across the 5 official CCME quality tiers: Poor, Marginal, Fair, Good, Excellent.<br>• Continuous CCME values bounded strictly between $0.0$ and $100.0$. |
| **`12_EDA_ccme_by_country_waterbody.png`** | **Slide: Quality Index Across Jurisdictions** | • Cross-tabulated performance of water quality by country and waterbody reach. |
| **`13_EDA_pairplot_key_features.png`** | **Slide: Multi-Parameter Interactions** | • Pairwise scatter and density interactions among key chemical parameters. |
| **`14_MODEL_comparison_mae_rmse_r2.png`** | **Slide: Quantitative Model Benchmarking** | • **Random Forest:** $R^2 = \mathbf{0.9999}$, $\text{MAE} = \mathbf{0.052}$, $\text{RMSE} = 0.208$.<br>• **XGBoost:** $R^2 = \mathbf{0.9994}$, $\text{MAE} = \mathbf{0.161}$, $\text{RMSE} = 0.437$ ($9\times$ faster inference latency).<br>• **Mean Baseline:** $R^2 = -0.0000$, $\text{MAE} = 14.130$. |
| **`15_MODEL_feature_importance_rf_xgb.png`** | **Slide: Built-in Feature Importances** | • Built-in split/gain feature importances for Random Forest vs. XGBoost.<br>• Confirms alignment: `orthophosphate_mg_l`, `log1p_orthophosphate`, `hist_ortho_lag1` dominate. |
| **`16_MODEL_shap_summary_beeswarm.png`** | **Slide: Explainable AI via TreeSHAP** | • Game-theoretic Shapley attribution beeswarm plot.<br>• **Orthophosphate drives >88% of model decisions**.<br>• **`country_code` importance is <0.01%**, proving model is governed by universal chemistry, not borders.<br>• Historical station lag (`hist_ortho_lag1`) is #2, confirming catchment memory. |
| **`18_AUDIT_outlier_screening_bounds.png`** | **Slide: Thermodynamic Sensor Screening** | • Purging of exactly 8,360 corrupted telemetry dropouts ($-999999$ codes, $\text{pH}=67,115$).<br>• Produces 2,819,617 pristine clean observations. |
| **`19_AUDIT_correlation_matrix.png`** | **Slide: Pre-Screening Correlation Matrix** | • Correlation matrix across raw parameters before feature engineering. |

---

## 🎯 Recommended Presentation Flow (10-Slide Deck Template)

1. **Slide 1: Title & Team** — Project 18: Water Quality Prediction via Gradient Boosting & Deep Neural Networks
2. **Slide 2: Problem Statement & Motivation** — Real-time telemetry vs. slow lab assays ($\text{BOD}_5$ takes 5 days)
3. **Slide 3: Multi-National Dataset Profile** — `01_EDA_country_distribution.png`, `03_EDA_temporal_coverage_1940_2023.png`
4. **Slide 4: Data Integrity & Physics Screening** — `18_AUDIT_outlier_screening_bounds.png`
5. **Slide 5: Chemical Dynamics & Skewness** — `04_EDA_chemical_boxplots_skewness.png`, `07_EDA_do_temperature_saturation.png`
6. **Slide 6: Feature Engineering (38 Features)** — Stoichiometric laws ($N:P$), station lags (`hist_*`), seasonal $\sin/\cos$
7. **Slide 7: Anti-Leakage Quarantine & Architecture** — Strict separation of features and CCME targets ($F_1, F_2, F_3$)
8. **Slide 8: Benchmark Evaluation & Results** — `14_MODEL_comparison_mae_rmse_r2.png` (RF $R^2=0.9999$, XGBoost $9\times$ speed)
9. **Slide 9: Explainable AI (XAI with TreeSHAP)** — `16_MODEL_shap_summary_beeswarm.png` (Phosphorus dominance, zero country bias)
10. **Slide 10: Conclusion & Future Scope** — IoT edge deployment, Phase 2 deep learning MLP benchmarks
