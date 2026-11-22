# Final Report Verification & Audit Handover

**Project Title:** Water Quality Prediction via Gradient Boosting and Deep Neural Networks  
**Project ID:** Project 18 (Semester 5 Applied Machine Learning & Deep Learning)  
**Academic Institution:** Department of Computer Science & Artificial Intelligence, Newton School of Technology, Rishihood University  
**Date of Verification:** November 22, 2026  
**Document Format:** IEEE Transactions / Conference Double-Column Format (`IEEEtran.cls`)  
**Compiled Artifact:** `Final_Report/Final_Project_Report.pdf` (3.9 MB, 10 Pages)

---

## 1. Inventory of Created Files and Paths

The final academic project report package is located in `Final_Report/`:

| File Path | Description | Format / Size | Status |
| :--- | :--- | :--- | :--- |
| `Final_Report/main.tex` | Complete 12-section IEEE LaTeX master document | LaTeX Source (50.1 KB, 755 lines) | Verified & Sanitized |
| `Final_Report/Final_Project_Report.pdf` | Compiled publication-ready research report | PDF (3.9 MB, 10 Pages) | Zero Errors / Clean Build |
| `Final_Report/IEEEtran.cls` | Official IEEE LaTeX document class (V1.8b) | Class File (275 KB) | Verified |
| `Final_Report/references.bib` | BibTeX bibliography (2 flagship peer-reviewed citations) | BibTeX (2 citations) | Cleanly Resolved |
| `Final_Report/figures/` | High-resolution publication figures (22 PNGs) | Directory (PNG images) | All embedded & rendered |
| `Final_Report/REPORT_VERIFICATION.md` | Verification audit and handover documentation | Markdown Audit | Current File |

---

## 2. Institutional Affiliation & Author Attribution

- **Institution:** Department of Computer Science & Artificial Intelligence, **Newton School of Technology, Rishihood University**.
- **Authors:** Aman Soni, Mayank Sharma, Aaryan Gera, Chaitanya Kumar, Ankit Kumar Pandey, and Pranav Sehgal.
- **Hierarchy Status:** All members are listed as equal academic peers; no member is listed as "lead" or with hierarchical designations.

---

## 3. Phase 2 Technical Work Packages & Viva Defense Matrix

As designated, Table V (`tab:team_contributions`) defines the technical ownership and viva voce defense specializations across all six team members:

| Member Name | Phase 2 Technical Work Packages | Technical Viva Defense Specialization |
| :--- | :--- | :--- |
| **Aman Soni** | **Tasks 1 & 6: DL Data Preparation, Feature Engineering, Explainability & Error Analysis**<br>• DL feature scaling (`StandardScaler` fit strictly on train).<br>• Categorical encoding and one-hot embeddings.<br>• Anti-leakage chronological/geographic quarantine.<br>• PyTorch `Dataset` and `DataLoader` implementation.<br>• TreeSHAP and Integrated Gradients ($N=1,000$, $m=50$).<br>• Prediction error analysis and residual diagnostic plots. | PyTorch DataLoaders, 38- vs 53-dimensional input representations, Integrated Gradients path integration, baseline selection ($F(x') = 86.7233$), and residual error diagnostics ($\hat{y} - y$). |
| **Mayank Sharma** | **Task 2: MLP Architecture Development**<br>• Implementing deep MLP using PyTorch.<br>• Designing hidden layer neurons ($53 \to 256 \to 128 \to 64 \to 32 \to 1$).<br>• `BatchNorm1d` and `Dropout(0.2)` regularizations.<br>• Regression output layer and parameter tuning. | Neural network topology, layer-wise inductive bias, Batch Normalization vs Dropout interactions, and tabular step boundary approximation limitations. |
| **Chaitanya Kumar** | **Task 3: MLP Training & Optimization**<br>• End-to-end training and validation loops.<br>• Loss functions (`SmoothL1Loss`, `MSELoss`) and AdamW optimizer.<br>• Learning-rate scheduling (`CosineAnnealingLR`).<br>• Early stopping protocols and checkpoint persistence.<br>• Overfitting diagnostics and learning dynamics. | Gradient descent optimization dynamics, learning rate decay schedules, validation loss tracking, and early stopping convergence criteria. |
| **Ankit Kumar Pandey** | **Task 4: Hybrid ML + DL Model Development**<br>• Ensembling existing base models ($0.5\text{RF} + 0.5\text{XGB}$).<br>• 5-fold out-of-fold (OOF) cross-validation predictions.<br>• Unbiased residual target generation ($r_i = y_i - \hat{y}_{\text{base}}^{\text{OOF}}$).<br>• Designing and training `ResidualMLPNet` ($54 \to 256 \to 128 \to 64 \to 1$).<br>• Additive hybrid inference ($\hat{y}_{\text{hybrid}} = \hat{y}_{\text{base}} + \hat{r}$). | Additive boosting mathematics, 5-fold OOF residual generation, data leakage avoidance, and peak error tail compression ($54.9\%$). |
| **Aaryan Gera** | **Task 5: Model Evaluation & Ablation Studies**<br>• Cross-model benchmark evaluation (RF, XGB, MLP, Hybrid).<br>• Metric calculation: MAE, RMSE, $R^2$, MedAE, MaxAE.<br>• Systematic 7-group feature ablation studies.<br>• Regularization and dropout sensitivity audits.<br>• Country-wise, site-wise, and temporal evaluations.<br>• Training and inference latency profiling (CPU vs GPU). | Statistical metrics trade-offs, catastrophic ablation failure modes, temporal-geographic distribution invariance, and inference latency profiling. |
| **Pranav Sehgal** | **Technical Documentation & Literature Review**<br>• Academic literature survey.<br>• CCME Water Quality Index mathematical formulation review.<br>• Environmental regulatory guideline synthesis.<br>• Technical reporting and presentation support. | Environmental regulatory guidelines, CCME index mathematical formulation ($F_1, F_2, F_3$), and technical report synthesis. |

---

## 4. Bibliography: The Two Flagship References

In accordance with instructions to retain exclusively the top two best references, the bibliography was pruned to the two foundational peer-reviewed machine learning papers underpinning the project's title and methodology:

1. **`chen2016xgboost`:**  
   Tianqi Chen and Carlos Guestrin, *"XGBoost: A Scalable Tree Boosting System"*, in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, pp. 785--794, 2016.  
   *(Foundational paper for Gradient Boosting, regularized objective functions, and second-order Taylor expansion tree learning).*
2. **`he2016deep`:**  
   Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, *"Deep Residual Learning for Image Recognition"*, in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR '16)*, pp. 770--778, 2016.  
   *(Seminal paper establishing residual learning, forming the mathematical foundation for the Phase 2 Residual Hybrid architecture).*

All secondary references and technical reports were removed from `references.bib` and their citations safely removed from `main.tex`, ensuring a completely clean build with **0 undefined citations**.

---

## 5. Structure and Coverage of the 12 Report Sections

1. **Section I: Introduction & Project Overview (`sec:intro`)** — Environmental telemetry context, multi-paradigm research problem, dataset scope (2.82M records), and executive summary of findings.
2. **Section II: Theoretical Foundations & Problem Formulation (`sec:theory`)** — Axiomatic derivation of CCME Water Quality Index ($F_1, F_2, F_3$), Random Forest ensemble principles, XGBoost gradient boosting loss Taylor expansion, deep residual learning theory, and axiomatic Integrated Gradients formulation.
3. **Section III: Data Pipeline & Preprocessing Architecture (`sec:data`)** — 5-country telemetry distribution (England 83.74%, France 12.01%, Spain 2.22%, Germany 1.15%, Italy 0.88%), physical plausibility screening thresholds, and strict anti-leakage temporal-geographic partitioning ($70\%$ Train, $15\%$ Val, $15\%$ Held-Out Test).
4. **Section IV: Domain-Specific Feature Engineering (`sec:features`)** — Taxonomy of the 38-feature tabular set, Benson--Krause Dissolved Oxygen saturation equations, Redfield stoichiometric N:P ratio with numerical stability constant, Emerson unionized ammonia formula, and 53-feature deep learning representation.
5. **Section V: Phase 1 — Tree Ensemble Benchmarks (`sec:phase1`)** — Baseline, Random Forest (100 trees, $\text{MAE} = 0.0514$), XGBoost (150 trees, $\text{MAE} = 0.1614$), and validation TreeSHAP game-theoretic analysis showing orthophosphate dominance ($> 88\%$ split importance).
6. **Section VI: Phase 2 — Deep Neural Network Development (`sec:phase2`)** — 5-layer MLP (`WaterQualityMLP`) topology ($53 \rightarrow 256 \rightarrow 128 \rightarrow 64 \rightarrow 32 \rightarrow 1$), AdamW optimization, training convergence trajectories, and mathematical analysis of why smooth continuous networks struggle with discontinuous tabular step thresholds ($\text{MAE} = 1.3800$).
7. **Section VII: Residual Hybrid Architecture (`sec:hybrid`)** — Additive boosting model ($\hat{y}_{\text{hybrid}} = \hat{y}_{\text{base}} + \hat{r}$), `ResidualMLPNet` topology ($54 \rightarrow 256 \rightarrow 128 \rightarrow 64 \rightarrow 1$), and strict 5-fold Out-Of-Fold (OOF) cross-validation training on the training split with zero test set target contamination.
8. **Section VIII: Comprehensive Model Comparison & Benchmark Results (`sec:comparison`)** — Uniform evaluation across all 6 architectures on the identical held-out test split ($N = 422,943$), highlighting the Residual Hybrid achieving the lowest RMSE ($0.1678$), highest $R^2$ ($0.999910$), and a **$54.9\%$ peak outlier error compression** ($12.6128 \rightarrow 5.6842$).
9. **Section IX: Feature Importance & Ablation Analysis (`sec:ablation`)** — Systematic 7-group ablation study isolating `base_sensors`, `log1p_norm`, `stoichiometry`, `env_stress`, `temporal`, `history_lags`, and `categoricals`, contrasting tree dependency on raw sensors with neural network dependency on logarithmic variance stabilization ($\Delta\text{MAE} = +12.16$).
10. **Section X: Interpretability & Prediction Error Diagnostics (`sec:explainability`)** — Axiomatic Integrated Gradients attribution ($N=1,000$ test observations, $m=50$ steps, mean completeness error $\Delta = 0.2529$) using the modal observed baseline ($F(x') = 86.7233$), and residual error diagnostics ($\text{Residual} = \hat{y} - y$).
11. **Section XI: Discussion, Critical Evaluation & Future Work (`sec:discussion`)** — Deterministic function approximation barriers, `exceedance_counter` feature leakage considerations, geographic imbalance audits, and roadmaps for spatial holdout splitting, online sensor drift calibration, and specialized tabular architectures.
12. **Section XII: Project Contributions, Author Attribution & Conclusion (`sec:team`, `sec:conclusion`)** — Six-member technical contribution matrix and viva voce defense specialization matrix, followed by final research synthesis.

---

## 6. Embedded Figures (15 Figures) & Formal Tables (5 Tables)

### Figures:
- `fig:eda_overview`: Geographic distribution across 5 European countries (`01_EDA_country_distribution.png`).
- `fig:station_sampling`: Temporal station sampling density (`03_EDA_temporal_station_sampling.png`).
- `fig:target_dist`: Target WQI multimodal distribution (`04_EDA_target_wqi_distribution.png`).
- `fig:feature_corr`: Feature correlation heatmap (`05_FE_feature_correlation_heatmap.png`).
- `fig:phase1_perf`: Phase 1 test performance metrics (`11_MODEL_test_performance_comparison.png`).
- `fig:shap`: TreeSHAP beeswarm attribution plot (`16_MODEL_shap_summary_beeswarm.png`).
- `fig:mlp_training`: MLP training and validation loss convergence curves (`21_DL_mlp_training_history.png`).
- `fig:hybrid_arch`: Residual Hybrid architecture schematic and OOF flow (`22_HYBRID_architecture_diagram.png`).
- `fig:residual_corr`: Target residuals vs predicted residual corrections (`26_HYBRID_residual_distribution.png`).
- `fig:all_models_mae`: Multi-model MAE and RMSE benchmark comparisons (`28_EVAL_model_comparison_bar_mae_rmse.png`).
- `fig:parity`: Prediction vs actual parity scatter plots (`29_EVAL_prediction_vs_actual_scatter.png`).
- `fig:error_cdf`: Cumulative error distribution function (CDF) (`30_EVAL_error_cdf_comparison.png`).
- `fig:ablation`: 7-group ablation $\Delta\text{MAE}$ bar chart (`31_EVAL_ablation_delta_mae.png`).
- `fig:ig_attr`: Integrated Gradients mean absolute feature attributions (`33_EXP_integrated_gradients_attributions.png`).
- `fig:residual_diagnostics`: Residual error distributions ($\hat{y} - y$) (`36_EXP_residual_error_distribution.png`).

### Tables:
- `tab:physics_bounds`: Screening thresholds and valid ranges for water telemetry.
- `tab:feature_taxonomy`: Taxonomy of the 38-dimensional tabular feature set.
- `tab:model_comparison`: Common held-out test set benchmark comparison across all 6 models.
- `tab:ablation_results`: Systematic 7-group feature ablation $\Delta\text{MAE}$ results.
- `tab:team_contributions`: Phase 2 Technical Work Packages & Viva Defense Specialization Matrix.

---

## 7. Numerical Consistency Verification

All metrics in Table III match the ground-truth test set benchmarks ($N = 422,943$ held-out observations) exactly:

| Model Architecture | MAE (WQI) | RMSE (WQI) | $R^2$ Score | MedAE (WQI) | MaxAE (WQI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.0514** | 0.2016 | 0.999870 | **0.0042** | 12.6128 |
| **Residual Hybrid** | 0.0710 | **0.1678** | **0.999910** | 0.0229 | **5.6842** |
| **Base Tree Ensemble (0.5 RF + 0.5 XGB)** | 0.0900 | 0.2464 | 0.999805 | 0.0261 | 8.7128 |
| **XGBoost Regressor** | 0.1614 | 0.4364 | 0.999389 | 0.0480 | 9.2769 |
| **Weighted Ensemble (0.4 RF + 0.4 XGB + 0.2 MLP)** | 0.3096 | 0.4155 | 0.999446 | 0.2653 | 8.8283 |
| **Deep Multi-Layer Perceptron (WaterQualityMLP)** | 1.3800 | 1.6720 | 0.991029 | 1.2893 | 31.1327 |

- **Peak Outlier Error Compression:** $\frac{12.6128 - 5.6842}{12.6128} \times 100\% = \mathbf{54.9\%}$ reduction from Random Forest to Residual Hybrid.
- **Variance Reduction:** $\frac{0.2464 - 0.1678}{0.2464} \times 100\% = \mathbf{31.9\%}$ reduction in RMSE over the base tree ensemble.
- **MLP Discrepancy Ratio:** $\frac{1.3800}{0.0514} \approx \mathbf{26.8\times}$ higher MAE than Random Forest on tabular telemetry.
- **Integrated Gradients Baseline & Completeness:** $F(x') = 86.7233$, $F(x) = 83.7431$, $\sum_j \text{IG}_j = -2.7273$, mean completeness error $\Delta = 0.2529$ across 1,000 test observations.

---

## 8. Compilation Status Summary

- **TeX Engine:** Tectonic 0.17.0.
- **Exit Code:** `0` (Zero compilation errors).
- **Page Count:** **Exactly 10 pages** formatted in two-column IEEE style.
- **Output File Size:** 3.9 MB (`Final_Report/Final_Project_Report.pdf`).
- **Citation Warnings:** `0` (All citations cleanly mapped to the 2 retained references).
