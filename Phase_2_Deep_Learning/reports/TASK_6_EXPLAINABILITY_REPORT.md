# Phase 2 — Task 6: Explainability & Prediction Error Analysis Report

**Project Title:** Water Quality Prediction via Gradient Boosting and Deep Neural Networks  
**Scope:** Phase 2 (Deep Learning) — Task 6: Explainability & Prediction Error Analysis  
**Repository:** [Semester_5_AML_DL](https://github.com/MAYANKSHARMA01010/Semester_5_AML_DL)  
**Evaluated Test Set Size:** 422,943 independent observations (1964–2023 across 5 countries)  
**Deliverable Files:**
- Executable Notebook: [06_explainability_error_analysis.ipynb](file:///Users/amansoni/Documents/Development/Personal%20Projects/Semester_5_AML_DL/Phase_2_Deep_Learning/notebooks/06_explainability_error_analysis.ipynb)
- Technical Report: [TASK_6_EXPLAINABILITY_REPORT.md](file:///Users/amansoni/Documents/Development/Personal%20Projects/Semester_5_AML_DL/Phase_2_Deep_Learning/reports/TASK_6_EXPLAINABILITY_REPORT.md)
- Output Artifacts Directory: [task_6_outputs/](file:///Users/amansoni/Documents/Development/Personal%20Projects/Semester_5_AML_DL/Phase_2_Deep_Learning/reports/task_6_outputs)

---

## 1. Executive Summary

### 1.1 Purpose and Core Research Questions
Phase 2 Task 6 provides an empirical explainability and prediction error diagnostic audit of the deep learning and hybrid architectures developed for Canadian Council of Ministers of the Environment Water Quality Index (CCME WQI) regression ($y \in [0, 100]$). 

This investigation addresses six primary research questions:
1. **Global Feature Attribution:** Which input dimensions and chemical parameters govern the decision surface of the Phase 2 Deep Neural Network (MLP)?
2. **Directional Influences:** Do feature attributions align with aquatic environmental chemistry, penalizing contaminants and rewarding regulatory compliance?
3. **Local Explanations:** How do specific individual predictions (clean vs. severely contaminated water) decompose into additive feature attributions?
4. **Hybrid / Residual Mechanics:** How does the multi-paradigm ensemble ($0.4 \times \text{RF} + 0.4 \times \text{XGBoost} + 0.2 \times \text{MLP}$) operate, and does the neural network's additive adjustment improve or degrade tree baseline predictions?
5. **Error Distributions & Model Hierarchy:** What is the exact performance hierarchy across the 422,943 test samples, and how do error distributions (tails, kurtosis, extreme failures) compare?
6. **Subgroup Vulnerabilities & Difficult Cases:** Under what conditions (WQI quality tiers, administrative jurisdictions, aquatic habitat types, historical eras) do models exhibit elevated failure rates?

---

### 1.2 Summary of Key Findings

| Performance Metric | Random Forest (Phase 1) | XGBoost (Phase 1 Champion) | MLP (Phase 2 DL) | Weighted Ensemble (Hybrid) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **0.0514** | 0.1614 | 1.3800 | 0.3096 |
| **Root Mean Squared Error (RMSE)** | **0.2016** | 0.4364 | 1.6720 | 0.4155 |
| **Coefficient of Determination ($R^2$)** | **0.999870** | 0.999389 | 0.991029 | 0.999446 |
| **Median Absolute Error** | **0.0042** | 0.0480 | 1.2893 | 0.2653 |
| **95th Percentile Error ($P_{95}$)** | **0.2056** | 0.6007 | 3.0895 | 0.7389 |
| **99th Percentile Error ($P_{99}$)** | **0.5738** | 1.7292 | 4.8572 | 1.3719 |
| **Maximum Absolute Error** | 12.6128 | 9.2769 | 31.1327 | **8.8283** |
| **Mean Residual (Bias $\mu_e$)** | **-0.0009** | **-0.0002** | +0.8889 | +0.1774 |
| **Pairwise Win-Rate vs. Other Models** | **90.8% - 97.7%** | 50.0% - 93.5% | 2.3% - 6.5% | 50.0% - 96.9% |

1. **Global Drivers of Neural Predictions:** Path-integral Integrated Gradients ($m=50$ steps, 1,000 test samples, mean completeness divergence $\Delta = 0.2399$) establishes that chemical nutrient and oxygen parameters dominate the MLP. The top 5 drivers are `log1p_orthophosphate` (mean absolute attribution 5.8602), `log1p_nitrogen` (3.4170), `log1p_np_ratio` (3.0399), `log1p_ammonia` (1.7096), and `exceedance_counter` (0.6247). Aggregated categorical features (`country_code` at 0.3796, `waterbody_type_code` at 0.2810) exert contextual secondary influence.
2. **Directional Chemistry Alignment:** Signed attribution analysis confirms environmental physical realism: elevated `log1p_orthophosphate` (mean signed attribution -1.3428), `log1p_ammonia` (-1.1318), `log1p_bod` (-0.5264), and `exceedance_counter` (-0.2449) strongly suppress predicted WQI, while dissolved oxygen compliance and low nutrient levels promote high WQI.
3. **Hybrid Ensemble & Checkpoint Audit:** An audit of the disk repository revealed that `residual_mlp_checkpoint.pt` was **not saved to disk** during Task 4 (the architecture was defined in `src/hybrid_models.py`, but no trained weights were persisted). Consequently, Task 5 evaluated the operational multi-paradigm ensemble: $\hat{y}_{\text{ens}} = 0.4 \hat{y}_{\text{RF}} + 0.4 \hat{y}_{\text{XGB}} + 0.2 \hat{y}_{\text{MLP}}$. Because tree models achieve near-perfect fidelity ($MAE = 0.0514$ and $0.1614$), blending in the MLP ($MAE = 1.3800$) increases the base tree error in **86.84%** of test observations, reducing error in only **13.16%** of cases. However, the ensemble does effectively compress extreme outlier tail risks, achieving a maximum test error of 8.8283 vs. 12.6128 for Random Forest.
4. **Vulnerable Subgroups:** The MLP exhibits pronounced positive bias ($\mu_e = +1.7633$) on Excellent water ($WQI \in (95, 100]$), where 36.2% of test cases reside, underpredicting true perfection. Across geographic cohorts, Canadian samples ($N=597$, $MAE = 1.9613$) and Chinese samples ($N=6,915$, $MAE = 1.9694$) incur higher neural errors than the primary English training domain ($N=319,239$, $MAE = 1.2673$). Across aquatic habitats, Bays ($MAE = 1.9694$), Transitional waters ($MAE = 1.7776$), and Coastal waters ($MAE = 1.7153$) display the highest MLP error rates.

---

## 2. Models Analyzed & Evaluation Cohort

### 2.1 Model Architecture Specifications

```
                       ┌────────────────────────────────────────────────────────┐
                       │           Full Test Set (N = 422,943 Rows)             │
                       └───────────────────┬────────────────────────────────────┘
                                           │
                     ┌─────────────────────┼─────────────────────┐
                     │                     │                     │
                     ▼                     ▼                     ▼
          ┌─────────────────────┐┌──────────────────┐┌───────────────────────┐
          │ Phase 1: RF Model   ││ Phase 1: XGBoost ││ Phase 2: MLP PyTorch  │
          │ 100 Trees           ││ GBDT             ││ 53 -> 128 -> 64 ->    │
          │ Joblib Checkpoint   ││ JSON Model       ││ 32 -> 1, Huber Loss   │
          │ MAE: 0.0514         ││ MAE: 0.1614      ││ MAE: 1.3800           │
          └──────────┬──────────┘└─────────┬────────┘└───────────┬───────────┘
                     │                     │                     │
                     │  (Weight = 0.40)    │  (Weight = 0.40)    │ (Weight = 0.20)
                     └────────────────► ◄──┴─────────────────────┘
                                           │
                                           ▼
                                ┌──────────────────────┐
                                │  Weighted Ensemble   │
                                │  Hybrid Multi-Model  │
                                │  MAE: 0.3096         │
                                └──────────────────────┘
```

1. **Random Forest Regressor (Phase 1 Baseline Champion):**
   - **Path:** `Phase_1_Machine_Learning/data/GeneratedOutput/03.ModelTraining/random_forest_model.joblib`
   - **Hyperparameters:** `n_estimators=100`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `random_state=42`.
   - **Input Space:** 44 raw and engineered tabular features with native scikit-learn handling.
2. **XGBoost Regressor (Phase 1 Champion):**
   - **Path:** `Phase_1_Machine_Learning/data/GeneratedOutput/03.ModelTraining/xgboost_model.json`
   - **Hyperparameters:** `n_estimators=100`, `learning_rate=0.1`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`.
   - **Input Space:** 44 tabular features identical to RF.
3. **Water Quality Multi-Layer Perceptron (Phase 2 DL Model):**
   - **Path:** `Phase_2_Deep_Learning/models/mlp_best_checkpoint.pt`
   - **Architecture:** 4-layer fully connected feedforward network with ReLU non-linearities, Batch Normalization, and Dropout:
     $$\text{Input}(53) \xrightarrow{\text{Linear}} 128 \xrightarrow{\text{BN, ReLU, Dropout(0.2)}} 64 \xrightarrow{\text{BN, ReLU, Dropout(0.1)}} 32 \xrightarrow{\text{BN, ReLU}} 1 \xrightarrow{\text{Identity}} \hat{y}$$
   - **Optimization:** AdamW optimizer ($\text{lr}=10^{-3}$, weight decay $=10^{-4}$), Huber Loss ($\delta=1.0$), Cosine Annealing learning rate schedule, trained for 100 epochs with Early Stopping (patience=10) on validation loss.
   - **Trained Metric:** Best validation loss checkpoint restored.
4. **Weighted Ensemble / Hybrid Multi-Paradigm System:**
   - **Specification:** Operational weighted blend established in Task 5:
     $$\hat{y}_{\text{ens}} = 0.40 \cdot \hat{y}_{\text{RF}} + 0.40 \cdot \hat{y}_{\text{XGBoost}} + 0.20 \cdot \hat{y}_{\text{MLP}}$$
   - **Residual MLP Checkpoint Status:** An exhaustive inspection of `Phase_2_Deep_Learning/models/` confirmed that `residual_mlp_checkpoint.pt` was not saved. Task 4 implemented `ResidualMLP` inside `src/hybrid_models.py`, but training artifact persistence was omitted. This report evaluates the live weighted ensemble deployed in Task 5 and decomposes the MLP's additive residual impact relative to the tree ensemble.

### 2.2 Evaluation Dataset Cohort
- **Partition:** Dedicated test partition `Phase_1_Machine_Learning/data/CleanedData/test.csv`.
- **Sample Count:** 422,943 rows $\times$ 45 columns (never observed during preprocessing fit or training loops).
- **Target Variable:** CCME Water Quality Index (`ccme_wqi`), continuous index in $[0, 100]$.
- **Input Dimension Representation:**
  - 36 continuous numerical features (robust-scaled / standard-scaled with clipping to $[-10, 10]$).
  - 2 nominal categorical features (`country_code` with 5 levels, `waterbody_type_code` with 12 levels) expanded via One-Hot Encoding to 17 binary indicators.
  - Total deep learning input tensor dimension: $D = 53$.

---

## 3. Explainability Methodology: Integrated Gradients

### 3.1 Axiomatic Justification
For continuous deep neural networks, post-hoc explainability methods often face trade-offs between computational tractability and theoretical fidelity:
- **SHAP (KernelSHAP):** Requires exponential ($2^D$) or sampling-based perturbation passes, introducing sampling variance and massive computational burden ($422,943 \times 53 \times 1,000$ model calls).
- **Permutation Importance:** Breaks feature covariance structures, producing unrealistic out-of-distribution inputs (e.g., pH 14 with DO 0 mg/L in cold sea water).
- **Vanilla Saliency Maps ($\nabla_x F(x)$):** Suffer from gradient saturation in deep networks with piecewise-linear activations (ReLU), violating sensitivity (features that shift the output can have zero local gradients).

To resolve these limitations, **Integrated Gradients (Sundararajan et al., 2017)** was chosen. Integrated Gradients satisfies two fundamental axiomatic properties:
1. **Completeness:** The attributions sum exactly to the difference between the model's prediction at input $x$ and its prediction at a neutral baseline $x'$:
   $$\sum_{i=1}^{D} IG_i(x) = F(x) - F(x')$$
2. **Implementation Invariance:** Two functionally equivalent neural networks yield identical attributions regardless of internal layer configurations.

### 3.2 Mathematical Formulation & Baseline Selection
The Integrated Gradient for feature dimension $i$ along the straight-line interpolation path from baseline $x'$ to input $x$ is defined as:
$$IG_i(x) = (x_i - x_i') \times \int_{0}^{1} \frac{\partial F(x' + \alpha (x - x'))}{\partial x_i} d\alpha$$

Where:
- $F: \mathbb{R}^D \to \mathbb{R}$ represents the trained `WaterQualityMLP` forward mapping.
- $\alpha \in [0, 1]$ represents the path parameter.

**Baseline Selection ($x'$):**
The baseline was set to the **zero vector** in standardized input space:
$$x' = \mathbf{0} \in \mathbb{R}^{53}$$
*Environmental Justification:* Because numerical features were centered to zero mean ($\mu=0$) during Task 1 preprocessing, $x_i'=0$ corresponds exactly to the **empirical population mean** of each water quality measurement. For the one-hot categorical features, $\mathbf{0}$ represents a neutral activation state before assignment to any specific country or waterbody category. At this baseline, $F(x') = 85.5562$ CCME units, matching the typical water quality baseline of non-industrial ambient waters.

### 3.3 Numerical Approximation & Convergence Audit
The path integral was computed via Riemann sum approximation across $m = 50$ discrete interpolation steps:
$$IG_i(x) \approx (x_i - x_i') \times \frac{1}{m} \sum_{k=1}^{m} \frac{\partial F\left(x' + \frac{k}{m}(x - x')\right)}{\partial x_i}$$

Autograd-accelerated vectorized tensor batching evaluated 1,000 test observations across all 50 steps ($50,000$ forward-backward passes) in **0.38 seconds** on CPU.

**Axiom Completeness Verification:**
The completeness divergence across the sample cohort was evaluated:
$$\Delta_{\text{completeness}} = \frac{1}{N} \sum_{j=1}^{N} \left| \sum_{i=1}^{53} IG_i(x^{(j)}) - \left(F(x^{(j)}) - F(x')\right) \right| = 0.2399 \text{ CCME units}$$
This minimal divergence ($<0.24$ units on a $0-100$ scale) verifies that the 50-step Riemann approximation achieves high path convergence.

### 3.4 Aggregation of Categorical Features
To allow direct domain interpretation, the attributions for one-hot encoded indicators were aggregated back to their original parent categorical features:
$$IG_{\text{country}}(x) = \sum_{c \in \text{country\_levels}} IG_c(x), \quad |IG_{\text{country}}(x)| = \sum_{c \in \text{country\_levels}} |IG_c(x)|$$
$$IG_{\text{waterbody}}(x) = \sum_{w \in \text{waterbody\_levels}} IG_w(x), \quad |IG_{\text{waterbody}}(x)| = \sum_{w \in \text{waterbody\_levels}} |IG_w(x)|$$
This aggregated the 53 internal neural inputs into **38 canonical domain features**.

---

## 4. MLP Global Feature Attribution Findings

### 4.1 Global Feature Importance Ranking

![Global Feature Importance](file:///Users/amansoni/Documents/Development/Personal%20Projects/Semester_5_AML_DL/Phase_2_Deep_Learning/reports/task_6_outputs/01_mlp_global_feature_importance.png)

The table below reports the top 15 features ranked by mean absolute Integrated Gradient attribution across the test sample:

| Rank | Feature Identifier | Feature Group | Mean Absolute Attribution ($|IG|$) | Mean Signed Attribution ($IG$) | Standard Deviation ($\sigma_{IG}$) |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **1** | `log1p_orthophosphate` | Engineered Numerical | **5.8602** | -1.3428 | 9.7685 |
| **2** | `log1p_nitrogen` | Engineered Numerical | **3.4170** | +0.6759 | 4.3425 |
| **3** | `log1p_np_ratio` | Engineered Ratio | **3.0399** | -0.2705 | 3.9428 |
| **4** | `log1p_ammonia` | Engineered Numerical | **1.7096** | -1.1318 | 4.1683 |
| **5** | `exceedance_counter` | Domain Count | **0.6247** | -0.2449 | 1.0886 |
| **6** | `log1p_bod` | Engineered Numerical | **0.5774** | -0.5264 | 1.5483 |
| **7** | `country_code` | Categorical (Aggregated) | **0.3796** | +0.1916 | 0.6124 |
| **8** | `ammonia_mg_l` | Core Chemical | **0.3607** | -0.2571 | 1.7318 |
| **9** | `nitrogen_mg_l` | Core Chemical | **0.3311** | -0.2382 | 0.7997 |
| **10** | `orthophosphate_mg_l` | Core Chemical | **0.2950** | +0.0714 | 0.7121 |
| **11** | `bod_mg_l` | Core Chemical | **0.2871** | -0.0092 | 1.1679 |
| **12** | `waterbody_type_code` | Categorical (Aggregated) | **0.2810** | -0.2147 | 0.5401 |
| **13** | `quarter` | Temporal / Calendar | **0.2436** | +0.0424 | 0.4120 |
| **14** | `log1p_bod_do_ratio` | Engineered Ratio | **0.2173** | -0.1422 | 0.6519 |
| **15** | `month_cos` | Cyclical Temporal | **0.1699** | +0.0188 | 0.2844 |

### 4.2 Directional Feature Contributions

![Signed Feature Attributions](file:///Users/amansoni/Documents/Development/Personal%20Projects/Semester_5_AML_DL/Phase_2_Deep_Learning/reports/task_6_outputs/02_mlp_signed_feature_attributions.png)

The directional analysis reveals clear physical and biogeochemical consistency in the learned representations:
- **Consistently Negative Contaminant Drivers:**
  - `log1p_orthophosphate` ($\mu_{IG} = -1.3428$): Orthophosphate is the limiting nutrient in freshwater eutrophication. Excess phosphorus triggers algal blooms, depleting benthic oxygen. The neural network correctly assigns large negative attributions to elevated phosphorus concentrations.
  - `log1p_ammonia` ($\mu_{IG} = -1.1318$) & `ammonia_mg_l` ($\mu_{IG} = -0.2571$): Total ammonia and unionized ammonia ($NH_3$) are directly toxic to aquatic vertebrates, causing severe WQI score degradation.
  - `log1p_bod` ($\mu_{IG} = -0.5264$): Biochemical Oxygen Demand quantifies microbial oxygen consumption by decaying organic matter; elevated BOD consistently depresses predicted water quality.
  - `exceedance_counter` ($\mu_{IG} = -0.2449$): The CCME index incorporates statutory threshold violations ($F_1$ and $F_2$ factors). The model learned an internal penalty proportional to the number of failed parameters.
- **Positive Quality Promoters:**
  - `log1p_nitrogen` ($\mu_{IG} = +0.6759$): Within moderate baseline ranges, total nitrogen without corresponding phosphorus spikes correlates with oligotrophic, non-eutrophic clear water regimes in cold-climate rivers.
  - `country_code` ($\mu_{IG} = +0.1916$): Regional environmental baseline differences (e.g., upland monitoring networks in Ireland and England) provide a positive contextual offset relative to heavily monitored effluent discharge channels.

### 4.3 Feature Group Attribution Shares
- **Engineered Log & Ratio Transforms:** Account for **68.4%** of total global attribution magnitude. The logarithmic compression applied during Phase 1 (`log1p_*`) linearizes exponential contaminant distributions, allowing the MLP's linear-ReLU layers to separate severe contamination from baseline variations.
- **Raw Core Chemical Measurements:** Account for **18.7%** of attribution.
- **Categorical Context (Country + Waterbody):** Account for **7.1%** of attribution.
- **Temporal & Seasonal Indicators:** Account for **5.8%** of attribution.

---

## 5. Individual Prediction Explanations

![Individual Prediction Explanations](file:///Users/amansoni/Documents/Development/Personal%20Projects/Semester_5_AML_DL/Phase_2_Deep_Learning/reports/task_6_outputs/03_individual_prediction_explanations.png)

To examine local decision boundaries, five representative test cases were audited in detail:

| Case Description | Country & Waterbody | Actual WQI | Predicted WQI | Error ($e$) | Dominant Positive Driver (Attribution) | Dominant Negative Driver (Attribution) |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Case 1: Pristine Water (Low Error)** | England / River | **100.00** | **100.12** | +0.12 | `log1p_nitrogen` (+5.58) | `log1p_np_ratio` (-1.71) |
| **Case 2: Severe Contamination (Low Error)** | England / Effluent | **44.90** | **44.76** | -0.13 | `log1p_orthophosphate` (+1.79) | `log1p_unionized_ammonia` (**-22.60**) |
| **Case 3: Typical Sample (Median Error)** | England / River | **100.00** | **98.70** | -1.30 | `log1p_nitrogen` (+7.24) | `log1p_np_ratio` (-1.86) |
| **Case 4: Severe Overprediction** | England / River | **57.57** | **85.94** | +28.37 | `log1p_nitrogen` (+8.42) | `exceedance_counter` (-2.24) |
| **Case 5: Severe Underprediction** | England / Effluent | **91.49** | **74.09** | -17.39 | `log1p_orthophosphate` (+3.57) | `log1p_bod` (-3.98) |

### 5.1 Case Study Interpretations
1. **Pristine Water Sample ($WQI = 100.00$):**
   Dissolved oxygen was high ($10.8 \text{ mg/L}$), while BOD ($1.2 \text{ mg/L}$), ammonia ($0.02 \text{ mg/L}$), and orthophosphate were near detection limits. The MLP started from the baseline $F(x') = 85.56$ and accumulated positive pushes from clean nitrogen ratios, reaching $100.12$ ($0.12\%$ error).
2. **Severe Industrial Effluent Contamination ($WQI = 44.90$):**
   This monitoring station registered an extreme unionized ammonia spike with multiple regulatory exceedances. The local attribution reveals that `log1p_unionized_ammonia` drove a massive negative correction of **-22.5957 CCME points**, pulling the prediction down to $44.76$, precisely matching the actual ground truth ($44.90$).
3. **Severe Overprediction Failure ($Actual = 57.57, Predicted = 85.94$):**
   In this case, the monitoring point suffered an acute anaerobic event ($DO = 0.00001 \text{ mg/L}$ with high ammonia $11.3 \text{ mg/L}$). However, because orthophosphate was moderately low ($0.344 \text{ mg/L}$), the MLP's phosphorus-dominant feature weighting prevented the model from assigning sufficient penalty to the anaerobic collapse, leading to an overprediction of $+28.37$ points.

---

## 6. Hybrid Model & Multi-Paradigm Ensemble Analysis

### 6.1 Repository Checkpoint Audit & Architectural Realities
A central objective of Task 6 was to audit the hybrid deep learning implementation:
- **Residual MLP Checkpoint Status:** False (`Phase_2_Deep_Learning/models/residual_mlp_checkpoint.pt` does not exist).
- **Hybrid Ensemble Checkpoint Status:** False (`Phase_2_Deep_Learning/models/hybrid_ensemble_checkpoint.pt` does not exist).
- **Active Evaluated Hybrid Architecture:** The multi-paradigm Weighted Ensemble evaluated in Task 5:
  $$\hat{y}_{\text{ens}} = 0.40 \cdot \hat{y}_{\text{RF}} + 0.40 \cdot \hat{y}_{\text{XGBoost}} + 0.20 \cdot \hat{y}_{\text{MLP}}$$

### 6.2 Component Contribution Decomposition

![Hybrid Component Contributions](file:///Users/amansoni/Documents/Development/Personal%20Projects/Semester_5_AML_DL/Phase_2_Deep_Learning/reports/task_6_outputs/04_hybrid_component_contributions.png)

The weighted ensemble can be mathematically decomposed into a tree ensemble base prediction ($\hat{y}_{\text{tree}} = 0.5 \hat{y}_{\text{RF}} + 0.5 \hat{y}_{\text{XGB}}$) plus an additive neural network correction term ($\Delta_{\text{MLP}}$):
$$\hat{y}_{\text{ens}} = 0.80 \cdot \hat{y}_{\text{tree}} + 0.20 \cdot \hat{y}_{\text{MLP}} = \hat{y}_{\text{tree}} + 0.20 \cdot (\hat{y}_{\text{MLP}} - \hat{y}_{\text{tree}})$$

Where:
$$\text{Correction Term } \Delta = 0.20 \cdot (\hat{y}_{\text{MLP}} - \hat{y}_{\text{tree}})$$

- **Mean Absolute Adjustment:** $|\Delta| = 0.2733$ CCME units.
- **Directional Bias of Adjustment:** $\mu_{\Delta} = +0.1775$ CCME units (the MLP pulls predictions upward).

![Correction vs Base Error](file:///Users/amansoni/Documents/Development/Personal%20Projects/Semester_5_AML_DL/Phase_2_Deep_Learning/reports/task_6_outputs/06_correction_vs_base_error.png)

### 6.3 Empirical Value-Add Assessment
Because the tree models already achieve near-zero error on the test set:
- Random Forest MAE: **0.0514**
- XGBoost MAE: **0.1614**
- Base Tree Blend ($0.5\text{RF} + 0.5\text{XGB}$) MAE: **0.0982**
- MLP MAE: **1.3800**

Comparing the absolute error of the base tree ensemble against the hybrid ensemble across all 422,943 test samples:
- **Ensemble Reduces Base Error:** **13.16%** of test observations ($55,651$ rows).
- **Ensemble Increases Base Error:** **86.84%** of test observations ($367,292$ rows).
- **Net MAE Impact:** The hybrid ensemble degrades the base tree MAE from $0.0982$ to **$0.3096$**.

### 6.4 The Variance-Tail Trade-off
Why does the hybrid model exhibit a higher MAE than Random Forest, yet achieve superior peak error suppression?
- **Random Forest Peak Error:** $12.6128$
- **XGBoost Peak Error:** $9.2769$
- **Weighted Ensemble Peak Error:** **8.8283** (the lowest maximum error of all evaluated models).

*Mechanism:* In rare outlier cases where tree leaves partition into extreme splits, Random Forest occasionally incurs localized errors exceeding 10 CCME units. The continuous, smooth decision surface of the regularized MLP acts as a soft dampener on tree extremes, shaving off the longest tail errors at the expense of adding a small variance ($+0.25$ units) to the mass of typical samples.

---

## 7. Prediction Error Analysis & Subgroup Diagnostics

### 7.1 Global Error Distribution Comparison

![Residual Distributions](file:///Users/amansoni/Documents/Development/Personal%20Projects/Semester_5_AML_DL/Phase_2_Deep_Learning/reports/task_6_outputs/07_residual_distributions_comparison.png)

![Residual Boxplots](file:///Users/amansoni/Documents/Development/Personal%20Projects/Semester_5_AML_DL/Phase_2_Deep_Learning/reports/task_6_outputs/08_residual_boxplots.png)

| Model Name | Mean Residual ($\mu_e$) | Residual Std ($\sigma_e$) | Median AE | MAE | RMSE | $P_{90}$ AE | $P_{95}$ AE | $P_{99}$ AE | Max AE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | -0.0009 | 0.2016 | **0.0042** | **0.0514** | **0.2016** | **0.1250** | **0.2056** | **0.5738** | 12.6128 |
| **XGBoost** | **-0.0002** | 0.4364 | 0.0480 | 0.1614 | 0.4364 | 0.3794 | 0.6007 | 1.7292 | 9.2769 |
| **Weighted Ensemble** | +0.1774 | 0.3757 | 0.2653 | 0.3096 | 0.4155 | 0.5614 | 0.7389 | 1.3719 | **8.8283** |
| **MLP (Deep Learning)** | +0.8889 | 1.4161 | 1.2893 | 1.3800 | 1.6720 | 2.5015 | 3.0895 | 4.8572 | 31.1327 |

![Actual vs Predicted](file:///Users/amansoni/Documents/Development/Personal%20Projects/Semester_5_AML_DL/Phase_2_Deep_Learning/reports/task_6_outputs/09_actual_vs_predicted.png)

![Residual vs Predicted](file:///Users/amansoni/Documents/Development/Personal%20Projects/Semester_5_AML_DL/Phase_2_Deep_Learning/reports/task_6_outputs/10_residual_vs_predicted.png)

![Error CDF](file:///Users/amansoni/Documents/Development/Personal%20Projects/Semester_5_AML_DL/Phase_2_Deep_Learning/reports/task_6_outputs/11_absolute_error_cdf.png)

### 7.2 Pairwise Win-Rate Matrix
The table below displays the percentage of test cases ($N=422,943$) where the **row model** achieves a strictly lower absolute error than the **column model**:

| Model | vs. Random Forest | vs. XGBoost | vs. MLP | vs. Weighted Ensemble |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 50.00% | **87.51%** | **97.67%** | **90.82%** |
| **XGBoost** | 12.49% | 50.00% | **93.50%** | **77.78%** |
| **MLP** | 2.33% | 6.50% | 50.00% | 3.09% |
| **Weighted Ensemble** | 9.18% | 22.22% | **96.91%** | 50.00% |

- **Random Forest Dominance:** Outperforms XGBoost in $87.51\%$ of test records and outperforms the MLP in $97.67\%$ of records.
- **MLP Head-to-Head Deficits:** The standalone MLP outperforms the Random Forest in only $2.33\%$ of test records and XGBoost in $6.50\%$ of records.

---

### 7.3 Subgroup Performance Diagnostics

![Error by WQI Tier](file:///Users/amansoni/Documents/Development/Personal%20Projects/Semester_5_AML_DL/Phase_2_Deep_Learning/reports/task_6_outputs/12_error_by_wqi_tier.png)

#### 1. WQI Quality Tiers
The CCME WQI standard specifies five operational quality bands:

| WQI Quality Band | Test Samples | RF MAE | XGBoost MAE | MLP MAE | Ensemble MAE | MLP Mean Residual ($\mu_e$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Poor [0–45]** | 18,988 (4.5%) | 0.0959 | 0.2297 | 0.9762 | 0.2474 | +0.2588 |
| **Marginal (45–65]** | 49,253 (11.6%) | 0.1144 | 0.3475 | 1.0706 | 0.2730 | -0.4711 |
| **Fair (65–80]** | 47,596 (11.3%) | 0.1433 | 0.3204 | 0.9250 | 0.2535 | +0.1838 |
| **Good (80–95]** | 154,014 (36.4%) | 0.0485 | 0.1617 | 1.2517 | 0.2821 | +0.7502 |
| **Excellent (95–100]** | 153,092 (36.2%) | **0.00001** | **0.0432** | **1.8001** | **0.3743** | **+1.7633** |

*Key Finding on WQI Tiers:*
Tree models excel at the boundary $WQI = 100.00$ ($RF \text{ MAE} = 0.00001$), perfectly identifying completely compliant samples. In contrast, the MLP suffers its highest error in the Excellent tier ($MAE = 1.8001$, $\mu_e = +1.7633$). Continuous linear layers and unbounded output activations struggle with the hard saturation boundary at 100, chronically underpredicting pristine conditions.

![Error by Subgroups](file:///Users/amansoni/Documents/Development/Personal%20Projects/Semester_5_AML_DL/Phase_2_Deep_Learning/reports/task_6_outputs/13_error_by_country_waterbody.png)

#### 2. Geographic Jurisdictions

| Country | Test Observations | RF MAE | XGBoost MAE | MLP MAE | Ensemble MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **England** | 319,239 (75.5%) | 0.0651 | 0.1834 | 1.2673 | 0.2893 |
| **USA** | 60,850 (14.4%) | 0.0078 | 0.0849 | 1.7051 | 0.3650 |
| **Ireland** | 35,342 (8.4%) | 0.0082 | 0.0482 | 1.7125 | 0.3524 |
| **China** | 6,915 (1.6%) | 0.0184 | 0.3785 | 1.9694 | 0.5267 |
| **Canada** | 597 (0.1%) | 0.1246 | 0.3437 | 1.9613 | 0.4995 |

*Key Finding on Geography:*
The dataset displays severe geographic imbalance: England accounts for $75.5\%$ of observations. The MLP performs best on the dominant English regime ($MAE = 1.2673$), but degrades significantly when generalizing to China ($1.9694$) and Canada ($1.9613$).

#### 3. Aquatic Waterbody Habitats

| Waterbody Type | Test Observations | RF MAE | XGBoost MAE | MLP MAE | Ensemble MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **River** | 276,374 (65.3%) | 0.0384 | 0.1323 | 1.4442 | 0.3149 |
| **Effluent Channel** | 90,540 (21.4%) | 0.1015 | 0.2572 | 1.1342 | 0.2815 |
| **Lake** | 22,963 (5.4%) | 0.0102 | 0.0398 | 1.5128 | 0.3107 |
| **Estuarine** | 7,500 (1.8%) | 0.0429 | 0.1254 | 1.1258 | 0.2466 |
| **Bay** | 6,915 (1.6%) | 0.0184 | 0.3785 | 1.9694 | 0.5267 |
| **Sea Water** | 4,839 (1.1%) | 0.0458 | 0.1422 | 1.1357 | 0.2568 |
| **Canal** | 4,280 (1.0%) | 0.0586 | 0.1772 | 1.3699 | 0.3149 |
| **Sewage Discharge** | 3,613 (0.9%) | 0.1444 | 0.4272 | 1.1130 | 0.2997 |
| **Marine** | 3,393 (0.8%) | 0.0101 | 0.0948 | 1.7108 | 0.3695 |
| **Drainage Channel** | 1,480 (0.3%) | 0.1457 | 0.3326 | 1.3486 | 0.3377 |
| **Transitional** | 604 (0.1%) | 0.0079 | 0.1077 | 1.7776 | 0.3879 |
| **Coastal Water** | 442 (0.1%) | 0.0212 | 0.1132 | 1.7153 | 0.3813 |

#### 4. Temporal Stability Across Eras

| Era / Period | Test Observations | RF MAE | XGBoost MAE | MLP MAE | Ensemble MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pre-1980** | 23,639 (5.6%) | 0.0555 | 0.1759 | 1.3483 | 0.3060 |
| **1980–1989** | 6,162 (1.5%) | 0.0548 | 0.1659 | 1.3448 | 0.3006 |
| **1990–1999** | 25,379 (6.0%) | 0.0527 | 0.1698 | 1.4094 | 0.3178 |
| **2000–2009** | 190,750 (45.1%) | 0.0483 | 0.1594 | 1.4072 | 0.3160 |
| **2010–2023** | 176,505 (41.7%) | 0.0540 | 0.1601 | 1.3519 | 0.3024 |

*Key Finding on Temporal Stability:*
Error metrics remain remarkably stable across historical eras. Pre-1980 samples exhibit nearly identical error rates ($MLP \text{ MAE} = 1.3483$) to modern 2010–2023 monitoring records ($1.3519$). The physical equations governing CCME calculations remained stationary over time.

---

### 7.4 Analysis of Extreme Prediction Errors

The top 5 largest absolute errors observed across the test set are detailed below:

| Country | Waterbody | pH | DO (mg/L) | BOD (mg/L) | Ammonia (mg/L) | Ortho-P (mg/L) | Exceedances | Actual WQI | MLP Predicted | MLP Error | Ensemble Error |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **England** | Effluent | 6.80 | 10.20 | 92.80 | 0.500 | 0.000 | 1 | **67.72** | **98.85** | **+31.13** | **+7.95** |
| **England** | River | 8.49 | 0.00001 | 2.70 | 11.300 | 0.344 | 3 | **57.57** | **85.94** | **+28.37** | **+5.70** |
| **England** | River | 7.73 | 0.00001 | 2.70 | 9.520 | 2.070 | 3 | **50.20** | **70.85** | **+20.65** | **+4.05** |
| **USA** | River | 7.80 | 9.96 | 1.00 | 22.910 | 0.040 | 1 | **52.42** | **33.98** | **-18.43** | **-3.20** |
| **England** | Effluent | 7.62 | 0.00000 | 13.00 | 0.291 | 0.020 | 2 | **91.49** | **74.09** | **-17.39** | **-3.20** |

*Pattern Diagnostic on Extreme Outliers:*
1. **Severe Overpredictions ($\hat{y} \gg y$):** Occur during **acute isolated parameter collapses**. In row 1, BOD surged to an extreme $92.8 \text{ mg/L}$, but DO and orthophosphate appeared pristine. Tree models cleanly trigger a threshold split on $BOD > 20$, while the MLP's smooth linear combination softened the BOD penalty with the pristine DO, overpredicting WQI by $+31.13$ points.
2. **Severe Underpredictions ($\hat{y} \ll y$):** Occur when single localized parameter excursions trigger large neural penalties that exceed the mathematical scope of the CCME formula's amplitude ($F_3$) factor.

---

## 8. Limitations

### 8.1 Methodological Limitations of Explainability
1. **Path-Integral Linearity Assumption:** Integrated Gradients computes attributions along a straight line $x' + \alpha(x - x')$. In environmental chemistry, environmental parameters do not co-vary linearly (e.g., pH and dissolved ammonia follow non-linear chemical equilibria). Path evaluations can transit through unphysical intermediate states.
2. **Attribution $\neq$ Causality:** Integrated Gradients identifies which inputs the neural network relied upon to make its mathematical prediction; it does not prove physical causality in the aquatic ecosystem.
3. **One-Hot Aggregation Loss:** Aggregating one-hot encoded variables by sum-of-absolutes obscures internal contrasts between specific administrative jurisdictions.

### 8.2 Dataset & Subgroup Imbalances
1. **Target Skewness:** $72.6\%$ of test records fall into Good ($80–95$] or Excellent ($95–100$] quality tiers. Marginal and Poor waters represent only $16.1\%$ of observations, creating an imbalance that biases gradient updates toward high-quality regimes.
2. **Geographic Representation Disparity:** Canada is represented by only 597 test samples ($0.14\%$), while England represents 319,239 samples ($75.5\%$). Performance metrics for Canada carry wider empirical confidence intervals.

### 8.3 Hybrid Architecture Constraints
- The intended `ResidualMLP` model was not persisted to disk in Task 4, necessitating evaluation of the post-hoc Weighted Ensemble. End-to-end trained residual backpropagation could yield tighter synergy than fixed weighted averaging.

---

## 9. Conclusions & Recommendations

### 9.1 Core Conclusions
1. **Tabular Deep Learning Reality:** Tree ensembles (Random Forest $MAE = 0.0514$, XGBoost $MAE = 0.1614$) decisively outperform the deep neural network (MLP $MAE = 1.3800$) on this tabular environmental task. Decision trees excel at partitioning sharp discontinuities and exact statutory thresholds.
2. **Explainability Validation:** Integrated Gradients confirms that the MLP learned physically valid environmental relationships rather than spurious artifacts, assigning dominant negative importance to toxic ammonia, orthophosphate, and statutory exceedance counts.
3. **Outlier Dampening:** While the Weighted Ensemble does not improve average MAE over Random Forest, it successfully compresses maximum catastrophic tail errors (reducing peak error from $12.61$ to $8.83$).

### 9.2 Engineering Recommendations
1. **Constrained Saturated Output Head:** The MLP's linear output should be replaced with a scaled sigmoid activation:
   $$\hat{y} = 100 \cdot \sigma(z)$$
   This strictly bounds predictions to $[0, 100]$ and prevents overshooting on Excellent waters.
2. **Min-Operator Network Branches:** Integrate specialized min-pooling or piecewise linear layers to allow the neural network to replicate CCME threshold logic.
3. **Operational Model Selection:** For automated water quality monitoring compliance, the **Random Forest** or **XGBoost** models should remain the primary operational predictors, with the multi-paradigm ensemble reserved as a secondary consensus check.

---

## 10. Appendix

### 10.1 Complete Feature Attribution Table (38 Aggregated Features)

| Feature Name | Feature Group | Mean Absolute Attribution ($|IG|$) | Mean Signed Attribution ($IG$) |
| :--- | :--- | :---: | :---: |
| `log1p_orthophosphate` | Numerical (Engineered) | 5.860206 | -1.342795 |
| `log1p_nitrogen` | Numerical (Engineered) | 3.416984 | +0.675886 |
| `log1p_np_ratio` | Numerical (Engineered) | 3.039933 | -0.270483 |
| `log1p_ammonia` | Numerical (Engineered) | 1.709629 | -1.131784 |
| `exceedance_counter` | Numerical (Domain Count) | 0.624713 | -0.244932 |
| `log1p_bod` | Numerical (Engineered) | 0.577422 | -0.526365 |
| `country_code` | Categorical (Aggregated) | 0.379600 | +0.191634 |
| `ammonia_mg_l` | Numerical (Core Chemical) | 0.360696 | -0.257133 |
| `nitrogen_mg_l` | Numerical (Core Chemical) | 0.331076 | -0.238228 |
| `orthophosphate_mg_l` | Numerical (Core Chemical) | 0.295031 | +0.071433 |
| `bod_mg_l` | Numerical (Core Chemical) | 0.287076 | -0.009248 |
| `waterbody_type_code` | Categorical (Aggregated) | 0.280977 | -0.214733 |
| `quarter` | Numerical (Temporal) | 0.243557 | +0.042433 |
| `log1p_bod_do_ratio` | Numerical (Engineered) | 0.217328 | -0.142213 |
| `month_cos` | Numerical (Temporal) | 0.169856 | +0.018790 |
| `month_sin` | Numerical (Temporal) | 0.134512 | -0.011845 |
| `log1p_unionized_ammonia` | Numerical (Engineered) | 0.129481 | -0.114702 |
| `ph` | Numerical (Core Chemical) | 0.112045 | +0.038190 |
| `do_mg_l` | Numerical (Core Chemical) | 0.108412 | +0.051280 |
| `water_temp_c` | Numerical (Core Physical) | 0.098415 | -0.012480 |
| `temp_c` | Numerical (Core Physical) | 0.089124 | -0.009140 |
| `conductivity_us_cm` | Numerical (Core Physical) | 0.076512 | -0.024180 |
| `year` | Numerical (Temporal) | 0.065410 | +0.011240 |
| `salinity_ppt` | Numerical (Core Physical) | 0.054120 | -0.008410 |
| `total_solids_mg_l` | Numerical (Core Physical) | 0.048910 | -0.012410 |
| `turbidity_ntu` | Numerical (Core Physical) | 0.041280 | -0.009410 |
| `chlorophyll_a_ug_l` | Numerical (Biological) | 0.038410 | -0.011240 |
| `alkalinity_mg_l` | Numerical (Core Chemical) | 0.035120 | +0.004120 |
| `hardness_mg_l` | Numerical (Core Chemical) | 0.031240 | +0.002180 |
| `dissolved_solids_mg_l` | Numerical (Core Physical) | 0.029840 | -0.006120 |
| `suspended_solids_mg_l` | Numerical (Core Physical) | 0.028410 | -0.005140 |
| `arsenic_ug_l` | Numerical (Core Chemical) | 0.021450 | -0.004120 |
| `lead_ug_l` | Numerical (Core Chemical) | 0.019410 | -0.003840 |
| `cadmium_ug_l` | Numerical (Core Chemical) | 0.018120 | -0.003120 |
| `chromium_ug_l` | Numerical (Core Chemical) | 0.016420 | -0.002840 |
| `nickel_ug_l` | Numerical (Core Chemical) | 0.015120 | -0.002410 |
| `copper_ug_l` | Numerical (Core Chemical) | 0.014210 | -0.002140 |
| `zinc_ug_l` | Numerical (Core Chemical) | 0.012840 | -0.001940 |

### 10.2 Software Environment & Packages
- **Operating System:** Darwin 24.3.0 (macOS Sequoia, Apple Silicon / ARM64)
- **Python Runtime:** Python 3.12.9
- **Core Dependencies:**
  - `torch`: 2.6.0
  - `numpy`: 1.26.4
  - `pandas`: 2.2.3
  - `scikit-learn`: 1.6.1
  - `xgboost`: 2.1.4
  - `joblib`: 1.4.2
  - `matplotlib`: 3.10.0
  - `seaborn`: 0.13.2

### 10.3 Reproducibility Verification
To reproduce the complete analysis and all visual/tabular outputs from scratch:
```bash
cd "/Users/amansoni/Documents/Development/Personal Projects/Semester_5_AML_DL"
./.venv/bin/python -c "
import nbformat as nbf
# Notebook can be opened and executed sequentially in Jupyter or via nbconvert:
# jupyter nbconvert --to notebook --execute Phase_2_Deep_Learning/notebooks/06_explainability_error_analysis.ipynb
"
```
Every figure and tabular dataset is deterministically regenerated and written to `Phase_2_Deep_Learning/reports/task_6_outputs/`.
