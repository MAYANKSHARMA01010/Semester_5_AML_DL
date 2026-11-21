# Water Quality Prediction via Gradient Boosting and Deep Neural Networks

> **Course:** Semester 5 Applied Machine Learning & Deep Learning (CSAI)  
> **Master Dataset:** 2,819,617 clean complete-case telemetry observations across 5 nations (1940–2023)  
> **Target Continuous Index:** Canadian Council of Ministers of the Environment Water Quality Index (**CCME WQI**, $y \in [0, 100]$)  
> **Repository:** [MAYANKSHARMA01010/Semester_5_AML_DL](https://github.com/MAYANKSHARMA01010/Semester_5_AML_DL)  
> **Test Set Evaluation Partition:** 422,943 held-out test observations  

---

## 1. Project Overview & Problem Statement

Freshwater ecosystems face severe ecological degradation from anthropogenic nutrient pollution, agricultural runoff, industrial discharge, and urban effluent. Effective water resource management requires continuous, accurate quantification of overall water quality from physical and chemical sensor telemetry.

Traditional laboratory assessments relying on the Canadian Council of Ministers of the Environment Water Quality Index (CCME WQI) require manual multi-parameter aggregation and non-linear formula evaluation. This project develops a unified machine learning and deep learning framework to predict continuous CCME Water Quality Index values directly from multi-sensor telemetry, comparing tree ensembles, deep neural networks, and multi-paradigm residual hybrids across 2.82 million observations.

---

## 2. Dataset Source & Description

The experimental corpus spans 83 years of environmental monitoring (1940–2023) aggregated across five nations:

| Country | Clean Observations | Proportion | Source Sensor Network |
| :--- | :---: | :---: | :--- |
| **England** | 2,361,042 | 83.74% | UK Environment Agency Water Quality Archive (`data.gov.uk`) |
| **USA** | 239,890 | 8.51% | USGS National Water Information System (NWIS) + US EPA WQP |
| **Ireland** | 200,980 | 7.13% | Environmental Protection Agency (EPA) Ireland Catchments |
| **China** | 11,985 | 0.43% | Major River Basin Environmental Monitoring Stations |
| **Canada** | 5,720 | 0.20% | Historical Reference Telemetry Basins |
| **Total** | **2,819,617** | **100.00%** | **Multi-National Water Quality Monitoring Corpus** |

### Data Partitions
- **Training Set (70%):** 1,973,730 observations (transparently sharded into 3 sub-parts under 60 MB each for GitHub compatibility).
- **Validation Set (15%):** 422,944 observations.
- **Held-Out Test Set (15%):** 422,943 observations.

---

## 3. Target Definition: CCME Water Quality Index

The target variable is the continuous CCME WQI score ($y \in [0, 100]$), representing overall aquatic health based on Canadian regulatory guidelines:

$$WQI = 100 - \frac{\sqrt{F_1^2 + F_2^2 + F_3^2}}{1.732}$$

Where:
- $F_1$ (**Scope**): Percentage of chemical parameters exceeding regulatory safety thresholds.
- $F_2$ (**Frequency**): Percentage of individual analytical tests failing compliance standards.
- $F_3$ (**Amplitude**): Normalized extent to which failed measurements deviate from guidelines.

WQI values correspond to standard environmental health classifications:
- **Excellent** ($95–100$): Pristine, near-natural aquatic conditions.
- **Good** ($80–94$): Minor degree of impairment; ecosystem protected.
- **Fair** ($65–79$): Frequently impaired conditions; parameters exceed guidelines.
- **Marginal** ($45–64$): Substantial aquatic stress; biological life degraded.
- **Poor** ($0–44$): Severely contaminated; ecosystem integrity threatened.

---

## 4. Feature Space & Input Representation

### 38 Tabular Features (Tree Ensembles)
1. **Base Physical & Chemical Sensors (8):** `ph`, `temperature_c`, `do_mg_l`, `bod_mg_l`, `ammonia_mg_l`, `orthophosphate_mg_l`, `nitrogen_mg_l`, `nitrate_mg_l`.
2. **Aquatic Stoichiometry & Thermodynamics (7):** Redfield stoichiometric $\text{N}:\text{P}$ ratio, Benson-Krause theoretical dissolved oxygen saturation %, Streeter-Phelps oxygen deficit, Emerson un-ionized toxic ammonia ($\text{NH}_3$).
3. **Log-Transformed Variance-Stabilized Features (7):** $\log(1+x)$ transformations on highly skewed nutrient parameters (orthophosphate skewness reduced from $\gamma_1 > 40$ to near-normal).
4. **Historical Temporal Lags (6):** Chronological shift-1 station lags and rolling-3 statistics with strict zero-lookahead prevention.
5. **Regulatory Stress Counter (1):** `exceedance_counter` tracking cumulative guideline threshold violations.
6. **Contextual & Categorical Identifiers (2):** `country_code` (integer-encoded jurisdiction) and `waterbody_type_code` (integer-encoded habitat: River, Lake, Canal, Estuary, etc.).
7. **Temporal & Cyclical Features (7):** Calendar year, month, season, and harmonic sinusoidal projections (`month_sin`, `month_cos`).

### 53 Deep Learning Features (MLP)
- 36 continuous features scaled using scikit-learn `StandardScaler` fitted strictly on training data.
- 17 binary indicators from One-Hot Encoding of `country_code` (5 categories) and `waterbody_type_code` (12 categories).

---

## 5. Model Architectures & Methodologies

### Phase 1: Machine Learning Baselines
- **Mean Baseline Regressor:** Predicts the constant training mean ($\bar{y} \approx 86.72$, $\text{MAE} = 14.130$).
- **Random Forest Regressor:** Bagging ensemble of 30 trees (`max_depth=14`, `min_samples_leaf=5`, `max_features=1.0`). Captures dense regional feature distributions.
- **XGBoost Regressor:** Gradient boosted decision trees (150 estimators, `learning_rate=0.08`, `max_depth=6`, histogram-based split finding). High throughput ($4.6\text{ ms}$ latency).

### Phase 2: Deep Learning Neural Network
- **WaterQualityMLP:** Deep multi-layer perceptron with funnel topology `53 → 512 → 256 → 128 → 64 → 1`. Incorporates Batch Normalization, Dropout ($p = 0.3$), ReLU activations, and Kaiming He normal weight initialization. Trained with AdamW optimizer ($\text{lr}=10^{-3}$, $\text{weight\_decay}=10^{-4}$) under MSE loss.

### Phase 2: Hybrid ML + DL Architecture
The **Residual Hybrid** architecture combines the strengths of tree ensembles and deep neural networks:

$$\hat{y}_{\text{hybrid}} = \hat{y}_{\text{base\_tree}} + \hat{r}$$

1. **Base Tree Ensemble ($\hat{y}_{\text{base\_tree}}$):** Equal-weighted average of Random Forest and XGBoost ($0.5\times\text{RF} + 0.5\times\text{XGB}$).
2. **Residual Neural Network (`ResidualMLPNet`):** Takes **54 inputs** (53 DL features + 1 base tree prediction scalar) and predicts the signed residual error ($\hat{r} \approx y - \hat{y}_{\text{base}}$).
3. **Topology:** `54 → 256 → 128 → 64 → 1` with BatchNorm1d, Dropout ($p=0.2$), trained for 50 epochs on leakage-free 5-fold Out-of-Fold (OOF) residuals using Smooth L1 (Huber) loss.

---

## 6. Comprehensive Test Set Performance Comparison

All models were evaluated on the exact same held-out test split (`test.parquet`, $N = 422,943$ records):

| Model Architecture | Paradigm | Input Dimension | MAE | RMSE | $R^2$ | MedAE | MaxAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Bagging Trees | 38 Tabular | **0.0514** | 0.2016 | 0.999870 | **0.0042** | 12.6128 |
| **Residual Hybrid** | Tree + Residual MLP | 38 Tab + 53 DL | 0.0710 | **0.1678** | **0.999910** | 0.0229 | **5.6842** |
| **Base Tree Ensemble** | Equal-Weighted Trees | 38 Tabular | 0.0900 | 0.2464 | 0.999805 | 0.0261 | 8.7128 |
| **XGBoost** | Gradient Boosting | 38 Tabular | 0.1614 | 0.4364 | 0.999389 | 0.0480 | 9.2769 |
| **Weighted Ensemble** | Convex Blend ($0.4\text{RF}+0.4\text{XGB}+0.2\text{MLP}$) | 38 Tab + 53 DL | 0.3096 | 0.4155 | 0.999446 | 0.2653 | 8.8283 |
| **WaterQualityMLP** | Standalone Deep Net | 53 (36 num + 17 OHE) | 1.3800 | 1.6720 | 0.991029 | 1.2893 | 31.1327 |

### Core Empirical Findings:
1. **Outlier Compression:** The Residual Hybrid cuts the maximum peak prediction error by **54.9%** (from $12.6128$ in Random Forest down to **$5.6842$**), eliminating severe step-function boundary errors in extreme pollutant regimes.
2. **Lowest Global Variance:** Residual Hybrid achieves the lowest RMSE across all evaluated models (**$0.1678$**, a **31.9% variance reduction** over the Base Tree Ensemble).
3. **Tabular Boundary Dynamics:** Standalone Random Forest achieves the lowest median absolute deviation across the dense central distribution ($\text{MAE}=0.0514$), while the Residual Hybrid dominates on distribution tails and high-error samples.

---

## 7. Model Explainability & Prediction Error Analysis

### Integrated Gradients Feature Attributions
Using path-integral Integrated Gradients with a valid modal observed reference baseline (clean river water in England, $F(x') = 86.7233$ WQI units) across $N = 1{,}000$ test samples ($m=50$ steps, mean completeness error $< 0.26$ WQI units):
- **Primary Negative Attributions (Pollutants):** `log1p_orthophosphate` (mean absolute attribution $2.9554$), `log1p_nitrogen` ($2.2359$), `log1p_np_ratio` ($1.8906$), and `log1p_ammonia` ($1.2046$) strongly suppress predicted WQI.
- **Primary Positive Attributions:** Dissolved oxygen saturation and low exceedance counters promote elevated water quality.
- **Secondary Contextual Drivers:** Waterbody habitat type ($1.0425$) and country code ($0.6698$) provide contextual regional calibration.

### Prediction Error Diagnostics ($\text{Residual} = \hat{y} - y$)
- **Zero-Bias Centering:** Residual distributions across tree models and the hybrid model exhibit near-zero mean residuals ($+0.0005$ to $+0.0055$ WQI units).
- **Subgroup Vulnerabilities:** The highest error rates occur in heavily polluted industrial effluent channels and transitional estuarine waters due to unobserved localized contaminants.

---

## 8. Repository Layout

```text
Semester_5_AML_DL/
├── Final_Visualizations/                      # Centralized directory for report and PPT figures
│   ├── EDA/                                   # 13 exploratory analysis figures
│   ├── Phase_1/                               # 6 Phase 1 model evaluation figures
│   ├── Phase_2/                               # 5 Phase 2 neural network & ablation figures
│   ├── Model_Comparison/                      # 3 unified 6-model comparison plots
│   ├── Explainability/                        # 3 Integrated Gradients attribution figures
│   ├── Error_Analysis/                        # 5 residual diagnostic plots
│   ├── Hybrid_Model/                          # 6 hybrid mechanics and error correlation plots
│   └── README.md                              # Comprehensive visual catalog
│
├── Phase_1_Machine_Learning/                  # Submitted Phase 1 Machine Learning pipeline
│   ├── notebooks/                             # 6 standardized notebooks (EDA to evaluation)
│   ├── data/                                  # Cleaned datasets, Parquet shards, metadata
│   ├── src/                                   # Core Phase 1 utility modules
│   ├── tests/                                 # 12 automated unit tests (100% passing)
│   └── reports/                               # Phase 1 IEEE report, presentation, and guides
│
├── Phase_2_Deep_Learning/                     # Phase 2 Deep Learning pipeline
│   ├── notebooks/                             # 6 standardized notebooks (Tasks 1–6)
│   ├── models/                                # Serialized checkpoints (MLP, ResidualMLP, Hybrid)
│   ├── artifacts/                             # Scalers, encoders, predictions, config JSONs
│   ├── src/                                   # Modular Python components (dl_models, hybrid, etc.)
│   ├── scripts/                               # Background runners for benchmarks and ablations
│   ├── tests/                                 # 104 automated unit tests (100% passing)
│   └── reports/                               # Technical reports and Task 6 output data
│
├── README.md                                  # Root project documentation
├── PROJECT_VIVA_GUIDE.md                      # Comprehensive academic viva defense handbook
├── FINAL_PROJECT_VERIFICATION.md              # Final submission verification report
└── requirements.txt                           # Project dependencies
```

---

## 9. Environment Setup & Notebook Execution

### Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/MAYANKSHARMA01010/Semester_5_AML_DL.git
cd Semester_5_AML_DL

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running Test Suites
```bash
# Run complete test suite (116 passing tests)
pytest Phase_1_Machine_Learning/tests/ -v
pytest Phase_2_Deep_Learning/tests/ -v
```

### Notebook Execution Sequence
All notebooks follow the strict formatting standard: **zero comments in code cells** and **single-line Markdown headings only**.

1. `Phase_1_Machine_Learning/notebooks/01.EDA/01_data_understanding.ipynb`: Initial schema and distributions.
2. `Phase_1_Machine_Learning/notebooks/01.EDA/02_data_patterns_and_target.ipynb`: Target behavior and correlations.
3. `Phase_1_Machine_Learning/notebooks/02.Preprocessing&FeatureSelection/01_data_preprocessing.ipynb`: Data cleaning and feature engineering.
4. `Phase_1_Machine_Learning/notebooks/03.ModelTraining/01_baseline_and_tree_models.ipynb`: RF and XGBoost baseline training.
5. `Phase_1_Machine_Learning/notebooks/04.ModelEvaluation/01_ml_model_comparison.ipynb`: Tree baseline comparisons.
6. `Phase_1_Machine_Learning/notebooks/04.ModelEvaluation/02_feature_importance.ipynb`: Gini feature importances.
7. `Phase_2_Deep_Learning/notebooks/01_dl_data_pipeline.ipynb`: PyTorch Dataset, DataLoader, scaling, and OHE.
8. `Phase_2_Deep_Learning/notebooks/02_mlp_architecture.ipynb`: MLP layer topology and parameter analysis.
9. `Phase_2_Deep_Learning/notebooks/03_mlp_training.ipynb`: PyTorch training loop and regularization diagnostics.
10. `Phase_2_Deep_Learning/notebooks/04_hybrid_model.ipynb`: OOF residual generation and ResidualMLPNet integration.
11. `Phase_2_Deep_Learning/notebooks/05_model_evaluation.ipynb`: 6-model benchmarking and 7-group ablation studies.
12. `Phase_2_Deep_Learning/notebooks/06_explainability_error_analysis.ipynb`: Integrated Gradients and prediction error analysis.

---

## 10. Important Scientific Limitations

1. **Deterministic Target Construction:** The CCME WQI is calculated through a deterministic mathematical formula based on threshold exceedances. Consequently, machine learning models achieve high coefficients of determination ($R^2 > 0.99$), reflecting function approximation rather than purely noisy observational prediction.
2. **Feature Leakage Consideration:** The feature `exceedance_counter` directly counts threshold violations, mirroring the $F_1$ parameter of CCME. While environmentally informative, ablation studies prove that even when `exceedance_counter` is completely removed, tree models retain $R^2 > 0.999$, driven primarily by raw physical parameters (`base_sensors`).
3. **Tabular Deep Learning Gaps:** Tabular continuous datasets with step-function regulatory boundaries favor axis-aligned decision trees over smooth neural network gradient descent. The standalone MLP ($\text{MAE} = 1.3800$) is less effective than Random Forest ($\text{MAE} = 0.0514$) on median deviations, but provides essential non-linear error correction when integrated into the Residual Hybrid.
