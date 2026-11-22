# Water Quality Prediction via Gradient Boosting and Deep Neural Networks

> **Course:** Semester 5 Applied Machine Learning & Deep Learning (CSAI)  
> **Institution:** Department of Computer Science & Artificial Intelligence, Newton School of Technology, Rishihood University  
> **Master Dataset:** 2,819,617 clean complete-case telemetry observations across 5 nations (1940–2023)  
> **Target Continuous Index:** Canadian Council of Ministers of the Environment Water Quality Index (**CCME WQI**, $y \in [0, 100]$)  
> **Repository:** [MAYANKSHARMA01010/Semester_5_AML_DL](https://github.com/MAYANKSHARMA01010/Semester_5_AML_DL)  
> **Held-Out Test Set:** 422,943 observations (Strict Zero-Leakage Quarantine)  

---

## Project Overview

Freshwater ecosystems face severe ecological pressures from municipal discharges, agricultural runoff, industrial effluents, and urban stormwater. Continuous, real-time assessment of water quality is critical for environmental protection and public safety.

Conventional laboratory testing and manual index evaluations, such as the Canadian Council of Ministers of the Environment Water Quality Index (**CCME WQI**), involve multi-day incubation delays (e.g., 5-day Biochemical Oxygen Demand, $\text{BOD}_5$) and complex non-linear mathematical formulas. 

This project establishes a unified, multi-paradigm machine learning and deep learning framework to predict continuous CCME WQI values directly from physical and chemical sensor telemetry. Spanning **2.82 million international observations**, the research benchmarks tree ensembles, deep neural networks, and additive residual hybrid architectures to evaluate accuracy, inference latency, outlier robustness, and feature attribution.

---

## Project Team & Responsibilities

**Newton School of Technology, Rishihood University**

| Team Member | Phase 1 Focus | Phase 2 Technical Work Packages |
| :--- | :--- | :--- |
| **Aman Soni** | Model Training & Baselines | **Tasks 1 & 6:** DL Data Preparation, Feature Engineering, Explainability & Prediction Error Diagnostics |
| **Mayank Sharma** | Pipeline Orchestration & EDA | **Task 2:** MLP Architecture Development (PyTorch, Funnel Topology, Regularization) |
| **Chaitanya Kumar** | Feature Engineering & Stoichiometry | **Task 3:** MLP Training & Optimization (Losses, Learning Rate Schedules, Early Stopping) |
| **Ankit Kumar Pandey** | Validation & Stratification | **Task 4:** Hybrid ML + DL Model Development (5-Fold OOF Residuals, ResidualMLPNet) |
| **Aaryan Gera** | Preprocessing & Chemical Bounds | **Task 5:** Model Evaluation & Ablation Studies (6-Model Benchmarking, 7-Group Ablation, Latency) |
| **Pranav Sehgal** | Regulatory Documentation | Technical Literature Review, CCME Mathematical Derivation & Presentation Coordination |

---

## Quick Navigation to Deliverables

### Reports & Documentation
- **Final Master Project Report (10-Page IEEE Paper):** [Final_Report/Final_Project_Report.pdf](Final_Report/Final_Project_Report.pdf)
- **Phase 1 Machine Learning Report (IEEE Format):** [Phase_1_Machine_Learning/Final/Project_Report_LaTeX_IEEE_Final.pdf](Phase_1_Machine_Learning/Final/Project_Report_LaTeX_IEEE_Final.pdf)
- **Phase 2 Deep Learning Report (IEEE Format):** [Phase_2_Deep_Learning/reports/Phase_2_Report.pdf](Phase_2_Deep_Learning/reports/Phase_2_Report.pdf)
- **Phase 1 Technical Specification:** [Phase_1_Machine_Learning/docs/PREPROCESSING_SPECIFICATION.md](Phase_1_Machine_Learning/docs/PREPROCESSING_SPECIFICATION.md)
- **Phase 1 Exploratory Data Analysis Report:** [Phase_1_Machine_Learning/docs/EDA_REPORT.md](Phase_1_Machine_Learning/docs/EDA_REPORT.md)
- **Task 5 Evaluation Technical Report:** [Phase_2_Deep_Learning/reports/TASK_5_EVALUATION_REPORT.md](Phase_2_Deep_Learning/reports/TASK_5_EVALUATION_REPORT.md)
- **Task 6 Explainability Technical Report:** [Phase_2_Deep_Learning/reports/TASK_6_EXPLAINABILITY_REPORT.md](Phase_2_Deep_Learning/reports/TASK_6_EXPLAINABILITY_REPORT.md)

### Presentations
- **Final Phase 2 PowerPoint Presentation:** [Final_Report/Phase2_Presentation.pptx](Final_Report/Phase2_Presentation.pptx)
- **Phase 1 PowerPoint Presentation:** [Phase_1_Machine_Learning/Final/Project_Presentation_Final.pptx](Phase_1_Machine_Learning/Final/Project_Presentation_Final.pptx)

### Visualizations Catalog
- **Centralized Figures Index (38 Visualizations):** [Final_Visualizations/README.md](Final_Visualizations/README.md)

---

## Phase Breakdown

### Phase 1: Machine Learning & Feature Engineering
Phase 1 establishes the geochemical, statistical, and classical machine learning foundation:
- **Data Ingestion & Physical Screening:** Audited 2,827,977 raw records across 5 nations (1940–2023). Filtered 8,360 unphysical readings using thermodynamic bounds (pH $0–14$, DO $0–30\text{ mg/L}$, water temperature $0–45^{\circ}\text{C}$), leaving 2,819,617 clean complete-case observations.
- **Biochemical Feature Engineering:** Engineered 38 tabular features including Benson-Krause Dissolved Oxygen saturation equations, Redfield stoichiometric $\text{N}:\text{P}$ ratios, Emerson unionized toxic ammonia ($\text{NH}_3$), $\log(1+x)$ variance-stabilizing transformations on skewed nutrients, historical chronological station lags, and guideline exceedance counters.
- **Tree-Based Benchmarks:** Evaluated Empirical Mean Baseline, Random Forest (100 estimators), and XGBoost (150 estimators, histogram-based split finding).
- **Interpretability:** Applied TreeSHAP game-theoretic attribution to quantify parameter sensitivity.

### Phase 2: Deep Learning, Residual Hybrid & Diagnostics
Phase 2 explores tabular deep learning, hybrid architectures, and axiomatic attribution:
- **Task 1 — DL Data Pipeline:** Implemented PyTorch `Dataset` and `DataLoader` pipelines with 53-dimensional inputs (36 standardized continuous features + 17 one-hot categorical indicators).
- **Task 2 — MLP Architecture:** Developed `WaterQualityMLP` using a funnel topology ($53 \to 256 \to 128 \to 64 \to 32 \to 1$) with `BatchNorm1d`, `Dropout(0.2)`, and ReLU activations.
- **Task 3 — Training & Optimization:** Optimized using AdamW, Smooth L1 and MSE losses, Cosine Annealing learning rate schedules, and early stopping.
- **Task 4 — Residual Hybrid Architecture:** Coupled an unweighted Base Tree Ensemble ($0.5\text{RF} + 0.5\text{XGB}$) with a 54-dimensional `ResidualMLPNet` trained on 5-fold out-of-fold (OOF) residuals using Smooth L1 loss ($\hat{y}_{\text{hybrid}} = \hat{y}_{\text{base}} + \hat{r}$).
- **Task 5 — Evaluation & Ablation Studies:** Conducted uniform test-set benchmarking across all 6 architectures and executed a systematic 7-group ablation study.
- **Task 6 — Explainability & Prediction Error Diagnostics:** Applied Integrated Gradients ($N=1,000$ test observations, $m=50$ integration steps) from a modal observed baseline ($F(x') = 86.7233$ WQI units) and conducted residual error diagnostics ($\text{Residual} = \hat{y} - y$).

---

## Dataset Overview & Target Formulation

### Data Provenance
| Country | Clean Records | Share | Authoritative Source |
| :--- | :---: | :---: | :--- |
| **England** | 2,361,042 | 83.74% | UK Environment Agency Water Quality Archive (`data.gov.uk`) |
| **USA** | 239,890 | 8.51% | USGS National Water Information System (NWIS) + US EPA WQP |
| **Ireland** | 200,980 | 7.13% | Environmental Protection Agency (EPA) Ireland Catchments |
| **China** | 11,985 | 0.43% | Major River Basin Environmental Monitoring Stations |
| **Canada** | 5,720 | 0.20% | Historical Reference Telemetry Basins |
| **Total** | **2,819,617** | **100.00%** | **Multi-National Water Quality Monitoring Corpus** |

### Data Partitions (Strict Temporal & Stratified Quarantine)
- **Training Set (70%):** 1,973,730 observations.
- **Validation Set (15%):** 422,944 observations.
- **Held-Out Test Set (15%):** 422,943 observations.

### Target Formulation: CCME Water Quality Index
The target $y \in [0, 100]$ is computed deterministically from water telemetry guidelines:

$$WQI = 100 - \frac{\sqrt{F_1^2 + F_2^2 + F_3^2}}{1.732}$$

- $F_1$ (**Scope**): Percentage of variables exceeding water quality guidelines.
- $F_2$ (**Frequency**): Percentage of individual tests that fail to meet guidelines.
- $F_3$ (**Amplitude**): Normalized deviation of failed values from guideline thresholds.

Quality tiers: **Excellent** ($95–100$), **Good** ($80–94$), **Fair** ($65–79$), **Marginal** ($45–64$), and **Poor** ($0–44$).

---

## Benchmark Performance Comparison

All models were evaluated on the identical held-out test split ($N = 422,943$ records):

| Model Architecture | Paradigm | Input Dimension | MAE | RMSE | $R^2$ | MedAE | MaxAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Bagging Trees | 38 Tabular | **0.0514** | 0.2016 | 0.999870 | **0.0042** | 12.6128 |
| **Residual Hybrid** | Tree + Residual MLP | 38 Tab + 53 DL | 0.0710 | **0.1678** | **0.999910** | 0.0229 | **5.6842** |
| **Base Tree Ensemble** | Equal-Weighted Trees ($0.5\text{RF}+0.5\text{XGB}$) | 38 Tabular | 0.0900 | 0.2464 | 0.999805 | 0.0261 | 8.7128 |
| **XGBoost Regressor** | Gradient Boosted Trees | 38 Tabular | 0.1614 | 0.4364 | 0.999389 | 0.0480 | 9.2769 |
| **Weighted Ensemble** | Convex Blend ($0.4\text{RF}+0.4\text{XGB}+0.2\text{MLP}$) | 38 Tab + 53 DL | 0.3096 | 0.4155 | 0.999446 | 0.2653 | 8.8283 |
| **WaterQualityMLP** | Standalone Deep Net | 53 (36 num + 17 OHE) | 1.3800 | 1.6720 | 0.991029 | 1.2893 | 31.1327 |

### Key Empirical Findings:
1. **Outlier Compression:** The Residual Hybrid compresses peak prediction error by **$54.9\%$** ($12.6128 \to 5.6842$), eliminating severe step-boundary errors produced by tree ensembles.
2. **Lowest Global Variance:** Residual Hybrid achieves the lowest overall RMSE (**$0.1678$**, representing a **$31.9\%$ variance reduction** over the Base Tree Ensemble).
3. **Tabular Boundary Dynamics:** Random Forest demonstrates superior precision across the dense central distribution ($\text{MAE} = 0.0514$), whereas the Residual Hybrid dominates on distribution tails and severe outlier cases.
4. **Divergent Ablation Sensitivities:** Ablation experiments reveal that tree ensembles degrade catastrophically when raw sensor telemetry is removed ($\Delta\text{MAE} = +21.65$), while neural networks collapse when $\log(1+x)$ variance stabilization is omitted ($\Delta\text{MAE} = +12.16$, $\Delta R^2 = -0.8511$).

---

## Repository Structure

```text
Semester_5_AML_DL/
├── Final_Report/                              # Master research report and presentation
│   ├── Final_Project_Report.pdf               # 10-page master IEEE conference report
│   ├── Phase2_Presentation.pptx               # Final Phase 2 PowerPoint presentation
│   ├── main.tex                               # Master LaTeX source
│   ├── references.bib                         # Pruned bibliography (XGBoost & ResNet)
│   ├── IEEEtran.cls                           # IEEE document class
│   └── figures/                               # 22 publication figures for main.tex
│
├── Final_Visualizations/                      # Centralized presentation & visual assets
│   ├── EDA/                                   # 13 exploratory analysis figures
│   ├── Phase_1/                               # 3 Phase 1 model evaluation figures
│   ├── Phase_2/                               # 4 Phase 2 neural network & ablation figures
│   ├── Model_Comparison/                      # 3 unified 6-model comparison plots
│   ├── Explainability/                        # 3 Integrated Gradients attribution figures
│   ├── Error_Analysis/                        # 5 residual diagnostic plots
│   ├── Hybrid_Model/                          # 6 hybrid mechanics & error correlation plots
│   └── README.md                              # Comprehensive visual catalog
│
├── Phase_1_Machine_Learning/                  # Phase 1 Machine Learning pipeline
│   ├── notebooks/                             # 6 standardized notebooks (EDA to evaluation)
│   ├── data/                                  # Cleaned datasets, Parquet partitions, outputs
│   ├── docs/                                  # EDA report, preprocessing specs, setup guide
│   ├── Final/                                 # Phase 1 PDF report, presentation, and docx
│   ├── graphs/                                # 19 presentation charts
│   ├── reports/                               # Phase 1 IEEE report source and PDF
│   ├── src/                                   # Core Phase 1 utility modules
│   └── tests/                                 # 12 automated unit tests (100% passing)
│
├── Phase_2_Deep_Learning/                     # Phase 2 Deep Learning pipeline
│   ├── notebooks/                             # 6 standardized notebooks (Tasks 1–6)
│   ├── models/                                # Checkpoints (MLP, ResidualMLP, Hybrid)
│   ├── artifacts/                             # Scalers, encoders, predictions, config JSONs
│   ├── configs/                               # Training history and feature configs
│   ├── reports/                               # Phase 2 IEEE report, Task 5/6 technical reports
│   ├── scripts/                               # Evaluation, ablation, and benchmark runners
│   ├── src/                                   # PyTorch modules (dl_models, hybrid, etc.)
│   └── tests/                                 # 104 automated unit tests (100% passing)
│
├── data/                                      # Shared data collection sources & metadata
├── requirements.txt                           # Project Python dependencies
├── LICENSE                                    # MIT License
└── README.md                                  # Repository documentation
```

---

## Environment Setup & Execution

### 1. Environment Installation
```bash
# Clone the repository
git clone https://github.com/MAYANKSHARMA01010/Semester_5_AML_DL.git
cd Semester_5_AML_DL

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or .venv\Scripts\activate on Windows

# Install project dependencies
pip install -r requirements.txt
```

### 2. Running Automated Test Suites
The project includes a 116-test automated verification suite covering data pipelines, aquatic chemistry bounds, zero-leakage partitions, model architectures, checkpoint persistence, and explainability:

```bash
# Run Phase 1 test suite (12 tests)
PYTHONPATH="Phase_1_Machine_Learning" pytest Phase_1_Machine_Learning/tests/ -v

# Run Phase 2 test suite (104 tests)
PYTHONPATH="Phase_2_Deep_Learning" pytest Phase_2_Deep_Learning/tests/ -v
```

### 3. Notebook Execution Flow
All 12 notebooks follow the project standard: **zero comments in code cells** and **single-line Markdown section headings**:

| # | Notebook Path | Primary Focus |
| :---: | :--- | :--- |
| **1** | [`Phase_1_Machine_Learning/notebooks/01.EDA/01_data_understanding.ipynb`](Phase_1_Machine_Learning/notebooks/01.EDA/01_data_understanding.ipynb) | Dataset schema, 5-country telemetry distribution |
| **2** | [`Phase_1_Machine_Learning/notebooks/01.EDA/02_data_patterns_and_target.ipynb`](Phase_1_Machine_Learning/notebooks/01.EDA/02_data_patterns_and_target.ipynb) | Aquatic chemistry correlation and CCME WQI target behavior |
| **3** | [`Phase_1_Machine_Learning/notebooks/02.Preprocessing&FeatureSelection/01_data_preprocessing.ipynb`](Phase_1_Machine_Learning/notebooks/02.Preprocessing&FeatureSelection/01_data_preprocessing.ipynb) | Physical plausibility filtering and 38-feature engineering |
| **4** | [`Phase_1_Machine_Learning/notebooks/03.ModelTraining/01_baseline_and_tree_models.ipynb`](Phase_1_Machine_Learning/notebooks/03.ModelTraining/01_baseline_and_tree_models.ipynb) | Baseline, Random Forest, and XGBoost training |
| **5** | [`Phase_1_Machine_Learning/notebooks/04.ModelEvaluation/01_ml_model_comparison.ipynb`](Phase_1_Machine_Learning/notebooks/04.ModelEvaluation/01_ml_model_comparison.ipynb) | Tree model comparison on test partition |
| **6** | [`Phase_1_Machine_Learning/notebooks/04.ModelEvaluation/02_feature_importance.ipynb`](Phase_1_Machine_Learning/notebooks/04.ModelEvaluation/02_feature_importance.ipynb) | Gini split importance and TreeSHAP attribution |
| **7** | [`Phase_2_Deep_Learning/notebooks/01_dl_data_pipeline.ipynb`](Phase_2_Deep_Learning/notebooks/01_dl_data_pipeline.ipynb) | PyTorch Dataset, DataLoader, 53-feature representation |
| **8** | [`Phase_2_Deep_Learning/notebooks/02_mlp_architecture.ipynb`](Phase_2_Deep_Learning/notebooks/02_mlp_architecture.ipynb) | `WaterQualityMLP` architecture design and parameter profiling |
| **9** | [`Phase_2_Deep_Learning/notebooks/03_mlp_training.ipynb`](Phase_2_Deep_Learning/notebooks/03_mlp_training.ipynb) | Training loops, AdamW, loss functions, learning rate schedules |
| **10** | [`Phase_2_Deep_Learning/notebooks/04_hybrid_model.ipynb`](Phase_2_Deep_Learning/notebooks/04_hybrid_model.ipynb) | 5-fold OOF residual collection and `ResidualMLPNet` training |
| **11** | [`Phase_2_Deep_Learning/notebooks/05_model_evaluation.ipynb`](Phase_2_Deep_Learning/notebooks/05_model_evaluation.ipynb) | 6-model held-out comparison and 7-group ablation studies |
| **12** | [`Phase_2_Deep_Learning/notebooks/06_explainability_error_analysis.ipynb`](Phase_2_Deep_Learning/notebooks/06_explainability_error_analysis.ipynb) | Integrated Gradients path attributions and residual diagnostics |

---

## Scientific Limitations & Methodological Notes

1. **Deterministic Function Approximation:** The CCME WQI is calculated through a deterministic mathematical formula based on threshold exceedances. Consequently, machine learning models achieve high coefficients of determination ($R^2 > 0.99$), reflecting continuous function approximation rather than purely stochastic observational forecasting.
2. **Feature Leakage Considerations:** The engineered feature `exceedance_counter` tracks cumulative regulatory threshold violations, mirroring the $F_1$ factor of the CCME formula. Systematic ablation experiments demonstrate that even when `exceedance_counter` is completely omitted, tree models retain $R^2 > 0.999$, driven by raw physical parameters (`base_sensors`).
3. **Tabular Deep Learning Discrepancy:** Continuous datasets characterized by axis-aligned, step-function threshold boundaries favor decision tree partitioning over smooth neural network gradient descent. While the standalone MLP ($\text{MAE} = 1.3800$) underperforms Random Forest ($\text{MAE} = 0.0514$) on central median deviations, its integration into the Residual Hybrid architecture provides crucial non-linear error compensation, cutting peak outlier errors by $54.9\%$.
4. **Geographic Distribution:** England represents $83.74\%$ of total observations, reflecting the extensive UK Environment Agency open-data monitoring network. Regional subgroup evaluations confirm high model stability across all 5 monitoring jurisdictions.

---

## License

This project is released under the [MIT License](LICENSE).
