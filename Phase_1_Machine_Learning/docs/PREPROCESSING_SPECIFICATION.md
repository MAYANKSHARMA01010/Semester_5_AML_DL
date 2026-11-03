# Water Quality Prediction ML Project — Data Cleaning & Preprocessing Specification

> **Module**: [`src/features.py`](../src/features.py), [`src/utils.py`](../src/utils.py)  
> **Executable Notebook**: [`notebooks/02.Preprocessing&FeatureSelection/01_data_preprocessing.ipynb`](../notebooks/02.Preprocessing&FeatureSelection/01_data_preprocessing.ipynb)  
> **Audited Dataset Source**: `data/Dataset/Parquet/Combined Data/Combined_dataset.parquet`  
> **Generated Outputs**: `data/GeneratedOutput/02.Preprocessing&FeatureSelection/`  
> **Evaluation Phase**: Phase 1 — Applied Machine Learning  
> **Strategy**: Complete Case Analysis (100% Ground Truth, Zero Synthetic Imputation Bias)  
> **Production Scale**: 2,827,977 raw rows $\to$ 2,819,617 clean observations across 5 nations  

---

## 1. Overview & Objective

This document formalizes the deterministic, reproducible 9-stage data cleaning and feature engineering pipeline for **Project 18: Water Quality Prediction (Phase 1 Machine Learning)**.

The pipeline ingests the authoritative multi-country surface water monitoring repository (2,827,977 records spanning 83 years across England, USA, Ireland, China, and Canada) and transforms raw sensor telemetry into a 38-feature machine learning predictor matrix while enforcing aquatic thermodynamic screening, non-linear normalization, anti-leakage target quarantine, and stratified 70/15/15 partitioning.

---

## 2. Ingestion Schema & Target Aliasing

### 2.1 Input Data Specification

- **Primary Source File**: `data/Dataset/Parquet/Combined Data/Combined_dataset.parquet` (Snappy parquet).
- **Raw Dimensions**: 2,827,977 rows × 14 columns.
- **Null Value Profile**: **0 missing values (100% complete ground truth)** across all 14 columns.

### 2.2 Column Standardisation Dictionary

| Raw Column Identifier | Canonical Alias | Data Type | Physical Dimension / Meaning |
|---|---|---|---|
| `Country` | `country` | `string` | Sovereign nation of monitoring authority |
| `Area` | `area` | `string` | Catchment, monitoring station, or river reach |
| `Waterbody Type` | `waterbody_type` | `string` | Ecological classification (River, Stream, Lake, Estuary) |
| `Date` | `date` | `datetime64[ns]` | Timestamp of telemetry sampling |
| `Ammonia (mg/l)` | `ammonia_mg_l` | `float64` | Ammonium / ammonia concentration |
| `Biochemical Oxygen Demand (mg/l)` | `bod_mg_l` | `float64` | 5-day Biochemical Oxygen Demand ($\text{BOD}_5$) |
| `Dissolved Oxygen (mg/l)` | `do_mg_l` | `float64` | In-situ polarographic / optical dissolved oxygen |
| `Orthophosphate (mg/l)` | `orthophosphate_mg_l`| `float64` | Dissolved reactive inorganic phosphorus |
| `pH (ph units)` | `ph` | `float64` | Hydrogen ion activity ($-\log_{10}[\text{H}^+]$) |
| `Temperature (cel)` | `temperature_c` | `float64` | Water temperature in Celsius |
| `Nitrogen (mg/l)` | `nitrogen_mg_l` | `float64` | Total dissolved nitrogen |
| `Nitrate (mg/l)` | `nitrate_mg_l` | `float64` | Nitrate ion concentration ($\text{NO}_3^-$) |
| `CCME_Values` | `ccme_values` | `float64` | Continuous index (0.0 – 100.0) — **Regression Target** |
| `CCME_WQI` | `ccme_wqi` | `Categorical` | Ordered quality tier — **Classification Target** |

---

## 3. The 9-Stage Preprocessing Pipeline

### Stage 1: Standardized snake_case Aliasing
- **Transformation**: Maps verbose, bracketed column headers to clean snake_case strings (`ph`, `temperature_c`, `do_mg_l`, `ammonia_mg_l`, etc.).
- **Rationale**: Eliminates quoting bugs across Polars, PyArrow, DuckDB, scikit-learn, and XGBoost while storing bi-directional mappings in metadata.

### Stage 2: Semantic Data Type Casting
- **Transformation**:
  - `date`: Converted via `pd.to_datetime(dayfirst=True)`.
  - `ccme_wqi`: Cast to `pd.CategoricalDtype(categories=["Poor", "Marginal", "Fair", "Good", "Excellent"], ordered=True)`.
  - Encoded integer alias `ccme_wqi_code` (0 for Poor through 4 for Excellent).
- **Rationale**: Guarantees correct ordinal ranking for multiclass classification metrics (Macro F1, confusion matrices).

### Stage 3: Data Quality & Null Value Verification (Complete Case Analysis)
- **Verification**: Evaluates `df.isnull().sum()`. The authentic 5-country dataset contains **0 null entries across all 14 columns**.
- **Decision**: Implements **Complete Case Analysis**. With 100% complete records, no synthetic imputation (mean/median/KNN) is injected, completely avoiding artificial distributional bias.

### Stage 4: Domain-Informed Physical Bounds & Outlier Screening
- **Screening Envelope**:
  - $\text{pH} \in [3.0, 12.0]$: Filters sensor scale corruptions (e.g., $\text{pH} = 67,115$).
  - $\text{Temperature} \in [0.0^\circ\text{C}, 45.0^\circ\text{C}]$: Enforces liquid water domain at atmospheric pressure, eliminating $-999999.0$ transmitter dropouts.
  - $\text{Dissolved Oxygen} \in [0.0, 25.0\text{ mg/L}]$: Enforces biological and thermodynamic solubility bounds.
  - $\text{Nutrients} \ge 0.0\text{ mg/L}$: Mass concentrations cannot physically be negative.
- **Yield**: Exactly **8,360 corrupted telemetry dropouts (0.296%)** are purged, producing **2,819,617 pristine complete-case observations**.

### Stage 5: Analytical Zero-Drift Calibration
- **Transformation**: Minor negative values within instrument detection noise limits ($[-0.05, 0.0)$ mg/L) are clipped to $0.0\text{ mg/L}$.
- **Rationale**: Reflects standard analytical chemistry practice for baseline optical/spectrophotometric zero calibration.

### Stage 6: Duplicate Verification & Catchment Grouping
- **Verification**: Evaluates exact identical records across all 14 features; dataset is 100% deduplicated.
- **Catchment Indexing**: Stations indexed by `Area` to enable temporal ordering for historical feature lag derivations.

### Stage 7: Feature Engineering & Geochemical Thermodynamics (38 Features)
- **Engineered Predictors**:
  1. **Non-Linear Skewness Normalization**: $\log(1+x)$ applied to skewed nutrients (`log1p_orthophosphate`, `log1p_ammonia`, `log1p_bod`, `log1p_nitrogen`, `log1p_nitrate`).
  2. **Henry's Law Gas Saturation**: Evaluates theoretical capacity $DO_{\text{sat}}(T)$ via Benson-Krause polynomial; derives $DO$ Saturation Percentage and signed Streeter-Phelps Dissolved Oxygen Deficit ($DO_{\text{sat}} - DO$).
  3. **Aquatic Stoichiometry**: Redfield Ratio ($N:P = \text{Nitrogen} / \text{Orthophosphate}$) and Organic Loading Ratio ($\text{BOD} / DO$).
  4. **Emerson Un-Ionized Ammonia Equilibrium**: Derives toxic un-ionized $\text{NH}_3$ fraction as a function of temperature and pH:
     $$pKa = 0.09018 + \frac{2729.92}{273.15 + T}, \quad f_{\text{NH}_3} = \frac{1}{10^{pKa - \text{pH}} + 1}$$
  5. **Chronological Station Dynamics**: Non-lookahead historical shift-1 lags and rolling-3 statistics within each monitoring reach (`Area`), capturing catchment environmental memory.
  6. **Cyclical Temporal Context**: Trigonometric harmonic seasonal features $\sin(2\pi m / 12)$ and $\cos(2\pi m / 12)$.

### Stage 8: Anti-Leakage Target Quarantine
- **Quarantine Enforcement**:
  - Isolated regression target: `ccme_values`.
  - Isolated classification target: `ccme_wqi`, `ccme_wqi_code`.
  - Strict exclusion of intermediate CCME terms ($F_1, F_2, F_3, nse$).
- **Verification**: Validated by automated `pytest` tests asserting zero target overlap in the predictor matrix.

### Stage 9: Stratified Partitioning & Sharded Parquet Storage
- **Class-Stratified Split**: 70% Train (1,973,731), 15% Validation (422,943), 15% Test (422,943).
- **Standardization**: `StandardScaler` fitted strictly on Train features only.
- **Parquet Sharding**: Training set (~190 MB) transparently sharded into three <60 MB files (`train_part0.parquet`, `train_part1.parquet`, `train_part2.parquet`), enabling 100% remote version control on GitHub within the 100 MB file limit.

---

## 4. Pipeline Before-versus-After Data Quality Verification

| Metric | Raw Ingested Source | Clean Processed Partitions | Production Status |
|---|---|---|---|
| **Total Observation Count** | 2,827,977 rows | 2,819,617 rows | 8,360 unphysical records purged (0.296%) |
| **Missing Values (Nulls)** | **0 nulls** | **0 nulls** | **100% Complete Case Ground Truth** |
| **Sovereign Nations** | 5 countries | 5 countries | England, USA, Ireland, China, Canada |
| **Hardware Dropouts ($-999999$)**| 81 records | 0 records | 100% neutralized |
| **Unphysical pH ($>12$ or $<3$)** | 1,581 records | 0 records | 100% neutralized |
| **Feature Dimensionality** | 8 raw predictors | 38 engineered features | Skewness normalized, stoichiometric laws |
| **Train / Val / Test Partition** | Unpartitioned | 70% / 15% / 15% | Stratified by `ccme_wqi` grade |
| **GitHub Storage Compliance** | Single 190 MB file (rejected) | 3 shards (<60 MB each) | 100% compliant with 100 MB file limit |

---

## 5. Execution & Reproduction Commands

```bash
# Execute unit testing suite verifying cleaning, features, and zero leakage
pytest tests/ -v

# Run Phase 1 ML Training (Baseline, Random Forest, XGBoost)
python scripts/run_notebook.py "notebooks/03.ModelTraining/01_baseline_and_tree_models.ipynb"

# Execute end-to-end verification and evaluation
python scripts/run_phase1_pipeline.py
```
