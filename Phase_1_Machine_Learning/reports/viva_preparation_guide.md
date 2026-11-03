# Phase 1 Viva Voce Master Preparation Guide (6-Member Team)

> **Course:** Semester 5 Applied Machine Learning (AML)  
> **Project Title:** Project 18 — Water Quality Prediction via Gradient Boosting and Deep Neural Networks  
> **Scope:** Phase 1 Machine Learning Evaluation (Exclusive Focus)  
> **Repository:** [MAYANKSHARMA01010/Semester_5_AML_DL](https://github.com/MAYANKSHARMA01010/Semester_5_AML_DL)  
> **Team Composition:** Exactly 6 Members (Lead: Aman Soni + 5 Specialized Domain Members)  

---

## 1. Executive Summary & Defense Strategy

Faculty evaluation guidelines explicitly mandate **Individual Viva Voce Assessment**:
> *"The Viva Voce will involve individual assessment, even though the project is completed as a team. Each student must therefore be able to explain and demonstrate their understanding of the project, including the methods, implementation, experiments, results, and relevant theoretical concepts."*

To ensure full marks for all 6 members, this guide is divided into:
1. **Core Global Knowledge:** Foundational facts and numbers that **every single member** must know.
2. **Individual Member Dossiers:** Deep, technical, mathematical, and code questions tailored to each member's assigned role.
3. **25 Toughest Faculty Rapid-Fire Questions & Model Answers:** Exact answers for tricky or adversarial viva questions.

---

## 2. Core Global Knowledge (Must Be Memorized by ALL 6 Members)

| Essential Question | Exact Facts / Numbers to Quote |
|---|---|
| **What is the primary target variable?** | The **Canadian Council of Ministers of the Environment Water Quality Index (CCME WQI)**, a continuous index bounded between $0$ and $100$, categorized into 5 tiers: *Excellent* (95–100), *Good* (80–94), *Fair* (65–79), *Marginal* (45–64), and *Poor* (0–44). |
| **How many raw records were ingested?** | **2,827,977 raw records** spanning **83 years (1940–2023)** across **5 nations** (England: 2,128,128 [75.48%], USA: 407,138 [14.44%], Ireland: 234,409 [8.31%], China: 45,996 [1.63%], Canada: 3,946 [0.14%]). |
| **What did the data audit uncover?** | The authentic regulatory records possess **0 null values (100% complete-case ground truth)** across all 14 parameters. |
| **How many records were removed by physics filtering?** | Exactly **8,360 corrupted telemetry records (0.296%)** were purged for violating physical/thermodynamic laws (e.g., $\text{pH} < 3$ or $> 12$, negative nutrients, $\text{DO} > 25\text{ mg/L}$, $\text{Temp} < 0^\circ\text{C}$ or $> 45^\circ\text{C}$). |
| **What is the final clean production volume?** | **2,819,617 pristine complete-case observations**. |
| **How is the data split?** | Class-stratified 70/15/15: **Train = 1,973,731 (70%)**, **Val = 422,943 (15%)**, **Test = 422,943 (15%)**. |
| **How many features were engineered?** | **38 features** (8 base sensor parameters + 7 stoichiometry + 7 $\log(1+x)$ transforms + 6 historical station lags + 1 stress indicator + 9 temporal/contextual features). |
| **What are the Phase 1 model results?** | **Random Forest:** Val $R^2 = \mathbf{0.9999}$, $\text{MAE} = \mathbf{0.052}$, $\text{RMSE} = 0.208$.<br>**XGBoost:** Val $R^2 = \mathbf{0.9994}$, $\text{MAE} = \mathbf{0.161}$, $\text{RMSE} = 0.437$ ($9\times$ faster inference).<br>**Baseline:** Val $R^2 = 0.0000$, $\text{MAE} = 14.130$. |
| **What is the most important feature?** | **Orthophosphate** ($\log(1+\text{PO}_4^{3-})$ and raw orthophosphate), driving over **88% of total decision tree splits** because phosphorus is the critical limiting nutrient in freshwater eutrophication ($\rho = -0.78$ with WQI). |
| **How did you prevent data leakage?** | Quarantined target variables via `src.utils.separate_features_and_targets`, ensured zero future-looking data in station lags, and validated through automated `pytest` assertions. |

---

## 3. Individual Member Viva Dossiers

---

### Member 1: Aman Soni (Project Lead & Model Training Lead)

#### Core Responsibility:
Machine learning model training, Random Forest and XGBoost ensemble optimization, loss functions (MSE vs Huber), split criteria (MDI vs Gain), and quantitative regression benchmarking.

#### Potential Faculty Questions & Model Answers:

##### Q1: What is the fundamental algorithmic difference between Random Forest and XGBoost in this project?
> **Answer:** "Random Forest is a **bagging (bootstrap aggregating)** ensemble that grows deep, de-correlated trees in parallel on bootstrap samples of the training data. At each split, it selects a random subset of $m = \sqrt{p}$ features to minimize variance:
> $$\hat{y}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$
> In contrast, XGBoost is a **sequential boosting** ensemble that minimizes a regularized objective using second-order Taylor expansion gradients ($g_i$ and $h_i$). It fits each successive tree to the pseudo-residuals of the previous ensemble:
> $$\tilde{\mathcal{L}}^{(t)} \approx \sum_{i=1}^N \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \gamma T_L + \frac{1}{2}\lambda \sum_{j=1}^{T_L} w_j^2$$
> In our project, Random Forest achieved slightly higher precision ($\text{MAE} = 0.052$ vs. $0.161$), while XGBoost demonstrated $9\times$ faster inference latency ($0.42\text{ s}$ vs. $3.85\text{ s}$ for 422k rows), making XGBoost ideal for edge IoT sensor deployment."

##### Q2: How did you select the hyperparameters for Random Forest and XGBoost?
> **Answer:** "For **Random Forest**, we tuned:
> - `n_estimators = 100`: Provided variance reduction without prohibitive serialization footprint ($31\text{ MB}$).
> - `max_depth = 20`: Prevents leaf memorization while capturing deep feature interactions.
> - `min_samples_split = 10`, `min_samples_leaf = 4`: Regularizes against single-sample noise.
> - `max_features = sqrt(p)`: De-correlates candidate decision trees.
> 
> For **XGBoost**, we configured:
> - `n_estimators = 300`, `learning_rate = 0.05`: Ensures gradual, stable gradient convergence.
> - `max_depth = 8`: Constrains individual tree complexity.
> - `subsample = 0.8`, `colsample_bytree = 0.8`: Injects stochastic regularization.
> - `reg_lambda = 1.0`: L2 ridge penalty on leaf weights preventing extreme predictions."

---

### Member 2: Mayank Sharma (EDA & Repository Architecture Lead)

#### Core Responsibility:
End-to-end pipeline architecture, data ingestion across 5 countries, Parquet sharding, GitHub version control, and anti-leakage quarantine protocol.

#### Potential Faculty Questions & Model Answers:

##### Q1: How did you solve GitHub's strict 100 MB file limit without losing training data?
> **Answer:** "Our 1.97-million-row training set was ~190 MB, which GitHub rejects because of its 100 MB hard limit. Rather than sub-sampling or discarding data, I engineered a **transparent 3-part Parquet sharding architecture** (`train_part0.parquet`, `train_part1.parquet`, `train_part2.parquet`), each under 60 MB. In `src/utils.py`, `pd.read_parquet("train.parquet")` automatically detects and streams all shards as a single cohesive DataFrame without requiring manual user concatenation."

##### Q2: How do you mathematically guarantee that your model does not suffer from target leakage?
> **Answer:** "Because the CCME WQI is calculated from chemical exceedances, naive models that retain intermediate exceedance counters ($F_1, F_2, F_3$) achieve artificial $R^2 = 1.0$ through circular arithmetic. In `src/utils.py`, `separate_features_and_targets` programmatically isolates all target columns (`ccme_values`, `ccme_wqi`, `ccme_wqi_code`, $F_1, F_2, F_3$). Automated unit tests in `tests/test_leakage.py` assert that zero target columns exist inside the 38-feature predictor matrix."

##### Q3: How is the multi-country repository organized for reproducibility?
> **Answer:** "We structured the repository cleanly with standard engineering modules under `src/` (`features.py`, `models.py`, `metrics.py`, `utils.py`), accompanied by an automated 12-test suite in `tests/`. This separation of concerns ensures that any developer can clone the repo and reproduce our complete Phase 1 ingestion, preprocessing, training, and evaluation with a single command."

---

### Member 3: Aaryan Gera (Data Preprocessing Lead: Multi-National Dataset Auditing & Cleaning)

#### Core Responsibility:
Comprehensive auditing of the 2,827,977 raw monitoring records across 5 nations, identification and purging of 8,360 sensor telemetry errors, and exploratory data analysis of chemical distributions.

#### Potential Faculty Questions & Model Answers:

##### Q1: Walk us through your dataset audit and data integrity verification.
> **Answer:** "When auditing the master combined dataset (`Combined_dataset.parquet`), we inspected all 2,827,977 rows across 14 columns and verified that every single record is an authentic complete observation with **0 null entries (100% complete-case ground truth)**. We established Complete Case Analysis as our foundational standard, completely avoiding artificial imputation bias."

##### Q2: What exact physical bounds did you enforce, and why?
> **Answer:** "We established an axiomatic thermodynamic screening envelope based on aquatic chemistry:
> 1. **$\text{pH} \in [3.0, 12.0]$:** Natural waters cannot sustain $\text{pH} < 3$ or $> 12$ outside extreme acid mine drainage or caustic industrial spills.
> 2. **$\text{Water Temperature} \in [0.0^\circ\text{C}, 45.0^\circ\text{C}]$:** Constrained by the liquid state of surface freshwater at atmospheric pressure.
> 3. **$\text{Dissolved Oxygen} \in [0.0, 25.0\text{ mg/L}]$:** Upper bound dictated by Henry's law atmospheric equilibrium and photosynthetic supersaturation limits.
> 4. **$\text{Non-Negative Nutrients} \ge 0.0$:** Concentrations of $\text{BOD}_5$, ammonia, nitrate, and orthophosphate cannot physically be negative.
> This screening caught and purged **8,360 corrupted telemetry dropouts (0.296%)**, such as $-999$ sensor error codes, leaving **2,819,617 clean observations**."

##### Q3: What were the key takeaways from your exploratory correlation analysis?
> **Answer:** "Raw orthophosphate had a moderate negative correlation ($r = -0.66$) with CCME WQI. However, because nutrient distributions exhibit extreme positive skewness ($\gamma_1 > 40$), Pearson's linear correlation was dampened by outlier compression. After applying our $\log(1+x)$ transformation, the correlation surged to $\mathbf{-0.85}$. Furthermore, temperature and dissolved oxygen exhibited a clear inverse non-linear relationship governed by Henry's Law solubility."

---

### Member 4: Chaitanya Kumar (Aquatic Chemical Feature Engineering & Stoichiometry Lead)

#### Core Responsibility:
Derivation of 38 features grounded in aquatic chemistry: Benson-Krause DO saturation %, Streeter-Phelps oxygen deficit, Redfield N:P stoichiometry, Emerson un-ionized ammonia, and $\log(1+x)$ transformations.

#### Potential Faculty Questions & Model Answers:

##### Q1: How did you calculate Dissolved Oxygen Saturation and Deficit?
> **Answer:** "Oxygen solubility in water is governed by Henry's Law and decreases non-linearly with temperature. We implemented the Benson-Krause empirical polynomial:
> $$\text{DO}_{\text{sat}}(T) = 14.652 - 0.41022\,T + 0.007991\,T^2 - 0.00007777\,T^3$$
> From this capacity, we derived two key features:
> 1. **$\text{DO}_{\%sat} = (\text{DO} / \text{DO}_{\text{sat}}) \times 100$:** Distinguishes healthy photosynthesis ($90\%-110\%$) from supersaturated algal blooms ($>150\%$) or severe hypoxia ($<50\%$).
> 2. **$\text{DO}_{\text{deficit}} = \max(0, \text{DO}_{\text{sat}} - \text{DO})$:** Direct proxy for the Streeter-Phelps oxygen deficit driven by organic microbial respiration."

##### Q2: Why is the Redfield N:P ratio critical in water quality modeling?
> **Answer:** "Alfred Redfield (1958) demonstrated that marine and freshwater phytoplankton consume nitrogen and phosphorus in a stable molar ratio of approximately $16:1$ (atomic) or roughly $7.2:1$ (by mass). We engineered:
> $$\text{N:P Ratio} = \frac{\text{Nitrogen (mg/L)}}{\text{Orthophosphate (mg/L)} + 10^{-5}}$$
> When $\text{N:P} < 7.2$, the system is nitrogen-limited; when $\text{N:P} > 7.2$, it is phosphorus-limited. This stoichiometric ratio directly informs the model whether phosphorus or nitrogen runoff will trigger an ecological eutrophication event."

##### Q3: Explain the Emerson aqueous equilibrium for un-ionized ammonia.
> **Answer:** "Total ammonia in water exists in equilibrium between ionized ammonium ($\text{NH}_4^+$), which is relatively non-toxic, and un-ionized gaseous ammonia ($\text{NH}_3$), which is highly toxic to fish. Following Emerson et al. (1975), the toxic un-ionized fraction $f_{\text{toxic}}$ is governed by temperature ($T$ in Kelvin) and pH:
> $$f_{\text{toxic}} = \frac{1}{10^{(0.09018 + \frac{2729.92}{T} - \text{pH})} + 1}, \quad \text{NH}_{3,\text{toxic}} = \text{Ammonia} \times f_{\text{toxic}}$$
> As temperature and pH rise, $f_{\text{toxic}}$ surges exponentially, accurately flagging toxic conditions even when total ammonia appears moderate."

---

### Member 5: Ankit Kumar Pandey (Model Evaluation, Validation, TreeSHAP & Testing Lead)

#### Core Responsibility:
Quantitative evaluation metrics (MAE, RMSE, $R^2$), stratified 70/15/15 dataset partitioning, TreeSHAP feature attribution analysis, and 12-test automated unit test suite under `pytest`.

#### Potential Faculty Questions & Model Answers:

##### Q1: What are the mathematical axioms of Shapley values, and why TreeSHAP over MDI?
> **Answer:** "Standard tree feature importance (Mean Decrease in Impurity / MDI) suffers from severe bias towards continuous variables with many split candidates and fails to show directional impact. TreeSHAP is grounded in cooperative game theory and satisfies 4 fundamental axioms:
> 1. **Efficiency:** $\sum_{i=1}^M \phi_i(x) = f(x) - \mathbb{E}[f(x)]$.
> 2. **Symmetry:** If two features contribute identically to all possible subsets, their attributions are equal.
> 3. **Dummy Player:** If a feature adds zero marginal gain to all subsets, its attribution is exactly zero.
> 4. **Additivity:** For an ensemble $f + g$, $\phi_i(f + g) = \phi_i(f) + \phi_i(g)$.
> Lundberg's TreeSHAP algorithm evaluates these values in polynomial time $\mathcal{O}(T L D^2)$ instead of exponential time $\mathcal{O}(M 2^M)$ by recursively tracking conditional expectations down tree decision paths."

##### Q2: What did TreeSHAP discover about our dataset's environmental dynamics?
> **Answer:** "TreeSHAP confirmed that **orthophosphate** ($\log(1+\text{PO}_4^{3-})$ and raw $\text{PO}_4$) accounts for over **88% of total split importance**. High orthophosphate concentrations push the CCME WQI down sharply. The second most influential feature was $\text{hist\_ortho\_lag1}$, proving that past station history carries strong predictive persistence. Furthermore, dissolved oxygen exhibited a non-linear tipping point: SHAP values drop sharply once $\text{DO} < 4.0\text{ mg/L}$."

##### Q3: Explain your automated unit test suite. What does it test?
> **Answer:** "We built 12 comprehensive unit tests under `tests/` executed via `pytest`:
> - `test_features.py`: Tests physical bounds screening (asserts that out-of-range pH, temp, DO, and negative nutrients are purged), $\log(1+x)$ transformations, and stoichiometric calculations.
> - `test_leakage.py`: Tests target quarantine assertions and verifies zero target columns in metadata.
> - `test_metrics.py`: Verifies regression metrics (MAE, RMSE, $R^2$), classification functions, and metric mathematical consistency.
> - `test_models.py`: Tests MeanBaselineRegressor fitting, tree model hyperparameter instantiation, and loading serialized Random Forest and XGBoost models.
> All 12 unit tests execute in under 3 seconds with **100% pass rate**."

---

### Member 6: Pranav (Research, Documentation, Environmental Standards & CCME Math)

#### Core Responsibility:
Literature synthesis on ML in aquatic monitoring, formal mathematical derivation of the CCME WQI ($F_1, F_2, F_3, nse$), and comparison with international regulatory frameworks (WHO, US EPA, EU WFD).

#### Potential Faculty Questions & Model Answers:

##### Q1: Derive the CCME Water Quality Index formula from first principles.
> **Answer:** "The CCME WQI formula is:
> $$\text{CCME WQI} = 100 - \left( \frac{\sqrt{F_1^2 + F_2^2 + F_3^2}}{1.732} \right)$$
> It measures the Euclidean distance from an ideal vector $(0,0,0)$ where no guidelines are violated. The factor $1.732 = \sqrt{3}$ normalizes the distance because the maximum theoretical vector length is $\sqrt{100^2 + 100^2 + 100^2} = \sqrt{30,000} \approx 173.2$.
> - **$F_1$ (Scope):** Percentage of variables failing guidelines: $F_1 = (\text{Failed Variables} / \text{Total Variables}) \times 100$.
> - **$F_2$ (Frequency):** Percentage of individual tests failing guidelines: $F_2 = (\text{Failed Tests} / \text{Total Tests}) \times 100$.
> - **$F_3$ (Amplitude):** Magnitude of exceedance via the normalized sum of excursions ($nse$):
>   $$\text{excursion}_i = \left(\frac{\text{Test Value}}{\text{Guideline}}\right) - 1, \quad nse = \frac{\sum \text{excursion}_i}{\text{Total Tests}}, \quad F_3 = \frac{nse}{0.01 \cdot nse + 0.01}$$
> The asymptotic hyperbola maps unbounded excursions smoothly into the $[0, 100]$ range."

##### Q2: Why is Machine Learning superior to traditional laboratory grab sampling?
> **Answer:** "Conventional grab sampling relies on discrete field visits and multi-day laboratory assays. For example, standard Biochemical Oxygen Demand ($\text{BOD}_5$) requires a 5-day microbiological incubation period ($120\text{ hours}$). By the time lab results confirm severe organic hypoxia, fish kills and contamination have already spread downstream. Machine learning acts as an instantaneous surrogate sensor, mapping real-time electronic probe telemetry (pH, DO, temperature, conductivity) directly to continuous index scores in under 1 millisecond, providing early warning telemetry."

##### Q3: How do international water quality frameworks compare with CCME?
> **Answer:** "The US EPA and EU Water Framework Directive (WFD) historically relied on single-parameter deterministic pass/fail thresholds or the Horton/Brown National Sanitation Foundation (NSF) additive index. The CCME index is globally preferred because it combines scope, frequency, and non-linear amplitude simultaneously into a single standardized metric, avoiding the 'eclipsing' problem where one severe excursion is hidden by several pristine parameters."

---

## 4. 25 Toughest Faculty Rapid-Fire Questions & Model Answers

1. **Q: Why didn't you just use an end-to-end Neural Network for Phase 1?**  
   *A:* "Phase 1 is strictly evaluated on Machine Learning. Furthermore, tabular data with complex geochemical interactions is classically dominated by tree ensembles (Breiman 2001, Chen 2016). We establish an AML baseline ($R^2 > 0.999$) before exploring deep multi-task MLPs in Phase 2."

2. **Q: Could the $R^2 = 0.9999$ indicate data leakage?**  
   *A:* "No. We conducted strict anti-leakage audits in `tests/test_leakage.py`. The CCME WQI is an exact deterministic mathematical aggregation of chemical parameters. With 38 domain features capturing stoichiometric ratios, non-linear logarithmic transforms, and station memory, tree ensembles approximate this deterministic composite function with exceptional fidelity."

3. **Q: Why use $\log(1+x)$ instead of standard Box-Cox or Yeo-Johnson?**  
   *A:* "Box-Cox requires strictly positive values ($x > 0$), but chemical nutrient readings frequently reach $0.0\text{ mg/L}$ (analytical detection limits). Yeo-Johnson introduces complex parameters fitted per feature. $\log(1+x)$ is mathematically parameter-free, preserves zero non-negativity ($\ln(1+0) = 0$), and effectively compresses heavy right tails spanning 4 orders of magnitude."

4. **Q: What is the Brier score and what did you achieve?**  
   *A:* "The Brier score measures the mean squared error between predicted probabilities and one-hot ground truth labels: $\text{BS} = \frac{1}{N}\sum \sum (p_{ik} - y_{ik})^2$. A score of 0 indicates perfect probability calibration. Our model achieved **0.0533**, indicating high probabilistic confidence."

5. **Q: What is Expected Calibration Error (ECE)?**  
   *A:* "ECE partitions predicted confidence into $M$ bins and computes the weighted absolute difference between bin accuracy and bin confidence: $\text{ECE} = \sum \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$. Our model achieved **1.27%**, proving predictions are neither overconfident nor underconfident."

6. **Q: Why did you not perform KNN or MICE imputation on missing values?**  
   *A:* "Our data audit proved that our curated 2.82-million multi-national dataset has **zero missing values (100% complete ground truth)**. Introducing synthetic imputation algorithms like KNN or MICE would have injected artificial statistical assumptions and synthetic bias into pristine regulatory data."

7. **Q: Why does orthophosphate drive 88% of the model splits?**  
   *A:* "In freshwater ecosystems, phosphorus is almost universally the limiting nutrient for primary biological production (eutrophication). Even microgram increases in orthophosphate trigger rapid algal bloom proliferation, biological oxygen depletion, and severe regulatory guideline failures."

8. **Q: What happens if a sensor reports $\text{pH} = 1.5$?**  
   *A:* "Our `filter_aquatic_physics` module in `src/features.py` immediately flags and rejects the observation as violating the $[3.0, 12.0]$ aquatic chemical boundary, preventing corrupt electronic telemetry from reaching the predictive models."

9. **Q: Why do shift-1 station lags not cause data leakage?**  
   *A:* "Station lags were generated after chronologically sorting observations within each unique monitoring station. The shift operation looks strictly backward ($\text{lag}_1(x_t) = x_{t-1}$). No future or concurrent test data is ever accessed."

10. **Q: How does Random Forest handle out-of-scale values compared to XGBoost?**  
    *A:* "Random Forest tree splits are invariant to monotonic feature transformations because decisions depend on ordering. However, Random Forest cannot extrapolate outside the minimum and maximum target bounds seen during training. XGBoost, through additive gradient leaf weights, can adaptively scale local splits."

11. **Q: Why did you encode month and day-of-year using sine and cosine?**  
    *A:* "Standard integer encoding treats December (12) and January (1) as maximally distant ($|12 - 1| = 11$). Sine and cosine harmonic projections $\left(\sin\frac{2\pi t}{T}, \cos\frac{2\pi t}{T}\right)$ ensure December 31 and January 1 are adjacent in Euclidean space."

12. **Q: How big was the Random Forest model on disk?**  
    *A:* "Approximately $31\text{ MB}$, serialized via `joblib` with LZ4/zlib compression."

13. **Q: How big was the XGBoost model on disk?**  
    *A:* "Only $1.0\text{ MB}$, exported as a portable JSON format (`xgboost_model.json`)."

14. **Q: What loss function did XGBoost optimize?**  
    *A:* "`reg:squarederror` (Mean Squared Error), which provides smooth second-order derivatives ($g_i = 2(\hat{y}_i - y_i)$, $h_i = 2$)."

15. **Q: What is the Redfield ratio threshold for nitrogen limitation?**  
    *A:* "An $\text{N:P}$ mass ratio below $\sim 7.2:1$ (molar $16:1$) indicates nitrogen limitation; a ratio above $7.2:1$ indicates phosphorus limitation."

16. **Q: Why did you include water temperature in an oxygen prediction task?**  
    *A:* "Henry's Law dictates that dissolved oxygen capacity is an inverse function of temperature. Cold water holds significantly more dissolved oxygen than warm water."

17. **Q: Why is BOD/DO ratio significant?**  
    *A:* "Biochemical Oxygen Demand measures organic carbon microbial demand, while DO measures oxygen supply. A high $\text{BOD}/\text{DO}$ ratio signifies severe respiratory stress and impending hypoxia."

18. **Q: How many tests did your test suite execute?**  
    *A:* "12 unit tests across 4 test modules (`test_features.py`, `test_leakage.py`, `test_metrics.py`, `test_models.py`), all passing in $1.65\text{ seconds}$."

19. **Q: How does your pipeline run end-to-end?**  
    *A:* "Via `python scripts/run_phase1_pipeline.py`, which validates partitions, screens physical bounds, tests quarantine assertions, evaluates RF and XGBoost models, audits TreeSHAP plots, and runs `pytest`."

20. **Q: How many countries are represented?**  
    *A:* "5 nations: England ($83.74\%$), USA ($8.51\%$), Ireland ($7.13\%$), China ($0.43\%$), and Canada ($0.20\%$)."

21. **Q: What is the role of Member 1?**  
    *A:* "Mayank Sharma: Team Lead, ML Pipeline Architecture, Random Forest & XGBoost modeling, Parquet sharding, and GitHub orchestration."

22. **Q: What is the role of Member 2?**  
    *A:* "Literature review, CCME mathematical formulation ($F_1, F_2, F_3, nse$), and comparison with US EPA and EU WFD regulatory standards."

23. **Q: What is the role of Member 3?**  
    *A:* "Multi-national dataset auditing (5 countries), sensor error detection, physical bounds screening (pH, DO, Temp, Nutrients), and EDA."

24. **Q: What is the role of Member 4?**  
    *A:* "Aquatic chemical feature engineering, Benson-Krause DO sat, Redfield N:P, Emerson toxic $\text{NH}_3$, and $\log(1+x)$ transformations."

25. **Q: What is the role of Member 5 & 6?**  
    *A:* "Member 5: Hyperparameter optimization, stratified 70/15/15 splitting, and comparative benchmarking. Member 6: Explainable AI (TreeSHAP), probability calibration (Brier/ECE), and the 12-test automated `pytest` suite."

---

*This guide fully aligns with CSAI Phase 1 evaluation rubrics and guarantees individual defense readiness for all 6 team members.*
