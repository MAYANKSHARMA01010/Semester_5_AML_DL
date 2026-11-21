# Water Quality Prediction: Student Viva Voce Preparation Guide

> **Project Title:** Water Quality Prediction via Gradient Boosting and Deep Neural Networks  
> **Course:** Semester 5 Applied Machine Learning & Deep Learning (CSAI)  
> **Target Continuous Index:** Canadian Council of Ministers of the Environment Water Quality Index (**CCME WQI**, continuous $0–100$)  
> **Dataset Size:** 2,819,617 clean complete-case observations across 5 nations (1940–2023)  
> **Test Set Size:** 422,943 observations  

---

## 1. What Problem Are We Solving?

Assessing freshwater quality traditionally requires collecting water samples, transporting them to a lab, measuring dozens of chemical parameters, and manually calculating regulatory water quality indices. This process is slow, expensive, and cannot provide real-time alerts.

**Our solution:** We build automated Machine Learning and Deep Learning models that take real-time multi-sensor telemetry (like pH, dissolved oxygen, nutrients) and immediately predict the continuous regulatory Water Quality Index score, allowing instant environmental monitoring and rapid pollution detection.

---

## 2. What Dataset Did We Use?

We compiled an 83-year environmental monitoring dataset (1940–2023) across 5 nations totaling **2,819,617 clean observations**:
- **England (83.74%):** UK Environment Agency Water Quality Archive (`data.gov.uk`) — 2,361,042 records.
- **USA (8.51%):** USGS National Water Information System (NWIS) and US EPA WQP — 239,890 records.
- **Ireland (7.13%):** EPA Ireland river and lake catchments — 200,980 records.
- **China (0.43%):** Major river basin monitoring stations — 11,985 records.
- **Canada (0.20%):** Historical reference telemetry basins — 5,720 records.

---

## 3. What Exactly Do We Predict and Why This Target?

We predict the **Canadian Council of Ministers of the Environment Water Quality Index (CCME WQI)**, a continuous scalar between **0 and 100**.

### Formula:
$$WQI = 100 - \frac{\sqrt{F_1^2 + F_2^2 + F_3^2}}{1.732}$$
- **$F_1$ (Scope):** Percentage of chemical parameters that failed the safety guidelines.
- **$F_2$ (Frequency):** Percentage of individual tests that violated standards.
- **$F_3$ (Amplitude):** The amount by which failed tests exceeded acceptable limits.
- **$1.732$ ($\sqrt{3}$):** Normalization factor ensuring the result stays between 0 and 100.

### Why this target?
1. It is globally recognized by environmental protection agencies.
2. It condenses complex multi-parameter chemistry into a single actionable score.
3. We avoided mixing incompatible national standards by standardizing on CCME WQI guidelines across all 5 nations.

---

## 4. How Was the Data Preprocessed and Split?

### Handling Missing Data:
- The raw telemetry contained 2,827,977 rows.
- We applied **Complete Case Analysis**: records missing essential chemical parameters or failing physical bounds screening (e.g., negative chemical concentrations or temperatures outside $[-5^\circ\text{C}, 45^\circ\text{C}]$) accounted for only 8,360 rows ($0.296\%$).
- These 8,360 corrupted rows were dropped, leaving **2,819,617 pristine records**. Zero imputation was needed, ensuring no artificial synthetic bias.

### Data Partitioning:
- **70% Training:** 1,973,730 records (sharded into 3 files under 60 MB for GitHub limits).
- **15% Validation:** 422,944 records (used for hyperparameter tuning and early stopping).
- **15% Test:** 422,943 records (quarantined and untouched until final benchmarking).
- Scalers and encoders were fitted **strictly on the training partition** to prevent any data leakage.

---

## 5. What Features Did We Engineer?

We engineered **38 tabular features** for tree models, expanded to **53 features** for Deep Learning:

1. **8 Raw Physical & Chemical Sensors:** `ph`, `temperature_c`, `do_mg_l`, `bod_mg_l`, `ammonia_mg_l`, `orthophosphate_mg_l`, `nitrogen_mg_l`, `nitrate_mg_l`.
2. **7 Stoichiometric & Thermodynamic Features:**
   - **Benson-Krause Dissolved Oxygen Saturation %:** Compares measured oxygen against the maximum oxygen water can physically dissolve at that temperature.
   - **Streeter-Phelps Oxygen Deficit:** Measures how much oxygen has been depleted by decomposing organic waste.
   - **Redfield N:P Ratio:** The molecular ratio of nitrogen to phosphorus, showing which nutrient triggers toxic algae blooms.
   - **Emerson Toxic Ammonia:** Calculates the fraction of ammonia in its toxic un-ionized $\text{NH}_3$ form using pH and temperature.
3. **7 Log-Transformed Features ($\log(1+x)$):** Nutrients like orthophosphate have extreme positive skewness ($\gamma_1 > 40$). The log-transform normalizes them, boosting linear correlation from $-0.66$ to $-0.85$.
4. **6 Historical Temporal Lags:** Previous measurement at the same monitoring station (shift-1 lag) and 3-step rolling averages. Strict zero-lookahead was enforced.
5. **1 Regulatory Stress Counter (`exceedance_counter`):** Counts how many parameters violated environmental thresholds for that sample.
6. **2 Categorical Features (17 One-Hot encoded for DL):** Jurisdiction (`country_code`) and habitat (`waterbody_type_code`).
7. **7 Temporal Cyclical Features:** Calendar year, month, season, and trigonometric sine/cosine harmonic transformations (`month_sin`, `month_cos`).

---

## 6. What Models Did We Build and Why?

1. **Random Forest (Phase 1 Baseline):** Bagging ensemble of 30 decision trees. Excellent for tabular data with sharp thresholds.
2. **XGBoost (Phase 1 Gradient Boosting):** Sequentially boosted trees. Fast inference ($4.6\text{ ms}$ latency), great for deployment.
3. **WaterQualityMLP (Phase 2 Deep Learning):** Funnel architecture `53 → 512 → 256 → 128 → 64 → 1` with BatchNorm and Dropout ($0.3$). Evaluates how standard neural networks model tabular environmental telemetry.
4. **Residual Hybrid Model (Phase 2 Synergy):**
   - Combines a **Base Tree Ensemble** ($0.5\times\text{RF} + 0.5\times\text{XGB}$) with a neural network (`ResidualMLPNet`).
   - The neural network takes 54 inputs (53 DL features + 1 base tree prediction scalar) and predicts the signed error ($\hat{r}$) made by the trees.
   - Final prediction: $\hat{y}_{\text{hybrid}} = \hat{y}_{\text{base\_tree}} + \hat{r}$.

---

## 7. How Did the Models Perform? (Key Results Table)

Evaluated on the exact same **422,943 held-out test records**:

| Model | MAE | RMSE | $R^2$ | Max AE | Best At |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | **0.0514** | 0.2016 | 0.999870 | 12.6128 | Lowest median error across normal water |
| **Residual Hybrid** | 0.0710 | **0.1678** | **0.999910** | **5.6842** | **Lowest overall variance & best outlier control** |
| **Base Tree Ensemble** | 0.0900 | 0.2464 | 0.999805 | 8.7128 | Solid tree baseline |
| **XGBoost** | 0.1614 | 0.4364 | 0.999389 | 9.2769 | Fastest inference latency ($4.6\text{ ms}$) |
| **Weighted Ensemble** | 0.3096 | 0.4155 | 0.999446 | 8.8283 | Linear combination ($0.4\text{RF}+0.4\text{XGB}+0.2\text{MLP}$) |
| **WaterQualityMLP** | 1.3800 | 1.6720 | 0.991029 | 31.1327 | Tabular deep learning baseline |

---

## 8. Why Do Tree Models and the MLP Perform Differently?

- **Why Random Forest wins on MAE:** The CCME index uses step-function threshold rules (e.g. if phosphorus $> 0.1$, add an exceedance). Decision trees split data along orthogonal, axis-aligned boundaries, which naturally matches threshold-based regulatory rules.
- **Why the MLP has higher MAE ($1.3800$ vs $0.0514$):** Neural networks use continuous smooth activation functions (ReLU, linear layers) and gradient descent, which tend to smooth out sharp tabular boundaries and struggle on irregular, non-smooth decision steps.
- **Why the Residual Hybrid is superior on RMSE ($0.1678$) and Max AE ($5.6842$):** Where trees make severe errors (outliers, rare combinations where trees produce step artifacts), the residual neural network learns these residual error patterns and corrects them, cutting peak error by **54.9%**!

---

## 9. What Do the Metrics Mean?

- **MAE (Mean Absolute Error):** Average magnitude of errors: $\frac{1}{N}\sum |y - \hat{y}|$. In our units (0–100 scale), an MAE of 0.0514 means on average the prediction is within 0.05 index points of the true score.
- **RMSE (Root Mean Squared Error):** Penalizes large outlier errors heavily: $\sqrt{\frac{1}{N}\sum (y - \hat{y})^2}$. A low RMSE means the model rarely makes huge mistakes.
- **$R^2$ (Coefficient of Determination):** Proportion of variance explained: $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$. Values near 1.0 indicate near-perfect correlation with ground truth.
- **Residual Sign Convention:** We strictly use $\text{Residual} = \hat{y} - y = \text{Prediction} - \text{Actual}$. Positive residual means overprediction; negative residual means underprediction.

---

## 10. What Does Explainability Tell Us?

### TreeSHAP (Phase 1):
Shows that **orthophosphate** accounts for over 88% of split decisions, followed by nitrogen and ammonia. Phosphorus is the primary limiting nutrient in freshwater bodies.

### Integrated Gradients (Phase 2):
Axiomatic path-integral feature attribution for the neural network using a valid baseline (clean river water in England, score $\approx 86.72$):
- **Negative Attributions (Pull Score Down):** High orthophosphate, high ammonia, high BOD, high nitrogen, and high exceedance counter.
- **Positive Attributions (Push Score Up):** High dissolved oxygen saturation and low pollutant levels.
- **Physical Realism:** The model learned real environmental chemistry, not spurious statistical shortcuts.

---

## 11. Top 15 Tough Viva Questions & How to Answer Them

### Q1: Why is your $R^2$ so high ($> 0.99$)? Did your model overfit or leak data?
**Answer:** The CCME Water Quality Index is calculated using a deterministic mathematical formula based on measured parameters and guideline thresholds. Our models are learning to approximate this underlying mathematical function from the chemical inputs. Furthermore, we verified zero leakage: scalers were fit only on training data, temporal station lags used zero-lookahead, and the $R^2$ is evaluated on a completely untouched test set of 422,943 records.

### Q2: Does the feature `exceedance_counter` cause data leakage?
**Answer:** `exceedance_counter` counts how many parameters failed thresholds in that sample, which is related to the $F_1$ factor in CCME. However, in Task 5 we ran a systematic 7-group ablation study: when `exceedance_counter` is completely removed, Random Forest still achieves $R^2 = 0.99987$ and XGBoost achieves $R^2 = 0.9989$. The raw physical sensors (`base_sensors`) carry almost all the predictive information.

### Q3: Why did you build a hybrid model if Random Forest already had the lowest MAE?
**Answer:** While Random Forest had lower median and mean absolute error across the dense center of the distribution, it suffered from large peak errors on extreme outliers ($\text{Max AE} = 12.6128$). The Residual Hybrid combines the tree ensemble with a neural network that predicts and corrects tree residual errors. The hybrid achieved the lowest RMSE ($0.1678$ vs $0.2016$) and reduced the maximum prediction error by **54.9%** (down to $5.6842$).

### Q4: How is the base prediction in the hybrid model computed?
**Answer:** The base tree prediction is the equal-weighted average of Phase 1 Random Forest and XGBoost: $\hat{y}_{\text{base\_tree}} = 0.5\times\hat{y}_{\text{RF}} + 0.5\times\hat{y}_{\text{XGB}}$. The `ResidualMLPNet` takes 54 inputs (53 DL features + 1 base tree prediction scalar) and outputs a signed correction scalar $\hat{r}$.

### Q5: What is Out-of-Fold (OOF) training and why was it necessary?
**Answer:** If we trained the residual network on the base trees' training predictions, the trees would have near-zero errors on their own training data (memorization), so the residual network would have nothing realistic to learn. Using 5-fold cross-validation on the training set, we generated out-of-fold predictions where each fold's predictions come from trees that never saw those rows during training. This gave the residual network realistic, leakage-free residual targets.

### Q6: Why did you use Integrated Gradients instead of standard saliency maps?
**Answer:** Standard gradient saliency maps suffer from gradient saturation (when a feature becomes very large, the gradient flattens to zero even though the feature is important). Integrated Gradients integrates gradients along a straight-line path from a neutral reference baseline to the input sample. It satisfies the mathematical axioms of **Completeness** (attributions sum to the difference in output) and **Implementation Invariance**.

### Q7: What baseline did you use for Integrated Gradients and why not all zeros?
**Answer:** An all-zero baseline would set all 17 one-hot categorical flags to 0.0, representing a non-existent country and waterbody. Instead, we used a **modal observed baseline**: 0.0 for standardized numerical features (their training mean) and one-hot vectors for England (`country_code=2`) and River (`waterbody_type_code=8`). This represents a real, physically plausible clean English river with a baseline WQI score of $86.72$.

### Q8: What was the purpose of the 7-group feature ablation study?
**Answer:** To measure the unique marginal contribution of each domain subsystem. We systematically neutralized one group at a time (setting features to their population mean) and measured the error increase. We found that `base_sensors` was the irreplaceable primary pillar (removing it caused XGBoost MAE to jump by $+21.65$), while `log1p_normalized` was critical for the neural network ($\Delta\text{MAE} = +12.16$).

### Q9: Why did you apply $\log(1+x)$ transformations?
**Answer:** Freshwater nutrient parameters like orthophosphate and ammonia have heavy positive skewness ($\gamma_1 > 40$) because most samples have low concentrations while a few agricultural runoff events have massive spikes. Neural networks and linear operations struggle with extreme skewness. $\log(1+x)$ compresses the scale without losing ordering, improving feature correlation with WQI from $-0.66$ to $-0.85$.

### Q10: What is the difference between your Residual Hybrid and the Weighted Ensemble?
**Answer:** The Weighted Ensemble is a simple linear convex blend: $0.4\times\text{RF} + 0.4\times\text{XGB} + 0.2\times\text{MLP}$ ($\text{MAE} = 0.3096$). The Residual Hybrid is a multi-paradigm architecture where the neural network takes the non-linear features plus the base tree prediction to learn a targeted residual correction ($\text{MAE} = 0.0710$, $\text{RMSE} = 0.1678$).

### Q11: How did you prevent data leakage during temporal modeling?
**Answer:** Historical station lags (`shift-1` and `rolling-3`) were computed strictly within chronological station groupings with zero lookahead (only past and current observations were used). Furthermore, all scalers and encoders were fit exclusively on the training partition.

### Q12: What loss function was used to train the Residual MLP and why?
**Answer:** Smooth L1 (Huber) loss. Residual errors have occasional extreme outliers. Standard MSE loss squares errors, causing large gradients that destabilize training on noisy samples. Smooth L1 behaves like MSE for small errors ($|e| < 1.0$) and like MAE for large errors, providing robustness against outlier shocks.

### Q13: What does the sign of a residual mean in your error analysis?
**Answer:** We followed the standard scientific convention: $\text{Residual} = \hat{y} - y = \text{Prediction} - \text{Actual}$. A positive residual means the model overpredicted (predicted better water quality than reality), while a negative residual means the model underpredicted (predicted worse quality than reality).

### Q14: Which model would you deploy in an edge IoT sensor and why?
**Answer:** **XGBoost**. It has a single-sample inference latency of only **$4.6\text{ ms}$** and requires minimal memory compared to Random Forest's large memory footprint ($150\text{ MB}$ uncompressed trees). It maintains strong accuracy ($\text{MAE}=0.1614$, $R^2=0.9994$) with high throughput ($> 50{,}000\text{ samples/sec}$).

### Q15: What are the main real-world limitations of this project?
**Answer:**
1. **Target Formulation:** Since CCME WQI is a deterministic index, the models approximate a formula rather than unobserved natural phenomena.
2. **Geographic Distribution:** England represents $83.7\%$ of the dataset; while cross-national transferability was confirmed, smaller geographic regions have higher relative uncertainty.
3. **Sensor Drift:** Real-world deployments must handle physical sensor degradation and calibration drift over time, which were not present in static archived benchmarks.
