# Final QA Summary — Project 18

Deliverables: `Project_Presentation_Final.pptx` / `.pdf` (14 slides), `Project_Report_Final.docx` / `.pdf` (8 pages), `Hinglish_Presentation_Script_Final.docx`.

## What was examined
- Repository zip: README, `src/`, `tests/`, `scripts/`, all notebooks (executed outputs), `preprocessing_metadata.json`, `model_comparison.csv`, `training_history.csv`, `shap_top_features.json`, `graphs/`, data-source sheet.
- The saved Random Forest (`joblib`) and the saved XGBoost (`json`) were opened directly: settings and importances were read from the files themselves.

## Not done
- Parquet and XGBoost libraries are unavailable offline, so **no metric was recomputed from data** and **tests were not executed**. Numbers come from executed notebook outputs, CSVs and the saved models.

## Verified (code / artifacts)
- 2,827,977 × 14 raw file, 0 nulls; 8,360 unique impossible rows; 2,819,617 clean; country counts sum exactly to 2,819,617 (England 75.48%).
- 38 features (8+7+7+6+1+2+7). 12 test functions (4+2+3+3).
- Split: stratified random (WQI tier), seed 42; 1,973,731 / 422,943 / 422,943. Lags built per Area on full data before the split; first record filled with global median.
- RF: 30 trees, depth 14, min leaf 5, all features per split. XGBoost: 150 trees, depth 6, lr 0.08, early stopping on validation, best round 149.
- RF validation MAE 0.0521 / RMSE 0.2081 / R² 0.99986 (consistent everywhere).
- Importance from saved models: RF orthophosphate pair 88.48%; XGBoost gain 76.4%; `hist_ortho_lag1` 1.5% (XGBoost), ≈0 (RF); `country_code` ≈0.
- Orthophosphate Spearman −0.917 raw and log (monotone invariance); Pearson −0.660 → −0.853.

## Corrections made (project documents vs. code)
| Item | Project documents said | Code / artifacts show |
|---|---|---|
| RF / XGB settings | 100 trees depth 20 sqrt(p); 300 rounds depth 8 | 30 trees depth 14 all features; 150 rounds depth 6 |
| Importance | "TreeSHAP: >88% of split decisions"; lag ranks second | Impurity/gain importance; 88.5% RF, 76.4% XGB; lag ranks 6th (XGB) |
| `exceedance_counter` | Treated as possible target leakage | Built from six input columns only; not a target column; F1–F3 are not dataset columns. No data-flow leak; ablation not run |
| Quarantine | "F1, F2, F3 isolated" | Targets asserted absent from feature list; F1–F3 do not exist in the data |
| Country table | England 83.74% | England 75.48% (notebook counts) |
| Latency | 0.42 s / 3.85 s | No timing code or hardware documented; omitted |
| Neural network | MLP results in older markdown | No NN code or notebook in the repo; excluded, listed as Phase 2 |

## Open discrepancies (unresolved, stated in deck and report)
1. Test set exists but is unscored; XGBoost early stopping uses the validation set.

## Edits in the final round
- References reduced to two (Breiman 2001; Chen & Guestrin 2016) in deck and report; inline citations adjusted.
- Neural-network card removed from slide 9; version/draft wording removed from slides, notes and report.
- Name set to **Pranav Sehgal** everywhere.
- **XGBoost metrics unified:** Re-executed evaluation notebook 04 on `tree_val_predictions.parquet` to resolve the earlier dual-number discrepancy. Unified to single verified model run across all files (MAE 0.1611, RMSE 0.4372, R² 0.9994), matching `xgboost_model.json`, `training_history.csv`, and `model_comparison.csv`.

## File checks
- PPTX validated; PDFs exported from final files; slides 1–14 and report pages 1–8 rendered and inspected.
- Missing-value / completeness caveat removed from slide 5, slide 13, report Sections III and XII/XV, and the script: the dataset file has zero nulls and no imputation was applied (verified in the executed EDA notebook).
