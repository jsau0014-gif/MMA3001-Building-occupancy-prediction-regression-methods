# Automated Numerical Verification and Test Report

**Project:** Predicting Building Occupancy Using Machine Learning: A Comparative Study of Regression Methods  
**Unit:** MMA3001 — Numerical Methods and Machine Learning  
**Report updated:** 26 October 2026  
**Repository:** https://github.com/jsau0014-gif/MMA3001-Building-occupancy-prediction-regression-methods

## 1. Objective and scope

This test suite verifies that numerical preprocessing and temporal feature engineering produce expected outputs for known inputs, separately from assessing predictive performance on measured occupancy data. It tests the actual room-specific preprocessing scripts (`src/03_preprocess_training.py` and `src/03b_preprocess_G20.py`) and both feature-engineering workflows (`src/04_feature_engineering.py` and `src/04b_feature_engineering_G20.py`).

Tests run the production scripts in isolated temporary directories with small synthetic CSV files, without modifying the original occupancy datasets or trained models. Exact occupancy averages are checked against analytical occupancy-duration calculations; an independently implemented interval-walking reference provides an additional randomized comparison.

Three further tests independently recompute the exported G.20 evaluation metrics from saved predictions. These verify **consistency of the reported calculations**, not predictive validity or the provenance of a saved model checkpoint.

## 2. Recorded test environment and execution

The definitive repository evidence is the student's local Windows run in [`docs/pytest_results.txt`](pytest_results.txt).

| Property | Recorded result |
| --- | --- |
| Operating system | Windows (`win32`) |
| Python | 3.13.14 |
| pytest | 9.1.1 |
| pluggy | 1.7.0 |
| Command | `python -m pytest tests -v -rA` |
| Tests collected | **19** |
| Passed / failed / skipped | **19 / 0 / 0** |
| Execution duration | **14.98 seconds** |
| Data for numerical tests | Synthetic readings with known reference answers |
| Data for metric-consistency tests | Locally exported G.20 prediction and comparison CSV files |
| Timestamp interpretation | UTC source timestamps converted to `Australia/Melbourne` for temporal features |

The committed console log shows that all 19 tests ran against the local project, including the three G.20 metric-consistency tests that were unavailable in the preliminary, isolated test environment. The log is evidence for that recorded run; if data or implementation files subsequently change, rerun the suite and commit the new log.

## 3. Test cases and results

| ID | Test and expected result | Coverage | Windows result |
| --- | --- | --- | --- |
| N01 | Exact zero-order-hold rectangle integration and duplicate timestamps (retain last reading): first minute **4.166667**, following minute **2.333333** occupants | G.25 and G.20 | **PASS (2)** |
| N02 | Initial partial minute uses measured duration: `(2×20 + 6×30)/50 = 4.4` occupants | G.25 and G.20 | **PASS (2)** |
| N03 | Minutes containing no valid recorded measurement are not synthesized | G.25 and G.20 | **PASS (2)** |
| N04 | Invalid timestamps and nonnumeric headcounts are excluded; zero occupancy is preserved | G.25 and G.20 | **PASS (2)** |
| N05 | One reading held to the end of its minute yields the recorded headcount (7 occupants) | G.25 and G.20 | **PASS (2)** |
| N06 | Seeded, irregularly spaced readings agree with an independently written interval-integration reference, including duplicates | G.25 and G.20 | **PASS (2)** |
| F01 | Melbourne-local calendar encoding at midnight, weekends and month/year boundaries | G.25 | **PASS** |
| F02 | Identical timestamps produce identical seven engineered features across room workflows | Both | **PASS** |
| F03 | Sine/cosine feature pairs satisfy `sin² + cos² = 1` within floating-point tolerance | G.25 | **PASS** |
| F04 | Invalid timestamps are omitted from the resulting feature table | G.25 | **PASS** |
| M01 | Recompute Linear Regression MAE, RMSE, R², signed mean residual and residual consistency from exported G.20 predictions | G.20 | **PASS** |
| M02 | Recompute the same metrics for Decision Tree Regression | G.20 | **PASS** |
| M03 | Recompute the same metrics for SVR | G.20 | **PASS** |

### Example analytical verification (N01)

For readings of two occupants at 00:00:00, six at 00:00:20 and four at 00:00:45, the analytical mean under zero-order hold is

```text
(2 × 20 + 6 × 25 + 4 × 15) / 60 = 4.166667 occupants
```

The fixture includes an earlier conflicting reading at 00:00:20; the final observation at that timestamp is retained. In the next minute, occupancy is three for 40 seconds and one for 20 seconds, so the expected average is `(3×40 + 1×20)/60 = 2.333333`. Both room-specific preprocessing scripts matched these known results within numerical tolerance.

### Results summary

| Status | Count |
| --- | ---: |
| **Passed** | **19** |
| Failed | 0 |
| Skipped | 0 |
| **Total** | **19** |

## 4. Saved SVR checkpoint consistency

An earlier review found that `05_train_models.py` saved the final SVR to `05_svr_model_full.joblib`, whereas `06b_validate_G20_minute.py` loaded `05_svr_model.joblib`. Inspection of the earlier loaded checkpoint showed a fitted shape of **(50,000, 7)**, while the full saved checkpoint had **(298,608, 7)**.

The training script was subsequently updated to save its full-training-set SVR as `Results/05_svr_model.joblib`. The full checkpoint was copied to that filename locally, and the user reran G.20 evaluation with the full model. The final reported SVR G.20 metrics were:

| Metric | Full-model G.20 validation |
| --- | ---: |
| Evaluation observations | 306,459 |
| MAE | 1.901582 |
| RMSE | 3.875962 |
| R² | −0.071607 |
| Mean residual (actual − predicted) | +1.208271 |
| Prediction time | 1,579.286926 seconds |

**Important scope distinction:** Tests M01–M03 verify that the values in the locally saved prediction and comparison CSV files agree. They do not independently prove the trained checkpoint contains 298,608 observations; that was checked separately using `SVR.shape_fit_`. Keep the refreshed G.20 CSVs and test log together when archiving final evidence.

## 5. Known limitations

1. **Numerical correctness is conditional on the model assumptions.** Zero-order hold may not describe true occupancy between widely spaced readings. The tests cannot independently establish sensor accuracy or ground-truth occupancy.
2. **Negative numeric headcounts are not explicitly rejected by the preprocessing scripts.** Invalid and missing values are excluded, but nonnegative-domain enforcement would need an additional check and test.
3. **The suite does not retrain the full SVR.** That would take hours; instead, it checks the numerical workflow and recomputes metrics from saved predictions.
4. **Random internal validation is not temporal forecasting validation.** Nearby G.25 observations can occur in both training and validation subsets, and the same validation partition informs model selection. Cross-room G.20 evaluation checks transfer to a different room but not future chronological robustness.
5. **CSV metric consistency does not guarantee engineering adequacy.** Negative G.20 R² and systematic underprediction limit the usefulness of the models for safety- or capacity-critical occupancy control.

## 6. Reproducing and preserving evidence

Run the following commands at the project root in Windows PowerShell:

```powershell
python -m pip install -r requirements-test.txt
python -m pytest tests -v -rA
python -m pytest tests -v -rA *> docs/pytest_results.txt
Get-Content docs/pytest_results.txt -Tail 5
```

The three metric-consistency tests require both the exported G.20 prediction CSV and corresponding summary CSV in the local `Results/` folder. If those files are unavailable, the three tests are skipped rather than failed. Run `src/06b_validate_G20_minute.py` to regenerate them, after ensuring the correct full SVR model is loaded.

The saved console log, this report, and the test scripts should be committed together. Successful tests demonstrate important implementation and reporting consistency, but do not on their own prove reliable real-world occupancy prediction.
