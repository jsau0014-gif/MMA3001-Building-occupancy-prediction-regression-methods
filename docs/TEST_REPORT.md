# Automated Numerical Verification and Test Report

**Project:** Predicting Building Occupancy Using Machine Learning: A Comparative Study of Regression Methods  
**Unit:** MMA3001 — Numerical Methods and Machine Learning  
**Report date:** 26 October 2026  
**Repository:** https://github.com/jsau0014-gif/MMA3001-Building-occupancy-prediction-regression-methods

## 1. Objective and scope

The aim was to verify that the numerical preprocessing and temporal feature-engineering algorithms produce the correct results for known inputs, separately from evaluating the predictive accuracy of the trained models. The tests exercise both room-specific preprocessing scripts (`03_preprocess_training.py` and `03b_preprocess_G20.py`) and both relevant feature-engineering scripts (`04_feature_engineering.py` and `04b_feature_engineering_G20.py`). They do not require the original, multi-gigabyte sensor dataset or re-training the SVR.

Tests execute project scripts in **isolated temporary folders** containing synthetic CSV files. The actual sensor datasets and training outputs are not modified. Expected occupancy averages are established using analytical sums of occupancy × duration. A separate, independently written interval-walking implementation is also used as a numerical reference for a reproducible randomized scenario.

## 2. Execution environment and scope of this run

| Item | Value |
| --- | --- |
| Verification environment | Linux, Python 3.13.5 |
| Framework | pytest 9.0.2 |
| Numerical libraries | pandas 2.2.3, NumPy 2.3.5, Matplotlib 3.10.8 |
| Evaluation library | scikit-learn 1.8.0 |
| Input data | Small, synthetic CSV measurements with predefined headcounts |
| Time representation | UTC input timestamps, converted to `Australia/Melbourne` for model features |
| Test execution | `python -m pytest tests -q -rA` |

**Evidence qualification:** This ChatGPT-session run used a local working mirror of the numerical logic retrieved from the current GitHub source, since the complete Git repository was not mounted in the test container. The delivered tests are designed to execute the **actual, unmodified scripts from `src/`** when copied into the student's local Git repository. Re-run the suite there and attach its console output as the definitive evidence for the submitted repository. The output file `docs/pytest_local_validation.txt` is evidence of the local dry run, not a claim of having tested the student's Windows working tree.

## 3. Test cases and results

| ID | Test and expected result | Rooms | Local dry-run result |
| --- | --- | --- | --- |
| N01 | Exact zero-order-hold rectangle integration; a duplicate timestamp takes the final recorded value. First minute = **4.166667 occupants**; next minute = **2.333333 occupants**. | G.25, G.20 | PASS (2) |
| N02 | Initial partially covered minute: `(2×20 + 6×30)/50` = **4.4 occupants** rather than dividing by a full 60 seconds. | G.25, G.20 | PASS (2) |
| N03 | A minute with no valid raw reading is not synthesized. When readings occur in minutes 0 and 2, only those minutes are exported. | G.25, G.20 | PASS (2) |
| N04 | Invalid timestamps and non-numeric headcounts are excluded; valid zero occupancy is preserved. | G.25, G.20 | PASS (2) |
| N05 | A single reading is held to the end of its retained minute; a headcount of 7 produces a minute average of 7. | G.25, G.20 | PASS (2) |
| N06 | Numerically compare event-based integration with an independently written ZOH rectangle-summing oracle on seeded randomly timed readings, including duplicated times. | G.25, G.20 | PASS (2) |
| F01 | Calendar encoding uses Melbourne local time; check midnight, Sunday/weekend and December–January boundary. | G.25 | PASS |
| F02 | Identical timestamps generate identical seven features in G.25, raw G.20 and minute-averaged G.20. | Both | PASS |
| F03 | Each sine/cosine feature pair satisfies `sin² + cos² = 1` to floating-point precision. | G.25 | PASS |
| F04 | Invalid timestamps do not enter the final model feature table. | G.25 | PASS |
| M01–M03 | Recalculate MAE, RMSE, R², signed mean residual and optional residual columns from saved G.20 prediction records and compare with the reported comparison CSV. | G.20 | SKIPPED (3): generated CSVs unavailable in local dry run |

### Example analytical verification (N01)

For occupancy readings of 2 occupants at 00:00:00, 6 at 00:00:20 and 4 at 00:00:45, the analytical zero-order-hold minute average is:

\[
\overline{H}=\frac{2(20)+6(25)+4(15)}{60}=4.166667\text{ occupants}.
\]

The fixture also includes an earlier duplicate reading at 00:00:20 with a different value; the last valid observation at that timestamp takes precedence. The subsequent minute contains 3 occupants for 40 seconds followed by 1 occupant for 20 seconds, giving `140/60 = 2.333333` occupants. These expected values were matched within numerical floating-point tolerance in the local dry run.

### Local dry-run outcome

| Status | Number |
| --- | ---: |
| **Passed** | **16** |
| **Failed** | **0** |
| Skipped (full-data metric checks) | 3 |
| Total collected | 19 |

The optional metrics tests should run on the student's Windows machine once the final `Results/06b_G20validation_predictions.csv` and `Results/06b_G20model_comparison.csv` are available. A passing comparison checks the internal consistency of those reported metrics, not the model's independent predictive validity.

## 4. Known issues and limitations

1. **Potential saved-model inconsistency (requires resolution).** The checked training script saved its full-data SVR as `Results/05_svr_model_full.joblib`, but the cross-room validation script loaded `Results/05_svr_model.joblib`. This can mean the report's full-data G.25 SVR result and its G.20 result come from *different fitted SVRs*. Align the filenames and re-run the G.20 evaluation before declaring final consistent model performance. If the results change, update the report tables/figures.
2. **Negative headcounts are not explicitly rejected.** Current preprocessing drops missing/non-numeric values but does not reject negative numeric headcounts. A physical input-validation check such as `headcount >= 0` is recommended, alongside a dedicated test, if the source dataset may contain negative values.
3. **Unmeasured periods remain uncertain.** Tests establish correctness **under the zero-order-hold reconstruction**; they cannot show that occupancy actually stayed constant between sensor events, especially over long gaps.
4. **No full-model retraining in these tests.** The large SVR took several hours to fit. The suite instead verifies the fast mathematical stages and, if available, recomputes metrics from the saved full-data predictions.
5. **Random internal validation is not a chronological generalisation test.** Model validation results and their methodological limitations remain documented in the main report.

## 5. How to reproduce and archive evidence

From the repository root in PowerShell:

```powershell
python -m pip install -r requirements-test.txt
python -m pytest tests -v
python -m pytest tests -v -rA *> docs/pytest_results.txt
```

Review the `pytest_results.txt` file and record the actual pass/fail/skip totals and relevant software versions. Resolve any failures before claiming complete numerical verification. Commit the `tests/` scripts, this report and the successful run log to GitHub.

A pytest test report and test cases provide meaningful evidence of implementation correctness; they do **not** alone establish the accuracy of occupancy ground truth or justify the operational use of the trained regression models.
