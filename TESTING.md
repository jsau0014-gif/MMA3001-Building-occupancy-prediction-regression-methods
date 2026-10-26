# Running the automated tests

Copy the `tests/` folder and `requirements-test.txt` into the **root of your
existing project** (the directory containing `src/`, `Data/`, `Results/`
and `requirements.txt`). Do not put `tests/` inside `src/`.

In VS Code / PowerShell, from the project root:

```powershell
python -m pip install -r requirements-test.txt
python -m pytest tests -v
```

To save an updated run log:

```powershell
python -m pytest tests -v -rA *> docs/pytest_results.txt
```

The preprocessing and feature-engineering tests each copy the **actual**
production script from `src/` into a temporary workspace with small,
synthetic CSV inputs. Your original datasets are never modified.

The `test_results_consistency.py` tests check the exported G.20 prediction
rows and comparison metrics, if those CSV files are present in your local
`Results/` folder. Otherwise these tests are skipped, not failed.

After a successful run, update `docs/TEST_REPORT.md` with the date,
Python/dependency versions and the actual passed/skipped/failed totals.

## Important: saved SVR model consistency

At the time these files were prepared, `src/05_train_models.py` saved the
full-data SVR to `Results/05_svr_model_full.joblib`, but
`src/06b_validate_G20_minute.py` loaded `Results/05_svr_model.joblib`.
Before reporting a single model's G.25 and G.20 results together, align
these filenames and re-run G.20 evaluation. Changing the filename can
change the reported G.20 metrics when different fitted models exist.

## Test scope

- Exact integration on known rectangles, including duplicate timestamps
- Handling the first partially measured minute
- Omitting unmeasured minutes without synthesising targets
- Dropping missing/non-numeric readings, preserving zero occupancy
- Single-reading zero-order hold to the minute boundary
- Independent integration oracle over reproducible random events
- Correct local-time, weekday, month and cyclic feature values
- Consistency between G.25 and G.20 feature-engineering scripts
- Recalculation of saved MAE, RMSE, R² and signed mean residual

**Not covered:** verifying sensor ground truth, validating the constant-
occupancy assumption across long gaps, or re-training the full SVR (which
requires several hours). These are modelling and computational limitations,
not proof that preprocessing is exact in the real world.
