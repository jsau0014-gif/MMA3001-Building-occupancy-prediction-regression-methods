# Predicting Building Occupancy Using Machine Learning

### A Comparative Study of Regression Methods

**MMA3001 — Numerical Methods and Machine Learning | Monash University | 2026**

This project investigates predicting university room occupancy using only **calendar-derived temporal features**. Three methods—**Multiple Linear Regression**, **Decision Tree Regression** and **radial basis function Support Vector Regression (SVR)**—are fitted on measurements from **Room G.25**. They are evaluated both on withheld G.25 observations and on independent measurements from **Room G.20**.

The work covers irregularly sampled sensor data, zero-order-hold integration, cyclical feature engineering, model optimisation, computational cost, numerical verification and cross-room generalisation.

## Objectives

- Construct one-minute **time-weighted occupancy targets** from irregular sensor events.
- Derive seven calendar features without using a measured headcount as a predictor.
- Compare three regression methods with a training-mean baseline using MAE, RMSE and R².
- Investigate model-complexity and computational trade-offs, particularly decision-tree leaf size and SVR fitting cost.
- Assess whether models trained in G.25 generalise to the unseen G.20 room.

## Data, inputs and outputs

The original occupancy-sensor CSV is stored under `Data/` and tracked using **Git LFS**. The extraction script selects G.25 and G.20 using the original `floorspaceid` values; the main raw columns used for this project are `collecteddate` (timestamp) and `headcount` (occupants).

Preprocessing:

1. Excludes rows with missing/invalid timestamps or nonnumeric/missing headcounts; valid zero occupancy is retained.
2. Orders readings by time, keeping the last recorded value at duplicate timestamps.
3. Reconstructs occupancy as piecewise constant between readings (**zero-order hold**).
4. Integrates occupancy multiplied by duration to calculate **minute-level time-weighted mean headcount**, including correct normalisation of an initially partially observed minute.
5. Retains **only minutes containing at least one valid original sensor reading**. It does not fill unmeasured gaps with synthetic targets.

Timestamps are converted to the `Australia/Melbourne` time zone before constructing these seven model inputs:

| Input | Description |
| --- | --- |
| `TimeSin`, `TimeCos` | Sine and cosine of time of day, including fractional hour |
| `DaySin`, `DayCos` | Sine and cosine of day of week |
| `MonthSin`, `MonthCos` | Sine and cosine of month of year |
| `IsWeekend` | Binary indicator: Saturday/Sunday versus weekday |

**Prediction output:** a floating-point estimate of the **one-minute time-weighted average headcount**, in occupants. Model predictions are not constrained to be integers or physically nonnegative. Only timestamp features enter the regression models; timetables, room characteristics and previous headcounts are not inputs.

## Repository structure

```text
.
├── Data/                         # Source datasets and metadata (Git LFS)
│   └── processed/                # Generated room-level/model-ready CSVs (Git-ignored)
├── src/                          # Extraction, preprocessing, models, validation, figures
├── Results/                      # Generated metrics, plots, predictions and models
│   └── report_figures/           # Final report figures (generated locally)
├── tests/                        # Automated numerical and metric-consistency tests
├── docs/
│   ├── TEST_REPORT.md            # Verification methods and results
│   ├── pytest_results.txt        # Recorded Windows test output
│   ├── html/index.html           # Offline HTML source-code documentation
│   └── HTML_DOCUMENTATION.md     # Documentation build instructions
├── tools/build_html_docs.py      # Safe AST-based documentation generator
├── archive/                      # Historical/experimental scripts
├── LICENSE                       # MIT licence for original software
├── DATA_LICENSE_NOTICE.md        # Dataset ownership / redistribution caveat
├── TESTING.md                    # Detailed instructions for running tests
├── requirements.txt              # Main dependencies
├── requirements-test.txt         # Test dependencies and joblib
└── README.md
```

Generated `Data/processed/` files and trained `.joblib` models are not version-controlled. The `Results/` directory may contain additional locally generated files that are not included in the remote repository. Reproduce those outputs using the workflow below.

## Installation (Windows PowerShell)

Prerequisites: a working Python 3 environment, `pip`, Git, Git LFS and enough free disk space/memory for the occupancy dataset. The final RBF SVR fit may take several hours.

```powershell
git lfs install
git clone https://github.com/jsau0014-gif/MMA3001-Building-occupancy-prediction-regression-methods.git
cd MMA3001-Building-occupancy-prediction-regression-methods
git lfs pull

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install joblib
```

`joblib` is required for saving/loading fitted models and is included in `requirements-test.txt`; it is not yet explicitly included in `requirements.txt`. For the test suite, install `requirements-test.txt` as described below. Package versions are not pinned, so minor runtime or numerical variation can occur.

## Reproducing the final modelling workflow

Run the following commands **from the repository root**, in sequence. Script paths are resolved relative to the scripts, so renaming the local outer project folder does not affect them.

### 1. Extract and inspect the room measurements

```powershell
python src/01_Extract_Occupancy_Data.py
python src/02_Inspect_Occupancy_Data.py
```

The extraction reads `Data/5occupancySensor_MayToDec2024_9MRows.csv` in chunks and writes `Data/processed/G25_Occupancy.csv` and `G20_Occupancy.csv`.

### 2. Calculate time-weighted one-minute averages

```powershell
python src/03_preprocess_training.py
python src/03b_preprocess_G20.py
```

Outputs: `Data/processed/G25_Minute_Training.csv` and `Data/processed/G20_Minute_validation.csv`. The study retained **373,260 G.25** and **306,459 G.20** processed observations.

### 3. Construct the model inputs

```powershell
python src/04_feature_engineering.py
python src/04b_feature_engineering_G20.py
```

Use `Data/processed/G25_Model_Train.csv` for model development and `Data/processed/G20_Model_Validation_Minute.csv` for final cross-room evaluation. Script 04 also exports `G20_Model_Validation.csv` based on the original *unaggregated* G.20 readings for a legacy workflow; it is **not** the final G.20 evaluation input.

### 4. Fit and internally validate the three models on G.25

```powershell
python src/05_train_models.py
```

The script uses a reproducible **random 80% training / 20% validation split** (`random_state=42`): **298,608 training** and **74,652 validation** observations. It compares the training-mean baseline, fits Multiple Linear Regression, tunes a Decision Tree and fits the selected full-data RBF SVR with standardised inputs.

**Model filenames are now consistent across scripts:**

| Fitted model | Saved path |
| --- | --- |
| Linear Regression | `Results/05_linear_model.joblib` |
| Decision Tree | `Results/05_decision_tree.joblib` |
| Full-training SVR | `Results/05_svr_model.joblib` |

The final SVR has training shape **(298608, 7)**. An older checkpoint using the same base filename was previously trained on 50,000 observations; the full-data checkpoint was separately verified and substituted before the final cross-room run. If using locally retained historical models, check that the current SVR file is the intended full fit:

```powershell
python -c "import joblib; m=joblib.load(r'Results\05_svr_model.joblib'); print(m.named_steps['svr'].shape_fit_)"
```

Expected: `(298608, 7)`. Re-running Script 05 recreates the full fit, but can take several hours.

### 5. Evaluate independent G.20 measurements

```powershell
python src/06b_validate_G20_minute.py
```

This loads the already fitted G.25 models and predicts G.20 **without retraining or G.20-based parameter adjustment**. The script's sampling cap is larger than the processed dataset, so it evaluates **all 306,459 G.20 observations**. It saves:

- `Results/06b_G20model_comparison.csv` — MAE, RMSE, R², mean residual and prediction time.
- `Results/06b_G20validation_predictions.csv` — timestamped actual and predicted occupancy with residuals.
- Actual-versus-predicted, residual and occupancy plots.

The older `src/06_validate_G20.py` evaluates unaggregated G.20 observations and is **not** part of the final like-for-like minute-averaged comparison.

### 6. Generate report figures

```powershell
python src/07_generate_report_figures.py
```

Report graphics are generated under `Results/report_figures/`. The script reads previously saved model outputs rather than refitting any model. Some optional figures use experimental comparison CSVs from earlier iterations (such as a 100,000-observation SVR experiment) and may be omitted if those files are absent.

## Final numerical results

The reported internal and independent evaluation results are:

| Model | G.25 internal RMSE (occupants) | G.25 internal R² | G.20 external RMSE (occupants) | G.20 external R² |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 1.8931 | 0.1481 | **3.8183** | −0.0400 |
| Decision Tree | **1.6633** | **0.3424** | 3.8390 | −0.0513 |
| SVR (298,608 G.25 training observations) | 1.7260 | 0.2919 | 3.8760 | −0.0716 |

G.20 full-model validation used 306,459 processed observations. Its final SVR metrics were **MAE 1.901582, RMSE 3.875962, R² −0.071607 and mean residual +1.208271 occupants**. Prediction took **1,579.29 seconds** on the reported run (versus approximately 0.17 seconds for Linear Regression and 0.02 seconds for the Decision Tree).

Decision Tree Regression performed best on internal G.25 validation, but Linear Regression achieved the lowest G.20 RMSE. All G.20 R² values were negative and all three models systematically underpredicted occupancy. Temporal features alone could not reliably capture high-occupancy events or transfer across rooms.

## Automated numerical tests

A `pytest` suite under [`tests/`](tests/) tests analytical and random-reference zero-order-hold averaging, duplicate and invalid timestamps, initial partial minutes, missing-minute handling, temporal feature invariants and recomputation of final G.20 model metrics.

The recorded **Windows / Python 3.13.14** run in [`docs/pytest_results.txt`](docs/pytest_results.txt) produced **19 passed, 0 failed, 0 skipped (14.98 seconds)**. The full methodology and limitations are documented in [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md) and [`TESTING.md`](TESTING.md).

Run or refresh the suite:

```powershell
python -m pip install -r requirements-test.txt
python -m pytest tests -v -rA
python -m pytest tests -v -rA *> docs/pytest_results.txt
```

The three exported-metric tests need the locally generated G.20 predictions and comparison CSVs; without them they **skip**, and that is not equivalent to a pass. The tests check numerical implementation and exported-metric consistency, not sensor truth, model-checkpoint size or independent forecast validity.

## HTML documentation

An offline, navigable, source-linked HTML reference for all **11 scripts** is available at [`docs/html/index.html`](docs/html/index.html). It includes script input/output information, source listings, function signatures, assumptions and limitations.

Generate the documentation from your currently checked-out source code (without importing scripts or running the costly models):

```powershell
python tools/build_html_docs.py
Start-Process .\docs\html\index.html
```

The AST-based generator is documented in [`docs/HTML_DOCUMENTATION.md`](docs/HTML_DOCUMENTATION.md).

## Limitations and reproducibility

- Zero-order hold is an assumption; measurements between events are unobserved and long gaps increase uncertainty.
- Only minutes with valid raw events are retained. The dataset is not a continuous, fully observed minute series.
- Random G.25 validation can include temporally correlated records in both subsets; it does not demonstrate genuine future-period forecasting. The validation subset also informed model selection.
- G.20 shows different occupancy characteristics, and room-specific schedules and characteristics are not provided to the models.
- The nonlinear methods did not consistently generalise better than Linear Regression, despite additional computational cost.
- Source datasets may have separate ownership or redistribution conditions; see [`DATA_LICENSE_NOTICE.md`](DATA_LICENSE_NOTICE.md). **The MIT [`LICENSE`](LICENSE) covers original project code, not third-party datasets or their contents.**
- Git LFS may be required to retrieve original large inputs. Processed CSVs, fitted models and many generated outputs must be recreated locally.

## Academic context

Developed for **MMA3001 — Numerical Methods and Machine Learning**, Monash University (2026). The accompanying report provides detailed methods, validation, optimisation, limitations, engineering implications and a critical reflection on AI use.
