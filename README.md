# Predicting Building Occupancy Using Machine Learning

### A Comparative Study of Regression Methods

**MMA3001 — Numerical Methods and Machine Learning | Monash University | 2026**

This project investigates the prediction of university room occupancy from calendar-derived features. Three regression methods—**Multiple Linear Regression**, **Decision Tree Regression**, and **Support Vector Regression (SVR)**—are trained on occupancy measurements from **Room G.25** and evaluated both on withheld G.25 observations and on an independent dataset from **Room G.20**.

The project explores numerical preprocessing of irregularly sampled sensor data, cyclical feature engineering, regression accuracy, hyperparameter selection, computational cost, and **cross-room generalisation**.

## Project objectives

- Convert irregularly timed sensor measurements into one-minute, time-weighted occupancy targets.
- Construct seven timestamp-derived features without using contemporaneous or historical headcounts as predictors.
- Compare linear and nonlinear regression methods against a training-mean baseline.
- Investigate decision-tree complexity and SVR parameter selection and computational cost.
- Assess how well models trained on G.25 transfer to the unseen room G.20.

## Data and numerical approach

The original occupancy dataset is stored under `Data/` and includes a large occupancy-sensor CSV tracked using **Git LFS**. Room-level records are extracted using their `floorspaceid` values.

The scripts use `collecteddate` for the timestamp and `headcount` for recorded occupancy. Invalid timestamps and nonnumeric or missing headcounts are excluded during preprocessing. For time-weighted averaging, readings are sorted chronologically, and when timestamps are duplicated the final recorded value is retained.

A **zero-order-hold** reconstruction assumes that occupancy stays constant between consecutive recorded measurements. A cumulative occupancy–time integral is evaluated at minute boundaries to calculate the time-weighted mean headcount. **Only minutes containing at least one valid original measurement are retained**; the preprocessing does not create a complete, gap-filled minute-by-minute time series. The first partial minute is normalised by its covered duration.

The seven model inputs, all derived from timestamps converted to **Australia/Melbourne** local time, are:

| Feature | Meaning |
| --- | --- |
| `TimeSin`, `TimeCos` | Cyclical time of day (24-hour period) |
| `DaySin`, `DayCos` | Cyclical day of week (7-day period) |
| `MonthSin`, `MonthCos` | Cyclical month of year (12-month period) |
| `IsWeekend` | Binary indicator for Saturday or Sunday |

The prediction target is the **one-minute time-weighted average headcount**, not a classification label or an instantaneous real-time sensor reading.

## Repository structure

```text
.
├── Data/                         # Original datasets and source metadata (Git LFS)
│   └── processed/                # Generated room-level and model-ready CSVs (ignored by Git)
├── src/                          # Extraction, preprocessing, modelling, and plotting scripts
├── Results/                      # Locally generated metrics, predictions, figures, and models
│   └── report_figures/           # Figures generated for the report
├── archive/                      # Earlier exploratory scripts
├── docs/                         # Reserved for project documentation
├── tests/                        # Reserved for automated tests
├── requirements.txt              # Listed Python dependencies
└── README.md
```

The `docs/` and `tests/` directories are currently placeholders; this repository does **not** yet provide a complete automated test suite. Generated `Data/processed/` datasets and `.joblib` model files are excluded from Git and must be recreated locally.

## Requirements and installation

- Python 3, with `pip` and a working scientific Python environment.
- Git and **Git LFS** to retrieve the original large CSV/XLSX files.
- Sufficient disk space and memory for the original dataset and model artefacts. Fitting the full RBF SVR can be computationally intensive.

Clone the repository:

```powershell
git lfs install
git clone https://github.com/jsau0014-gif/MMA3001-Building-occupancy-prediction-regression-methods.git
cd MMA3001-Building-occupancy-prediction-regression-methods
git lfs pull
```

Create and activate a virtual environment in **PowerShell**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install joblib
```

`joblib` is imported by the training and validation scripts but is not currently listed explicitly in `requirements.txt`, so it is installed separately above. An exact Python/package-version lockfile is not included; small numerical or runtime differences may occur between environments.

## Reproduce the workflow

Run commands **from the repository root**, in the order below. The scripts derive their file paths relative to their own location, so the local name of the outer project folder does not affect them.

### 1. Extract and inspect the two rooms

```powershell
python src/01_Extract_Occupancy_Data.py
python src/02_Inspect_Occupancy_Data.py
```

The extraction script reads `Data/5occupancySensor_MayToDec2024_9MRows.csv` in chunks and writes:

- `Data/processed/G25_Occupancy.csv`
- `Data/processed/G20_Occupancy.csv`

### 2. Produce one-minute, time-weighted datasets

```powershell
python src/03_preprocess_training.py
python src/03b_preprocess_G20.py
```

These create `G25_Minute_Training.csv` and `G20_Minute_validation.csv` in `Data/processed/`. The scripts print occupancy statistics and display visual comparisons of original readings against minute averages.

### 3. Engineer the seven temporal input features

```powershell
python src/04_feature_engineering.py
python src/04b_feature_engineering_G20.py
```

**Use the following minute-averaged model-ready files:**

- `Data/processed/G25_Model_Train.csv` — G.25 model development.
- `Data/processed/G20_Model_Validation_Minute.csv` — independent, like-for-like G.20 evaluation.

Note: `04_feature_engineering.py` also creates `G20_Model_Validation.csv` from the *raw* G.20 observations for an earlier workflow. That file is **not** the target for the final minute-averaged cross-room evaluation.

### 4. Train and internally validate the models

```powershell
python src/05_train_models.py
```

This script uses a **random 80% training / 20% validation split** of G.25, with `random_state=42`. It fits the linear regression, compares decision-tree minimum leaf sizes, fits an RBF SVR with scaled inputs, and reports **MAE, RMSE, and R²** relative to a training-mean baseline. Results and plots are saved under `Results/`.

The full G.25 training partition contains **298,608 observations**. The full-data RBF SVR can require several hours to fit, depending on the computer.

**Current model-filename discrepancy:** The training script saves the full SVR as `Results/05_svr_model_full.joblib`, whereas the G.20 validation script loads `Results/05_svr_model.joblib`. Before the next step, either update the SVR entry in `src/06b_validate_G20_minute.py` to reference `05_svr_model_full.joblib`, **or** make a copy of the newly trained full model using:

```powershell
Copy-Item .\Results\05_svr_model_full.joblib .\Results\05_svr_model.joblib
```

The copy command preserves the full model and ensures that the subsequent evaluation uses that same saved fit. An older file named `05_svr_model.joblib` may otherwise refer to an earlier, smaller training experiment.

### 5. Perform independent G.20 validation

```powershell
python src/06b_validate_G20_minute.py
```

The three saved G.25 models predict **G.20 without retraining**. The script records MAE, RMSE, R², mean residual (`actual − predicted`), prediction runtime, and plots. Its current `G20_SAMPLE_SIZE` setting exceeds the processed dataset size, so it evaluates **all 306,459 available G.20 minute observations**.

Outputs include:

- `Results/06b_G20model_comparison.csv`
- `Results/06b_G20validation_predictions.csv`
- Actual-versus-predicted, residual, and occupancy-over-time figures.

For this final comparison, do **not** substitute the older `src/06_validate_G20.py`, which evaluates the unaggregated G.20 target using a different workflow.

### 6. Generate figures for the report

```powershell
python src/07_generate_report_figures.py
```

Figures are generated in `Results/report_figures/` (with a fallback to `Results/` if a figure cannot be saved there). This stage uses saved outputs from previous stages; it does not refit the models. Some optional figures require historical comparison files from earlier experiments, such as `Results/05_model_comparison_100k.csv`, and are skipped when those files are absent.

## Validation and key findings

The final study used **373,260 processed G.25 observations** and **306,459 processed G.20 observations**. Internal evaluation used a shuffled split within G.25; independent evaluation used the separately preprocessed G.20 room.

- **Decision Tree Regression** achieved the best internal G.25 RMSE (**1.6633 occupants**; R² **0.3424**).
- G.20 cross-room validation was notably weaker: RMSE was **3.8183** for Linear Regression, **3.8390** for Decision Tree Regression, and **3.8788 occupants** for the reported SVR evaluation. All G.20 R² values were negative, and all models showed positive mean residuals (systematic underprediction).
- The results suggest that timestamp-only inputs describe some recurring utilisation patterns but do not reliably capture class-specific or irregular high-occupancy events.

**Comparability note:** Results depend on the particular saved SVR checkpoint. The G.20 figures quoted above are from the recorded full-room validation run; when reproducing the work, ensure the validation script points to the intended SVR fit and report the checkpoint used.

## Limitations and reproducibility notes

- The zero-order-hold assumption may be inaccurate across long intervals between sensor readings.
- Minutes without valid original measurements are excluded; the resulting dataset is **not** a continuous occupancy record.
- A random within-room split can place temporally related observations in both subsets. It is not a chronological forecast evaluation.
- Cross-room performance may differ because rooms have different utilisation patterns, timetables, and occupancy distributions; these variables are not model inputs.
- The repository does not include automated regression tests or pinned package versions. The preprocessing scripts print diagnostic statistics and show visual checks, but these are not a substitute for formal verification tests.
- Original data is tracked with Git LFS; generated datasets and trained `.joblib` models are not committed. Large exported per-observation prediction CSVs are also best regenerated locally rather than uploaded unnecessarily.

## Academic context

Developed for **MMA3001 — Numerical Methods and Machine Learning**, Monash University (2026). The accompanying report contains further details on the numerical method, hyperparameter experiments, model comparison, engineering implications, and critical reflection on AI-assisted development.
