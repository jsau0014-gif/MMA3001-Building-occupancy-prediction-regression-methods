"""This script cross validates the models trained on the g25 data
on the g20 data and records residuals
"""

from pathlib import Path
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

#setting up file paths

project_dir = Path(__file__).resolve().parent.parent

data_dir = project_dir / "Data" / "processed"
results_dir = project_dir / "Results"

results_dir.mkdir(parents=True,exist_ok=True)

features = [
    "TimeSin",
    "TimeCos",
    "DaySin",
    "DayCos",
    "MonthSin",
    "MonthCos",
    "IsWeekend"
]

#G20 observations for faster SVR prediction
G20_SAMPLE_SIZE = 1000000

RANDOM_STATE = 42
PLOT_DAYS = 7

#loading fitted g25 models

model_paths = {
    "Linear Regression": results_dir / "05_linear_model.joblib",
    "Decision Tree": results_dir / "05_decision_tree.joblib",
    "SVR": results_dir / "05_svr_model.joblib"
}

# Check that model files exist
for model_name, path in model_paths.items():

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Save the fitted "
            f"{model_name} model from script 05 first."
        )
#load g20 val data
models = {
    name: joblib.load(path)
    for name, path in model_paths.items()
}

g20 = pd.read_csv(
    data_dir / "G20_Model_Validation_Minute.csv"
)

required_columns = [
    "Timestamp",
    "Headcount",
    *features
]

missing_columns = [
    col for col in required_columns
    if col not in g20.columns
]

if missing_columns:
    raise ValueError(
        f"Missing G20 columns: {missing_columns}"
    )

#removing incomplete observations
g20 = g20.dropna(
    subset=required_columns
).copy()

#convert timestamps to melb local time
g20["Timestamp"] = pd.to_datetime(
    g20["Timestamp"],
    utc=True
).dt.tz_convert("Australia/Melbourne")

#selecting the validation sample

sample_size = min(
    G20_SAMPLE_SIZE,
    len(g20)
)

g20_sample = (
    g20.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    )
    .sort_values("Timestamp")
    .reset_index(drop=True)
)

#inputs and actual heacounts
X_g20 = g20_sample[features]

y_g20 = g20_sample["Headcount"].to_numpy()

print("\nMinute-weighted G20 observations:", len(g20))
print("G20 evaluation observations:", sample_size)

print("\nG20 date range:")
print(g20_sample["Timestamp"].min())
print(g20_sample["Timestamp"].max())

print("\nG20 mean headcount:", y_g20.mean())
print("G20 standard deviation:", y_g20.std())
print("G20 maximum headcount:", y_g20.max())

#precit g20 using g25 trained models
comparison_rows = []

validation_results = g20_sample[
    ["Timestamp", "Headcount"]
].rename(
    columns={"Headcount": "Actual"}
).copy()

prediction_columns = {
    "Linear Regression": "LinearPredicted",
    "Decision Tree": "TreePredicted",
    "SVR": "SVRPredicted"
}

for model_name, model in models.items():
    print(
        f"\nPredicting G20 using {model_name}..."
    )

    #Check feature compatibility
    trained_features = getattr(
        model,
        "feature_names_in_",
        None
    )

    if trained_features is not None:
        if list(trained_features) != features:
            raise ValueError(
                f"Feature mismatch for {model_name}"
            )

    #General Predictions
    start_time = time.perf_counter()

    y_prediction = model.predict(X_g20)

    prediction_time = (
        time.perf_counter() - start_time
    )

    #store predictions and residuals
    column = prediction_columns[model_name]
    validation_results[column] = y_prediction
    residual_column = column.replace(
        "Predicted",
        "Residual"
    )

    validation_results[residual_column] = (
        y_g20 - y_prediction
    )

    #Calculate performance
    mae = mean_absolute_error(
        y_g20,
        y_prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_g20,
            y_prediction
        )
    )

    r2 = r2_score(
        y_g20,
        y_prediction
    )

    mean_residual = np.mean(
        y_g20 - y_prediction
    )

    comparison_rows.append({
        "Model": model_name,
        "Observations": len(y_g20),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MeanResidual": mean_residual,
        "PredictionSeconds": prediction_time
    })

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    print(
        f"Mean residual: {mean_residual:.4f}"
    )

    print(
        f"Prediction time: {prediction_time:.2f}s"
    )

#Compare G20 model performance

comparison = pd.DataFrame(
    comparison_rows
)

print("\nG20 CROSS-ROOM VALIDATION")
print(comparison.to_string(index=False))

comparison.to_csv(
    results_dir / "06b_G20model_comparison.csv",
    index=False
)


#saving G20 predictions and residuals

validation_results.to_csv(
    results_dir / "06b_G20validation_predictions.csv",
    index=False
)

#plot actual vs predicted and residuals
for model_name, column in prediction_columns.items():

    safe_name = model_name.lower().replace(
        " ",
        "_"
    )

    actual = validation_results["Actual"]
    predicted = validation_results[column]

    residual = actual - predicted

    #Actual vs predicted

    fig, ax = plt.subplots(figsize=(8,6))

    plot = ax.hexbin(
        actual,
        predicted,
        gridsize=55,
        bins="log",
        mincnt=1,
        cmap="viridis"
    )

    fig.colorbar(
        plot,
        ax=ax,
        label="Log observation count"
    )

    # Perfect prediction line
    lower = min(
        float(actual.min()),
        float(predicted.min()),
        0
    )

    upper = max(
        float(actual.max()),
        float(predicted.max())
    )

    ax.plot(
        [lower, upper],
        [lower, upper],
        "r--",
        label="Perfect prediction"
    )

    ax.set_xlabel("Actual G20 headcount")
    ax.set_ylabel("Predicted G20 headcount")

    ax.set_title(
        f"G20: {model_name} - Actual vs Predicted"
    )

    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        results_dir /
        f"06b_G20{safe_name}_actual_vs_predicted.png",
        dpi=200
    )

    plt.close(fig)

    # ---------------------------------------------
    # Residual plot
    # ---------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        predicted,
        residual,
        s=3,
        alpha=0.15
    )

    ax.axhline(
        0,
        color="red",
        linestyle="--"
    )

    ax.set_xlabel("Predicted G20 headcount")

    ax.set_ylabel(
        "Residual (Actual - Predicted)"
    )

    ax.set_title(
        f"G20: {model_name} Residuals"
    )

    ax.grid(True, alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        results_dir /
        f"06b_G20{safe_name}_residuals.png",
        dpi=200
    )

    plt.close(fig)

#G2O occupancy predictions over time

# Select first seven days of sampled observations
start = validation_results["Timestamp"].min()

end = start + pd.Timedelta(
    days=PLOT_DAYS
)

sample = validation_results[
    (validation_results["Timestamp"] >= start)
    & (validation_results["Timestamp"] < end)
].sort_values("Timestamp")

fig, ax = plt.subplots(figsize=(13, 5))

# Actual occupancy
ax.scatter(
    sample["Timestamp"],
    sample["Actual"],
    s=12,
    alpha=0.65,
    label="Actual G20"
)

# Model predictions
for model_name, column in prediction_columns.items():

    ax.scatter(
        sample["Timestamp"],
        sample[column],
        s=8,
        alpha=0.6,
        label=model_name
    )

ax.set_xlabel("Time (Melbourne)")
ax.set_ylabel("Headcount")

ax.set_title(
    "G20: Actual vs Predicted Occupancy - 7 Days"
)

ax.legend()
ax.grid(True, alpha=0.2)

fig.autofmt_xdate()
fig.tight_layout()

fig.savefig(
    results_dir / "06b_G20occupancy_over_time.png",
    dpi=200
)

plt.close(fig)


# --------------------------------------------------
# Complete
# --------------------------------------------------

print(
    "\nG20 results saved to:",
    results_dir
)

