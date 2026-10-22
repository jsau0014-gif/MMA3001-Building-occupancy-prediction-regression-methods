"""
This script will train a a multiple linear regression model for the first
80% of the training data (from g25) leaving the final 20% for validation.
It uses LinearRegression functions from scikit

The script will produce a predicted vs acutal headcounts
A residual plot to identify systematic patterns

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

#setting up file locations

project_dir = Path(__file__).resolve().parent.parent

data_dir = project_dir / "Data" / "processed"
results_dir = project_dir / "Results"

results_dir.mkdir(parents=True, exist_ok=True)

inpute_file = data_dir / "G25_Model_Train.csv"

#pull the training dataset
data = pd.read_csv(inpute_file)

data["Timestamp"] = pd.to_datetime(
    data["Timestamp"],
    utc=True
).dt.tz_convert("Australia/Melbourne")

#Model Inputs from script 4
features = [
    "TimeSin",
    "TimeCos",
    "DaySin",
    "DayCos",
    "MonthSin",
    "MonthCos",
    "IsWeekend"
]

#placing inputs in X array and outputs in Y array
X = data[features]
y = data["Headcount"]

print("\nTotal observations:", len(data))
print("Number of input features:", X.shape[1])

#splitting data chronoligically 80/20
data = data.sort_values("Timestamp").reset_index(drop=True)

X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

print("\nTraining observations:", len(X_train))
print("Validation observations:", len(X_validation))

print("\nTraining mean:", y_train.mean())
print("Validation mean:", y_validation.mean())

print("\nTraining standard deviation:", y_train.std())
print("Validation standard deviation:", y_validation.std())

print("\nTraining date range:")
print(data.loc[X_train.index, "Timestamp"].min())
print(data.loc[X_train.index, "Timestamp"].max())

print("\nValidation date range:")
print(data.loc[X_validation.index, "Timestamp"].min())
print(data.loc[X_validation.index, "Timestamp"].max())

print("\nTraining observations:", len(X_train))
print("Validation observations:", len(X_validation))

#Training Linear Regression Model
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

#predict witheld observations
linear_prediction = linear_model.predict(
    X_validation
)

#calculating baseline predictions
#goal is to predict the mean training headcount

training_mean = y_train.mean()

baseline_prediction = np.full(
    shape=len(y_validation),
    fill_value=training_mean
)

#creating a function to calculate regression metrics
def regression_metrics(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)

    return{
        "MAE":mean_absolute_error(y_true, y_pred),
        "RMSE" : np.sqrt(mse),
        "R2" : r2_score(y_true, y_pred)

    }
#compare model against baseline
metrics = pd.DataFrame({
    "Linear Regression": regression_metrics(
        y_validation,
        linear_prediction
    ),
    "Mean Baseline" : regression_metrics(
        y_validation,
        baseline_prediction
    )
}).T

print("\nRegression performance:")
print(metrics.to_string())

print("\nMean training headcount:", round(training_mean,4))
print("\nTraining mean:", y_train.mean())
print("Validation mean:", y_validation.mean())

print("\nTraining standard deviation:", y_train.std())
print("Validation standard deviation:", y_validation.std())

metrics.to_csv(
    results_dir / "05_linear_regression_metrics.csv"
)

#checking learned coefficinets
coefficients = pd.DataFrame({
    "Feature" : features,
    "Coefficient" : linear_model.coef_
})

print("\nLinear regression intercept:")
print(linear_model.intercept_)

print("\nFeature coefficients:")
print(coefficients.to_string(index=False))

coefficients.to_csv(
    results_dir / "05_linear_coefficients.csv",
    index=False
)

#storing validation results
validation_results = pd.DataFrame({
    "Timestamp" : data.loc[
        X_validation.index, "Timestamp"
    ],
    "Actual" : y_validation,
    "Predicted" : linear_prediction
})

validation_results["Residual"] = (
    validation_results["Actual"]
    - validation_results["Predicted"]
)

validation_results.to_csv(
    results_dir / "05_linear_validation_results.csv",
    index=False
)

#Plotting predicted vs actual headcount
plt.figure(figsize=(8, 6))

plt.hexbin(
    validation_results["Actual"],
    validation_results["Predicted"],
    gridsize=55,
    bins="log",
    mincnt=1,
    cmap="viridis"
)

plt.colorbar(label="Log observation count")

# Ideal prediction line
upper = max(
    validation_results["Actual"].max(),
    validation_results["Predicted"].max()
)

plt.plot(
    [0, upper],
    [0, upper],
    "r--",
    label="Perfect prediction"
)

plt.xlabel("Actual headcount")
plt.ylabel("Predicted headcount")
plt.title("G25 Linear Regression: Predicted vs Actual")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    results_dir / "05_predicted_vs_actual.png",
    dpi=200
)

#plotting residuals
plt.figure(figsize=(9, 5))

plt.scatter(
    validation_results["Predicted"],
    validation_results["Residual"],
    s=3,
    alpha=0.15
)

plt.axhline(
    0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted headcount")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("G25 Linear Regression Residuals")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    results_dir / "05_residuals.png",
    dpi=200
)
#comparing predictions over time
start = validation_results["Timestamp"].min()
end = start + pd.Timedelta(days=2)

sample = validation_results[
    (validation_results["Timestamp"] >= start) &
    (validation_results["Timestamp"] < end)
]

plt.figure(figsize=(13, 5))

plt.scatter(
    sample["Timestamp"],
    sample["Actual"],
    s=8,
    alpha=0.6,
    label="Actual"
)

plt.scatter(
    sample["Timestamp"],
    sample["Predicted"],
    s=8,
    alpha=0.6,
    label="Linear regression prediction"
)

plt.xlabel("Time")
plt.ylabel("Headcount")
plt.title("G25 Validation: Actual vs Predicted Occupancy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    results_dir / "05_occupancy_prediction.png",
    dpi=200
)

print("\nResults saved to:", results_dir)

plt.show()