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
import time

from pathlib import Path

from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
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




#splitting data chronoligically 80/20
data = data.sort_values("Timestamp").reset_index(drop=True)

#placing inputs in X array and outputs in Y array
X = data[features]
y = data["Headcount"]

print("\nTotal observations:", len(data))
print("Number of input features:", X.shape[1])

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

#Setting up Decision Tree Regression
# ---------------------------------------------
# Decision Tree Regression - Leaf Size Tuning
# ---------------------------------------------

# Keep maximum depth constant
max_depth = 30

# Test different minimum leaf sizes
leaf_sizes = [10, 20, 50, 75, 100, 150, 200, 300, 500]

tree_models = {}
tree_results = []

for leaf_size in leaf_sizes:

    # Create decision tree
    tree_model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=leaf_size,
        random_state=42
    )

    # Train model
    tree_model.fit(X_train, y_train)

    # Generate predictions
    train_prediction = tree_model.predict(X_train)
    validation_prediction = tree_model.predict(X_validation)

    # Calculate training RMSE
    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_prediction)
    )

    # Calculate validation metrics
    val_metrics = regression_metrics(
        y_validation,
        validation_prediction
    )

    # Store fitted tree
    tree_models[leaf_size] = tree_model

    # Store performance and complexity
    tree_results.append({
        "LeafSize": leaf_size,
        "ActualDepth": tree_model.get_depth(),
        "Leaves": tree_model.get_n_leaves(),
        "Train_RMSE": train_rmse,
        "Validation_MAE": val_metrics["MAE"],
        "Validation_RMSE": val_metrics["RMSE"],
        "Validation_R2": val_metrics["R2"]
    })

    # Add to overall model comparison
    metrics.loc[
        f"Decision Tree (leaf={leaf_size})"
    ] = val_metrics


# ---------------------------------------------
# Compare leaf sizes
# ---------------------------------------------

leaf_comparison = pd.DataFrame(tree_results)

print("\nDecision Tree Leaf Size Comparison:")
print(leaf_comparison.to_string(index=False))

# Find leaf size with lowest validation RMSE
best_row = leaf_comparison.loc[
    leaf_comparison["Validation_RMSE"].idxmin()
]

best_leaf = int(best_row["LeafSize"])

# Retrieve best decision tree
best_tree = tree_models[best_leaf]

best_depth = best_tree.get_depth()

# Predict with best tree
tree_prediction = best_tree.predict(X_validation)

print("\nBest minimum leaf size:", best_leaf)
print("Actual tree depth:", best_depth)
print("Number of leaves:", best_tree.get_n_leaves())

# Save comparison
leaf_comparison.to_csv(
    results_dir / "05_leaf_size_comparison.csv",
    index=False
)


# ---------------------------------------------
# Plot leaf size vs training/validation error
# ---------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    leaf_comparison["LeafSize"],
    leaf_comparison["Train_RMSE"],
    marker="o",
    label="Training RMSE"
)

plt.plot(
    leaf_comparison["LeafSize"],
    leaf_comparison["Validation_RMSE"],
    marker="o",
    label="Validation RMSE"
)

# Reference: linear regression performance
plt.axhline(
    y=metrics.loc["Linear Regression", "RMSE"],
    linestyle="--",
    color="red",
    label="Linear regression RMSE"
)

plt.xlabel("Minimum Samples per Leaf")
plt.ylabel("RMSE (occupants)")
plt.title("Decision Tree: Leaf Size vs Prediction Error")

plt.xticks(leaf_sizes, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    results_dir / "05_leaf_size_comparison.png",
    dpi=200
)

#Setting Up SVR Regression 
#Number of observations used to train svr
svr_sample_size = 50000

#selecting random sample from g25 traiing data
X_svr_train = X_train.sample(
    n=svr_sample_size,
    random_state=42 #should pull the same 5000 data set each run
)

#select corresponding target values with mask
y_svr_train = y_train.loc[X_svr_train.index]

print("\nSupport Vector Regression")
print("SVR training observations:", len(X_svr_train))
print("SVR validation observations", len(X_validation))

#Create the SVR model with feature scaling
svr_model = make_pipeline(
    StandardScaler(),
    SVR(
        kernel = "rbf", #for non-linear solving,
        C=10,
        epsilon=0.5,
        gamma="scale"
    )
)

#Measure training time
start_time = time.perf_counter()

#Train SVR
svr_model.fit(X_svr_train, y_svr_train)
training_time = time.perf_counter() - start_time

print(f"SVR training time: {training_time:.2f} seconds")

#Generate predictions for all validation observations (not just the 5000)
prediction_start = time.perf_counter()

svr_prediction = svr_model.predict(X_validation)

prediction_time = time.perf_counter() - prediction_start

print(f"SVR prediction time {prediction_time:.2f} seconds")

#Calculate Validation Performance
svr_metrics = regression_metrics(
    y_validation,
    svr_prediction
)

#Additng to existing regression comparison dataframe
metrics.loc[f"SVR (RBF, n={svr_sample_size})"] = svr_metrics

#report number of support vectors
n_support_vectors = len(
    svr_model.named_steps["svr"].support_
)

print("Number of support vectors", n_support_vectors)
print("\nSVR Validation Performance:")
print(svr_metrics)

print("\nRegression performance:")
print(metrics.to_string())

print("\nMean training headcount:", round(training_mean,4))
print("\nTraining mean:", y_train.mean())
print("Validation mean:", y_validation.mean())

print("\nTraining standard deviation:", y_train.std())
print("Validation standard deviation:", y_validation.std())

metrics.to_csv(
    results_dir / "05_model_comparison_random.csv"
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
    "Predicted" : linear_prediction,
    "TreePredicted" : tree_prediction
})

validation_results["Residual"] = (
    validation_results["Actual"]
    - validation_results["Predicted"]
)
validation_results["TreeResidual"] = (
    validation_results["Actual"]
    - validation_results["TreePredicted"]
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
#decision tree residual plot
plt.figure(figsize=(9, 5))

plt.scatter(
    validation_results["TreePredicted"],
    validation_results["TreeResidual"],
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
plt.title(f"G25 Decision Tree Residuals (Depth {best_depth})")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    results_dir / "05_tree_residuals.png",
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

plt.scatter(
    sample["Timestamp"],
    sample["TreePredicted"],
    s=8,
    alpha=0.6,
    label=f"Decision tree (depth={best_depth})"
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