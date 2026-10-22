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
Y = data["Headcount"]

print("\nTotal observations:", len(data))
print("Number of input features:", X.shape[1])

#splitting data chronoligically 80/20
data = data.sort_values("Timestamp").reset_index(drop=True)

# Recreate X and y in chronological order
X = data[features]
y = data["Headcount"]

# First 80% for training, final 20% for validation
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_validation = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_validation = y.iloc[split_index:]

print("\nTraining date range:")
print(data.loc[X_train.index, "Timestamp"].min())
print(data.loc[X_train.index, "Timestamp"].max())

print("\nValidation date range:")
print(data.loc[X_validation.index, "Timestamp"].min())
print(data.loc[X_validation.index, "Timestamp"].max())

print("\nTraining observations:", len(X_train))
print("Validation observations:", len(X_validation))

#Training Linear Regression Model

