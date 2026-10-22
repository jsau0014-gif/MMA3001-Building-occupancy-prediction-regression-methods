"""
Creating a time-based feature/variable to enable occupancy regression.

Using the one-minute weighted averages from g25 for training.
using untouched g20 original headcount measurements for evaluation

No resampling or occupancy modification is performed.
"""

import pandas as pd
import numpy as np
from pathlib import Path

#Setting up file locations

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "Data" / "processed"

train_file = data_dir / "G25_Minute_Training.csv"
validation_file = data_dir / "G20_Occupancy.csv"

train_output = data_dir / "G25_Model_Train.csv"
validation_output = data_dir / "G20_Model_Validation"

#pulling g25 training data from csv files
train = pd.read_csv(train_file)

train["Timestamp"] = pd.to_datetime(
    train["Timestamp"],
    utc=True,
    errors = "coerce"
).dt.tz_convert("Australia/Melbourne")

#pulling g20 validation data from csv files
validation = pd.read_csv(
    validation_file,
    usecols = ["collecteddate","headcount"]
)

validation["Timestamp"] = pd.to_datetime(
    validation["collecteddate"],
    utc=True,
    errors = "coerce"
).dt.tz_convert("Australia/Melbourne")

validation = validation.rename(
    columns={"headcount": "Headcount"}
)

#constructing feature engineering function
def create_features(data):

    data= data.copy()

    #converting to a fractional hour, including mins and sec
    time_of_day = (
        data["Timestamp"].dt.hour
        + data["Timestamp"].dt.minute / 60
        + data["Timestamp"].dt.second / 3600
    )

    # Daily cyclical features
    data["Timesin"] = np.sin(
        2 * np.pi * time_of_day / 24
    )

    # Day of week: Monday = 0 Sunday = 6
    day = data["Timestamp"].dt.dayofweek

    #weekly cyclical features
    data["DaySin"] = np.sin(
        2 * np.pi * day / 7
    )

    data["DayCos"] = np.cos(
        2 * np.pi * day / 7
    )

    #Month: January = 1 December = 12
    month = data["Timestamp"].dt.month

    data["MonthSin"] = np.sin(
        2 * np.pi * (month-1) / 12
    )

    data["MonthCos"] = np.cos(
        2 * np.pi * (month-1) / 12
    )

    #Weekend indicator
    data["IsWeekend"] = (day >= 5).astype(int)

    return data
#applying transformation to validation and training data
train = create_features(train)
validation = create_features(validation)

#selecting identical model input columns
features = [
    "TimeSin",
    "TimeCos",
    "DaySin",
    "DayCos",
    "MonthSin",
    "MonthCos",
    "IsWeekend"
]

#retaining timestampt and target for comparison and debugging
output_columns = [
    "Timestamp",
    "Headcount"
] + features

train = train[output_columns].copy()
validation = train[output_columns].copy()

#removing rows with invalid timestamps or targets
train = train.dropna(subset=output_columns)
validation = train.dropna(subset=output_columns)

#saving data files
train.to_csv(train_output, index=False)
validation.to_csv(validation_output, index=False)

print("\nFeature engineering complete!")

print("\nG25 training observations:", len(train))
print("G20 validation observations:", len(validation))

print("\nFeature columns:")
print(features)

print("\nG25 sample:")
print(train.head(5).to_string(index=False))

print("\nG20 sample:")
print(validation.head(5).to_string(index=False))

print("\nSaved training file:", train_output)
print("Saved validation file:", validation_output)
print("\nOriginal occupancy files unchanged.")