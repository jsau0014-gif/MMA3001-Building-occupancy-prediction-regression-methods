"""
This script extracts the occupancy count at the end of each hour to 
standardise an hourly resolution accross both rooms which currenty are 
recorded at irregular inteverals.
This allows for unbiased modelling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

project_Dir = Path(__file__).resolve().parent.parent
data_dir = project_Dir / "Data" / "processed"


input_file = data_dir / "G25_Occupancy.csv"
output_file = data_dir / "G25_Minute_Training.csv"

#loading training data
data = pd.read_csv(
    input_file,
    usecols=["collecteddate","headcount"]

)

data["Timestamp"] = pd.to_datetime(
    data["collecteddate"],
    utc=True,
    errors="coerce"
)

# Ensure headcount is numeric
data["headcount"] = pd.to_numeric(
    data["headcount"],
    errors="coerce"
)
# Remove invalid measurements
data = data.dropna(
    subset=["Timestamp", "headcount"]
)
#sorting chronologically
data=data.sort_values("Timestamp",kind="stable")

print("\nFirst 15 original measurements:")
print(
    data[["Timestamp", "headcount"]]
    .head(15)
    .to_string(index=False)
)

print("\nOriginal headcount statistics:")
print(data["headcount"].describe())

minute_counts = data.groupby(
    data["Timestamp"].dt.floor("min")
).size()

#If multiple readings have identical timestamps,
#kep the last for the time weighted calc

events = data.drop_duplicates(
    subset="Timestamp",
    keep="last"
)

#calculating time weighted occupany per min

times=(
    events["Timestamp"]
    .astype("int64")
    .to_numpy() / 1e9
)

headcounts = events["headcount"].to_numpy()

#cumulative area under the headcount-time urve

durations = np.diff(times)

areas = np.concatenate((
    [0.0],
    np.cumsum(headcounts[:-1]*durations)
))

def cumulative_occupancy(boundaries):
    """Integrated headcount from first reading to each boundary."""

    indices = np.searchsorted(
        times, boundaries, side="right"
    ) - 1

    safe_indicies = np.maximum(indices,0)

    values = (
        areas[safe_indicies]
        + headcounts[safe_indicies]
        *(boundaries - times[safe_indicies])
    )

    return np.where(indices >=0,values, 0.0)

minute_start = minute_counts.index.asi8 / 1e9
minute_end = minute_start + 60

weighted_area = (
    cumulative_occupancy(minute_end)
    - cumulative_occupancy(minute_start)
)

covered_seconds = (
    minute_end - np.maximum(minute_start,times[0])
)

average_headcount = weighted_area/covered_seconds

#creating the training dataset

training = pd.DataFrame({
    "Timestamp": minute_counts.index.tz_convert(
        "Australia/Melbourne"
    ),
    "Headcount": average_headcount,
    "MeasurementCount": minute_counts.to_numpy()
})

#extract time-based features
training["Hour"] = training["Timestamp"].dt.hour
training["Minute"] = training["Timestamp"].dt.minute
training["DayOfWeek"] = training["Timestamp"].dt.dayofweek
training["Month"] = training["Timestamp"].dt.month

training["IsWeekend"] = (
    training["DayOfWeek"] >= 5
).astype(int)

#saving the processed data
training.to_csv(output_file, index=False)

#displaying results
print("\nG25 training preprocessing complete")

print("Original measurements:", len(data))
print("One-minute observations:", len(training))

print("\nTraining headcount statistics:")
print(training["Headcount"].describe())

print("\nFirst 10 rows:")
print(training.head(10).to_string(index=False))

print("\nSaved to:", output_file)
print("G20 validation data was not modified.")
