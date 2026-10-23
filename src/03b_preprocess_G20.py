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


input_file = data_dir / "G20_Occupancy.csv"
output_file = data_dir / "G20_Minute_validation.csv"

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

# Timestamp representation in integer nanoseconds
time_ns = (
    events["Timestamp"]
    .dt.as_unit("ns")
    .astype("int64")
    .to_numpy()
)

headcounts = events["headcount"].to_numpy(dtype=np.int64)

# Duration between consecutive readings
duration_ns = np.diff(time_ns)

# Cumulative area under the occupancy-time curve
cumulative_area = np.concatenate((
    np.array([0], dtype=np.int64),
    np.cumsum(
        headcounts[:-1] * duration_ns,
        dtype=np.int64
    )
))

def integral_at(boundaries_ns):
    """Cumulative headcount × time at each boundary."""

    indices = np.searchsorted(
        time_ns,
        boundaries_ns,
        side="right"
    ) - 1

    results = np.zeros(
        len(boundaries_ns),
        dtype=np.int64
    )

    valid = indices >= 0

    results[valid] = (
        cumulative_area[indices[valid]]
        + headcounts[indices[valid]]
        * (
            boundaries_ns[valid]
            - time_ns[indices[valid]]
        )
    )

    return results


# Start and end of every retained minute
minute_start_ns = minute_counts.index.as_unit("ns").asi8
minute_end_ns = minute_start_ns + 60_000_000_000

# Time-weighted area during each minute
minute_area = (
    integral_at(minute_end_ns)
    - integral_at(minute_start_ns)
)

# Account for the first partial minute
covered_ns = (
    minute_end_ns
    - np.maximum(minute_start_ns, time_ns[0])
)

# Calculate minute-average occupancy
average_headcount = minute_area / covered_ns

print("\nFirst 10 calculated minute averages:")

for timestamp, avg in zip(
    minute_counts.index[:10],
    average_headcount[:10]
):
    print(timestamp, round(avg, 4))

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
print("\nG20 training preprocessing complete")

print("Original measurements:", len(data))
print("One-minute observations:", len(training))

print("\nTraining headcount statistics:")
print(training["Headcount"].describe())

print("\nFirst 10 rows:")
print(training.head(10).to_string(index=False))

print("\nSaved to:", output_file)
print("G20 validation data was not modified.")



# ---------------------------------------------
# Visual verification of preprocessing
# ---------------------------------------------

# Select the first 30mins
start = training["Timestamp"].iloc[0]
end = start + pd.Timedelta(minutes=30)

# Original sensor readings
raw_sample = data[
    (data["Timestamp"] >= start) &
    (data["Timestamp"] < end)
].copy()

# Processed minute averages
minute_sample = training[
    (training["Timestamp"] >= start) &
    (training["Timestamp"] < end)
].copy()

# Convert raw timestamps to Melbourne time
raw_sample["Timestamp"] = (
    raw_sample["Timestamp"]
    .dt.tz_convert("Australia/Melbourne")
)

plt.figure(figsize=(14, 6))

# Raw occupancy (assumed constant between readings)
plt.step(
    raw_sample["Timestamp"],
    raw_sample["headcount"],
    where="post",
    label="Original headcount",
    color="steelblue",
    linewidth=1.2,
    alpha=0.7
)

# One-minute time-weighted averages
plt.scatter(
    minute_sample["Timestamp"] + pd.Timedelta(seconds=30),
    minute_sample["Headcount"],
    label="One-minute weighted average",
    color="darkorange",
    s=18,
    zorder=3
)

plt.title("G.20: Original vs One-Minute Occupancy")
plt.xlabel("Time")
plt.ylabel("Headcount")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()

