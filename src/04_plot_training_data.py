import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# File locations
# --------------------------------------------------

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "Data" / "processed"

raw_file = data_dir / "G25_Occupancy.csv"
processed_file = data_dir / "G25_Minute_Training.csv"

# --------------------------------------------------
# Load data
# --------------------------------------------------

raw = pd.read_csv(
    raw_file,
    usecols=["collecteddate", "headcount"]
)

processed = pd.read_csv(processed_file)

# Convert timestamps
raw["Timestamp"] = pd.to_datetime(
    raw["collecteddate"],
    utc=True,
    errors="coerce"
).dt.tz_convert("Australia/Melbourne")

processed["Timestamp"] = pd.to_datetime(
    processed["Timestamp"],
    utc=True,
    errors="coerce"
)

# Numeric headcount
raw["headcount"] = pd.to_numeric(raw["headcount"], errors="coerce")
processed["Headcount"] = pd.to_numeric(processed["Headcount"], errors="coerce")

# Drop invalid rows
raw = raw.dropna(subset=["Timestamp", "headcount"])
processed = processed.dropna(subset=["Timestamp", "Headcount"])

# Sort
raw = raw.sort_values("Timestamp")
processed = processed.sort_values("Timestamp")

# --------------------------------------------------
# Choose time window to inspect
# --------------------------------------------------

start = raw["Timestamp"].min()
end = start + pd.Timedelta(hours=6)

raw_sample = raw[
    (raw["Timestamp"] >= start) &
    (raw["Timestamp"] < end)
]

processed_sample = processed[
    (processed["Timestamp"] >= start) &
    (processed["Timestamp"] < end)
]

# --------------------------------------------------
# Plot 1: raw data
# --------------------------------------------------

plt.figure(figsize=(12, 5))
plt.scatter(
    raw_sample["Timestamp"],
    raw_sample["headcount"],
    s=8,
    alpha=0.6
)
plt.title("G25 Raw Occupancy Readings (First 6 Hours)")
plt.xlabel("Time")
plt.ylabel("Headcount")
plt.grid(True)
plt.tight_layout()

# --------------------------------------------------
# Plot 2: processed one-minute data
# --------------------------------------------------

plt.figure(figsize=(12, 5))
plt.plot(
    processed_sample["Timestamp"],
    processed_sample["Headcount"],
    marker=".",
    markersize=4,
    linewidth=1
)
plt.title("G25 One-Minute Training Data (First 6 Hours)")
plt.xlabel("Time")
plt.ylabel("Average Headcount")
plt.grid(True)
plt.tight_layout()

# --------------------------------------------------
# Plot 3: histogram of processed minute headcount
# --------------------------------------------------

plt.figure(figsize=(12, 5))
plt.hist(processed["Headcount"], bins=50)
plt.title("Distribution of G25 One-Minute Average Headcount")
plt.xlabel("Average Headcount")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()

plt.show()