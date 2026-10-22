"""
This script extracts the occupancy count at the end of each hour to 
standardise an hourly resolution accross both rooms which currenty are 
recorded at irregular inteverals.
This allows for unbiased modelling
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

project_Dir = Path(__file__).resolve().parent.parent

input_dir = project_Dir / "Data" / "processed"

rooms = ["G20","G25"]

for room in rooms:

    #extracting date and headcount data
    file = input_dir / f"{room}_Occupancy.csv"
    data = pd.read_csv(
        file,
        usecols=["collecteddate","headcount"]
    )

    #converting timestampls
    data["Timestamp"] = pd.to_datetime(
        data["collecteddate"],
        utc=True,
        errors="coerce"
    ).dt.tz_convert("Australia/Melbourne")

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

    #converting into one minute occupancy measurments (minute resolution)
    occupancy = data.set_index("Timestamp")["headcount"]

    #take the latest measurement within each minute
    minute_data = occupancy.resample("1min").last()

    minute_data = minute_data.ffill()

    hourly = minute_data.resample("1h").mean()

    #count actual measurements in each hour
    hourly_counts = occupancy.resample("1h").count()

    #count minutes with esitmate occupancy values
    coverage = minute_data.resample("1h").count()

    #combine into one dataframe
    hourly = pd.DataFrame({
        "Headcount": hourly,
        "MeasurementCount": hourly_counts,
        "CoverageMinutes": coverage
    })

    #keep hours with atleast one measurement
    hourly = hourly[
        hourly["MeasurementCount"] >0 
    ].copy()

    #Calculate approximate percentage of hour covvered using weighted averages
    hourly["CoveragePct"] = (
        hourly["CoverageMinutes"] / 60*100
    )

    #Restore timestamp as a column
    hourly = hourly.reset_index()

    #Extracting time-based features
    hourly["Hour"] = hourly["Timestamp"].dt.hour

    hourly["DayOfWeek"] = (
        hourly["Timestamp"].dt.dayofweek
    )

    hourly["Month"] = hourly["Timestamp"].dt.month

    hourly["IsWeekend"] = (
        hourly["DayOfWeek"] >=5
    ).astype(int)


    
    #saving output data
    output_file = input_dir / f"{room}_Hourly.csv"

    hourly.to_csv(output_file, index=False)

    print(f"\n========== {room} ==========")

    print("Raw measurements:", len(data))

    print("Total hours in date range:", len(hourly_counts))

    print("Hours containing measurements:", len(hourly))

    print("\nHourly headcount statistics:")
    print(hourly["Headcount"].describe())

    print("\nHourly measurement coverage:")
    print(hourly["CoveragePct"].describe())

    print("\nFirst 10 hourly measurements:")
    print(hourly.head(10).to_string(index=False))

    # --------------------------------------------------
    # Plot first week of occupancy
    # --------------------------------------------------

    start = hourly["Timestamp"].min()
    end = start + pd.Timedelta(days=7)

    sample = hourly[
        (hourly["Timestamp"] >= start) &
        (hourly["Timestamp"] < end)
    ]

    plt.figure(figsize=(12, 5))

    plt.plot(
        sample["Timestamp"],
        sample["Headcount"],
        marker=".",
        markersize=3,
        linewidth=1
    )

    plt.title(
        f"{room} - Hourly Average Headcount"
    )

    plt.xlabel("Time")
    plt.ylabel("Average Headcount")
    plt.grid(True)
    plt.tight_layout()

plt.show()

