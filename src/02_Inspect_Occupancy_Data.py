"""
    Inspecting the extracted data to guage what time domain each room has, 
    the spread of occupancy and status data across the domain
"""

import pandas as pd
from pathlib import Path


#locating project directory

project_dir = Path(__file__).resolve().parent.parent

#processing both room g20 and g25
rooms = ["G20","G25"]

for room in rooms:

    file = project_dir / "Data" / "processed" / f"{room}_Occupancy.csv"

    #load the extracted room occupancy data
    data = pd.read_csv(file)

    #convert time stamps into normalised melbourne time
    data["Timestamp"] = pd.to_datetime(
        data["collecteddate"],
        utc=True,
        errors="coerce"
    ).dt.tz_convert("Australia/Melbourne")

    print(f"\n========== {room} ==========")

    #Total Measurments
    print("\nFirst Measurement:")
    print(data["Timestamp"].min())

    print("\nLast measurement:")
    print(data["Timestamp"].max())

    #Headcount distribution
    print("\nHeadcount distribution:")
    print(data["headcount"].value_counts().sort_index())

    #occupancy status distribution
    print("\nOccupancy status distribution:")
    print(data["occupancystatus"].value_counts())

    # Missing values
    print("\nMissing values:")
    print(
        data[["Timestamp", "headcount", "occupancystatus"]]
        .isna()
        .sum()
    )