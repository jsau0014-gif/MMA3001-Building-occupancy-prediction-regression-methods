import pandas as pd
import json
import re


# --------------------------------------------------
# Files
# --------------------------------------------------

environment_file = "Data/5EnvSensor_MayToDec2024_180kRows.csv"
mapping_file = "Room_Sensor_Mapping.csv"


# --------------------------------------------------
# Load and clean mapping file
# --------------------------------------------------

mapping = pd.read_csv(mapping_file)


def extract_sensor_ids(value):
    """
    Extract environmental sensor IDs from strings such as:
    [np.int64(6012003000486)]
    """

    if pd.isna(value):
        return []

    numbers = re.findall(r"\d{10,}", str(value))

    return [int(number) for number in numbers]


mapping["Environment Sensor ID"] = (
    mapping["Environment Sensor ID"]
    .apply(extract_sensor_ids)
)


# Create one row per sensor
mapping = mapping.explode(
    "Environment Sensor ID"
)


mapping = mapping.dropna(
    subset=["Environment Sensor ID"]
)


mapping["Environment Sensor ID"] = (
    mapping["Environment Sensor ID"]
    .astype("int64")
)


sensor_rooms = mapping[
    [
        "Building",
        "Room",
        "Environment Sensor ID"
    ]
].copy()


# Create a set for quick sensor lookup
mapped_ids = set(
    sensor_rooms[
        "Environment Sensor ID"
    ]
)


# --------------------------------------------------
# Function to extract CO2 measurement timestamp
# --------------------------------------------------

def extract_co2_timestamp(json_string):
    """
    Find the timestamp associated with the CO2
    measurement inside the environmental JSON.
    """

    try:

        data = json.loads(json_string)

        for measurement in data:

            variable = measurement.get(
                "variable",
                {}
            )

            if variable.get("name") == "Carbon dioxide":

                values = measurement.get(
                    "values",
                    []
                )

                if len(values) > 0:

                    return values[0].get("time")

        return None

    except (json.JSONDecodeError, TypeError):

        return None


# --------------------------------------------------
# Read environmental file
# --------------------------------------------------

results = []


for chunk in pd.read_csv(
    environment_file,
    chunksize=50000,
    usecols=[
        "sensorid",
        "jsondata"
    ]
):

    chunk["CO2_Timestamp"] = (
        chunk["jsondata"]
        .apply(extract_co2_timestamp)
    )

    results.append(
        chunk[
            [
                "sensorid",
                "CO2_Timestamp"
            ]
        ]
    )


environment = pd.concat(
    results,
    ignore_index=True
)


environment["sensorid"] = (
    environment["sensorid"]
    .astype("int64")
)


# --------------------------------------------------
# Count raw rows and unique measurements
# --------------------------------------------------

summary = (
    environment
    .groupby("sensorid")
    .agg(
        RawRows=(
            "CO2_Timestamp",
            "size"
        ),

        UniqueDataPoints=(
            "CO2_Timestamp",
            "nunique"
        )
    )
    .reset_index()
)


summary = summary.rename(
    columns={
        "sensorid": "OriginalSensorID"
    }
)


# --------------------------------------------------
# Check mapping WITHOUT correction
# --------------------------------------------------

summary["Mapped_Original"] = (
    summary["OriginalSensorID"]
    .isin(mapped_ids)
)


# --------------------------------------------------
# Try +999,999 correction
# --------------------------------------------------

summary["CorrectionCandidateID"] = (
    summary["OriginalSensorID"]
    + 999999
)


summary["CandidateMapped"] = (
    summary["CorrectionCandidateID"]
    .isin(mapped_ids)
)


# If already mapped directly, don't change the ID.
# Otherwise use +999,999 only if that candidate exists.
def choose_corrected_id(row):

    original = row["OriginalSensorID"]
    candidate = row["CorrectionCandidateID"]

    if original in mapped_ids:

        return original

    elif candidate in mapped_ids:

        return candidate

    else:

        return original


summary["ID_After_Correction"] = (
    summary.apply(
        choose_corrected_id,
        axis=1
    )
)


summary["Mapped_After_Correction"] = (
    summary["ID_After_Correction"]
    .isin(mapped_ids)
)


# --------------------------------------------------
# Find building and room
# --------------------------------------------------

def find_location(sensor_id):

    matches = sensor_rooms[
        sensor_rooms[
            "Environment Sensor ID"
        ] == sensor_id
    ]

    if matches.empty:

        return "NOT MAPPED"

    locations = []

    for _, row in matches.iterrows():

        locations.append(
            f"Building {row['Building']}, Room {row['Room']}"
        )

    return "; ".join(locations)


summary["OriginalLocation"] = (
    summary["OriginalSensorID"]
    .apply(find_location)
)


summary["CorrectedLocation"] = (
    summary["ID_After_Correction"]
    .apply(find_location)
)


# --------------------------------------------------
# Rank sensors by amount of useful data
# --------------------------------------------------

summary = summary.sort_values(
    "UniqueDataPoints",
    ascending=False
)


summary["Rank"] = range(
    1,
    len(summary) + 1
)


# Put rank first
summary = summary[
    [
        "Rank",
        "OriginalSensorID",
        "RawRows",
        "UniqueDataPoints",
        "Mapped_Original",
        "OriginalLocation",
        "CorrectionCandidateID",
        "CandidateMapped",
        "ID_After_Correction",
        "Mapped_After_Correction",
        "CorrectedLocation"
    ]
]


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n==============================================")
print("ENVIRONMENTAL SENSOR DATA SUMMARY")
print("==============================================\n")


for _, row in summary.iterrows():

    print(
        f"RANK {row['Rank']} "
        f"- SENSOR {row['OriginalSensorID']}"
    )

    print(
        f"  Raw rows:           "
        f"{row['RawRows']}"
    )

    print(
        f"  Unique data points: "
        f"{row['UniqueDataPoints']}"
    )


    # Original mapping
    if row["Mapped_Original"]:

        print(
            "  Original mapping:   "
            "YES ✓"
        )

        print(
            f"  Location:           "
            f"{row['OriginalLocation']}"
        )

    else:

        print(
            "  Original mapping:   "
            "NO ✗"
        )


    # Corrected mapping
    print(
        f"  +999999 candidate:  "
        f"{row['CorrectionCandidateID']}"
    )


    if row["Mapped_After_Correction"]:

        if (
            row["ID_After_Correction"]
            != row["OriginalSensorID"]
        ):

            print(
                "  Mapping after correction: "
                "YES ✓"
            )

            print(
                f"  Corrected ID:       "
                f"{row['ID_After_Correction']}"
            )

        else:

            print(
                "  Mapping after correction: "
                "YES ✓ (original already matched)"
            )

        print(
            f"  Location:           "
            f"{row['CorrectedLocation']}"
        )

    else:

        print(
            "  Mapping after correction: "
            "NO ✗"
        )

    print()


# --------------------------------------------------
# Save summary
# --------------------------------------------------

summary.to_csv(
    "Sensor_Data_Summary.csv",
    index=False
)


print(
    "Summary saved as "
    "'Sensor_Data_Summary.csv'"
)