import pandas as pd
import json
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

#calling files
environment_file = "Data/5EnvSensor_MayToDec2024_180kRows.csv"
mapping_file = "Room_Sensor_Mapping.csv"

mapping = pd.read_csv(mapping_file)

def extract_sensor_ids(value):
    """
    Extracting the sensor ID from the mapping files
    handles int64 values from csv file and creates a list of integer
    sesor IDs
    """
    if pd.isna(value):
        return[]#for any empty values return nothing
    numbers = re.findall(r"\d{10,}",str(value))
    return [int(number) for number in numbers]

#pulling sensor ID values from data 
mapping["Environment Sensor ID"] = mapping["Environment Sensor ID"].apply(extract_sensor_ids)

#given the same room has multiple sensors
#explode will create a individual row for each sensor
mapping = mapping.explode("Environment Sensor ID")

mapping = mapping.dropna(subset=["Environment Sensor ID"])

#convert to integers
mapping["Environment Sensor ID"] = mapping["Environment Sensor ID"].astype("int64")

#mask for columns that contain building room and sensor IDs

sensor_rooms = mapping[
    ["Building","Room","Environment Sensor ID"]
]

print("\nSensor to room lookup:")
print(sensor_rooms.head())

def extract_environment(json_string):
    """
    Extracting C02, temperature, rel humidity Data and timestamp of the measurement
    from the environment sensor

    
    """
    try:
        data = json.loads(json_string)

        co2 = None
        temperature = None
        humidity = None
        unix_time = None

        for measurement in data:
            variable = measurement.get("variable",{}) #pulling json dictionary using get is safer as it returns {} rather than a keyerror
            variable_name = variable.get("name")
            values = measurement.get("values",[]) #applying mask
            if len(values) == 0:
                continue
            value = values[0].get("value")
            measurement_time = values[0].get("time")
            if unix_time is None:
                unix_time = measurement_time

            if variable_name == "Carbon dioxide":
                co2 = value
                

            elif variable_name == "Temperature":
                temperature = value

            elif variable_name =="Humidity":
                humidity = value

        return co2, temperature, humidity, unix_time

    except (json.JSONDecodeError,TypeError):
        return None, None, None, None

def calc_abs_humidity(temp_c, relative_humidity):
    """
    calculate the absolute humidity in g/m^3
    from temperatuire in deg C and RH in %
    """

    saturation_pressure = (
        6.112* np.exp((17.67*temp_c)/(temp_c+243.5))
    )

    vapour_pressure = (
        relative_humidity/100.0
    )*saturation_pressure

    absolute_humidity = (
        216.7*vapour_pressure/(temp_c+273.15)
    )
    return absolute_humidity

#create results array
results = []

#reading environment file in chunks so the method works even if the file is extemely large
for chunk in pd.read_csv(
    environment_file,
    chunksize=50000, #setting chunk size
    usecols=["sensorid","jsondata"]
):
    #pulling CO2 and timestamp from the environment file chunk from extract_co2 func
    extracted = chunk["jsondata"].apply(extract_environment)
    #splitting extracted tuple from extracted into Co2 and timestamp
    chunk["CO2_ppm"] = extracted.apply(lambda x: x[0])
    chunk["Temperature_C"] = extracted.apply(lambda x: x[1])
    chunk["RelativeHumidity_pct"] = extracted.apply(lambda x: x[2])
    chunk["UnixTime"] = extracted.apply(lambda x: x[3]) 

   

    chunk["Timestamp"] =(
        pd.to_datetime(
            chunk["UnixTime"],
            unit="s",
            utc=True
        )
        .dt.tz_convert("Australia/Melbourne")
    )

    #calc absolute humidity

    chunk["AbsoluteHumidity_gm3"]=(
        calc_abs_humidity(
            chunk["Temperature_C"],
            chunk["RelativeHumidity_pct"]
        )
    )

    #renaming to match mapping table
    chunk = chunk.rename(
        columns={"sensorid":"EnvironmentSensorID"}
    )

    chunk["EnvironmentSensorID"] = (
        chunk["EnvironmentSensorID"].astype("int64")
    )
    
    #keeping required variables
    chunk = chunk[
        [
            "EnvironmentSensorID",
            "Timestamp",
            "CO2_ppm",
            "Temperature_C",
            "RelativeHumidity_pct",
            "AbsoluteHumidity_gm3"
        ]
    ]

    results.append(chunk)

#combining processed chunks
environment_data = pd.concat(
    results,
    ignore_index=True
)

print("\nRAW DATA BEFORE DUPLICATE REMOVAL:")

print(
    environment_data.groupby(
        "EnvironmentSensorID"
    ).agg(
        RawRows=("Timestamp", "size"),
        UniqueTimestamps=("Timestamp", "nunique"),
        FirstTimestamp=("Timestamp", "min"),
        LastTimestamp=("Timestamp", "max")
    )
)

timestamp_duplicates = environment_data.duplicated(
    subset=[
        "EnvironmentSensorID",
        "Timestamp"
    ]
).sum()

exact_duplicates = environment_data.duplicated(
    subset=[
        "EnvironmentSensorID",
        "Timestamp",
        "CO2_ppm",
        "Temperature_C",
        "RelativeHumidity_pct"
    ]
).sum()

print("Repeated sensor/timestamps:", timestamp_duplicates)
print("Identical environmental readings:", exact_duplicates)
print("Potentially conflicting readings:",
      timestamp_duplicates - exact_duplicates)

""" #removing duplicate data
has_timestamp = environment_data["Timestamp"].notna()

with_timestamp = (
    environment_data[has_timestamp]
    .drop_duplicates(
        subset=[
            "EnvironmentSensorID",
            "Timestamp"
        ]
    )
)

without_timestamp = environment_data[
    ~has_timestamp
]

environment_data = pd.concat(
    [
        with_timestamp,
        without_timestamp
    ],
    ignore_index=True
) """
sensor_id_corrections = {
    6012002000227: 6012003000226,
    6012002000241: 6012003000240,
    6012002000777: 6012003000776,
    6012002000869: 6012002000869
}

# Keep original ID so we don't lose the source information
environment_data["OriginalSensorID"] = (
    environment_data["EnvironmentSensorID"]
)

# Replace IDs where a correction is known
environment_data["EnvironmentSensorID"] = (
    environment_data["EnvironmentSensorID"]
    .replace(sensor_id_corrections)
)

environment_data = environment_data.merge(
    sensor_rooms[
        [
            "Environment Sensor ID",
            "Building",
            "Room"
        ]
    ],
    left_on="EnvironmentSensorID",
    right_on="Environment Sensor ID",
    how="left"
)

environment_data = environment_data.drop(
    columns=["Environment Sensor ID"]
)

environment_data = environment_data.sort_values(
    [
        "EnvironmentSensorID",
        "Timestamp"
    ]
)

#saving final csv file

environment_data.to_csv(
    "5_Sensor_Environment_Data.csv",
    index=False
)

print("\nFinished.")

print(
    f"{len(environment_data)} "
    "environmental measurements extracted."
)

print("\nFirst 20 rows:")

print(
    environment_data
    .head(20)
    .to_string(index=False)
)


# --------------------------------------------------
# Environmental sensor summary
# --------------------------------------------------

env_sensor_ids = (
    environment_data[
        "EnvironmentSensorID"
    ]
    .dropna()
    .unique()
)



print("\nNumber of unique environmental sensors:")
print(len(env_sensor_ids))


print("\nMeasurements per sensor:")

print(
    environment_data[
        "EnvironmentSensorID"
    ]
    .value_counts()
    .sort_index()
)


print("\nEnvironmental sensor IDs:")

for sensor in sorted(env_sensor_ids):
    print(sensor)


print("\nCorrected sensor mapping check:")

for sensor in sorted(
    environment_data["EnvironmentSensorID"].unique()
):

    room = sensor_rooms[
        sensor_rooms["Environment Sensor ID"] == sensor
    ]

    if not room.empty:

        print(
            sensor,
            "->",
            room[
                ["Building", "Room"]
            ].to_dict("records")
        )

    else:

        print(
            sensor,
            "-> NOT FOUND IN MAPPING"
        )

for sensor in sorted(
    environment_data["OriginalSensorID"].unique()
):

    sensor_data = environment_data[
        environment_data["OriginalSensorID"] == sensor
    ].sort_values("Timestamp")

    plt.figure(figsize=(12, 6))

    plt.scatter(
        sensor_data["Timestamp"],
        sensor_data["CO2_ppm"],
        s=5,
        alpha=0.3
    )

    plt.xlabel("Time")
    plt.ylabel("CO2 (ppm)")

    plt.title(
        f"Raw CO2 measurements - Sensor {sensor}"
    )

    plt.grid()

plt.show()
for sensor in sorted(
    environment_data["OriginalSensorID"].unique()
):

    sensor_data = environment_data[
        environment_data["OriginalSensorID"] == sensor
    ]

    timestamp_counts = (
        sensor_data
        .groupby("Timestamp")
        .size()
        .reset_index(name="NumberOfRecords")
        .sort_values("Timestamp")
    )

    plt.figure(figsize=(12, 6))

    plt.scatter(
        timestamp_counts["Timestamp"],
        timestamp_counts["NumberOfRecords"],
        s=10
    )

    plt.xlabel("Measurement timestamp")
    plt.ylabel("Number of raw records")

    plt.title(
        f"Duplicate records per timestamp - Sensor {sensor}"
    )

    plt.grid()

plt.show()

for sensor in sorted(
    environment_data["OriginalSensorID"].unique()
):

    sensor_data = environment_data[
        environment_data["OriginalSensorID"] == sensor
    ]

    counts = (
        sensor_data
        .groupby("Timestamp")
        .size()
        .sort_values(ascending=False)
    )

    print(
        f"\nSensor {sensor} - "
        "most repeated timestamps:"
    )

    print(counts.head(10))