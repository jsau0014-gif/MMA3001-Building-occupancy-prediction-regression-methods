import pandas as pd

file = "Data/5EnvSensor_MayToDec2024_180kRows.csv"

df = pd.read_csv(file,nrows=10000)

print("\nColumn Names:")
for col in df.columns:
    print(col)

print("\nFirst 100 Rows:")
print(df.head(10).to_string())

df.head(100).to_csv("Data/Data Samples/First100_environment_MaytoDec2024 .csv", index=False)