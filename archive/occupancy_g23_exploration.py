
import pandas as pd
from collections import Counter

occupancy_file = "Data/5occupancySensor_MayToDec2024_9MRows.csv"

space_counts = Counter()
device_counts = Counter()

for chunk in pd.read_csv(
    occupancy_file,
    usecols=["deviceid", "floorspaceid"],
    chunksize=100000,
    dtype=str
):
    space_counts.update(
        chunk["floorspaceid"].dropna().str.strip()
    )

    device_counts.update(
        chunk["deviceid"].dropna().str.strip()
    )

print("\nNumber of unique floor spaces:", len(space_counts))
print("Number of unique devices:", len(device_counts))

print("\nTop 20 floor spaces:")
for space, count in space_counts.most_common(20):
    print(space, count)

print("\nDevice IDs:")
for device, count in device_counts.most_common():
    print(device, count)

# Check known G.23 SpaceId
g23_id = "2387067e-5349-422a-a4f5-08d93ef52a6e"

print("\nG.23 records:", space_counts[g23_id])
