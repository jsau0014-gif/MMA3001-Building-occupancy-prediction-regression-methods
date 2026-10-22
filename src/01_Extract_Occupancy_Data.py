"""
 This script extracts the occupancy data from rooms g20 and g25 with Space ID's  cca1ac3e-f81f-4ebb-aa2b-8b0bf3e9e53b and 227f1068-9e41-469f-bef0-275f27a0bdba respectively
    It runs in chunks to avoid memory issues and tracks the numbers of extractions completed
 """

import pandas as pd
from pathlib import Path

#Defining the project directiory
project_dir = Path(__file__).resolve().parent.parent

#defining the occupancy file
occupancy_file = (project_dir/ "Data" /"5occupancySensor_MayToDec2024_9MRows.csv")

#setting the output directory
output_dir = project_dir / "Data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

#setting room keys

rooms = {
 "G20":"cca1ac3e-f81f-4ebb-aa2b-8b0bf3e9e53b",
 "G25":"227f1068-9e41-469f-bef0-275f27a0bdba"
}

#tracking the number of measurements extracted setting it to 0 initially
counts = {room: 0 for room in rooms}


#Tracking whether data has been written to the output CSV for each room
first_write = {room: True for room in rooms}

#Extracting the occupancy file in chunks, so that memory isnt overloaded

for chunk in pd.read_csv(
 occupancy_file,
 chunksize=100000,
 dtype={"floorspaceid":str}
):

 for room, space_id in rooms.items():

  #extract records for the called room
  room_data = chunk[
   chunk["floorspaceid"] == space_id
  ].copy()

  if room_data.empty:
   continue

  #add the room name
  room_data["Room"] = room

  #setting output location
  output_file = output_dir / f"{room}_Occupancy.csv"

  #save each chunk after it is processed
  room_data.to_csv(
   output_file,
   mode="w" if first_write[room] else "a",
   header=first_write[room],
   index=False
  )

  first_write[room] = False

  counts[room] += len(room_data)

#displaying data exxtraction summary
print("\nOccupancy extraction completed")

for room in rooms:
 print(f"{room}: {counts[room]:,} records")

print(f"\nFiles saved in: {output_dir}")

