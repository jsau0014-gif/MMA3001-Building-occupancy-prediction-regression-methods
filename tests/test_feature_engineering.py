"""Verify feature values, time zones, periodic boundaries and room consistency."""

from __future__ import annotations

import math
import pandas as pd
import pytest


# Expected Melbourne-local dates are not the same as the UTC dates in the files.
RAW_TIMES = [
    "2024-03-03T13:00:00+00:00",  # Monday 4 March 2024, 00:00 AEDT
    "2024-03-02T13:00:00+00:00",  # Sunday 3 March 2024, 00:00 AEDT
    "2024-12-31T13:00:00+00:00",  # Wednesday 1 January 2025, 00:00 AEDT
    "2024-12-30T12:15:00+00:00",  # Monday 30 December 2024, 23:15 AEDT
]
FEATURE_NAMES = [
    "TimeSin", "TimeCos", "DaySin", "DayCos",
    "MonthSin", "MonthCos", "IsWeekend",
]


def write_feature_inputs(root):
    data = root / "Data" / "processed"
    pd.DataFrame({"Timestamp": RAW_TIMES, "Headcount": [2.5, 1.0, 6.0, 4.0]}).to_csv(
        data / "G25_Minute_Training.csv", index=False
    )
    pd.DataFrame({"Timestamp": RAW_TIMES, "Headcount": [3.5, 2.0, 7.0, 5.0]}).to_csv(
        data / "G20_Minute_validation.csv", index=False
    )
    pd.DataFrame({"collecteddate": RAW_TIMES, "headcount": [3, 2, 7, 5]}).to_csv(
        data / "G20_Occupancy.csv", index=False
    )


def feature_outputs(workspace):
    root, run = workspace
    write_feature_inputs(root)
    run("04_feature_engineering.py")
    run("04b_feature_engineering_G20.py")
    data = root / "Data" / "processed"
    return (
        pd.read_csv(data / "G25_Model_Train.csv"),
        pd.read_csv(data / "G20_Model_Validation_Minute.csv"),
        pd.read_csv(data / "G20_Model_Validation.csv"),
    )


def test_expected_calendar_values_and_melbourne_timezone(workspace):
    g25, _, _ = feature_outputs(workspace)
    assert set(g25.columns) == {"Timestamp", "Headcount", *FEATURE_NAMES}
    assert g25["Timestamp"].iloc[0].startswith("2024-03-04 00:00:00")
    assert g25["IsWeekend"].tolist() == [0, 1, 0, 0]
    assert g25["TimeSin"].iloc[0] == pytest.approx(0, abs=1e-12)
    assert g25["TimeCos"].iloc[0] == pytest.approx(1, abs=1e-12)
    assert g25["DaySin"].iloc[0] == pytest.approx(0, abs=1e-12)
    assert g25["DayCos"].iloc[0] == pytest.approx(1, abs=1e-12)
    assert g25["MonthSin"].iloc[0] == pytest.approx(math.sin(2 * math.pi * 2 / 12))
    assert g25["MonthSin"].iloc[2] == pytest.approx(0, abs=1e-12)  # January 2025
    assert g25["MonthCos"].iloc[2] == pytest.approx(1, abs=1e-12)
    assert g25["TimeSin"].iloc[3] == pytest.approx(math.sin(2 * math.pi * 23.25 / 24))
    assert g25["TimeCos"].iloc[3] == pytest.approx(math.cos(2 * math.pi * 23.25 / 24))


def test_identical_feature_definitions_for_both_rooms(workspace):
    g25, g20minute, g20raw = feature_outputs(workspace)
    for column in FEATURE_NAMES:
        assert g25[column].to_numpy() == pytest.approx(g20minute[column].to_numpy())
        assert g25[column].to_numpy() == pytest.approx(g20raw[column].to_numpy())
    # Targets remain room-specific and may be fractional after preprocessing.
    assert g25["Headcount"].iloc[0] == pytest.approx(2.5)
    assert g20minute["Headcount"].iloc[0] == pytest.approx(3.5)


def test_sine_cosine_unit_circle_invariant(workspace):
    g25, _, _ = feature_outputs(workspace)
    for prefix in ["Time", "Day", "Month"]:
        value = g25[f"{prefix}Sin"] ** 2 + g25[f"{prefix}Cos"] ** 2
        assert value.to_numpy() == pytest.approx([1] * len(g25), abs=1e-12)


def test_invalid_timestamp_is_excluded_from_engineered_features(workspace):
    root, run = workspace
    write_feature_inputs(root)
    file_path = root / "Data" / "processed" / "G25_Minute_Training.csv"
    original = pd.read_csv(file_path)
    extra = pd.DataFrame([{"Timestamp": "not-a-time", "Headcount": 100}])
    pd.concat([original, extra], ignore_index=True).to_csv(file_path, index=False)
    run("04_feature_engineering.py")
    output = pd.read_csv(root / "Data" / "processed" / "G25_Model_Train.csv")
    assert len(output) == 4
    assert output[FEATURE_NAMES].notna().all().all()
