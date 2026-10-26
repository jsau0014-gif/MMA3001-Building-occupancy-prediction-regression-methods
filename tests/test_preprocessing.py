"""Numerical verification of G.25 and G.20 zero-order-hold averaging.

Expected results come from independent analytical rectangle sums rather than
from the cumulative-integral implementation in the production scripts.
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timedelta, timezone
import math
import random

import pandas as pd
import pytest


ROOMS = [
    ("G25", "03_preprocess_training.py", "G25_Minute_Training.csv"),
    ("G20", "03b_preprocess_G20.py", "G20_Minute_validation.csv"),
]
BASE = datetime(2024, 3, 3, tzinfo=timezone.utc)


def row(seconds: int, headcount: object) -> dict:
    """A synthetic raw reading, timestamped in UTC."""
    return {
        "collecteddate": (BASE + timedelta(seconds=seconds)).isoformat(),
        "headcount": headcount,
    }


def evaluate(workspace, room: str, script: str, output: str, records):
    root, run = workspace
    pd.DataFrame(records).to_csv(root / "Data" / "processed" / f"{room}_Occupancy.csv", index=False)
    run(script)
    return pd.read_csv(root / "Data" / "processed" / output)


def independently_integrated_expected(records):
    """Compute expected minute averages by walking individual ZOH rectangles.

    This deliberately does NOT use np.cumsum or np.searchsorted: a distinct
    algorithm is useful for detecting arithmetic/indexing errors in the project.
    Assume the last observation at an identical timestamp supersedes prior ones.
    """
    observations = []
    for record in records:
        try:
            dt = datetime.fromisoformat(str(record["collecteddate"]).replace("Z", "+00:00"))
            value = float(record["headcount"])
            if not math.isfinite(value):
                continue
        except (ValueError, TypeError):
            continue
        observations.append((dt, value))
    observations.sort(key=lambda pair: pair[0])  # Stable sort.
    unique = {}
    for dt, value in observations:
        unique[dt] = value  # Keep the latest record at duplicate times.
    times = sorted(unique)
    values = [unique[dt] for dt in times]
    retained = sorted({t.replace(second=0, microsecond=0) for t in times})
    results = {}
    for minute in retained:
        lower = max(minute, times[0])
        upper = minute + timedelta(minutes=1)
        cursor = lower
        area = 0.0
        while cursor < upper:
            index = bisect_right(times, cursor) - 1
            assert index >= 0
            next_breakpoint = times[index + 1] if index + 1 < len(times) else upper
            limit = min(upper, next_breakpoint)
            area += values[index] * (limit - cursor).total_seconds()
            cursor = limit
        results[minute] = area / (upper - lower).total_seconds()
    return results


@pytest.mark.parametrize("room,script,output", ROOMS)
def test_exact_known_rectangle_areas_and_duplicate_timestamp(workspace, room, script, output):
    readings = [
        row(45, 4), row(0, 2), row(20, 999), row(60, 3),
        row(20, 6), row(100, 1), row(120, 1),
    ]
    # At 00:00:20 UTC the later value 6 supersedes the earlier value 999.
    # Minute 0: (2*20 + 6*25 + 4*15)/60 = 4.166666... occupants.
    # Minute 1: (3*40 + 1*20)/60 = 2.333333... occupants.
    actual = evaluate(workspace, room, script, output, readings)
    assert len(actual) == 3
    assert actual["Headcount"].to_numpy() == pytest.approx([250 / 60, 140 / 60, 1])
    assert actual["MeasurementCount"].tolist() == [4, 2, 1]


@pytest.mark.parametrize("room,script,output", ROOMS)
def test_initial_partial_minute_uses_covered_duration(workspace, room, script, output):
    readings = [row(10, 2), row(30, 6), row(60, 4)]
    actual = evaluate(workspace, room, script, output, readings)
    # Only 50 seconds are covered in the initial minute.
    assert actual["Headcount"].iloc[0] == pytest.approx((2 * 20 + 6 * 30) / 50)
    assert actual["Headcount"].iloc[1] == pytest.approx(4)


@pytest.mark.parametrize("room,script,output", ROOMS)
def test_unmeasured_minutes_are_not_added(workspace, room, script, output):
    readings = [row(0, 2), row(120, 4), row(150, 6)]
    actual = evaluate(workspace, room, script, output, readings)
    assert len(actual) == 2  # The minute starting at 00:01 has no readings.
    assert actual["Headcount"].to_numpy() == pytest.approx([2, 5])
    assert actual["MeasurementCount"].tolist() == [1, 2]


@pytest.mark.parametrize("room,script,output", ROOMS)
def test_invalid_rows_removed_and_zero_is_valid(workspace, room, script, output):
    readings = [row(0, 0), row(30, 4), row(60, 2)]
    readings.extend([
        {"collecteddate": "invalid-timestamp", "headcount": 400},
        {"collecteddate": row(10, 0)["collecteddate"], "headcount": "not-a-number"},
        {"collecteddate": None, "headcount": 77},
    ])
    actual = evaluate(workspace, room, script, output, readings)
    assert len(actual) == 2
    assert actual["Headcount"].to_numpy() == pytest.approx([2, 2])
    assert actual["MeasurementCount"].tolist() == [2, 1]


@pytest.mark.parametrize("room,script,output", ROOMS)
def test_single_reading_is_held_until_minute_end(workspace, room, script, output):
    actual = evaluate(workspace, room, script, output, [row(15, 7)])
    assert len(actual) == 1
    assert actual["Headcount"].iloc[0] == pytest.approx(7)
    assert actual["MeasurementCount"].iloc[0] == 1


@pytest.mark.parametrize("room,script,output", ROOMS)
def test_randomised_reference_comparison(workspace, room, script, output):
    random_source = random.Random(20261026)
    readings = [row(0, random_source.randint(0, 16))]
    for _ in range(65):
        readings.append(row(random_source.randint(1, 1199), random_source.randint(0, 16)))
    readings.extend([readings[12].copy(), readings[25].copy()])
    random_source.shuffle(readings)
    actual = evaluate(workspace, room, script, output, readings)
    expected = independently_integrated_expected(readings)
    assert len(actual) == len(expected)
    for _, record in actual.iterrows():
        utc_minute = pd.Timestamp(record["Timestamp"]).tz_convert("UTC").to_pydatetime()
        assert record["Headcount"] == pytest.approx(expected[utc_minute], abs=1e-10)
    assert (actual["Headcount"] >= 0).all()
    assert (actual["Headcount"] <= 16).all()
