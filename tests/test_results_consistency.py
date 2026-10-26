"""Recompute saved G.20 metrics directly from exported predictions.

This suite is optional when published results are absent: generated Results/
files are deliberately not committed by default in the student's repository.
When present, this verifies the recorded metrics without retraining SVR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@pytest.fixture(scope="module")
def g20_results():
    root = Path(__file__).resolve().parents[1] / "Results"
    prediction_path = root / "06b_G20validation_predictions.csv"
    comparison_path = root / "06b_G20model_comparison.csv"
    if not prediction_path.exists() or not comparison_path.exists():
        pytest.skip("Generated G.20 prediction and comparison CSVs not present; run script 06b first")
    predictions = pd.read_csv(prediction_path)
    comparison = pd.read_csv(comparison_path).set_index("Model")
    return predictions, comparison


@pytest.mark.parametrize(
    "model,predicted",
    [
        ("Linear Regression", "LinearPredicted"),
        ("Decision Tree", "TreePredicted"),
        ("SVR", "SVRPredicted"),
    ],
)
def test_exported_g20_metrics_recompute_correctly(g20_results, model, predicted):
    predictions, summary = g20_results
    assert len(predictions) == len(predictions.dropna(subset=["Actual", predicted]))
    assert model in summary.index
    actual = predictions["Actual"].to_numpy(dtype=float)
    estimate = predictions[predicted].to_numpy(dtype=float)
    residual_column = predicted.replace("Predicted", "Residual")
    if residual_column in predictions.columns:
        assert predictions[residual_column].to_numpy(dtype=float) == pytest.approx(
            actual - estimate, abs=1e-8
        )
    expected = {
        "Observations": len(actual),
        "MAE": mean_absolute_error(actual, estimate),
        "RMSE": np.sqrt(mean_squared_error(actual, estimate)),
        "R2": r2_score(actual, estimate),
        "MeanResidual": np.mean(actual - estimate),
    }
    for name, value in expected.items():
        assert float(summary.loc[model, name]) == pytest.approx(value, abs=1e-8), (
            f"{model} {name} disagrees with predictions"
        )
    assert float(summary.loc[model, "PredictionSeconds"]) >= 0
