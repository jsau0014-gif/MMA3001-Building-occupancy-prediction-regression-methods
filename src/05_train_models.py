"""
This script will train a a multiple linear regression model for the first
80% of the training data (from g25) leaving the final 20% for validation.
It uses LinearRegression functions from scikit

The script will produce a predicted vs acutal headcounts
A residual plot to identify systematic patterns

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

#setting up file locations

project_dir = Path(__file__).resolve().parent.parent

data_dir = project_dir / "Data" / "processed"
results_dir = project_dir / "Results"

results_dir.mkdir(parents=True, exist_ok=True)

inpute_file = data_dir / "G25_Model_Train.csv"