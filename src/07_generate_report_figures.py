
"""
MMA3001 - Generate Report Figures

Creates plots comparing:
- G25 internal validation and G20 external validation
- Linear Regression, Decision Tree and SVR
- Prediction errors and occupancy patterns
- Computational performance and optimisation

No model training or prediction is performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "Data" / "processed"
results_dir = project_dir / "Results"

figures_dir = results_dir / "report_figures"
figures_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 250
})


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def find_file(*filenames):
    """Return first existing result file."""
    for filename in filenames:
        path = results_dir / filename

        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find any of these files:\n"
        + "\n".join(filenames)
    )


def save_figure(filename):
    """Save current figure and close it."""

    figures_dir.mkdir(parents=True, exist_ok=True)

    output_path = figures_dir / filename

    print(f"Saving: {output_path}", flush=True)

    try:
        plt.tight_layout()

        plt.savefig(
            str(output_path),
            dpi=200,
            format="png",
            bbox_inches="tight"
        )

        print(f"Saved: {filename}", flush=True)

    except OSError as e:
        print(f"\nCould not save to: {output_path}")
        print(f"Error: {e}")

        # Fallback: save directly into Results
        fallback = results_dir / filename

        print(f"Trying alternative: {fallback}")

        plt.savefig(
            str(fallback),
            dpi=200,
            format="png",
            bbox_inches="tight"
        )

        print(f"Saved to alternative location: {fallback}")

    finally:
        plt.close()


# ============================================================
# 3. LOAD DATA
# ============================================================

# G25 internal model comparison
g25_metrics = pd.read_csv(
    results_dir / "05_model_comparison_random.csv",
    index_col=0
)

# G20 independent validation metrics
g20_metrics = pd.read_csv(
    find_file(
        "06b_G20_model_comparison.csv",
        "06b_G20model_comparison.csv"
    )
)

# G20 actual and predicted occupancy
g20_predictions = pd.read_csv(
    find_file(
        "06b_G20_validation_predictions.csv",
        "06b_G20validation_predictions.csv"
    )
)

# Convert timestamps
g20_predictions["Timestamp"] = pd.to_datetime(
    g20_predictions["Timestamp"],
    utc=True,
    errors="coerce"
).dt.tz_convert("Australia/Melbourne")

# Model names
models = [
    "Linear Regression",
    "Decision Tree",
    "SVR"
]

prediction_columns = {
    "Linear Regression": "LinearPredicted",
    "Decision Tree": "TreePredicted",
    "SVR": "SVRPredicted"
}

# Extract the final selected model from script 05
def get_g25_row(prefix):
    matches = [
        index for index in g25_metrics.index
        if str(index).startswith(prefix)
    ]

    if not matches:
        raise ValueError(
            f"Could not find G25 model starting with {prefix}"
        )

    return g25_metrics.loc[matches[-1]]


g25_selected = pd.DataFrame({
    "Linear Regression": g25_metrics.loc["Linear Regression"],
    "Decision Tree": get_g25_row("Decision Tree (leaf=100)"),
    "SVR": get_g25_row("SVR Tuned")
}).T

g20_selected = (
    g20_metrics
    .set_index("Model")
    .loc[models]
)

print("\nLoaded G25 and G20 results successfully.")
print("\nG25:")
print(g25_selected[["MAE", "RMSE", "R2"]])

print("\nG20:")
print(g20_selected[["MAE", "RMSE", "R2"]])


# ============================================================
# FIGURE 1: INTERNAL VS EXTERNAL RMSE
# ============================================================

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    x - width / 2,
    g25_selected["RMSE"],
    width,
    label="G25 Internal Validation"
)

ax.bar(
    x + width / 2,
    g20_selected["RMSE"],
    width,
    label="G20 External Validation"
)

ax.set_xticks(x)
ax.set_xticklabels(models)

ax.set_ylabel("RMSE (occupants)")
ax.set_title("Internal vs External Validation RMSE")

ax.legend()
ax.grid(axis="y", alpha=0.25)

save_figure("01_internal_vs_external_RMSE.png")


# ============================================================
# FIGURE 2: INTERNAL VS EXTERNAL R2
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    x - width / 2,
    g25_selected["R2"],
    width,
    label="G25 Internal Validation"
)

ax.bar(
    x + width / 2,
    g20_selected["R2"],
    width,
    label="G20 External Validation"
)

ax.axhline(0, color="black", linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(models)

ax.set_ylabel("R²")
ax.set_title("Model Generalisation: R² Comparison")

ax.legend()
ax.grid(axis="y", alpha=0.25)

save_figure("02_internal_vs_external_R2.png")


# ============================================================
# FIGURE 3: MEAN RESIDUAL / PREDICTION BIAS
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    models,
    g20_selected["MeanResidual"]
)

ax.axhline(
    0,
    color="black",
    linestyle="--"
)

ax.set_ylabel("Mean Residual (occupants)")
ax.set_title("G20 Systematic Prediction Bias")

ax.grid(axis="y", alpha=0.25)

save_figure("03_G20_prediction_bias.png")


# ============================================================
# FIGURE 4: ACTUAL VS MEAN PREDICTED HEADCOUNT
# ============================================================

labels = ["Actual"] + models

values = [
    g20_predictions["Actual"].mean()
]

for column in prediction_columns.values():
    values.append(
        g20_predictions[column].mean()
    )

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(labels, values)

ax.set_ylabel("Mean Headcount")
ax.set_title("G20 Actual vs Predicted Mean Occupancy")

ax.tick_params(
    axis="x",
    rotation=15
)

ax.grid(axis="y", alpha=0.25)

save_figure("04_G20_mean_headcount.png")


# ============================================================
# FIGURE 5: OCCUPANCY PATTERNS BY HOUR
# ============================================================

g20_predictions["Hour"] = (
    g20_predictions["Timestamp"].dt.hour
)

hourly = g20_predictions.groupby("Hour").mean(
    numeric_only=True
)

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    hourly.index,
    hourly["Actual"],
    linewidth=2.5,
    color="black",
    label="Actual G20"
)

for model, column in prediction_columns.items():

    ax.plot(
        hourly.index,
        hourly[column],
        marker=".",
        label=model
    )

ax.set_xlabel("Hour of Day (Melbourne)")
ax.set_ylabel("Mean Headcount")
ax.set_title("G20 Average Daily Occupancy Profile")

ax.set_xticks(range(0, 24, 2))

ax.legend()
ax.grid(True, alpha=0.25)

save_figure("05_G20_daily_occupancy.png")


# ============================================================
# FIGURE 6: MAE BY HOUR OF DAY
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5))

for model, column in prediction_columns.items():

    errors = (
        g20_predictions["Actual"]
        - g20_predictions[column]
    ).abs()

    hourly_error = errors.groupby(
        g20_predictions["Hour"]
    ).mean()

    ax.plot(
        hourly_error.index,
        hourly_error.values,
        marker=".",
        label=model
    )

ax.set_xlabel("Hour of Day (Melbourne)")
ax.set_ylabel("Mean Absolute Error (occupants)")
ax.set_title("G20 Prediction Error by Hour")

ax.set_xticks(range(0, 24, 2))

ax.legend()
ax.grid(True, alpha=0.25)

save_figure("06_G20_MAE_by_hour.png")


# ============================================================
# FIGURE 7: MAE BY ACTUAL OCCUPANCY LEVEL
# ============================================================

bins = [0, 1, 2, 4, 8, 16, np.inf]

band_labels = [
    "0–1",
    "1–2",
    "2–4",
    "4–8",
    "8–16",
    "16+"
]

g20_predictions["OccupancyBand"] = pd.cut(
    g20_predictions["Actual"],
    bins=bins,
    labels=band_labels,
    right=False,
    include_lowest=True
)

fig, ax = plt.subplots(figsize=(10, 5))

positions = np.arange(len(band_labels))

for i, (model, column) in enumerate(
    prediction_columns.items()
):

    temp = g20_predictions[
        ["OccupancyBand", "Actual", column]
    ].copy()

    temp["AbsoluteError"] = (
        temp["Actual"] - temp[column]
    ).abs()

    errors = temp.groupby(
        "OccupancyBand",
        observed=True
    )["AbsoluteError"].mean()

    errors = errors.reindex(band_labels)

    ax.bar(
        positions + (i - 1) * 0.25,
        errors.to_numpy(),
        width=0.25,
        label=model
    )

ax.set_xticks(positions)
ax.set_xticklabels(band_labels)

ax.set_xlabel("Actual Headcount Range")
ax.set_ylabel("Mean Absolute Error")
ax.set_title("G20 Prediction Error by Occupancy Level")

ax.legend()
ax.grid(axis="y", alpha=0.25)

save_figure("07_G20_MAE_by_occupancy.png")


# ============================================================
# FIGURE 8: CUMULATIVE ABSOLUTE ERROR DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

for model, column in prediction_columns.items():

    errors = np.sort(
        np.abs(
            g20_predictions["Actual"]
            - g20_predictions[column]
        )
    )

    cumulative = (
        np.arange(1, len(errors) + 1)
        / len(errors)
    )

    ax.plot(
        errors,
        cumulative,
        label=model
    )

ax.set_xlim(0, 12)
ax.set_ylim(0, 1)

ax.set_xlabel("Absolute Prediction Error (occupants)")
ax.set_ylabel("Cumulative Proportion")
ax.set_title("G20 Cumulative Absolute Error")

ax.legend()
ax.grid(True, alpha=0.25)

save_figure("08_G20_cumulative_error.png")


# ============================================================
# FIGURE 9: ACTUAL VS PREDICTED (ALL THREE MODELS)
# ============================================================

for i, (model, column) in enumerate(
    prediction_columns.items(),
    start=1
):

    fig, ax = plt.subplots(figsize=(7, 6))

    plot = ax.hexbin(
        g20_predictions["Actual"],
        g20_predictions[column],
        gridsize=60,
        bins="log",
        mincnt=1,
        cmap="viridis"
    )

    fig.colorbar(
        plot,
        ax=ax,
        label="Log Observation Count"
    )

    upper = max(
        float(g20_predictions["Actual"].max()),
        float(g20_predictions[column].max())
    )

    lower = min(
        0,
        float(g20_predictions[column].min())
    )

    ax.plot(
        [lower, upper],
        [lower, upper],
        "r--",
        label="Perfect Prediction"
    )

    ax.set_xlabel("Actual G20 Headcount")
    ax.set_ylabel("Predicted G20 Headcount")

    ax.set_title(
        f"G20 Actual vs Predicted: {model}"
    )

    ax.legend()
    ax.grid(True, alpha=0.15)

    save_figure(
        f"09_{i}_{model.lower().replace(' ', '_')}.png"
    )


# ============================================================
# FIGURE 10: MODEL PREDICTION RUNTIME
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

times = g20_selected["PredictionSeconds"]

ax.bar(models, times)

ax.set_yscale("log")

ax.set_ylabel("Prediction Time (seconds, log scale)")
ax.set_title("G20 Full-Dataset Prediction Runtime")

ax.grid(axis="y", alpha=0.25)

save_figure("10_G20_prediction_runtime.png")


# ============================================================
# FIGURE 11: DECISION TREE LEAF SIZE OPTIMISATION
# ============================================================

leaf_file = results_dir / "05_leaf_size_comparison.csv"

if leaf_file.exists():

    leaf_data = pd.read_csv(leaf_file)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        leaf_data["LeafSize"],
        leaf_data["Train_RMSE"],
        marker="o",
        label="Training RMSE"
    )

    ax.plot(
        leaf_data["LeafSize"],
        leaf_data["Validation_RMSE"],
        marker="o",
        label="Validation RMSE"
    )

    ax.set_xlabel("Minimum Samples per Leaf")
    ax.set_ylabel("RMSE (occupants)")
    ax.set_title("Decision Tree Leaf Size Optimisation")

    ax.legend()
    ax.grid(True, alpha=0.25)

    save_figure("11_tree_leaf_optimisation.png")


# ============================================================
# FIGURE 12: SVR TRAINING SIZE SENSITIVITY
# ============================================================

# Results from the previous 100k SVR experiment
old_metrics_file = (
    results_dir / "05_model_comparison_100k.csv"
)

if old_metrics_file.exists():

    old_metrics = pd.read_csv(
        old_metrics_file,
        index_col=0
    )

    old_svr_rows = [
        index for index in old_metrics.index
        if str(index).startswith("SVR Tuned")
    ]

    if old_svr_rows:

        old_rmse = old_metrics.loc[
            old_svr_rows[-1],
            "RMSE"
        ]

        new_rmse = g25_selected.loc[
            "SVR", "RMSE"
        ]

        fig, ax = plt.subplots(figsize=(8, 5))

        sizes = [100000, 298608]
        rmses = [old_rmse, new_rmse]

        ax.plot(
            sizes,
            rmses,
            marker="o",
            markersize=8
        )

        ax.set_xticks(sizes)
        ax.set_xticklabels([
            "100,000",
            "298,608"
        ])

        ax.set_xlabel("SVR Training Observations")
        ax.set_ylabel("G25 Validation RMSE")
        ax.set_title("SVR Training Size Sensitivity")

        ax.grid(True, alpha=0.25)

        save_figure("12_SVR_training_size.png")

else:
    print(
        "\nSkipping SVR training-size graph:"
        " 100k comparison CSV not found."
    )


# ============================================================
# FINISHED
# ============================================================

print("\n========================================")
print("REPORT FIGURES COMPLETE")
print("========================================")

print("\nSaved to:")
print(figures_dir)

print("\nNo models were retrained.")




rooms = {
    "G25": "G25_Model_Train.csv",
    "G20": "G20_Model_Validation_Minute.csv"
}

bands = [-np.inf, 0, 1, 2, 4, 8, 16, 32, np.inf]
labels = [
    "0", "(0,1]", "(1,2]", "(2,4]",
    "(4,8]", "(8,16]", "(16,32]", ">32"
]

distribution = {}

for room, filename in rooms.items():
    data = pd.read_csv(data_dir / filename)
    y = data["Headcount"].dropna()

    print(f"\n{'='*40}")
    print(f"{room} HEADCOUNT DISTRIBUTION")
    print(f"{'='*40}")

    print(y.describe(percentiles=[
        .25, .50, .75, .90, .95, .99
    ]))

    print(f"\nExactly zero: {(y == 0).mean()*100:.2f}%")
    print(f"Above 10 occupants: {(y > 10).mean()*100:.2f}%")

    counts = pd.cut(
        y, bins=bands, labels=labels
    ).value_counts(normalize=True).reindex(labels) * 100

    distribution[room] = counts

    print("\nOccupancy bands (% of observations):")
    print(counts.round(2).to_string())

# Compare occupancy distributions

# ============================================================
# FIGURE 13: RAW HEADCOUNT DISTRIBUTION - G25 VS G20
# ============================================================

raw_files = {
    "G25": data_dir / "G25_Occupancy.csv",
    "G20": data_dir / "G20_Occupancy.csv"
}

distributions = {}

for room, filepath in raw_files.items():

    # Load original, unprocessed headcount readings
    data = pd.read_csv(
        filepath,
        usecols=["headcount"]
    )

    # Convert to numeric without averaging or resampling
    headcount = pd.to_numeric(
        data["headcount"],
        errors="coerce"
    ).dropna()

    print(f"\n{room} RAW HEADCOUNT DISTRIBUTION")
    print(headcount.describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ))

    print(
        f"Zero occupancy: {(headcount == 0).mean() * 100:.2f}%"
    )

    distributions[room] = headcount

# Headcount bands, including a separate category for zero
bins = [-0.5, 0.5, 1.5, 2.5, 4.5, 8.5, 16.5, 32.5, np.inf]

labels = [
    "0", "1", "2", "3-4",
    "5-8", "9-16", "17-32", "33+"
]

comparison = {}

for room, headcount in distributions.items():

    counts = pd.cut(
        headcount,
        bins=bins,
        labels=labels
    ).value_counts(normalize=True)

    comparison[room] = (
        counts.reindex(labels).fillna(0) * 100
    )

comparison = pd.DataFrame(comparison)

# Plot percentage rather than raw counts
fig, ax = plt.subplots(figsize=(10, 5.5))

comparison.plot(
    kind="bar",
    ax=ax,
    width=0.75
)

ax.set_title("Distribution of Original Headcount Measurements")
ax.set_xlabel("Recorded Headcount (occupants)")
ax.set_ylabel("Percentage of Sensor Readings (%)")

ax.tick_params(axis="x", rotation=0)
ax.legend(title="Room")
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

plt.savefig(
    figures_dir / "13_raw_headcount_distribution.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()

print("\nSaved Figure 13: Raw headcount distribution")

