
"""
Stage 7: Prediction Error Visualization
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

error_file = OUTPUT_DIR / "prediction_error_analysis.csv"

df = pd.read_csv(error_file)

# Graph 1: Actual vs Predicted (first 100 rows)
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["Actual Appliances"].head(100).to_numpy(),
    label="Actual"
)

ax.plot(
    df["Predicted Appliances"].head(100).to_numpy(),
    label="Predicted"
)

ax.set_title("Actual vs Predicted Consumption")
ax.set_xlabel("Test Sample")
ax.set_ylabel("Appliances Consumption (Wh)")
ax.legend()
ax.grid(alpha=0.3)

fig.savefig(
    OUTPUT_DIR / "actual_vs_predicted_analysis.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

# Graph 2: Distribution of absolute errors
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    df["Absolute Error"],
    bins=30,
    edgecolor="black"
)

ax.set_title("Distribution of Prediction Errors")
ax.set_xlabel("Absolute Error (Wh)")
ax.set_ylabel("Number of Predictions")
ax.grid(axis="y", alpha=0.3)

fig.savefig(
    OUTPUT_DIR / "prediction_error_distribution.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

print("Error visualization completed!")
print("Saved: actual_vs_predicted_analysis.png")
print("Saved: prediction_error_distribution.png")
