
"""
Smart Energy Consumption Monitoring & Prediction System
Stage 7: Prediction Error Analysis
"""

from pathlib import Path

import pandas as pd
import numpy as np

# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

# 2. Load predictions
prediction_file = OUTPUT_DIR / "best_model_predictions.csv"

if not prediction_file.exists():
    raise FileNotFoundError(
        "Predictions file not found. Run Stage 5-6 first."
    )

df = pd.read_csv(prediction_file)

# 3. Calculate prediction errors
df["Error"] = (
    df["Actual Appliances"] - df["Predicted Appliances"]
)

df["Absolute Error"] = df["Error"].abs()

# Percentage error: avoid division by zero
df["Percentage Error"] = np.where(
    df["Actual Appliances"] != 0,
    df["Absolute Error"] / df["Actual Appliances"] * 100,
    np.nan
)

# 4. Display results
print("\nFirst 10 prediction errors:")
print(df.head(10).round(2).to_string(index=False))

print("\nError summary:")
print(df["Absolute Error"].describe().round(2))

print("\nMean Absolute Error:", round(df["Absolute Error"].mean(), 2), "Wh")

# 5. Save results
output_file = OUTPUT_DIR / "prediction_error_analysis.csv"

df.to_csv(output_file, index=False)

print("\nError analysis saved to:")
print(output_file)
