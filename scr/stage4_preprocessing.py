
# code ko run karne ki command

# $env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; .\venv\Scripts\python.exe scr/stage4_preprocessing.py

"""
Smart Energy Consumption Monitoring & Prediction System
Stage 4: Data Preprocessing
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_URL = (
    "https://raw.githubusercontent.com/LuisM78/"
    "Appliances-energy-prediction-data/master/energydata_complete.csv"
)

LOCAL_DATA = DATA_DIR / "energy.csv"

# Load the real dataset
if LOCAL_DATA.exists() and LOCAL_DATA.stat().st_size > 0:
    df = pd.read_csv(LOCAL_DATA)
else:
    print("Local CSV is empty. Loading real dataset...")
    df = pd.read_csv(DATA_URL)

if "Appliances" not in df.columns:
    print("Loading real energy dataset...")
    df = pd.read_csv(DATA_URL)

# Use the real dataset if the local practice file has different columns
if "Appliances" not in df.columns:
    print("Loading the real energy dataset...")
    df = pd.read_csv(DATA_URL)

print("Original dataset shape:", df.shape)

# Check required columns
required_columns = ["date", "Appliances", "T1", "RH_1", "T2", "RH_2"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Required columns missing: {missing_columns}")

# Remove duplicate rows
df = df.drop_duplicates().copy()

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Convert required numeric columns
numeric_columns = ["Appliances", "T1", "RH_1", "T2", "RH_2"]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows with missing required values
df = df.dropna(subset=required_columns).copy()

# Sort by date before splitting
df = df.sort_values("date").reset_index(drop=True)

# Create time-based features
df["hour"] = df["date"].dt.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["day_of_week"] = df["date"].dt.dayofweek

# Input features and target
feature_columns = [
    "T1",
    "RH_1",
    "T2",
    "RH_2",
    "T3",
    "T_out",
    "T6",
    "Windspeed",
    "lights",
    "hour",
    "hour_sin",
    "hour_cos",
    "day_of_week"
]

X = df[feature_columns]
y = df["Appliances"]

# Split chronologically: first 80% for training, last 20% for testing
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()

print("\nCleaned dataset shape:", df.shape)
print("Input features (X):", X.shape)
print("Target (y):", y.shape)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

print("\nTraining date range:")
print(df["date"].iloc[0], "to", df["date"].iloc[split_index - 1])

print("\nTesting date range:")
print(df["date"].iloc[split_index], "to", df["date"].iloc[-1])

# Save prepared data
df.to_csv(OUTPUT_DIR / "cleaned_energy_data.csv", index=False)
X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

print("\nStage 4 preprocessing completed successfully!")
print("Prepared files saved in:", OUTPUT_DIR)
