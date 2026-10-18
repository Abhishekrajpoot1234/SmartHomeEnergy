# code run karne ki command 

# $env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; .\venv\Scripts\python.exe scr/stage3_analysis.py


"""
SmartHomeEnergy Project
Stage 3: Data Analysis and Visualization
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_URL = (
    "https://raw.githubusercontent.com/LuisM78/"
    "Appliances-energy-prediction-data/master/energydata_complete.csv"
)

# 2. Load dataset
LOCAL_DATA = DATA_DIR / "energy.csv"

if LOCAL_DATA.exists() and LOCAL_DATA.stat().st_size > 0:
    df = pd.read_csv(LOCAL_DATA)
else:
    print("Local CSV is empty. Loading real dataset...")
    df = pd.read_csv(DATA_URL)

if "Appliances" not in df.columns:
    print("Loading real energy dataset...")
    df = pd.read_csv(DATA_URL)

# 3. Ensure required columns exist
if "Appliances" not in df.columns:
    print("Practice dataset detected. Loading real dataset...")
    df = pd.read_csv(DATA_URL)

required_columns = [
    "date", "Appliances", "T1", "RH_1", "T2", "RH_2"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# 4. Check missing values and duplicates
print("Original shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# 5. Remove duplicate rows
df = df.drop_duplicates().copy()

# 6. Convert date and remove invalid dates/target values
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["Appliances"] = pd.to_numeric(
    df["Appliances"], errors="coerce"
)
df = df.dropna(subset=["date", "Appliances"]).copy()

# 7. Create time features
df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek

print("\nCleaned shape:", df.shape)
print("\nHourly counts:\n", df["hour"].value_counts().sort_index())
print(
    "\nDay of week counts:\n",
    df["day_of_week"].value_counts().sort_index()
)

# 8. Average consumption by hour
hourly_avg = df.groupby("hour")["Appliances"].mean()

plt.figure(figsize=(9, 5))
hourly_avg.plot(marker="o")
plt.title("Average Appliances Consumption by Hour")
plt.xlabel("Hour (0–23)")
plt.ylabel("Average Appliances (Wh)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "average_by_hour.png", dpi=150)
plt.close()

# 9. Average consumption by weekday
weekday_avg = df.groupby("day_of_week")["Appliances"].mean()

plt.figure(figsize=(9, 5))
weekday_avg.plot(kind="bar")
plt.title("Average Appliances Consumption by Weekday")
plt.xlabel("Day (Monday=0, Sunday=6)")
plt.ylabel("Average Appliances (Wh)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "average_by_weekday.png", dpi=150)
plt.close()

# 10. Correlation analysis
numeric_df = df.select_dtypes(include="number")
correlations = numeric_df.corr(numeric_only=True)["Appliances"]
correlations = correlations.drop("Appliances").sort_values(
    ascending=False
)

print("\nCorrelation with Appliances:\n")
print(correlations.round(3))

plt.figure(figsize=(10, 6))
correlations.sort_values().plot(kind="barh")
plt.title("Feature Correlation with Appliances")
plt.xlabel("Correlation")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_correlations.png", dpi=150)
plt.close()

# 11. Save analysis results
df.to_csv(OUTPUT_DIR / "cleaned_energy_data.csv", index=False)
correlations.to_csv(OUTPUT_DIR / "appliances_correlations.csv")

print("\nStage 3 analysis completed.")
print("Graphs and reports saved in:", OUTPUT_DIR)



PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Load cleaned dataset
df = pd.read_csv(OUTPUT_DIR / "cleaned_energy_data.csv")

# Calculate average appliance consumption by hour
hourly_energy = df.groupby("hour")["Appliances"].mean()

print("\nAverage Appliance Consumption by Hour:")
print(hourly_energy.round(2))

# Plot the hourly average
hourly_energy.plot(kind="bar", figsize=(12, 5))

plt.title("Average Appliance Energy Consumption by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Appliances (Wh)")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "hourly_energy_analysis.png")
plt.show()

print("\nHourly analysis completed successfully!")

