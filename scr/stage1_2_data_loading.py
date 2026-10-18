
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Dataset URL
DATA_URL = "https://raw.githubusercontent.com/LuisM78/Appliances-energy-prediction-data/master/energydata_complete.csv"

# Step 2: Load dataset
try:
    energy_df = pd.read_csv(DATA_URL)
    print("Dataset loaded successfully!")
except Exception as exc:
    raise RuntimeError(
        "Dataset load nahi hua. Internet connection check karo."
    ) from exc

# Step 3: Display first 5 rows
print("\nFirst 5 rows:")
print(energy_df.head())

# Step 4: Check rows and columns
print("\nDataset shape:")
print(energy_df.shape)

# Step 5: Display column names
print("\nColumn names:")
print(energy_df.columns.tolist())

# Step 6: Check data types and non-null values
print("\nDataset information:")
energy_df.info()

# Step 7: Statistical summary
print("\nStatistical summary:")
print(energy_df.describe())

# Step 8: Check missing values
print("\nMissing values:")
print(energy_df.isnull().sum())

# Step 9: Check duplicate rows
print("\nDuplicate rows:")
print(energy_df.duplicated().sum())

# Step 10: Plot energy consumption distribution
plt.figure(figsize=(8, 5))
plt.hist(energy_df["Appliances"], bins=30)
plt.title("Appliances Energy Consumption Distribution")
plt.xlabel("Appliances consumption (Wh)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
