"""
Smart Energy Consumption Monitoring & Prediction System
Stage 5-6: Model Training and Evaluation
"""

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load training and testing data
X_train_full = pd.read_csv(OUTPUT_DIR / "X_train.csv")
X_test = pd.read_csv(OUTPUT_DIR / "X_test.csv")

y_train_full = pd.read_csv(
    OUTPUT_DIR / "y_train.csv"
).squeeze("columns")

y_test = pd.read_csv(
    OUTPUT_DIR / "y_test.csv"
).squeeze("columns")

# 3. Create chronological training and validation sets
# Earlier 80% of the training data is used for training.
# Later 20% is used for validation.
split_index = int(len(X_train_full) * 0.8)

X_train = X_train_full.iloc[:split_index].copy()
y_train = y_train_full.iloc[:split_index].copy()

X_val = X_train_full.iloc[split_index:].copy()
y_val = y_train_full.iloc[split_index:].copy()

print("\nTraining data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Final testing data:", X_test.shape)

print("\nTraining columns:")
print(X_train.columns.tolist())

print("\nTraining target statistics:")
print(y_train.describe())

print("\nValidation target statistics:")
print(y_val.describe())

print("\nMissing values in training data:", X_train.isnull().sum().sum())
print("Missing values in validation data:", X_val.isnull().sum().sum())
print("Missing values in testing data:", X_test.isnull().sum().sum())

# 4. Create baseline predictions using training mean
baseline_prediction = np.full(
    len(y_val),
    y_train.mean()
)

baseline_mae = mean_absolute_error(
    y_val, baseline_prediction
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_val, baseline_prediction)
)

baseline_r2 = r2_score(
    y_val, baseline_prediction
)

print("\nValidation Baseline:")
print("MAE:", round(baseline_mae, 3))
print("RMSE:", round(baseline_rmse, 3))
print("R2 Score:", round(baseline_r2, 3))

# 5. Define machine learning models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

# 6. Initialize validation results
results = [{
    "Model": "Baseline",
    "MAE": baseline_mae,
    "RMSE": baseline_rmse,
    "R2 Score": baseline_r2
}]

# Store validation predictions
validation_predictions = {}

# 7. Train and evaluate models using validation data
for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    validation_predictions[model_name] = y_val_pred

    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)

    print("Validation MAE:", round(mae, 3))
    print("Validation RMSE:", round(rmse, 3))
    print("Validation R2:", round(r2, 3))

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

# 8. Feature importance using validation data
# This is an exploratory diagnostic, not proof of causation.
print("\nChecking feature importance...")

rf_model = models["Random Forest"]

importance = permutation_importance(
    rf_model,
    X_val,
    y_val,
    n_repeats=3,
    random_state=42,
    scoring="neg_mean_absolute_error",
    n_jobs=1
)

importance_df = pd.DataFrame({
    "Feature": X_val.columns,
    "Importance": importance.importances_mean
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df.round(3).to_string(index=False))

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

# 9. Display and save validation model comparison
results_df = pd.DataFrame(results)

print("\nValidation Model Comparison:")
print(results_df.round(3).to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

# 10. Select best ML model using validation MAE
ml_results_df = results_df[
    results_df["Model"] != "Baseline"
]

best_model_name = ml_results_df.loc[
    ml_results_df["MAE"].idxmin(),
    "Model"
]

print("\nBest ML model based on validation MAE:", best_model_name)

# 11. Retrain the selected model on ALL original training data
# The final test data remains untouched until this step.
best_model = models[best_model_name]

best_model.fit(X_train_full, y_train_full)

# 12. Evaluate the selected model on the final test set
y_test_pred = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("\nFinal Test Results:")
print("Best model:", best_model_name)
print("Test MAE:", round(test_mae, 3))
print("Test RMSE:", round(test_rmse, 3))
print("Test R2 Score:", round(test_r2, 3))

# Save final test metrics separately
test_metrics_df = pd.DataFrame([{
    "Best Model": best_model_name,
    "Test MAE": test_mae,
    "Test RMSE": test_rmse,
    "Test R2 Score": test_r2
}])

test_metrics_df.to_csv(
    OUTPUT_DIR / "final_test_metrics.csv",
    index=False
)

# 13. Save final test predictions
prediction_df = pd.DataFrame({
    "Actual Appliances": y_test.to_numpy(),
    "Predicted Appliances": y_test_pred
})

prediction_df.to_csv(
    OUTPUT_DIR / "best_model_predictions.csv",
    index=False
)

print("\nFirst 20 final test predictions:")
print(prediction_df.head(20).round(2).to_string(index=False))

print("\nActual test target distribution:")
print(y_test.describe())

print("\nResults saved in:", OUTPUT_DIR)
print("\nStage 5-6 completed successfully!")