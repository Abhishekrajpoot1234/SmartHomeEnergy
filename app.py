
#  code run karne ki command
# cd E:\SmartHomeEnergy
# .\venv\Scripts\python.exe -m streamlit run app.py


"""
Smart Energy Consumption Monitoring & Prediction System
Stage 7: Streamlit Dashboard
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

# 2. Page configuration
st.set_page_config(
    page_title="Smart Energy Monitoring",
    page_icon="⚡",
    layout="wide"
)

# 3. Dashboard heading
st.title("⚡ Smart Energy Consumption Monitoring")
st.write(
    "Analyze appliance energy consumption and view "
    "machine-learning prediction results."
)

# 4. Result file paths
comparison_file = OUTPUT_DIR / "model_comparison.csv"
predictions_file = OUTPUT_DIR / "best_model_predictions.csv"
test_metrics_file = OUTPUT_DIR / "final_test_metrics.csv"
cleaned_file = OUTPUT_DIR / "cleaned_energy_data.csv"
importance_file = OUTPUT_DIR / "feature_importance.csv"

# 5. Check required files
required_files = [
    comparison_file,
    predictions_file,
    test_metrics_file
]

missing_files = [
    file.name for file in required_files
    if not file.exists()
]

if missing_files:
    st.warning(
        "Required result files are missing: "
        + ", ".join(missing_files)
    )
    st.info(
        "Run Stage 4 preprocessing and Stage 5-6 "
        "model training first."
    )
    st.stop()

# 6. Load results
results = pd.read_csv(comparison_file)
predictions = pd.read_csv(predictions_file)
test_metrics = pd.read_csv(test_metrics_file)

if results.empty or predictions.empty or test_metrics.empty:
    st.error(
        "One or more result files are empty. "
        "Run Stage 5-6 again."
    )
    st.stop()

# 7. Final test performance
st.header("Final Test Performance")
st.caption(
    "These metrics evaluate the selected model on the "
    "separate final test dataset."
)

final_row = test_metrics.iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Selected Model",
    str(final_row["Best Model"])
)

col2.metric(
    "Test MAE",
    f'{final_row["Test MAE"]:.2f} Wh'
)

col3.metric(
    "Test R² Score",
    f'{final_row["Test R2 Score"]:.3f}'
)

st.metric(
    "Test RMSE",
    f'{final_row["Test RMSE"]:.2f} Wh'
)

st.caption(
    "A negative R² means the model performed worse than "
    "a baseline that predicts the test-set mean."
)

# 8. Validation model comparison
st.header("Validation Model Comparison")
st.caption(
    "Use these validation results to compare models. "
    "They are not the final test results."
)

st.dataframe(
    results.round(3),
    width="stretch",
    hide_index=True
)

# 9. Actual versus predicted graph
st.header("Actual vs Predicted Consumption")

st.write(
    "The graph below shows the first 100 observations "
    "from the final test predictions."
)

plot_data = predictions.head(100)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    plot_data["Actual Appliances"].to_numpy(),
    label="Actual Consumption",
    linewidth=2
)

ax.plot(
    plot_data["Predicted Appliances"].to_numpy(),
    label="Predicted Consumption",
    linewidth=2
)

ax.set_xlabel("Test Sample")
ax.set_ylabel("Appliances Consumption (Wh)")
ax.set_title("Actual vs Predicted Values")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

# 10. Prediction table
st.header("Prediction Details")

st.dataframe(
    predictions.head(100).round(2),
    width="stretch",
    hide_index=True
)

prediction_csv = predictions.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Predictions",
    data=prediction_csv,
    file_name="best_model_predictions.csv",
    mime="text/csv"
)

# 11. Feature importance
if importance_file.exists():
    importance_df = pd.read_csv(importance_file)

    st.header("Feature Importance")
    st.caption(
        "Permutation importance indicates how model performance "
        "changes when a feature is shuffled. It does not prove causation."
    )

    st.dataframe(
        importance_df.round(3),
        width="stretch",
        hide_index=True
    )

# 12. Dataset preview and download
if cleaned_file.exists():
    st.header("Cleaned Dataset")

    cleaned_df = pd.read_csv(cleaned_file)

    st.write("Dataset rows and columns:", cleaned_df.shape)

    st.dataframe(
        cleaned_df.head(10),
        width="stretch",
        hide_index=True
    )

    csv_bytes = cleaned_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Cleaned Dataset",
        data=csv_bytes,
        file_name="cleaned_energy_data.csv",
        mime="text/csv"
    )


    
# Prediction Error Analysis
st.subheader("Prediction Error Analysis")

error_file = OUTPUT_DIR / "prediction_error_analysis.csv"
actual_pred_plot = OUTPUT_DIR / "actual_vs_predicted_analysis.png"
error_dist_plot = OUTPUT_DIR / "prediction_error_distribution.png"

if error_file.exists():
    error_df = pd.read_csv(error_file)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Mean Absolute Error",
            f'{error_df["Absolute Error"].mean():.2f} Wh'
        )

    with col2:
        st.metric(
            "Maximum Absolute Error",
            f'{error_df["Absolute Error"].max():.2f} Wh'
        )

    if actual_pred_plot.exists():
        st.image(
            str(actual_pred_plot),
            caption="Actual vs Predicted Consumption",
            width="stretch"
        )

    if error_dist_plot.exists():
        st.image(
            str(error_dist_plot),
            caption="Distribution of Prediction Errors",
            width="stretch"
        )

    st.download_button(
        label="Download Error Analysis CSV",
        data=error_df.to_csv(index=False).encode("utf-8"),
        file_name="prediction_error_analysis.csv",
        mime="text/csv"
    )
else:
    st.info("Run Stage 7 error analysis to view these results.")


# 13. Footer
st.success("Dashboard loaded successfully.")
