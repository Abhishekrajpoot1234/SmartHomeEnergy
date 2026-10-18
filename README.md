# Smart Energy Consumption Monitoring & Prediction System

## About the Project
This project analyzes household energy consumption data and uses Machine Learning to predict appliance energy consumption.

The project is built using Python, Pandas, NumPy, Matplotlib, Scikit-learn, and Streamlit.

## Features
- Load and clean energy consumption data
- Analyze energy consumption patterns
- Visualize consumption trends
- Train and compare Machine Learning models
- Evaluate predictions using MAE, RMSE, and R²
- Analyze prediction errors
- View results through an interactive Streamlit dashboard
- Download datasets and analysis results as CSV files

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Git and GitHub (for version control and sharing)

## Project Structure
```text
SmartHomeEnergy/
├── data/
├── outputs/
├── scr/
├── app.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone or download this repository.
2. Open the project folder in a terminal.
3. Create a virtual environment:

   `python -m venv venv`

4. Install the required libraries:

   `python -m pip install -r requirements.txt`

## Run the Dashboard

From the project root folder, run:

`python -m streamlit run app.py`

## Model Evaluation
The project compares multiple Machine Learning models using validation data and evaluates the selected model on a separate test dataset.

**Note:** The current model still needs improvement. Its test results show that prediction accuracy is not yet reliable enough for real-world use.

## Dataset
The project uses household appliance energy consumption data. If the local dataset is unavailable, the data-loading scripts may retrieve it from the configured online source.

## Future Improvements
- Improve model accuracy
- Add more informative dashboard visualizations
- Experiment with additional features and models
- Improve prediction performance during high-consumption periods
