# regression.py
# -----------------------------------
# Predict student final score using Regression
# -----------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import joblib


def load_data(filepath):
    """
    Load cleaned student performance data
    """
    return pd.read_csv(filepath)


def prepare_features(df):
    """
    Select features and target variable
    """
    # Example features (change if needed)
    features = [
        "attendance_percentage",
        "study_hours",
        "midterm_score"
    ]

    target = "final_score"

    X = df[features]
    y = df[target]

    return X, y


def train_model(X, y):
    """
    Train Linear Regression model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("Regression Model Performance")
    print("-----------------------------")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")


def save_model(model, path):
    """
    Save trained model to disk
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def main():
    # Path to processed dataset
    data_path = "data/processed/merged_data.csv"
    model_path = "models/regression_model.pkl"

    # Load and prepare data
    df = load_data(data_path)
    X, y = prepare_features(df)

    # Train model
    model, X_test, y_test = train_model(X, y)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, model_path)


if __name__ == "__main__":
    main()
# regression.py
# -----------------------------------