# classification.py
# -----------------------------------
# Predict Pass / Fail using Classification
# -----------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import joblib


def load_data(filepath):
    """
    Load cleaned student performance data
    """
    return pd.read_csv(filepath)


def prepare_features(df):
    """
    Prepare features and classification target
    """
    features = [
        "attendance_percentage",
        "study_hours",
        "midterm_score"
    ]

    # Create binary target: Pass (1) / Fail (0)
    # Adjust threshold if needed
    df["pass_fail"] = np.where(df["final_score"] >= 40, 1, 0)

    X = df[features]
    y = df["pass_fail"]

    return X, y


def train_model(X, y):
    """
    Train Logistic Regression model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate classification model
    """
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print("Classification Model Performance")
    print("--------------------------------")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


def save_model(model, path):
    """
    Save trained model
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def main():
    data_path = "data/processed/merged_data.csv"
    model_path = "models/classification_model.pkl"

    # Load data
    df = load_data(data_path)

    # Prepare features
    X, y = prepare_features(df)

    # Train model
    model, X_test, y_test = train_model(X, y)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, model_path)


if __name__ == "__main__":
    main()