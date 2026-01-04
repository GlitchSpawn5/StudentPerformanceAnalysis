# time_series.py
# -----------------------------------
# Forecast student attendance trend using Time Series (ARIMA)
# -----------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

import joblib


def load_data(filepath):
    """
    Load cleaned student attendance data
    """
    return pd.read_csv(filepath)


def prepare_time_series(df):
    """
    Prepare time series data
    Assumes columns:
    - date
    - attendance_percentage
    """
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    ts = df.set_index("date")["attendance_percentage"]
    return ts


def train_arima_model(ts, order=(1, 1, 1)):
    """
    Train ARIMA model
    """
    model = ARIMA(ts, order=order)
    fitted_model = model.fit()
    return fitted_model


def forecast(model, steps=10):
    """
    Forecast future attendance
    """
    forecast_values = model.forecast(steps=steps)
    return forecast_values


def evaluate_model(actual, predicted):
    """
    Evaluate forecast using MAE
    """
    mae = mean_absolute_error(actual, predicted)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")


def plot_forecast(ts, forecast_values):
    """
    Plot original time series and forecast
    """
    plt.figure(figsize=(10, 5))
    plt.plot(ts, label="Historical Attendance")
    plt.plot(
        forecast_values.index,
        forecast_values,
        label="Forecast",
        linestyle="--"
    )
    plt.xlabel("Date")
    plt.ylabel("Attendance Percentage")
    plt.title("Attendance Time Series Forecast")
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_model(model, path):
    """
    Save trained ARIMA model
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def main():
    data_path = "data/processed/merged_data.csv"
    model_path = "models/arima_attendance_model.pkl"

    # Load data
    df = load_data(data_path)

    # Prepare time series
    ts = prepare_time_series(df)

    # Train ARIMA model
    model = train_arima_model(ts)

    # Forecast future values
    forecast_values = forecast(model, steps=10)

    print("Forecasted Attendance:")
    print(forecast_values)

    # Plot results
    plot_forecast(ts, forecast_values)

    # Save model
    save_model(model, model_path)


if __name__ == "__main__":
    main()
