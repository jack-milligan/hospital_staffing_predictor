import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


def create_images_folder(folder: str = "images") -> None:
    """Create a folder for saving images if it does not exist."""
    os.makedirs(folder, exist_ok=True)


def simulate_data(seed: int = 42) -> pd.DataFrame:
    """Simulate hourly patient volume and staffing levels for one year."""
    np.random.seed(seed)

    # Define the time range: hourly data for the year 2024.
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')

    # Extract time components for trend simulation.
    hour_array = date_range.hour.values
    day_of_year_array = date_range.dayofyear.values

    # Generate synthetic patient volume.
    patient_volume = (
            50  # Base patient count
            + 10 * np.sin(2 * np.pi * hour_array / 24)  # Daily pattern
            + 5 * np.cos(2 * np.pi * day_of_year_array / 365)  # Seasonal pattern
            + np.random.normal(0, 5, len(date_range))  # Random noise
    ).clip(0)  # Ensure non-negative values

    # Calculate staffing levels based on a 10:1 patient-to-staff ratio.
    staffing_levels = (patient_volume / 10).round().clip(1)

    # Create a DataFrame.
    df = pd.DataFrame({
        'timestamp': date_range,
        'patient_volume': patient_volume,
        'staffing_levels': staffing_levels
    })

    # Set the timestamp as the index.
    df.set_index('timestamp', inplace=True)
    return df


def visualize_simulated_data(df: pd.DataFrame, image_path: str = "images/patient_staffing_trend.png") -> None:
    """Plot and save the simulated patient volumes and staffing levels over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['patient_volume'], label='Patient Volume', alpha=0.7)
    plt.plot(df['staffing_levels'], label='Staffing Levels', linestyle='--', alpha=0.7)
    plt.title('Simulated Patient Volumes and Staffing Levels Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved to: {image_path}")
    plt.close()


def perform_adf_test(df: pd.DataFrame, column: str = 'patient_volume') -> None:
    """Perform the Augmented Dickey-Fuller test and print the results."""
    result = adfuller(df[column])
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    if result[1] < 0.05:
        print("The time series is stationary (no differencing needed).")
    else:
        print("The time series is non-stationary. Differencing may be required for ARIMA.")


def forecast_patient_volume(df: pd.DataFrame, periods: int = 7 * 24) -> (Prophet, pd.DataFrame):
    """
    Prepare data for Prophet, fit the forecasting model, and predict future patient volumes.

    Parameters:
        df: DataFrame containing patient_volume time series.
        periods: Number of future periods (hours) to forecast.

    Returns:
        model: Fitted Prophet model.
        forecast: DataFrame containing forecasted values.
    """
    # Prepare data for Prophet (columns 'ds' and 'y').
    prophet_df = df['patient_volume'].reset_index()
    prophet_df.columns = ['ds', 'y']

    model = Prophet()
    model.fit(prophet_df)

    # Create a dataframe for future predictions.
    future = model.make_future_dataframe(periods=periods, freq='h')
    forecast = model.predict(future)
    return model, forecast


def plot_forecast(model: Prophet, forecast: pd.DataFrame,
                  image_path: str = "images/patient_volume_forecast.png") -> None:
    """Plot and save the forecasted patient volume."""
    model.plot(forecast)
    plt.xlabel("Time")
    plt.ylabel("Volume")
    plt.title("Forecasted Patient Volume for Next 7 Days")
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Forecast visualization saved to: {image_path}")


def predict_hourly_staffing_levels(forecast: pd.DataFrame, ratio: int = 10) -> pd.DataFrame:
    """
    Group forecasted patient volume by hour and calculate recommended staffing levels.

    Parameters:
        forecast: DataFrame with Prophet forecast (must include 'ds' and 'yhat').
        ratio: Patient-to-staff ratio used for recommendation.

    Returns:
        hourly_forecast_df: DataFrame containing average forecasted patient volume and recommended staffing by hour.
    """
    forecast['hour'] = forecast['ds'].dt.hour
    hourly_forecast = forecast.groupby('hour')['yhat'].mean()
    hourly_forecast_df = pd.DataFrame(hourly_forecast)
    hourly_forecast_df['recommended_staffing'] = (hourly_forecast_df['yhat'] / ratio).round().clip(lower=1)

    print("\nHourly Staffing Recommendations:")
    print(hourly_forecast_df)
    return hourly_forecast_df


def save_staffing_recommendations_csv(hourly_forecast_df: pd.DataFrame,
                                      csv_filename: str = "hourly_staffing_recommendations.csv") -> None:
    """Save the hourly staffing recommendations to a CSV file."""
    hourly_forecast_df.to_csv(csv_filename, index=True)
    print(f"\nHourly staffing recommendations saved to: {csv_filename}")


def plot_hourly_staffing_recommendations(hourly_forecast_df: pd.DataFrame,
                                         image_path: str = "images/hourly_staffing_recommendations.png") -> None:
    """Plot and save hourly recommended staffing levels based on forecasted demand."""
    plt.figure(figsize=(10, 5))
    plt.plot(hourly_forecast_df.index, hourly_forecast_df['recommended_staffing'],
             marker='o', linestyle='-', label='Recommended Staffing')
    plt.title("Hourly Recommended Staffing Levels Based on Forecasted Demand")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Recommended Staff Count")
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.legend()
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Hourly staffing recommendations visualization saved to: {image_path}")
    plt.close()


def evaluate_model(df: pd.DataFrame, forecast: pd.DataFrame) -> None:
    """
    Evaluate the forecasting model using a naive baseline and the Prophet forecast.

    Compares the patient_volume time series against:
      - A naive forecast (previous value).
      - Prophet forecast values (matched by timestamp).
    """
    # --- Naive Forecast Evaluation ---
    naive_forecast = df['patient_volume'].shift(1)
    valid_rows = df['patient_volume'].notna() & naive_forecast.notna()
    naive_mae = mean_absolute_error(df['patient_volume'][valid_rows], naive_forecast[valid_rows])
    naive_rmse = np.sqrt(mean_squared_error(df['patient_volume'][valid_rows], naive_forecast[valid_rows]))

    print("\nNaive Model Baseline Evaluation:")
    print(f"Naïve Model MAE: {naive_mae:.2f}")
    print(f"Naïve Model RMSE: {naive_rmse:.2f}")

    # --- Prophet Model Evaluation ---
    # Merge actual values with the forecasted yhat by matching timestamps.
    df_reset = df.reset_index()  # bring 'timestamp' back as a column
    actual_vs_pred = pd.merge(df_reset, forecast[['ds', 'yhat']],
                              left_on='timestamp', right_on='ds', how='inner')

    mae = mean_absolute_error(actual_vs_pred['patient_volume'], actual_vs_pred['yhat'])
    mse = mean_squared_error(actual_vs_pred['patient_volume'], actual_vs_pred['yhat'])
    rmse = np.sqrt(mse)

    print("\nProphet Model Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


def plot_recommended_staffing_levels(forecast: pd.DataFrame,
                                     image_path: str = "images/recommended_staffing_levels.png",
                                     ratio: int = 10) -> None:
    """
    Plot and save a comparison between the forecasted patient volume and the recommended staffing levels.

    Parameters:
        forecast: DataFrame with Prophet forecast.
        image_path: File path for the saved image.
        ratio: Patient-to-staff ratio.
    """
    # Calculate recommended staffing levels using the forecast.
    forecast['recommended_staffing'] = (forecast['yhat'] / ratio).round().clip(lower=1)

    plt.figure(figsize=(12, 6))
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Patient Volume', alpha=0.7)
    plt.plot(forecast['ds'], forecast['recommended_staffing'],
             label='Recommended Staffing Levels', linestyle='--', alpha=0.7)
    plt.title("Recommended Staffing Levels Based on Predicted Patient Volume")
    plt.xlabel("Timestamp")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Recommended staffing levels visualization saved to: {image_path}")
    plt.close()


def main() -> None:
    # Setup: Create necessary folders.
    create_images_folder()

    # Step 1: Data Simulation.
    df = simulate_data()
    print("Simulated Data Sample:")
    print(df.head())

    # Step 2: Visualize the simulated data.
    visualize_simulated_data(df)

    # Step 3: Stationarity Test.
    perform_adf_test(df)

    # Step 4: Forecasting using Prophet.
    model, forecast = forecast_patient_volume(df)
    plot_forecast(model, forecast)

    # Step 5: Predicting Optimal Staffing Levels.
    hourly_forecast_df = predict_hourly_staffing_levels(forecast)
    save_staffing_recommendations_csv(hourly_forecast_df)
    plot_hourly_staffing_recommendations(hourly_forecast_df)

    # Step 6: Model Evaluation.
    evaluate_model(df, forecast)

    # Step 7: Plotting Recommended Staffing Levels Over Time.
    plot_recommended_staffing_levels(forecast)


if __name__ == "__main__":
    main()
