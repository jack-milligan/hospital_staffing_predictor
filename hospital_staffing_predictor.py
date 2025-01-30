# Import necessary libraries
import os  # For directory creation
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For data visualization
from datetime import datetime  # For date handling
from statsmodels.tsa.stattools import adfuller  # For stationarity testing
from prophet import Prophet  # For time-series forecasting

# --- Step 1: Setup & Data Simulation ---

# Set a random seed for reproducibility of results
np.random.seed(42)

# Create an "images" folder if it doesn't exist (to save generated plots)
os.makedirs("images", exist_ok=True)

# Define the time range: Generate hourly data for one full year (2024)
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='h')

# Extract hour and day of the year as NumPy arrays for trend simulation
hour_array = date_range.hour.values
day_of_year_array = date_range.dayofyear.values

# Generate synthetic patient volume data
patient_volume = (
        50  # Base number of patients per hour
        + 10 * np.sin(2 * np.pi * hour_array / 24)  # Daily pattern: More patients during peak hours
        + 5 * np.cos(2 * np.pi * day_of_year_array / 365)  # Seasonal pattern: Higher demand in flu season
        + np.random.normal(0, 5, len(date_range))  # Add random noise
).clip(0)  # Ensure patient volumes are non-negative

# Simulate staffing levels based on a base patient-to-staff ratio of 10
staffing_levels = (patient_volume / 10).round().clip(1)

# Create a DataFrame to store simulated data
df = pd.DataFrame({
    'timestamp': date_range,
    'patient_volume': patient_volume,
    'staffing_levels': staffing_levels
})

# Set the timestamp as the index for time-series analysis
df.set_index('timestamp', inplace=True)

# Display the first few rows of the dataset
print(df.head())

# --- Step 2: Visualizing Simulated Data ---

# Create a plot to visualize patient volumes and staffing levels over time
plt.figure(figsize=(12, 6))
plt.plot(df['patient_volume'], label='Patient Volume', alpha=0.7)
plt.plot(df['staffing_levels'], label='Staffing Levels', linestyle='--', alpha=0.7)
plt.title('Simulated Patient Volumes and Staffing Levels Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Count')
plt.legend()

# Save the plot as an image
image_path = "images/patient_staffing_trend.png"
plt.savefig(image_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Visualization saved to: {image_path}")

# --- Step 3: Stationarity Test (Required for ARIMA) ---

# Conduct an Augmented Dickey-Fuller (ADF) test to check if the time series is stationary
result = adfuller(df['patient_volume'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# Interpretation of the results
if result[1] < 0.05:
    print("The time series is stationary (no differencing needed).")
else:
    print("The time series is non-stationary. Differencing may be required for ARIMA.")

# --- Step 4: Forecasting Patient Volumes with Prophet ---

# Prepare the data for Prophet (Prophet requires column names 'ds' for datetime and 'y' for values)
prophet_df = df['patient_volume'].reset_index()
prophet_df.columns = ['ds', 'y']

# Initialize and fit the Prophet forecasting model
model = Prophet()
model.fit(prophet_df)

# Create a future dataframe for predictions (next 7 days, hourly intervals)
future = model.make_future_dataframe(periods=7 * 24, freq='h')  # Forecast for the next 7 days
forecast = model.predict(future)

# Plot the forecasted patient volume
plt.figure(figsize=(12, 6))
model.plot(forecast)
plt.title("Forecasted Patient Volume for Next 7 Days")

# Save the forecast plot
forecast_image_path = "images/patient_volume_forecast.png"
plt.savefig(forecast_image_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Forecast visualization saved to: {forecast_image_path}")

# --- Step 5: Predicting Optimal Staffing Levels ---
# Extract hour from the forecasted timestamps
forecast['hour'] = forecast['ds'].dt.hour

# Group by hour of the day to calculate the average patient volume over the next 7 days
hourly_forecast = forecast.groupby('hour')['yhat'].mean()

# Calculate recommended staffing levels based on patient-to-staff ratio (10:1)
hourly_forecast_df = pd.DataFrame(hourly_forecast)
hourly_forecast_df['recommended_staffing'] = (hourly_forecast_df['yhat'] / 10).round().clip(lower=1)

# Print the table in the console
print("\nHourly Staffing Recommendations:")
print(hourly_forecast_df)

# Save the table to a CSV file for future reference
csv_filename = "hourly_staffing_recommendations.csv"
hourly_forecast_df.to_csv(csv_filename, index=True)
print(f"\nHourly staffing recommendations saved to: {csv_filename}")

# Plot hourly staffing recommendations
plt.figure(figsize=(10, 5))
plt.plot(hourly_forecast_df.index, hourly_forecast_df['recommended_staffing'], marker='o', linestyle='-', label='Recommended Staffing')
plt.title("Hourly Recommended Staffing Levels Based on Forecasted Demand")
plt.xlabel("Hour of the Day")
plt.ylabel("Recommended Staff Count")
plt.xticks(range(0, 24))
plt.grid(True)
plt.legend()

# Save the hourly staffing recommendations plot
hourly_staffing_image_path = "images/hourly_staffing_recommendations.png"
plt.savefig(hourly_staffing_image_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\nHourly staffing recommendations visualization saved to: {hourly_staffing_image_path}")

# --- Step 6: Plotting Recommended Staffing Levels Over Time ---

# Calculate recommended staffing levels using a patient-to-staff ratio of 10
forecast['recommended_staffing'] = (forecast['yhat'] / 10).round().clip(lower=1)

# Plot recommended staffing levels based on predicted patient volume
plt.figure(figsize=(12, 6))
plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Patient Volume', alpha=0.7)
plt.plot(forecast['ds'], forecast['recommended_staffing'], label='Recommended Staffing Levels', linestyle='--', alpha=0.7)
plt.title("Recommended Staffing Levels Based on Predicted Patient Volume")
plt.xlabel("Timestamp")
plt.ylabel("Count")
plt.legend()

# Save the recommended staffing levels plot
staffing_image_path = "images/recommended_staffing_levels.png"
plt.savefig(staffing_image_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Recommended staffing levels visualization saved to: {staffing_image_path}")
