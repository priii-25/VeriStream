import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error

data = pd.read_csv('/content/indian_misinformation_dataset.csv')

data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by='date')

time_series_data = data.groupby('date')['spread_intensity'].sum()

plt.figure(figsize=(10, 6))
plt.plot(time_series_data, label='Original Spread Intensity', color='blue')
plt.title('Spread Intensity Over Time')
plt.xlabel('Date')
plt.ylabel('Spread Intensity')
plt.legend()
plt.show()

train_size = int(len(time_series_data) * 0.8)
train_data = time_series_data[:train_size]
test_data = time_series_data[train_size:]

model = ARIMA(train_data, order=(5, 1, 0))
model_fit = model.fit()

print(model_fit.summary())

forecast_steps = len(test_data)
forecast = model_fit.forecast(steps=forecast_steps)

plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data, label='Actual Spread Intensity', color='blue')
plt.plot(test_data.index, forecast, label='Forecasted Spread Intensity', color='orange')
plt.title('Actual vs Forecasted Spread Intensity')
plt.xlabel('Date')
plt.ylabel('Spread Intensity')
plt.legend()
plt.show()

mse = mean_squared_error(test_data, forecast)
print(f"Mean Squared Error: {mse}")

future_forecast = model_fit.forecast(steps=30)
future_dates = pd.date_range(start=time_series_data.index[-1], periods=30, freq='D')

plt.figure(figsize=(10, 6))
plt.plot(time_series_data, label='Historical Spread Intensity', color='blue')
plt.plot(future_dates, future_forecast, label='Future Forecast', color='green')
plt.title('Future Spread Intensity Forecast')
plt.xlabel('Date')
plt.ylabel('Spread Intensity')
plt.legend()
plt.show()

future_forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Spread Intensity': future_forecast})
future_forecast_df.to_csv('future_forecast.csv', index=False)
print("Future forecasts saved to 'future_forecast.csv'")