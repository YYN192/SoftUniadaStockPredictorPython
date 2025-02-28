import os
import math
import datetime
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Apply a clean style to plots
style.use('ggplot')

# Fetch stock data
try:
    df = yf.download('GOOGL')
    print("Data successfully fetched!")
except Exception as e:
    print(f"Error fetching stock data: {e}")
    exit()

# Ensure single-level columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

# Use 'Adj Close' if available; otherwise, use 'Close'
close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

# Select required columns
df = df[['Open', 'High', 'Low', close_col, 'Volume']].copy()

# Calculate additional features
df['HL_PCT'] = (df['High'] - df[close_col]) / df[close_col] * 100.0
df['PCT_change'] = (df[close_col] - df['Open']) / df['Open'] * 100.0

# Final DataFrame for training
df = df[[close_col, 'HL_PCT', 'PCT_change', 'Volume']]

# Define forecast parameters
forecast_out = int(math.ceil(0.01 * len(df)))
print(f"Forecasting {forecast_out} days into the future.")

# Label column for prediction
df['label'] = df[close_col].shift(-forecast_out)

# Handle missing data
df.fillna(-9999, inplace=True)

df.dropna(inplace=True)

x = np.array(df.drop(columns=['label']))
y = np.array(df['label'])

print(f"Length of x: {len(x)}, Length of y: {len(y)}")

x = preprocessing.scale(x)

# Ensure forecast_out is valid
if forecast_out >= len(df):
    raise ValueError(f"forecast_out ({forecast_out}) is too large for dataset of size {len(df)}.")

x_lately = x[-forecast_out:]
x = x[:-forecast_out]
y = y[:-forecast_out]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)

# Save model
model_path = 'linearregression.pickle'
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)
print("Model saved successfully!")

# Load trained model
with open(model_path, 'rb') as f:
    clf = pickle.load(f)

# Evaluate model performance
accuracy = clf.score(x_test, y_test)
forecast_set = clf.predict(x_lately)

print(f"Forecasted values:\n{forecast_set}")
print(f"Model Accuracy: {accuracy:.4f}")

# Extend DataFrame with forecasted data
df['Forecast'] = np.nan
last_date = df.index[-1]
next_date = last_date + pd.Timedelta(days=1)

# Append forecasted values to the DataFrame
forecast_dates = [next_date + pd.Timedelta(days=i) for i in range(forecast_out)]
forecast_df = pd.DataFrame({'Forecast': forecast_set}, index=forecast_dates)

df = pd.concat([df, forecast_df])

# Plot historical and predicted prices
plt.figure(figsize=(12, 6))
df[close_col].plot(label="Historical Price")
df['Forecast'].plot(label="Predicted Price", linestyle="dashed")

# Save and display plot
output_folder = 'Graphs'
os.makedirs(output_folder, exist_ok=True)
plt.legend(loc="best")
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Forecasting')
plt.savefig(os.path.join(output_folder, 'GOOGL_forecast.png'), dpi=300)
plt.show()

print(f"Plot saved in {output_folder}/GOOGL_forecast.png")