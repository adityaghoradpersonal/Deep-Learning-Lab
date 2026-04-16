# Run once (only if not installed)
# pip install yfinance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import json
import pickle

data = yf.download("AAPL", start="2015-01-01", end="2024-01-01")

df = data[['Close']]
print(df.head())

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df.values)

X = []
y = []

time_steps = 60

for i in range(time_steps, len(scaled_data)):
    X.append(scaled_data[i-time_steps:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

print("Shape of X:", X.shape)

split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

y_pred = model.predict(X_test)

# Inverse transform
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
print("RMSE:", rmse)

plt.figure(figsize=(10,6))
plt.plot(y_test_actual, label="Actual Price")
plt.plot(y_pred, label="Predicted Price")
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.savefig("stock_prediction.png")
plt.show()

model.save("stock_lstm_model.keras")

results = {
    "rmse": float(rmse)
}

with open("stock_results.json", "w") as f:
    json.dump(results, f)

with open("stock_predictions.pkl", "wb") as f:
    pickle.dump(y_pred, f)