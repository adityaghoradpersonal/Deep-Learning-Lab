import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import json
import pickle

# =========================
# STEP 1: LOAD CSV DATA
# =========================

df = pd.read_csv("DailyDelhiClimateTrain.csv")

# Use mean temperature
data = df[['meantemp']].dropna()

print(data.head())

# =========================
# STEP 2: NORMALIZE DATA
# =========================

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data.values)

# =========================
# STEP 3: CREATE SEQUENCES
# =========================

X = []
y = []
time_steps = 30

for i in range(time_steps, len(scaled_data)):
    X.append(scaled_data[i-time_steps:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

# =========================
# STEP 4: TRAIN-TEST SPLIT
# =========================

split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================
# STEP 5: BUILD LSTM MODEL
# =========================

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

# =========================
# STEP 6: COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# =========================
# STEP 7: TRAIN MODEL
# =========================

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# =========================
# STEP 8: PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# Convert back to original values
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

# =========================
# STEP 9: EVALUATE MODEL
# =========================

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
print("RMSE:", rmse)

# =========================
# STEP 10: PLOT RESULTS
# =========================

plt.figure(figsize=(10,6))
plt.plot(y_test_actual, label="Actual Temperature")
plt.plot(y_pred, label="Predicted Temperature")
plt.title("Weather Prediction using LSTM")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()
plt.savefig("weather_prediction.png")
plt.show()

# =========================
# STEP 11: SAVE MODEL
# =========================

model.save("weather_lstm_model.keras")

# =========================
# STEP 12: SAVE RESULTS
# =========================

results = {
    "rmse": float(rmse)
}

with open("weather_results.json", "w") as f:
    json.dump(results, f)

with open("weather_predictions.pkl", "wb") as f:
    pickle.dump(y_pred, f)

# =========================
# STEP 13: PREDICT NEXT DAY
# =========================

last_sequence = scaled_data[-30:]
last_sequence = last_sequence.reshape(1, 30, 1)

next_day_pred = model.predict(last_sequence)
next_day_pred = scaler.inverse_transform(next_day_pred)

print("Next Day Predicted Temperature:", next_day_pred[0][0])