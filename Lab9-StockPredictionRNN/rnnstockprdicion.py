# Run once if yfinance not installed 
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

data = yf.download("AAPL", start = "2015-01-01", end = "2024-01-10")
df = data[['Close']]
print(df.head())