import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# 10 Indian stocks
stocks = [
    "POWERGRID.NS",
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "LT.NS",
    "ITC.NS",
    "BHARTIARTL.NS"
]

x_train = []
y_train = []

scaler = MinMaxScaler(feature_range=(0, 1))

print("Downloading & preparing data...")

for stock in stocks:
    print(f"Processing {stock}")
    
    df = yf.download(stock, start="2000-01-01", end="2024-10-01")
    close_prices = df[['Close']].dropna()

    scaled = scaler.fit_transform(close_prices)

    for i in range(100, len(scaled)):
        x_train.append(scaled[i-100:i])
        y_train.append(scaled[i])

x_train = np.array(x_train)
y_train = np.array(y_train)

print("Training samples:", x_train.shape)

# Build model (1 feature only)
model = Sequential([
    Input(shape=(100, 1)),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

print("Training model...")
model.fit(x_train, y_train, epochs=5, batch_size=32)

model.save("stock_dl_model.h5")
print("Model saved successfully!")