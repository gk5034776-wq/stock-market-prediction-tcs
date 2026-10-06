import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# TCS feature dataset load karna
data = pd.read_csv("data/TCS_features.csv")

# Date ko datetime format mein convert karna
data["Date"] = pd.to_datetime(data["Date"])

# Features
features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Previous_Close",
    "MA_5",
    "MA_20",
    "Daily_Return",
    "Volatility_5"
]

X = data[features]
y = data["Target"]

# 80% training, 20% testing
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Testing dates
test_dates = data["Date"].iloc[split_index:]

# Graph
plt.figure(figsize=(12, 6))

plt.plot(test_dates, y_test.values, label="Actual Price")
plt.plot(test_dates, predictions, label="Predicted Price")

plt.title("TCS Actual vs Predicted Stock Price")
plt.xlabel("Date")
plt.ylabel("Price (₹)")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()