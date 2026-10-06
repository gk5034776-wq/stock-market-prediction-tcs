import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("data/TCS_features.csv")

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)
data = data.dropna().reset_index(drop=True)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# Final 10 features
features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "MA_5",
    "MA_20",
    "Volatility_5",
    "RSI_14"
]

print("\nFeatures used:")
print(features)

# Prepare data
X = data[features]
y = data["Target"]

# Chronological 80/20 split
split = int(len(data) * 0.8)

X_train = X.iloc[:split]
y_train = y.iloc[:split]

print("\nTraining data:", X_train.shape)

# Train final Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model bundle
model_bundle = {
    "model": model,
    "features": features
}

with open("tcs_linear_regression_model.pkl", "wb") as file:
    pickle.dump(model_bundle, file)

print("\nLinear Regression model trained successfully!")
print("Trained model saved successfully!")

# Latest available row
latest_row = data.iloc[-1]

latest_date = latest_row["Date"]
latest_close = latest_row["Close"]

X_latest = data.loc[[data.index[-1]], features]

prediction = model.predict(X_latest)[0]

print("\n========================================")
print("       NEXT-DAY STOCK PREDICTION")
print("========================================")
print("Latest Available Date:", latest_date.date())
print(f"Latest Actual Close: ₹{latest_close:.2f}")
print("----------------------------------------")
print(f"Predicted Next-Day Close: ₹{prediction:.2f}")
print("========================================")

# Compare prediction with actual next-day price
latest_index = data.index[-1]

if latest_index + 1 < len(data):
    actual_next_day = data.iloc[latest_index + 1]["Close"]

    error = actual_next_day - prediction
    absolute_error = abs(error)
    percentage_error = (absolute_error / actual_next_day) * 100

    print("\n========================================")
    print("       PREDICTION vs ACTUAL")
    print("========================================")
    print(f"Predicted Price: ₹{prediction:.2f}")
    print(f"Actual Price:    ₹{actual_next_day:.2f}")
    print(f"Absolute Error:  ₹{absolute_error:.2f}")
    print(f"Percentage Error: {percentage_error:.2f}%")
    print("========================================")
else:
    print("\nActual next-day price is not available yet.")