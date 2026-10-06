import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle
import matplotlib.pyplot as plt


# -----------------------------------------
# 1. Load Feature Dataset
# -----------------------------------------

data = pd.read_csv("data/TCS_features.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# -----------------------------------------
# 2. Convert Date
# -----------------------------------------

data["Date"] = pd.to_datetime(data["Date"])

data = data.dropna()
# -----------------------------------------
# 3. Select Features
# -----------------------------------------

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
X = data[features]

# Target = next day's closing price
y = data["Target"]


# -----------------------------------------
# 4. Train-Test Split
# -----------------------------------------

# Time-series data mein random splitting nahi karenge.
# Pehle 80% = Training
# Last 20% = Testing

split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_dates = data["Date"].iloc[split_index:]


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# -----------------------------------------
# 5. Create Random Forest Model
# -----------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# -----------------------------------------
# 6. Train Model
# -----------------------------------------

model.fit(X_train, y_train)

# Save Random Forest model with feature names
model_bundle = {
    "model": model,
    "features": features
}

with open("tcs_random_forest_model.pkl", "wb") as file:
    pickle.dump(model_bundle, file)

print("Saved Random Forest model successfully!")
print("\nRandom Forest training completed successfully!")


# -----------------------------------------
# 7. Make Predictions
# -----------------------------------------

predictions = model.predict(X_test)


# -----------------------------------------
# 8. Calculate Performance
# -----------------------------------------

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

r2 = r2_score(y_test, predictions)


print("\nRandom Forest Model Performance:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# -----------------------------------------
# 9. Actual vs Predicted
# -----------------------------------------

results = pd.DataFrame({
    "Date": test_dates.values,
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

print("\nActual vs Predicted:")
print(results.head(10))


# -----------------------------------------
# 10. Plot Graph
# -----------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    results["Date"],
    results["Actual Price"],
    label="Actual Price"
)

plt.plot(
    results["Date"],
    results["Predicted Price"],
    label="Random Forest Prediction"
)

plt.title("TCS Actual vs Random Forest Predicted Price")

plt.xlabel("Date")

plt.ylabel("Price (₹)")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()