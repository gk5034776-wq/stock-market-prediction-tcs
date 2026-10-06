import pandas as pd
from sklearn.linear_model import LinearRegression

# Load feature dataset
data = pd.read_csv("data/TCS_features.csv")

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)
data = data.dropna().reset_index(drop=True)

# Features used by the final model
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

# Prepare data
X = data[features]
y = data["Target"]

# Chronological 80/20 split
split = int(len(data) * 0.8)

X_train = X.iloc[:split]
y_train = y.iloc[:split]

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Generate predictions for validation period
X_test = X.iloc[split:]
y_test = y.iloc[split:]

predictions = model.predict(X_test)

# Create prediction history table
prediction_history = pd.DataFrame({
    "Date": data["Date"].iloc[split:].values,
    "Actual_Close": y_test.values,
    "Predicted_Close": predictions
})

# Calculate errors
prediction_history["Absolute_Error"] = (
    prediction_history["Actual_Close"]
    - prediction_history["Predicted_Close"]
).abs()

prediction_history["Error_Percentage"] = (
    prediction_history["Absolute_Error"]
    / prediction_history["Actual_Close"]
) * 100

# Round values
prediction_history["Actual_Close"] = prediction_history["Actual_Close"].round(2)
prediction_history["Predicted_Close"] = prediction_history["Predicted_Close"].round(2)
prediction_history["Absolute_Error"] = prediction_history["Absolute_Error"].round(2)
prediction_history["Error_Percentage"] = prediction_history["Error_Percentage"].round(2)

# Save results
prediction_history.to_csv(
    "data/prediction_history.csv",
    index=False
)

print("Prediction history generated successfully!")
print("Records:", len(prediction_history))
print("\nLatest predictions:")
print(prediction_history.tail(10).to_string(index=False))

print("\nSaved as: data/prediction_history.csv")