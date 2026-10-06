import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load feature dataset
data = pd.read_csv("data/TCS_features.csv")

# Date sort
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)

# Features
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

# Remove rows with missing values
data = data.dropna().reset_index(drop=True)

# Features and target
X = data[features]
y = data["Target"]

# Walk-forward validation
initial_train_size = int(len(data) * 0.8)

actual_values = []
predicted_values = []

for i in range(initial_train_size, len(data)):

    X_train = X.iloc[:i]
    y_train = y.iloc[:i]

    X_test = X.iloc[i:i+1]
    y_test = y.iloc[i:i+1]

    model = LinearRegression()
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)[0]

    actual_values.append(y_test.iloc[0])
    predicted_values.append(prediction)

# Evaluation
mae = mean_absolute_error(actual_values, predicted_values)
rmse = mean_squared_error(actual_values, predicted_values) ** 0.5
r2 = r2_score(actual_values, predicted_values)

print("\n===== WALK-FORWARD VALIDATION =====")
print("Validation samples:", len(actual_values))
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
# Actual vs Predicted Price Graph

plt.figure(figsize=(12, 6))

plt.plot(actual_values, label="Actual Price")
plt.plot(predicted_values, label="Predicted Price")

plt.title("Actual vs Predicted TCS Closing Price")
plt.xlabel("Validation Sample")
plt.ylabel("Closing Price (₹)")

plt.legend()
plt.tight_layout()

plt.savefig("data/walk_forward_validation.png", dpi=300)

plt.show()