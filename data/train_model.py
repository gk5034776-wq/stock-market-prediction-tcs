import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle

# 1. Feature dataset load karna
data = pd.read_csv("data/TCS_features.csv")

# 2. Features select karna
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
y = data["Target"]

# 3. Time-series data ko 80% training aur 20% testing mein divide karna
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# 4. Linear Regression model banana
model = LinearRegression()

# 5. Model ko training data se train karna
model.fit(X_train, y_train)

# Save trained model with feature names
model_bundle = {
    "model": model,
    "features": features
}

with open("tcs_linear_regression_model.pkl", "wb") as file:
    pickle.dump(model_bundle, file)

print("Saved model updated successfully!")

print("\nModel training completed successfully!")

# 6. Test data par prediction karna
predictions = model.predict(X_test)

# 7. Model performance calculate karna
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Performance:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# 8. Actual aur predicted prices ko compare karna
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

print("\nActual vs Predicted:")
print(results.head(10))