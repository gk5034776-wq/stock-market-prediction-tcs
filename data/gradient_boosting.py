import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. Load Feature Dataset
# ==========================================

data = pd.read_csv("data/TCS_features.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ==========================================
# 2. Prepare Data
# ==========================================

data["Date"] = pd.to_datetime(data["Date"])

# Remove rows with missing values
data = data.dropna()


# ==========================================
# 3. Select Features
# ==========================================

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


# ==========================================
# 4. Chronological Train-Test Split
# ==========================================

split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 5. Create Gradient Boosting Model
# ==========================================

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)


# ==========================================
# 6. Train Model
# ==========================================

model.fit(X_train, y_train)

print("\nGradient Boosting training completed successfully!")


# ==========================================
# 7. Make Predictions
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 8. Calculate Performance
# ==========================================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)


print("\nGradient Boosting Model Performance:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# ==========================================
# 9. Save Model
# ==========================================

model_bundle = {
    "model": model,
    "features": features
}

with open("tcs_gradient_boosting_model.pkl", "wb") as file:
    pickle.dump(model_bundle, file)

print("\nSaved Gradient Boosting model successfully!")