import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load feature dataset
data = pd.read_csv("data/TCS_features.csv")

# Features used by the models
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
data = data.dropna()

X = data[features]
y = data["Target"]

# Chronological 80-20 split
split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# =========================
# Linear Regression
# =========================

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_prediction = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_prediction)
linear_rmse = mean_squared_error(
    y_test, linear_prediction
) ** 0.5
linear_r2 = r2_score(y_test, linear_prediction)


# =========================
# Random Forest
# =========================

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)

rf_prediction = random_forest.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_prediction)
rf_rmse = mean_squared_error(
    y_test, rf_prediction
) ** 0.5
rf_r2 = r2_score(y_test, rf_prediction)

# =========================
# Gradient Boosting
# =========================

gradient_boosting = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gradient_boosting.fit(X_train, y_train)

gb_prediction = gradient_boosting.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_prediction)
gb_rmse = mean_squared_error(
    y_test, gb_prediction
) ** 0.5
gb_r2 = r2_score(y_test, gb_prediction)

# =========================
# Model Comparison
# =========================

comparison = pd.DataFrame({
    "MAE": [linear_mae, rf_mae, gb_mae],
    "RMSE": [linear_rmse, rf_rmse, gb_rmse],
    "R2 Score": [linear_r2, rf_r2, gb_r2]
}, index=[
    "Linear Regression",
    "Random Forest",
    "Gradient Boosting"
])

print("\n===== MODEL COMPARISON =====")
print(comparison.round(4))


# =========================
# Best Model
# =========================

best_mae = comparison["MAE"].idxmin()
best_rmse = comparison["RMSE"].idxmin()
best_r2 = comparison["R2 Score"].idxmax()

print("\n===== BEST MODELS =====")
print("Best MAE Model:", best_mae)
print("Best RMSE Model:", best_rmse)
print("Best R2 Score Model:", best_r2)


# =========================
# R2 Score Graph
# =========================

plt.figure(figsize=(8, 5))

plt.bar(
    comparison.index,
    comparison["R2 Score"]
)

plt.title("Model Comparison - R2 Score")
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.ylim(0.90, 1.00)

plt.show()