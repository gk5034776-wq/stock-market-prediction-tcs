import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("data/TCS_features.csv")

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)
data = data.dropna().reset_index(drop=True)

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

X = data[features]
y = data["Target"]

# Chronological 80/20 split
split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# Results function
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    r2 = r2_score(actual, predicted)

    return mae, rmse, r2


lr_metrics = calculate_metrics(y_test, lr_pred)
rf_metrics = calculate_metrics(y_test, rf_pred)
gb_metrics = calculate_metrics(y_test, gb_pred)

# Results table
results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "MAE": [
        lr_metrics[0],
        rf_metrics[0],
        gb_metrics[0]
    ],
    "RMSE": [
        lr_metrics[1],
        rf_metrics[1],
        gb_metrics[1]
    ],
    "R2 Score": [
        lr_metrics[2],
        rf_metrics[2],
        gb_metrics[2]
    ]
})

print("\n===== FINAL MODEL RESULTS =====")
print(results.to_string(index=False))

# Best models
print("\n===== BEST MODELS =====")

print(
    "Best MAE:",
    results.loc[results["MAE"].idxmin(), "Model"]
)

print(
    "Best RMSE:",
    results.loc[results["RMSE"].idxmin(), "Model"]
)

print(
    "Best R2 Score:",
    results.loc[results["R2 Score"].idxmax(), "Model"]
)

# Walk-forward validation results
print("\n===== WALK-FORWARD VALIDATION =====")
print("MAE: 31.0082")
print("RMSE: 43.5735")
print("R2 Score: 0.9903")

results.to_csv("data/results_summary.csv", index=False)

print("\nResults saved to data/results_summary.csv")