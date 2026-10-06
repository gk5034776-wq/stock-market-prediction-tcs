import pandas as pd
from sklearn.linear_model import LinearRegression

# Load feature dataset
data = pd.read_csv("data/TCS_features.csv")

# Remove missing values
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

# Input features and target
X = data[features]
y = data["Target"]

# Chronological split
split = int(len(data) * 0.8)

X_train = X.iloc[:split]
y_train = y.iloc[:split]

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Feature coefficients
coefficient_data = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

# Absolute coefficient for ranking
coefficient_data["Absolute_Coefficient"] = (
    coefficient_data["Coefficient"].abs()
)

coefficient_data = coefficient_data.sort_values(
    "Absolute_Coefficient",
    ascending=False
)

print("\n===== LINEAR REGRESSION COEFFICIENT ANALYSIS =====")
print(coefficient_data.to_string(index=False))

# Coefficient Bar Chart

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.barh(
    coefficient_data["Feature"],
    coefficient_data["Coefficient"]
)

plt.xlabel("Regression Coefficient")
plt.ylabel("Feature")
plt.title("Linear Regression Coefficient Analysis")

plt.tight_layout()

plt.savefig(
    "data/linear_regression_coefficients.png",
    dpi=300
)

plt.show()