import pandas as pd
from sklearn.linear_model import LinearRegression

# -----------------------------------
# 1. Load feature dataset
# -----------------------------------

data = pd.read_csv("data/TCS_features.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# -----------------------------------
# 2. Prepare data
# -----------------------------------

# Remove Date because machine learning
# model cannot directly use date as a number
X = data.drop(columns=["Date", "Close"])

# Target variable
y = data["Close"]


# -----------------------------------
# 3. Train Linear Regression model
# -----------------------------------

model = LinearRegression()

model.fit(X, y)

print("\nLinear Regression model trained successfully!")


# -----------------------------------
# 4. Get latest available data
# -----------------------------------

latest_data = X.iloc[[-1]]

latest_date = data["Date"].iloc[-1]

print("\nLatest available date:", latest_date)


# -----------------------------------
# 5. Predict price
# -----------------------------------

predicted_price = model.predict(latest_data)[0]


# -----------------------------------
# 6. Display prediction
# -----------------------------------

print("\n================================")
print("       STOCK PRICE PREDICTION")
print("================================")

print("Latest Date:", latest_date)

print("Latest Actual Price: ₹", round(y.iloc[-1], 2))

print("Predicted Price: ₹", round(predicted_price, 2))

print("================================")