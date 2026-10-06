import pandas as pd

# Cleaned TCS data read karna
data = pd.read_csv("data/TCS_cleaned.csv")

# Date ko datetime format mein convert karna
data["Date"] = pd.to_datetime(data["Date"])

# Date ke according data sort karna
data = data.sort_values("Date")

# Previous day's closing price
data["Previous_Close"] = data["Close"].shift(1)

# 5-day moving average
data["MA_5"] = data["Close"].rolling(window=5).mean()

# 20-day moving average
data["MA_20"] = data["Close"].rolling(window=20).mean()

# Daily percentage return
data["Daily_Return"] = data["Close"].pct_change() * 100

# 5-day volatility
data["Volatility_5"] = data["Daily_Return"].rolling(window=5).std()

# 14-day Relative Strength Index (RSI)
delta = data["Close"].diff()

gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

average_gain = gain.rolling(window=14).mean()
average_loss = loss.rolling(window=14).mean()

rs = average_gain / average_loss

data["RSI_14"] = 100 - (100 / (1 + rs))

# Tomorrow's closing price - our target
data["Target"] = data["Close"].shift(-1)

# Features create hone ke baad missing rows remove karna
data = data.dropna()

# Feature dataset save karna
data.to_csv("data/TCS_features.csv", index=False)

print("Feature engineering completed successfully!")

print("\nNew columns:")
print(data.columns)

print("\nFirst 5 rows:")
print(data.head())

print("\nFeature dataset shape:")
print(data.shape)

print("\nFeature dataset saved as data/TCS_features.csv")