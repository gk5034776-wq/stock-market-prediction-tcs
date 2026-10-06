import yfinance as yf

data = yf.download("TCS.NS", start="2018-01-01", end="2026-01-01")

print(data.head())