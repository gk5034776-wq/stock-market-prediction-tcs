import yfinance as yf
from datetime import datetime, timedelta

# Automatically use tomorrow as the end date
end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

# Download TCS historical stock data
data = yf.download(
    "TCS.NS",
    start="2020-01-01",
    end=end_date,
    auto_adjust=False
)

# Remove extra ticker level from column names
if hasattr(data.columns, "levels"):
    data.columns = data.columns.get_level_values(0)

# Convert Date from index into a normal column
data.reset_index(inplace=True)

# Save the data
data.to_csv("data/TCS.csv", index=False)

print("TCS data downloaded and saved successfully!")
print("Latest date:", data["Date"].max())
print("Records:", len(data))
print(data.tail())