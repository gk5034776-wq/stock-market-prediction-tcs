import pandas as pd
import matplotlib.pyplot as plt

# TCS dataset read karna
data = pd.read_csv("data/TCS.csv")

# Date ko datetime format mein convert karna
data["Date"] = pd.to_datetime(data["Date"])

# Date ko X-axis aur Close ko Y-axis par plot karna
plt.figure(figsize=(12, 6))

plt.plot(data["Date"], data["Close"])

plt.title("TCS Stock Price History")
plt.xlabel("Date")
plt.ylabel("Closing Price (₹)")

plt.grid(True)
plt.tight_layout()

plt.show()