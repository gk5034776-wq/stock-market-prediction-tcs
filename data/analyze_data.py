import pandas as pd

# TCS dataset ko read karna
data = pd.read_csv("data/TCS.csv")

# Dataset ki first 5 rows dekhna
print("First 5 rows:")
print(data.head())

# Dataset mein kitni rows aur columns hain
print("\nDataset Shape:")
print(data.shape)

# Dataset ke columns ke naam
print("\nColumn Names:")
print(data.columns)

# Dataset ki basic information
print("\nDataset Information:")
data.info()

# Missing values check karna
print("\nMissing Values:")
print(data.isnull().sum())
