import pandas as pd

# TCS dataset ko read karna
data = pd.read_csv("data/TCS.csv")

print("Original data:")
print(data.head())

# Date ko datetime format mein convert karna
data["Date"] = pd.to_datetime(data["Date"])

# Date ke according data sort karna
data = data.sort_values("Date")

# Duplicate rows check karna
print("\nDuplicate rows:", data.duplicated().sum())

# Missing values check karna
print("\nMissing values:")
print(data.isnull().sum())

# Missing values ko remove karna
data = data.dropna()

# Duplicate rows ko remove karna
data = data.drop_duplicates()

# Cleaned data ko save karna
data.to_csv("data/TCS_cleaned.csv", index=False)

print("\nData cleaning completed successfully!")
print("Cleaned data saved as data/TCS_cleaned.csv")

print("\nCleaned data shape:")
print(data.shape)