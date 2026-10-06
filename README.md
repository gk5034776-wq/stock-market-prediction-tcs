# 📈 Stock Market Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting the next-day closing price of Tata Consultancy Services (TCS) using Machine Learning.

Historical stock market data is collected, cleaned, transformed into useful features, and used to train multiple regression models.

The project also includes model comparison, walk-forward validation, feature analysis, and an interactive Streamlit dashboard.

---

## 🎯 Objectives

- Collect historical TCS stock market data
- Clean and preprocess the dataset
- Perform feature engineering
- Predict the next-day closing price
- Compare multiple Machine Learning models
- Validate the best model using walk-forward validation
- Analyze Linear Regression coefficients
- Build an interactive dashboard for predictions

---

## 📊 Dataset

The project uses historical TCS stock market data.

### Main Features

- Open
- High
- Low
- Close
- Volume
- Daily Return
- 5-Day Moving Average
- 20-Day Moving Average
- 5-Day Volatility
- 14-Day RSI

### Target

The target variable is the **next-day closing price**.

---

## 🤖 Machine Learning Models

Three regression models were evaluated:

1. Linear Regression
2. Random Forest
3. Gradient Boosting

---

## 📈 Model Results

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 31.2935 | 43.6628 | 0.9902 |
| Random Forest | 53.0911 | 71.0184 | 0.9741 |
| Gradient Boosting | 57.1578 | 76.6378 | 0.9698 |

Based on the evaluated test data, **Linear Regression** achieved the best MAE, RMSE, and R² Score.

---

## 🔄 Walk-Forward Validation

Linear Regression was additionally evaluated using walk-forward validation.

Results:

- MAE: 31.0082
- RMSE: 43.5735
- R² Score: 0.9903

This provides an additional validation of the model using a chronological training and prediction process.

---

## 🔮 Latest Prediction

The latest available data point used by the prediction system is:

- Date: 2026-09-30
- Latest Closing Price: ₹2050.60
- Predicted Next-Day Closing Price: ₹2059.27

The actual next-day price was not available in the current dataset at the time of prediction.

---

## 🖥️ Dashboard

The project includes an interactive Streamlit dashboard with:

- Latest TCS closing price
- Next-day predicted price
- Expected price direction
- Model evaluation metrics
- Historical price trend
- Interactive date-based prediction
- Actual vs Predicted Price analysis
- Walk-Forward Validation visualization
- Linear Regression Coefficient Analysis
- Model comparison

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- yfinance
- Streamlit
- Pickle

---

## 📂 Project Structure

```text
Stock Market Prediction/
│
├── data/
│   ├── TCS.csv
│   ├── TCS_cleaned.csv
│   ├── TCS_features.csv
│   ├── download_data.py
│   ├── clean_data.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── random_forest.py
│   ├── gradient_boosting.py
│   ├── model_comparison.py
│   ├── walk_forward_validation.py
│   ├── walk_forward_validation.png
│   ├── linear_regression_coefficients.py
│   ├── linear_regression_coefficients.png
│   └── results_summary.py
│
├── app.py
├── next_day_prediction.py
├── tcs_linear_regression_model.pkl
└── README.md