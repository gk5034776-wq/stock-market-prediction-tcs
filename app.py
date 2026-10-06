import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="TCS Stock Market Prediction",
    page_icon="📈",
    layout="wide"
)

with open("tcs_linear_regression_model.pkl", "rb") as file:
    saved_model = pickle.load(file)

# ==========================================
# TITLE
# ==========================================

st.title("📈 TCS Stock Market Prediction")
st.subheader("Machine Learning Based Stock Price Prediction System")

st.markdown(
    """
    **Tata Consultancy Services (TCS)**

    This dashboard analyzes historical TCS stock data and uses
    Machine Learning to estimate the next-day closing price.

    📊 Historical Data &nbsp; | &nbsp;
    🤖 Machine Learning &nbsp; | &nbsp;
    🔮 Next-Day Prediction
    """
)

st.divider()


# ==========================================
# LOAD DATA
# ==========================================

try:

    data = pd.read_csv("data/TCS_features.csv")

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values("Date").reset_index(drop=True)

    latest_prediction_row = data.iloc[-1].copy()

except Exception as e:

    st.error(f"Unable to load dataset: {e}")
    st.stop()


# ==========================================
# CREATE NEXT-DAY TARGET
# ==========================================

data["Next_Day_Close"] = data["Close"].shift(-1)

data = data.dropna().reset_index(drop=True)


# ==========================================
# SELECT FEATURES
# ==========================================

possible_features = [
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

features = [
    column
    for column in possible_features
    if column in data.columns
]


# ==========================================
# TRAIN MODEL
# ==========================================

X = data[features]
y = data["Next_Day_Close"]

split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Load saved trained model
with open("tcs_linear_regression_model.pkl", "rb") as file:
    saved_model = pickle.load(file)["model"]

st.success("✅ Saved Linear Regression Model Loaded Successfully")

# Keep compatibility with the existing model file
if isinstance(saved_model, dict):
    model = saved_model["model"]
    features = saved_model["features"]
else:
    model = saved_model

  
# ==========================================
# MODEL EVALUATION
# ==========================================

y_pred = saved_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(y_test, y_pred) ** 0.5

r2 = r2_score(y_test, y_pred)


# ==========================================
# NEXT-DAY PREDICTION
# ==========================================

latest_row = latest_prediction_row

latest_date = latest_row["Date"]

latest_close = latest_row["Close"]

latest_features = latest_row[features].to_frame().T

predicted_price = saved_model.predict(latest_features)[0]

if predicted_price > latest_close:
    st.success("📈 Expected Direction: UP")
elif predicted_price < latest_close:
    st.error("📉 Expected Direction: DOWN")
else:
    st.info("➡️ Expected Direction: NEUTRAL")

price_change = predicted_price - latest_close

price_change_percent = (
    price_change / latest_close
) * 100

st.metric("Price Change", f"₹{price_change:.2f}")
st.metric("Price Change %", f"{price_change_percent:.2f}%")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📊 Project Information")

st.sidebar.write("**Stock:** TCS")

st.sidebar.write("**Model:** Linear Regression")

st.sidebar.write("**Prediction:** Next-Day Closing Price")

st.sidebar.write(
    f"**Dataset Records:** {len(data)}"
)


# ==========================================
# MAIN INFORMATION CARDS
# ==========================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Latest Closing Price",
        f"₹{latest_close:.2f}"
    )


with col2:

    st.metric(
        "Predicted Next-Day Price",
        f"₹{predicted_price:.2f}"
    )


with col3:

    st.metric(
        "R² Score",
        f"{r2 * 100:.2f}%"
    )


with col4:

    st.metric(
        "MAE",
        f"{mae:.2f}"
    )

st.divider()

st.subheader("📊 Prediction Summary")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.metric(
        "Expected Price Change",
        f"₹{price_change:.2f}"
    )

with summary_col2:
    st.metric(
        "Expected Change %",
        f"{price_change_percent:.2f}%"
    )

st.divider()


# ==========================================
# PREDICTION RESULT
# ==========================================

st.header("🔮 Next-Day Stock Price Prediction")


st.success(
    f"Predicted next-day closing price for TCS: "
    f"₹{predicted_price:.2f}"
)


st.write(
    f"Latest available date: **{latest_date.strftime('%Y-%m-%d')}**"
)

st.write(
    f"Latest actual closing price: **₹{latest_close:.2f}**"
)
# ==========================================
# INTERACTIVE DATE-BASED PREDICTION
# ==========================================

st.divider()

st.header("🎯 Interactive Stock Price Prediction")

st.write(
    "Select a date from the historical dataset to predict "
    "the following day's closing price."
)

selected_date = st.date_input(
    "📅 Select a Date",
    value=data["Date"].max().date(),
    min_value=data["Date"].min().date(),
    max_value=data["Date"].max().date()
)

selected_rows = data[
    data["Date"].dt.date == selected_date
]

if len(selected_rows) > 0:

    selected_row = selected_rows.iloc[0]

    selected_features = selected_row[features].to_frame().T

    interactive_prediction = model.predict(
        selected_features
    )[0]

    st.write(
        f"Selected Date Closing Price: "
        f"**₹{selected_row['Close']:.2f}**"
    )

    st.success(
        f"🔮 Predicted Next-Day Closing Price: "
        f"₹{interactive_prediction:.2f}"
    )

else:

    st.warning(
        "No stock data is available for the selected date."
    )
# ==========================================
# HISTORICAL PRICE TREND
# ==========================================

st.header("📈 TCS Historical Closing Price")

historical_data = data.set_index("Date")[["Close"]]

st.line_chart(
    historical_data,
    width="stretch"
)
# ==========================================
# PRICE COMPARISON
# ==========================================

st.header("📊 Actual vs Predicted Price")


comparison_data = pd.DataFrame({

    "Actual Price": y_test.values,

    "Predicted Price": y_pred

})


st.line_chart(comparison_data)

# ==========================================
# WALK-FORWARD VALIDATION
# ==========================================

st.header("🔄 Walk-Forward Validation")

st.write(
    "This graph compares actual and predicted TCS closing prices "
    "during walk-forward validation."
)

try:
    st.image(
        "data/walk_forward_validation.png",
        caption="Actual vs Predicted TCS Closing Price",
        width="stretch"
    )
except Exception:
    st.warning("Walk-forward validation graph is not available.")

# ==========================================
# LINEAR REGRESSION COEFFICIENT ANALYSIS
# ==========================================

st.header("📊 Linear Regression Coefficient Analysis")

st.write(
    "This chart shows the fitted coefficients of the "
    "features used by the Linear Regression model."
)

try:
    st.image(
        "data/linear_regression_coefficients.png",
        caption="Linear Regression Coefficient Analysis",
        width="stretch"
    )
except Exception:
    st.warning("Coefficient analysis chart is not available.")
# ==========================================
# MODEL COMPARISON
# ==========================================

st.header("🏆 Machine Learning Model Comparison")

# Train Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5
rf_r2 = r2_score(y_test, rf_pred)


# Train Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = mean_squared_error(y_test, gb_pred) ** 0.5
gb_r2 = r2_score(y_test, gb_pred)


# Comparison table
model_comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "MAE": [
        mae,
        rf_mae,
        gb_mae
    ],
    "RMSE": [
        rmse,
        rf_rmse,
        gb_rmse
    ],
    "R² Score": [
        r2,
        rf_r2,
        gb_r2
    ]
})

st.dataframe(
    model_comparison,
    width="stretch"
)

# Random Forest predictions
rf_pred = rf_model.predict(X_test)

# Random Forest metrics
rf_mae = mean_absolute_error(y_test, rf_pred)

rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5

rf_r2 = r2_score(y_test, rf_pred)

# Train Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)

# Gradient Boosting predictions
gb_pred = gb_model.predict(X_test)

# Gradient Boosting metrics
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = mean_squared_error(y_test, gb_pred) ** 0.5
gb_r2 = r2_score(y_test, gb_pred)

# Comparison table
model_comparison = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ],

    "MAE": [
        mae,
        rf_mae,
        gb_mae
    ],

    "RMSE": [
        rmse,
        rf_rmse,
        gb_rmse
    ],

    "R² Score": [
        r2,
        rf_r2,
        gb_r2
    ]

})


st.dataframe(
    model_comparison,
    width="stretch"
)
# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.header("🤖 Model Performance")


performance = pd.DataFrame({

    "Metric": [
        "MAE",
        "RMSE",
        "R² Score"
    ],

    "Value": [
        mae,
        rmse,
        r2
    ]

})


st.table(performance)


# ==========================================
# RECENT STOCK DATA
# ==========================================

st.header("📋 Recent TCS Stock Data")

st.dataframe(
    data[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    ].tail(10),
    width="stretch"
)


# ==========================================
# ABOUT PROJECT
# ==========================================

st.divider()

st.header("ℹ️ About This Project")

st.write(
    """
    **Stock Market Prediction using Machine Learning** is a project
    that analyzes historical TCS stock market data and uses Machine
    Learning techniques to predict the next-day closing price.

    The system uses historical stock features such as Open, High,
    Low, Close, Volume, moving averages, RSI and daily returns.

    Linear Regression is used as the prediction model because it
    achieved the best performance among the tested models.
    """
)


st.info(
    "⚠️ This prediction is for educational/project purposes only "
    "and should not be considered financial advice."
)
# ==========================================
# MODEL COMPARISON
# ==========================================

st.divider()

st.header("📊 Model Comparison")
comparison_models = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    
    "R² Score": [
        r2,
        rf_r2,
        gb_r2
    ]
})

st.bar_chart(
    comparison_models.set_index("Model")
)

# PREDICTION HISTORY
st.header("📋 Prediction History")

st.write(
    "Historical next-day predictions generated using the Linear Regression model."
)

prediction_history = pd.read_csv("data/prediction_history.csv")
prediction_history["Date"] = pd.to_datetime(prediction_history["Date"])

# Clean date format
prediction_history["Date"] = prediction_history["Date"].dt.date

# Actual vs Predicted Price Chart
st.subheader("📈 Actual vs Predicted Price")

chart_data = prediction_history.set_index("Date")[
    ["Actual_Close", "Predicted_Close"]
]

st.line_chart(chart_data)

# PREDICTION ERROR ANALYSIS
st.subheader("📉 Prediction Error Analysis")

prediction_history["Prediction_Error"] = (
    prediction_history["Actual_Close"]
    - prediction_history["Predicted_Close"]
)

average_error = prediction_history["Prediction_Error"].mean()
average_absolute_error = prediction_history["Absolute_Error"].mean()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Average Prediction Error",
        f"₹{average_error:.2f}"
    )

with col2:
    st.metric(
        "Average Absolute Error",
        f"₹{average_absolute_error:.2f}"
    )

error_chart = prediction_history.set_index("Date")[
    ["Prediction_Error"]
]

st.line_chart(error_chart)

st.caption(
    "Positive values indicate under-prediction, while negative values indicate over-prediction."
)

# Prediction History Table
st.subheader("📋 Detailed Prediction Records")

st.dataframe(
    prediction_history,
    use_container_width=True,
    hide_index=True
)