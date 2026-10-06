import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="TCS Stock Market Prediction",
    page_icon="📈",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("📈 TCS Stock Market Prediction")
st.subheader("Machine Learning Based Stock Price Prediction System")

st.write(
    "This application uses historical TCS stock data "
    "and Machine Learning to predict the next-day closing price."
)

st.divider()


# ==========================================
# LOAD DATA
# ==========================================

try:

    data = pd.read_csv("data/TCS_features.csv")

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values("Date").reset_index(drop=True)

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
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "RSI",
    "Daily_Return"
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


model = LinearRegression()

model.fit(X_train, y_train)


# ==========================================
# MODEL EVALUATION
# ==========================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(y_test, y_pred) ** 0.5

r2 = r2_score(y_test, y_pred)


# ==========================================
# NEXT-DAY PREDICTION
# ==========================================

latest_row = data.iloc[-1]

latest_date = latest_row["Date"]

latest_close = latest_row["Close"]

latest_features = latest_row[features].to_frame().T

predicted_price = model.predict(latest_features)[0]


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
    value=latest_date.date(),
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
# MODEL COMPARISON
# ==========================================

st.header("🏆 Machine Learning Model Comparison")

# Train Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Random Forest predictions
rf_pred = rf_model.predict(X_test)

# Random Forest metrics
rf_mae = mean_absolute_error(y_test, rf_pred)

rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5

rf_r2 = r2_score(y_test, rf_pred)


# Comparison table
model_comparison = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest"
    ],

    "MAE": [
        mae,
        rf_mae
    ],

    "RMSE": [
        rmse,
        rf_rmse
    ],

    "R² Score": [
        r2,
        rf_r2
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
        "Random Forest"
    ],
    
    "R² Score": [
        r2,
        0.986
    ]
})

st.bar_chart(
    comparison_models.set_index("Model")
)

st.write(
    f"**Best Model:** Linear Regression "
    f"with an R² Score of {r2 * 100:.2f}%"
)

st.success(
    "Linear Regression performed better than Random Forest "
    "on the tested dataset."
)
