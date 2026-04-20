import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="NAV Analysis", layout="wide")

# -------------------- TITLE --------------------
st.title("📊 Mutual Fund / Stock Analysis Dashboard")
st.markdown("Interactive dashboard for trend, distribution & forecasting")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

data = load_data()

# -------------------- SIDEBAR --------------------
st.sidebar.header("🔍 Filters")

min_date = st.sidebar.date_input("Start Date", data['Date'].min())
max_date = st.sidebar.date_input("End Date", data['Date'].max())

filtered_data = data[
    (data['Date'] >= pd.to_datetime(min_date)) &
    (data['Date'] <= pd.to_datetime(max_date))
]

# -------------------- DATA PREVIEW --------------------
st.subheader("📁 Dataset Preview")
st.dataframe(filtered_data.head(), use_container_width=True)

st.markdown("---")

# -------------------- METRICS --------------------
st.subheader("📊 Key Insights")

col1, col2, col3 = st.columns(3)

avg_price = filtered_data['AAPL.Close'].mean()
max_price = filtered_data['AAPL.Close'].max()
min_price = filtered_data['AAPL.Close'].min()

col1.metric("Average Price", round(avg_price, 2))
col2.metric("Max Price", round(max_price, 2))
col3.metric("Min Price", round(min_price, 2))

st.markdown("---")

# -------------------- GRAPHS --------------------
col1, col2 = st.columns(2)

# 📈 Trend
with col1:
    st.subheader("📈 Price Trend")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(filtered_data['Date'], filtered_data['AAPL.Close'])
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

# 📊 Distribution
with col2:
    st.subheader("📊 Price Distribution")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.hist(filtered_data['AAPL.Close'], bins=25)
    ax2.set_xlabel("Price")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

st.markdown("---")

# -------------------- NAV CHECKER --------------------
st.subheader("🧠 Price Filter (Interactive)")

value = st.slider("Filter Price Above", 0, 5000, 1000)

filtered_by_value = filtered_data[filtered_data['AAPL.Close'] >= value]

st.write(f"Showing {len(filtered_by_value)} records above {value}")

# Show updated graph
fig, ax = plt.subplots(figsize=(6,3))
ax.plot(filtered_by_value['Date'], filtered_by_value['AAPL.Close'])
st.pyplot(fig)
# -------------------- FORECAST --------------------
st.subheader("🔮 Price Forecast")

if st.button("Generate Forecast"):

    # Set index
    temp = filtered_data.set_index('Date')

    # KEEP ONLY NUMERIC COLUMN (VERY IMPORTANT)
    temp = temp[['AAPL.Close']]

    # Resample monthly
    temp = temp.resample('ME').mean()

    # Fill missing values
    temp = temp.ffill()

    # Model
    model = ExponentialSmoothing(temp['AAPL.Close'], trend='add')
    fit = model.fit()

    # Forecast next 12 months
    forecast = fit.forecast(12)

    # Plot
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(temp['AAPL.Close'], label="Actual")
    ax3.plot(forecast, label="Forecast", linestyle='dashed')
    ax3.legend()

    st.pyplot(fig3)

st.markdown("---")
# -------------------- FOOTER --------------------
st.caption("Developed for BDA Course Project 🚀")
