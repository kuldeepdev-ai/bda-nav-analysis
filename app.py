import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="NAV Analysis", layout="wide")

# -------------------- TITLE --------------------
st.title("📊 Mutual Fund / Stock Analysis Dashboard")
st.markdown("Clean, interactive analytics dashboard")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date')

data = load_data()

# -------------------- SIDEBAR --------------------
st.sidebar.header("🔍 Filters")

min_date = st.sidebar.date_input("Start Date", data['Date'].min())
max_date = st.sidebar.date_input("End Date", data['Date'].max())

filtered_data = data[
    (data['Date'] >= pd.to_datetime(min_date)) &
    (data['Date'] <= pd.to_datetime(max_date))
]

# -------------------- TOP METRICS --------------------
st.subheader("📊 Key Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Avg Price", round(filtered_data['AAPL.Close'].mean(), 2))
col2.metric("Max Price", round(filtered_data['AAPL.Close'].max(), 2))
col3.metric("Min Price", round(filtered_data['AAPL.Close'].min(), 2))

st.markdown("---")

# -------------------- MAIN CHARTS (SIDE BY SIDE) --------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Price Trend")
    fig, ax = plt.subplots(figsize=(5, 3))
   ax.plot(filtered_data['Date'], filtered_data['AAPL.Close'])
ax.tick_params(axis='x', rotation=45)
fig.autofmt_xdate()
    ax.grid(True)
    st.pyplot(fig)

with col2:
    st.subheader("📊 Distribution")
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.hist(filtered_data['AAPL.Close'], bins=25)
    ax2.grid(True)
    st.pyplot(fig2)

st.markdown("---")

# -------------------- FILTER + FORECAST SIDE BY SIDE --------------------
col1, col2 = st.columns(2)

# 🧠 FILTER
with col1:
    st.subheader("🧠 Price Filter")

    min_val = int(filtered_data['AAPL.Close'].min())
    max_val = int(filtered_data['AAPL.Close'].max())

    value = st.slider(
        "Filter Price Above",
        min_val,
        max_val,
        int((min_val + max_val) / 2)
    )

    filtered_by_value = filtered_data[filtered_data['AAPL.Close'] >= value]

    st.caption(f"{len(filtered_by_value)} records above {value}")

    if not filtered_by_value.empty:
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.plot(filtered_by_value['Date'], filtered_by_value['AAPL.Close'])
        ax3.grid(True)
        st.pyplot(fig3)
    else:
        st.warning("No data in this range")

# 🔮 FORECAST
with col2:
    st.subheader("🔮 Forecast")

    if st.button("Generate Forecast"):

        temp = filtered_data.set_index('Date')
        temp = temp[['AAPL.Close']].resample('ME').mean().ffill()

        model = ExponentialSmoothing(temp['AAPL.Close'], trend='add')
        fit = model.fit()

        forecast = fit.forecast(12)

        fig4, ax4 = plt.subplots(figsize=(5, 3))
        ax4.plot(temp['AAPL.Close'], label="Actual")
        ax4.plot(forecast, linestyle='dashed', label="Forecast")
        ax4.legend()
        ax4.grid(True)

        st.pyplot(fig4)

st.markdown("---")

# -------------------- DATA TABLE --------------------
with st.expander("📁 View Full Dataset"):
    st.dataframe(filtered_data, use_container_width=True)
