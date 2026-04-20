import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="NAV Analysis", layout="wide")

st.title("📊 Mutual Fund NAV Analysis Dashboard")

# Load data
data = pd.read_csv("https://raw.githubusercontent.com/yourusername/bda-nav-analysis/main/mutual_fund_nav.csv")
data['Year'] = pd.to_datetime(data['Year'], dayfirst=True)
data = data.sort_values('Year')

# Sidebar filters
st.sidebar.header("Filters")

min_date = st.sidebar.date_input("Start Date", data['Year'].min())
max_date = st.sidebar.date_input("End Date", data['Year'].max())

filtered_data = data[(data['Year'] >= pd.to_datetime(min_date)) & 
                     (data['Year'] <= pd.to_datetime(max_date))]

# Show dataset
st.subheader("📁 Dataset Preview")
st.dataframe(filtered_data.head())

# Metrics
st.subheader("📌 Key Insights")
col1, col2, col3 = st.columns(3)

col1.metric("Average NAV", round(filtered_data['NAV'].mean(), 2))
col2.metric("Max NAV", round(filtered_data['NAV'].max(), 2))
col3.metric("Min NAV", round(filtered_data['NAV'].min(), 2))

# Line chart
st.subheader("📈 NAV Trend Over Time")
fig, ax = plt.subplots()
ax.plot(filtered_data['Year'], filtered_data['NAV'])
st.pyplot(fig)

# Histogram
st.subheader("📊 NAV Distribution")
fig2, ax2 = plt.subplots()
ax2.hist(filtered_data['NAV'], bins=30)
st.pyplot(fig2)

# Interactive input
st.subheader("🧠 NAV Category Checker")
value = st.slider("Select NAV Value", 0, 5000, 1000)

if value > 2000:
    st.success("High NAV 💰")
elif value > 1000:
    st.warning("Moderate NAV ⚠")
else:
    st.info("Low NAV 📉")
    
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.subheader("🔮 Forecast NAV")

if st.button("Generate Forecast"):
    temp = filtered_data.set_index('Year')
    temp = temp.resample('ME').mean().ffill()

    model = ExponentialSmoothing(temp['NAV'], trend='add')
    fit = model.fit()

    forecast = fit.forecast(12)

    fig3, ax3 = plt.subplots()
    ax3.plot(temp['NAV'], label="Actual")
    ax3.plot(forecast, label="Forecast", linestyle='dashed')
    ax3.legend()
    st.pyplot(fig3)