import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Page configuration
st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")

# Title
st.title("Retail Sales Forecasting Dashboard")
st.write("Forecast Walmart weekly sales using SARIMA model")

# Load dataset
df = pd.read_csv("Walmart.csv")

# Convert date column
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values("Date")

# Sidebar
st.sidebar.title("Dashboard Controls")

# Store selection
store_list = sorted(df['Store'].unique())

store_id = st.sidebar.selectbox(
    "Select Store",
    store_list
)

# Forecast weeks slider
weeks = st.sidebar.slider(
    "Select number of weeks to forecast",
    4, 20, 12
)

# Filter selected store
store_data = df[df['Store'] == store_id]
store_data = store_data.set_index("Date")

sales = store_data['Weekly_Sales']

# KPI Metrics
col1, col2, col3 = st.columns(3)

col1.metric(
    "Last Weekly Sales",
    f"${sales.iloc[-1]:,.0f}"
)

col2.metric(
    "Average Weekly Sales",
    f"${sales.mean():,.0f}"
)

col3.metric(
    "Maximum Weekly Sales",
    f"${sales.max():,.0f}"
)

# Train SARIMA model
model = SARIMAX(
    sales,
    order=(2,0,1),
    seasonal_order=(1,0,1,52)
)

model_fit = model.fit()

forecast = model_fit.forecast(steps=weeks)

# Historical Sales Graph
st.subheader(f"Historical Weekly Sales — Store {store_id}")

fig1, ax1 = plt.subplots(figsize=(10,5))

ax1.plot(sales, color="blue")
ax1.set_title("Historical Sales Trend")
ax1.set_xlabel("Date")
ax1.set_ylabel("Weekly Sales")

st.pyplot(fig1)

# Forecast Graph
st.subheader(f"Sales Forecast — Store {store_id}")

recent_sales = sales[-52:]

fig2, ax2 = plt.subplots(figsize=(10,5))

ax2.plot(recent_sales, label="Recent Sales", color="blue")
ax2.plot(forecast.index, forecast, label="Forecast", color="orange")

ax2.legend()
ax2.set_title("Future Sales Forecast")
ax2.set_xlabel("Date")
ax2.set_ylabel("Weekly Sales")

st.pyplot(fig2)

# Forecast Table
st.subheader("Forecasted Sales")

forecast_df = pd.DataFrame({
    "Date": forecast.index,
    "Predicted Sales": forecast.values
})

st.dataframe(forecast_df)

# Download forecast
csv = forecast_df.to_csv(index=False)

st.download_button(
    label="Download Forecast Data",
    data=csv,
    file_name="sales_forecast.csv",
    mime="text/csv"
)

# Dataset preview
with st.expander("View Dataset"):
    st.dataframe(df.head(20))

# Footer
st.markdown("---")
st.write("Retail Sales Forecasting Dashboard")
st.write("Developed by Prarthna | Time Series Analysis Project")