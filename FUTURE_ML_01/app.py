import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Page Title
st.set_page_config(page_title="AI Sales Forecasting Dashboard")

st.title("📈 AI-Powered Sales Forecasting Dashboard")
st.write("Predict future sales using Machine Learning.")

# Sample Dataset
data = {
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "Sales": [1200, 1500, 1800, 2200, 2500, 2800]
}

df = pd.DataFrame(data)

st.subheader("Historical Sales Data")
st.dataframe(df)

# Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"₹{df['Sales'].sum():,}")
col2.metric("Average Sales", f"₹{df['Sales'].mean():.0f}")
col3.metric("Highest Sales", f"₹{df['Sales'].max():,}")

# Prepare Data
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = df["Sales"]

# Train Model
model = LinearRegression()
model.fit(X, y)

# Future Forecast
future_months = np.array([7, 8, 9, 10, 11, 12]).reshape(-1, 1)
predictions = model.predict(future_months)

forecast_df = pd.DataFrame({
    "Month": ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "Predicted Sales": predictions.astype(int)
})

st.subheader("Future Sales Forecast")
st.dataframe(forecast_df)

# Graph
st.subheader("Sales Forecast Chart")

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    df["Sales"],
    marker="o",
    label="Actual Sales"
)

ax.plot(
    forecast_df["Month"],
    forecast_df["Predicted Sales"],
    marker="o",
    linestyle="--",
    label="Forecasted Sales"
)

ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.legend()

st.pyplot(fig)

# Bar Chart
st.subheader("Monthly Sales Overview")
chart_data = pd.DataFrame({
    "Sales": df["Sales"]
}, index=df["Month"])

st.bar_chart(chart_data)

# Download Forecast
csv = forecast_df.to_csv(index=False)

st.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name="sales_forecast.csv",
    mime="text/csv"
)

st.success("Forecast generated successfully!")