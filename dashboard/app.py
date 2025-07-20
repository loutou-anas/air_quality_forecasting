import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# App title
st.set_page_config(page_title="Air Quality Forecast Dashboard", layout="wide")
st.title("🌫️ Air Quality Forecasting Dashboard")

# Load processed data
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed_air_quality.csv")
    csv_path = os.path.abspath(csv_path)
    return pd.read_csv(csv_path, parse_dates=["datetime"])

df = load_data()

# Sidebar - Select pollutant
targets = ['CO(GT)', 'NO2(GT)', 'NOx(GT)', 'C6H6(GT)']
selected_target = st.sidebar.selectbox("Select pollutant to forecast", targets)

# Load corresponding model
model_file = f"models/{selected_target.replace('(GT)', '').lower()}_model.pkl"
if os.path.exists(model_file):
    model = joblib.load(model_file)

    # Prepare features
    feature_cols = df.drop(columns=targets).select_dtypes(include='number').columns
    X = df[feature_cols]
    y = df[selected_target]
    y_pred = model.predict(X)

    # Display actual vs predicted
    st.subheader(f"📈 Actual vs Predicted: {selected_target}")
    plot_df = pd.DataFrame({
        "Actual": y[-200:].values,
        "Predicted": y_pred[-200:]
    }, index=y.index[-200:])
    st.line_chart(plot_df)

    # Show recent predictions
    st.subheader("🔍 Recent Predictions Table")
    st.dataframe(plot_df.tail(20).style.highlight_max(axis=0))
else:
    st.warning(f"No model found for {selected_target}. Please train it first.")
