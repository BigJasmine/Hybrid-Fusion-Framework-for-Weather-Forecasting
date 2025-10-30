# app.py
import streamlit as st  # pyright: ignore[reportMissingImports]
import pandas as pd
import matplotlib.pyplot as plt

from data_fetcher import fetch_weather_data
from model_builder import train_arima, train_random_forest, evaluate as eval_basic
from hybrid_model import build_hybrid_model, evaluate as eval_hybrid

st.set_page_config(page_title="Weather Forecasting", layout="centered")
st.title("🌦️ API-Based Weather Forecasting System")
st.markdown("This system uses ARIMA, Machine Learning, and Hybrid models to predict weather using real-time API data.")

# --- Fetch Data ---
st.sidebar.header("⚙️ Settings")
city = st.sidebar.text_input("Enter City", value="London")
run_button = st.sidebar.button("Fetch & Run Models")

if run_button:
    st.success(f"Fetching weather data for **{city}**...")

    try:
        df = fetch_weather_data()
        st.write("### 📈 Raw Weather Data")
        st.dataframe(df.head())

        model_type = st.selectbox("Choose Model", ["ARIMA", "Machine Learning", "Hybrid (ARIMA + ML)"])
        st.write(f"## 🔍 Model: {model_type}")

        if model_type == "ARIMA":
            true, pred = train_arima(df)
            rmse, mae, r2 = eval_basic(true, pred)
        elif model_type == "Machine Learning":
            true, pred = train_random_forest(df)
            rmse, mae, r2 = eval_basic(true, pred)
        else:
            true, pred = build_hybrid_model(df)
            rmse, mae, r2 = eval_hybrid(true, pred)

        # --- Metrics ---
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("MAE", f"{mae:.2f}")
        st.metric("R² Score", f"{r2:.2f}")

        # --- Plotting ---
        fig, ax = plt.subplots()
        ax.plot(true.index, true.values, label="Actual", color="black", linewidth=2)
        ax.plot(true.index, pred, label="Predicted", linestyle="--")
        ax.set_title(f"{model_type} Model Forecast")
        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        st.pyplot(fig)

        # --- Export CSV ---
        export_df = pd.DataFrame({"Actual": true.values, "Predicted": pred}, index=true.index)
        csv = export_df.to_csv().encode('utf-8')
        st.download_button("📤 Download Predictions as CSV", data=csv, file_name="predictions.csv", mime='text/csv')

    except Exception as e:
        st.error(f"An error occurred: {e}")
