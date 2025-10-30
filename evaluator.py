# evaluator.py

import matplotlib.pyplot as plt
import seaborn as sns
from model_builder import train_lstm, train_random_forest, evaluate
from data_fetcher import fetch_weather_data

sns.set(style="whitegrid")

def compare_models():
    df = fetch_weather_data()

    # === Feature Setup ===
    features = ['temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm', 'cloud']
    target = 'temperature_celsius'

    # === LSTM ===
    lstm_true, lstm_pred = train_lstm(df, features=features, target=target)
    rmse_lstm, mae_lstm, r2_lstm = evaluate(lstm_true, lstm_pred)

    # === Random Forest ===
    rf_true, rf_pred = train_random_forest(df, features=features, target=target)
    rmse_rf, mae_rf, r2_rf = evaluate(rf_true, rf_pred)

    # === Results Table ===
    print("\n--- Performance Comparison ---")
    print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print(f"{'LSTM':<20} {rmse_lstm:<10.2f} {mae_lstm:<10.2f} {r2_lstm:<10.2f}")
    print(f"{'Random Forest':<20} {rmse_rf:<10.2f} {mae_rf:<10.2f} {r2_rf:<10.2f}")

    # === Plotting ===
    plt.figure(figsize=(12, 6))
    plt.plot(lstm_true.index, lstm_true.values, label='Actual', color='black', linewidth=2)
    plt.plot(lstm_true.index, lstm_pred, label='LSTM Prediction', linestyle='--', color='blue')
    plt.title('LSTM Model vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_models()