import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_fetcher import prepare_master_dataframe
from hybrid_model import build_hybrid_model, evaluate as eval_hybrid
from lstm_model import train_lstm
from model_builder import train_random_forest, evaluate as eval_traditional
from fusion_utils import (
    align_validation_targets,
    optimize_fusion_weight,
    evaluate_predictions,
    log_mismatch
)

def plot_model_outputs(y_true, y_pred, model_name="model"):
    import seaborn as sns
    from datetime import datetime
    import os

    if isinstance(y_pred, (list, np.ndarray)):
        y_pred = pd.Series(y_pred, index=y_true.index)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("charts", exist_ok=True)

    def save_chart(fig, name):
        filename = os.path.abspath(f"charts/{model_name}_{name}_{timestamp}.png")
        fig.savefig(filename, dpi=300)
        print(f"✅ Saved: {filename}")
        plt.close(fig)

    # Forecast vs Actual
    fig = plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true.values, label='Actual', linewidth=2)
    plt.plot(y_pred.index, y_pred.values, label='Predicted', linestyle='--')
    plt.title(f'{model_name.upper()} Forecast vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.tight_layout()
    save_chart(fig, "forecast_vs_actual")

    # Forecast with Residuals
    residuals = y_true - y_pred
    fig = plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true.values, label='Actual', linewidth=2)
    plt.plot(y_pred.index, y_pred.values, label='Predicted', linestyle='--')
    plt.fill_between(y_true.index, residuals, color='gray', alpha=0.3, label='Residuals')
    plt.title(f'{model_name.upper()} Forecast with Residuals')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.tight_layout()
    save_chart(fig, "forecast_with_residuals")

    # Scatter Plot
    fig = plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.title(f'{model_name.upper()} Forecast: Actual vs Predicted (Scatter)')
    plt.tight_layout()
    save_chart(fig, "forecast_scatter")

    # Zoomed-in Forecast
    fig = plt.figure(figsize=(12, 6))
    zoom_range = slice(0, 100)
    plt.plot(y_true.index[zoom_range], y_true.values[zoom_range], label='Actual', linewidth=2)
    plt.plot(y_pred.index[zoom_range], y_pred.values[zoom_range], label='Predicted', linestyle='--')
    plt.title(f'{model_name.upper()} Forecast (Zoomed In)')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.tight_layout()
    save_chart(fig, "forecast_zoomed")

def run_lstm():
    df = prepare_master_dataframe()
    features = ['temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm', 'cloud']
    y_true, y_pred = train_lstm(df, features=features, target='temperature_celsius', n_steps=24, epochs=20)
    rmse, mae, r2 = eval_hybrid(y_true, y_pred)

    print("\n📊 LSTM Evaluation Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"R²  : {r2:.2f}")
    plot_model_outputs(y_true, y_pred, model_name="lstm")

def run_hybrid():
    df = prepare_master_dataframe()
    features = ['temperature_celsius', 'humidity', 'pressure_mb']
    y_true, y_pred = build_hybrid_model(df, features=features, target='temperature_celsius', seq_length=24, epochs=20)
    rmse, mae, r2 = eval_hybrid(y_true, y_pred)

    print("\n🔀 Hybrid Model Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"R²  : {r2:.2f}")
    plot_model_outputs(y_true, y_pred, model_name="hybrid")

def run_ml():
    df = prepare_master_dataframe()
    features = ['humidity', 'wind_kph', 'pressure_mb', 'precip_mm', 'cloud']
    y_true, y_pred = train_random_forest(df, features=features, target='temperature_celsius')
    rmse, mae, r2 = eval_traditional(y_true, y_pred)

    print("\n🌲 Random Forest Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"R²  : {r2:.2f}")
    plot_model_outputs(y_true, y_pred, model_name="ml")

def run_fusion():
    df = prepare_master_dataframe()
    target = "temperature_celsius"
    val_size = 96

    # Align targets
    y_true, val_index = align_validation_targets(df, target_column=target, val_size=val_size)

    # Predict
    rf_features = ['humidity', 'wind_kph', 'pressure_mb', 'precip_mm', 'cloud']
    lstm_features = ['temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm', 'cloud']

    rf_pred = train_random_forest(df.loc[val_index], features=rf_features, target=target)[1]
    lstm_pred = train_lstm(df.loc[val_index], features=lstm_features, target=target, n_steps=24, epochs=20)[1]

    # Reset index
    rf_pred = pd.Series(rf_pred).reset_index(drop=True)
    lstm_pred = pd.Series(lstm_pred).reset_index(drop=True)
    y_true = y_true.reset_index(drop=True)

    # Align lengths
    min_len = min(len(rf_pred), len(lstm_pred), len(y_true))
    rf_pred = rf_pred[:min_len]
    lstm_pred = lstm_pred[:min_len]
    y_true = y_true[:min_len]

    # Optional debug
    log_mismatch(rf_pred, lstm_pred, "RF", "LSTM")

    # Fuse
    hybrid_pred = optimize_fusion_weight(rf_pred, lstm_pred, y_true)

    # Evaluate
    metrics = evaluate_predictions(y_true, hybrid_pred)
    print("\n🔗 Late Fusion Results:")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAE : {metrics['MAE']:.2f}")
    print(f"R²  : {metrics['R²']:.2f}")
    plot_model_outputs(y_true, pd.Series(hybrid_pred), model_name="fusion")

def main():
    parser = argparse.ArgumentParser(description="Run weather forecasting models.")
    parser.add_argument("--model", choices=["ml", "hybrid", "lstm", "fusion"], required=True)
    args = parser.parse_args()

    if args.model == "lstm":
        run_lstm()
    elif args.model == "ml":
        run_ml()
    elif args.model == "hybrid":
        run_hybrid()
    elif args.model == "fusion":
        run_fusion()

if __name__ == "__main__":
    main()