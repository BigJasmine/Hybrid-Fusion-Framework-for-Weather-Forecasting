# model_builder.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore

def train_random_forest(df, features, target):
    df = df.dropna()

    X = df[features]
    y = df[target]

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 🔍 Plot feature importance
    importances = model.feature_importances_
    plt.figure(figsize=(8, 4))
    plt.barh(X.columns, importances, color='steelblue')
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    os.makedirs("charts", exist_ok=True)
    plt.savefig("charts/ml_feature_importance.png", dpi=300)
    plt.close()
    print("🖼️ ML feature importance chart saved as: charts/ml_feature_importance.png")

    return pd.Series(y_test.values, index=y_test.index), pd.Series(y_pred, index=y_test.index)

def train_lstm(df, target='temperature_celsius', n_steps=24, epochs=20):
    df = df.dropna()
    df = df[[target]].copy()

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(n_steps, len(data_scaled)):
        X.append(data_scaled[i - n_steps:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=2)

    print("✅ Training completed. All epochs finished.")  # 👈 Add this directly below

    y_pred_scaled = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y)

    y_true = pd.Series(y_true.flatten(), index=df.index[n_steps:])
    y_pred = pd.Series(y_pred.flatten(), index=df.index[n_steps:])

    return y_true, y_pred

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2