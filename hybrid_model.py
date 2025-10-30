# hybrid_model.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def create_sequences(data, target_col='temperature_celsius', seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length].values)
        y.append(data.iloc[i+seq_length][target_col])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def build_hybrid_model(df, features=None, target='temperature_celsius', seq_length=24, epochs=20):
    if features is None:
        features = ['temperature_celsius', 'humidity', 'pressure_mb']

    df = df[features].dropna()

    # Scale all features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=features, index=df.index)

    # Create sequences
    X, y = create_sequences(scaled_df, target_col=target, seq_length=seq_length)

    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Build and train model
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)

    # Predict
    y_pred = model.predict(X_test).flatten()

    # Inverse transform temperature only
    temp_scaler = MinMaxScaler()
    temp_scaler.fit(df[[target]])
    y_test_inv = temp_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = temp_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Align with datetime index
    index = df.index[-len(y_test):]
    y_test_series = pd.Series(y_test_inv, index=index)
    y_pred_series = pd.Series(y_pred_inv, index=index)

    return y_test_series, y_pred_series