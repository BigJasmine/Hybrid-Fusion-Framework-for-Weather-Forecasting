import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
import matplotlib.pyplot as plt

def prepare_data(df, features=None, target='temperature_celsius', n_steps=24):
    if features is None:
        features = ['temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm', 'cloud']

    df = df[features].dropna().copy()
    print(f"🔢 Total rows after dropna: {len(df)}")

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    target_index = features.index(target)
    X, y = [], []
    for i in range(n_steps, len(data_scaled)):
        X.append(data_scaled[i - n_steps:i])
        y.append(data_scaled[i, target_index])

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    print(f"🧮 Total sequences: {len(X)}")
    return X, y, scaler, df.index[n_steps:]

def inverse_transform_target(scaler, scaled_target, features, target):
    stacked = np.vstack([
        scaled_target.T if i == features.index(target) else np.zeros_like(scaled_target.T)
        for i in range(len(features))
    ]).T
    return scaler.inverse_transform(stacked)[:, features.index(target)]

def train_lstm(df, features=None, target='temperature_celsius', n_steps=24, epochs=20, batch_size=32):
    X, y, scaler, index = prepare_data(df, features, target, n_steps)

    print(f"📦 Batch size: {batch_size}")
    print(f"📊 Batches per epoch: {len(X) // batch_size}")

    model = Sequential([
        LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])

    # Optional: Plot loss curve
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    y_pred_scaled = model.predict(X)
    y_pred = inverse_transform_target(scaler, y_pred_scaled, features, target)
    y_true = inverse_transform_target(scaler, y, features, target)

    y_true_series = pd.Series(y_true.flatten(), index=index)
    y_pred_series = pd.Series(y_pred.flatten(), index=index)

    return y_true_series, y_pred_series