import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Validation Alignment
# -----------------------------
def align_validation_targets(df, target_column, val_size):
    """
    Aligns validation targets for consistent evaluation across models.
    Returns cleaned y_true and shared validation index.
    """
    val_index = df.index[-val_size:]
    y_true = df.loc[val_index, target_column]
    y_true_clean = y_true.dropna().reset_index(drop=True)
    return y_true_clean, val_index

# -----------------------------
# Fusion Logic
# -----------------------------
def weighted_fusion(rf_pred, lstm_pred, weight_rf=0.5):
    """
    Performs weighted averaging of RF and LSTM predictions.
    Returns fused prediction.
    """
    weight_lstm = 1 - weight_rf
    return weight_rf * np.array(rf_pred) + weight_lstm * np.array(lstm_pred)

def optimize_fusion_weight(rf_pred, lstm_pred, y_true, weights=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    Searches for the best fusion weight based on RMSE.
    Returns fused prediction using optimal weight.
    """
    best_rmse = float('inf')
    best_weight = None
    for w in weights:
        hybrid = weighted_fusion(rf_pred, lstm_pred, weight_rf=w)
        rmse = np.sqrt(mean_squared_error(y_true, hybrid))
        print(f"RF weight: {w:.1f}, LSTM weight: {1-w:.1f}, RMSE: {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = w
    print(f"\n✅ Best fusion weight: RF={best_weight:.2f}, LSTM={1-best_weight:.2f}, RMSE={best_rmse:.4f}")
    return weighted_fusion(rf_pred, lstm_pred, weight_rf=best_weight)

# -----------------------------
# Evaluation
# -----------------------------
def evaluate_predictions(y_true, y_pred):
    """
    Computes MAE, RMSE, and R² for given predictions.
    Returns a dictionary of metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R²": r2}

# -----------------------------
# Debugging
# -----------------------------
def log_mismatch(y1, y2, label1="y1", label2="y2"):
    """
    Logs mismatches between two Series for debugging.
    """
    if not y1.equals(y2):
        print(f"[Mismatch] {label1} vs {label2}")
        print(f"{label1} shape: {y1.shape}, {label2} shape: {y2.shape}")
        print(f"{label1} head:\n{y1.head()}")
        print(f"{label2} head:\n{y2.head()}")