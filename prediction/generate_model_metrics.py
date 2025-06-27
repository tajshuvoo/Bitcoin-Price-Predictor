import os
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Import root_mean_squared_error if available (sklearn ≥ 1.4)
try:
    from sklearn.metrics import root_mean_squared_error
    HAS_ROOT_RMSE = True
except ImportError:
    HAS_ROOT_RMSE = False

import tensorflow as tf
from tensorflow import keras

# 📁 Configuration
MODEL_DIR = "model/btc_only"
SCALER_PATH = os.path.join(MODEL_DIR, "btc_scaler_1day.pkl")
DATA_PATH = "data/raw/btc_price_1y.csv"
FEATURES = [
    "open", "high", "low", "close", "volume",
    "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume"
]

# 🧠 Mapping model filenames to sequence lengths
MODEL_CONFIG = {
    "transformer_model_15min.keras": 1,
    "transformer_model_1hour.keras": 4,
    "transformer_model_6hour.keras": 24,
    "transformer_model_12hour.keras": 48,
    "transformer_model_18hour.keras": 72,
    "transformer_model_1day.keras": 96,
    "transformer_model_2days.keras": 192,
    "transformer_model_3days.keras": 288,
    "transformer_model_4days.keras": 384,
    "transformer_model_5days.keras": 480,
    "transformer_model_6days.keras": 576,
    "transformer_model_7days.keras": 672,
}

# 📄 Load and scale data
btc_data = pd.read_csv(DATA_PATH, parse_dates=['timestamp'], index_col='timestamp')
feature_data = btc_data[FEATURES]
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
feature_data_scaled = scaler.transform(feature_data)

# 🔁 Sequence creator
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 3])  # "close" price
    return np.array(X), np.array(y)

# 📊 Evaluate models
results = {}
for model_file, seq_len in MODEL_CONFIG.items():
    model_path = os.path.join(MODEL_DIR, model_file)
    print(f"✅ Evaluating {model_file} with sequence length {seq_len}")
    try:
        model = keras.models.load_model(model_path)
        X_all, y_all = create_sequences(feature_data_scaled, seq_len)

        # Use last 20% as test set
        test_size = int(len(X_all) * 0.2)
        X_test = X_all[-test_size:]
        y_test = y_all[-test_size:]
        y_pred = model.predict(X_test, verbose=0)

        # 🧮 Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        if HAS_ROOT_RMSE:
            from sklearn.metrics import root_mean_squared_error
            rmse = root_mean_squared_error(y_test, y_pred)
        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100

        results[model_file.replace(".keras", "")] = {
            "MAE": round(mae, 6),
            "RMSE": round(rmse, 6),
            "R2": round(r2, 6),
            "MAPE (%)": round(mape, 6)
        }
    except Exception as e:
        results[model_file.replace(".keras", "")] = {"error": str(e)}

# 💾 Save results
output_path = os.path.join(MODEL_DIR, "model_metrics.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n📂 Metrics saved to {output_path}")
