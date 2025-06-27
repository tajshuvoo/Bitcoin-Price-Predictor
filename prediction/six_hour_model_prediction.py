import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from datetime import datetime, timedelta
from binance.client import Client

def predict_next_day_with_six_hour_model():
    model = keras.models.load_model("model/btc_only/transformer_model_6hour.keras")
    with open("model/btc_only/btc_scaler_1day.pkl", "rb") as f:
        scaler = pickle.load(f)

    client = Client()
    symbol = "BTCUSDT"
    interval = Client.KLINE_INTERVAL_15MINUTE
    limit = 24

    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)

    features = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume"
    ]
    data = df[features]
    scaled_data = scaler.transform(data)

    SEQ_LENGTH = 24
    current_sequence = scaled_data.copy()
    predictions_scaled = []
    timestamps = []

    last_timestamp = df.index[-1]

    for step in range(96):
        input_seq = np.expand_dims(current_sequence[-SEQ_LENGTH:], axis=0)
        pred_scaled = model.predict(input_seq, verbose=0)[0]
        predictions_scaled.append(pred_scaled)

        new_row = np.zeros((len(features),))
        new_row[3] = pred_scaled
        new_row[0] = current_sequence[-1][0]
        new_row[1] = current_sequence[-1][1]
        new_row[2] = current_sequence[-1][2]
        new_row[4:] = current_sequence[-1][4:]
        current_sequence = np.vstack([current_sequence, new_row])
        timestamps.append(last_timestamp + timedelta(minutes=15 * (step + 1)))

    dummy_array = np.zeros((96, len(features)))
    dummy_array[:, 3] = predictions_scaled
    inverse = scaler.inverse_transform(dummy_array)
    predicted_close_prices = inverse[:, 3]

    return [{"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "predicted_price": float(price)}
            for ts, price in zip(timestamps, predicted_close_prices)]
