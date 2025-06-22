# fetchers/fetch_btc_data.py

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

def fetch_binance_klines(symbol="BTCUSDT", interval="15m", start_str=None, end_str=None):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": 1000  # Binance max per call
    }

    df_all = pd.DataFrame()

    if end_str:
        end_time = int(pd.Timestamp(end_str).timestamp() * 1000)
    else:
        end_time = int(time.time() * 1000)

    if start_str:
        start_time = int(pd.Timestamp(start_str).timestamp() * 1000)
    else:
        start_time = end_time - (365 * 24 * 60 * 60 * 1000)

    while start_time < end_time:
        params["startTime"] = start_time
        params["endTime"] = start_time + 1000 * 15 * 60 * 1000  # 1000 candles of 15min
        response = requests.get(url, params=params)
        data = response.json()

        if not data:
            break

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df_all = pd.concat([df_all, df], ignore_index=True)

        start_time = int(df["timestamp"].iloc[-1].timestamp() * 1000) + 1
        time.sleep(0.1)  # respect Binance API

    return df_all

if __name__ == "__main__":
    print("Fetching 1 year of BTC price data (15-minute intervals)...")
    df = fetch_binance_klines()
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/btc_price_1y.csv", index=False)
    print("Saved to data/raw/btc_price_1y.csv")
