import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta

def fetch_last_seven_days():
    client = Client()
    symbol = "BTCUSDT"
    interval = Client.KLINE_INTERVAL_15MINUTE
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)

    klines = client.get_historical_klines(symbol, interval, start_str=start_time, end_str=end_time)
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)

    result = [
        {
            "timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"),
            "close_price": row["close"]
        }
        for idx, row in df.iterrows()
    ]
    return result
