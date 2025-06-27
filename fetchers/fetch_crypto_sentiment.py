# fetchers/stocktwits_data.py
import os, time, csv
from datetime import datetime, timedelta
import requests

OUT = "data/raw/stocktwits_15min.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

INTERVAL = timedelta(minutes=15)
START = datetime(2024,6,21,13,15)
END   = datetime(2025,6,21,13,0)
MAX_PER_INT = 2

def fetch_interval(start):
    since = int(start.timestamp())
    r = requests.get("https://api.stocktwits.com/api/2/streams/symbol/BTC.json", params={"since": since})
    r.raise_for_status()
    return r.json().get("messages", [])[:MAX_PER_INT]

def run():
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["interval_start", "user", "body", "likes"])
        cur = START
        while cur < END:
            msgs = fetch_interval(cur)
            for m in msgs:
                w.writerow([cur, m["user"]["username"], m["body"], m["likes"]["total"]])
            print(cur, "→", cur+INTERVAL, ":", len(msgs))
            time.sleep(1)
            cur += INTERVAL

if __name__ == "__main__":
    run()
