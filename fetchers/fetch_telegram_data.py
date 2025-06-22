from telethon.sync import TelegramClient
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from config import api_id, api_hash, session_name, channels

SAVE_PATH = "data/raw"
os.makedirs(SAVE_PATH, exist_ok=True)
OUTPUT_FILE = os.path.join(SAVE_PATH, "telegram_1year_grouped.csv")

def fetch_messages_last_year(client, channel_username):
    messages = []
    one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)
    print(f"Fetching from @{channel_username} starting {one_year_ago.date()}...")

    try:
        client.get_entity(channel_username)  # validate channel existence
    except Exception as e:
        print(f"Error accessing @{channel_username}: {e}. Skipping.")
        return messages

    try:
        for message in client.iter_messages(channel_username, reverse=True):
            if message.date < one_year_ago:
                continue
            if message.text:
                messages.append({
                    "channel": channel_username,
                    "date": message.date,
                    "text": message.text,
                    "views": message.views,
                    "forwards": message.forwards,
                    "replies": message.replies.replies if message.replies else None
                })
    except Exception as e:
        print(f"Error fetching messages from @{channel_username}: {e}. Continuing with next channel.")
    
    print(f"Fetched {len(messages)} messages from @{channel_username}")
    return messages


def group_by_15_min_intervals(messages, max_per_interval=2):
    if not messages:
        return pd.DataFrame()  # empty df if no messages

    df = pd.DataFrame(messages)
    df['interval'] = pd.to_datetime(df['date']).dt.floor('15min')
    
    grouped = (
        df.sort_values("date")
          .groupby(["channel", "interval"])
          .head(max_per_interval)
          .reset_index(drop=True)
    )
    return grouped

if __name__ == "__main__":
    all_messages = []
    with TelegramClient(session_name, api_id, api_hash) as client:
        for channel in channels:
            channel_msgs = fetch_messages_last_year(client, channel)
            all_messages.extend(channel_msgs)

    print("✅ Grouping messages into 15-minute intervals...")
    grouped_df = group_by_15_min_intervals(all_messages, max_per_interval=2)

    grouped_df.to_csv(OUTPUT_FILE, index=False)
    print(f"📁 Saved {len(grouped_df)} grouped messages to {OUTPUT_FILE}")
