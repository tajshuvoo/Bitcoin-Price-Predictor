import praw
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# Ensure save path exists
SAVE_PATH = "data/raw"
os.makedirs(SAVE_PATH, exist_ok=True)

# Output CSV file
OUTPUT_FILE = os.path.join(SAVE_PATH, "reddit_bitcoin_posts.csv")

def fetch_posts(start_time, end_time, limit=2):
    """
    Fetches Reddit submissions from r/Bitcoin between start_time and end_time.
    Returns a list of dictionaries with post metadata.
    """
    posts = []
    for submission in reddit.subreddit("Bitcoin").search(
        query="bitcoin",
        sort="new",
        time_filter="all",
        limit=100
    ):
        created_utc = datetime.utcfromtimestamp(submission.created_utc)
        if start_time <= created_utc < end_time:
            posts.append({
                "id": submission.id,
                "title": submission.title,
                "selftext": submission.selftext,
                "created_utc": created_utc,
                "score": submission.score,
                "url": submission.url,
                "num_comments": submission.num_comments
            })
        if len(posts) >= limit:
            break
    return posts

def scrape_reddit():
    # 1 year = 365 days * 24 hours * 4 = 35040 intervals
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365)

    interval = timedelta(minutes=15)
    current_start = start_date
    all_posts = []

    total_intervals = int((end_date - start_date) / interval)
    print(f"Fetching ~{total_intervals} intervals of 15-minute posts...")

    for i in range(total_intervals):
        current_end = current_start + interval
        print(f"Interval {i+1}/{total_intervals}: {current_start} → {current_end}")

        try:
            posts = fetch_posts(current_start, current_end)
            all_posts.extend(posts)
        except Exception as e:
            print(f"Error fetching interval {i+1}: {e}")

        # Be nice to Reddit
        time.sleep(1)
        current_start = current_end

    # Save to CSV
    df = pd.DataFrame(all_posts)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} posts to {OUTPUT_FILE}")

if __name__ == "__main__":
    scrape_reddit()
