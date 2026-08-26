import yfinance as yf
import json

asset = yf.Ticker("AAPL")
news_items = asset.news

print(f"Number of news items: {len(news_items)}\n")

if news_items:
    print("--- Raw structure of first news item ---")
    print(json.dumps(news_items[0], indent=2, default=str))