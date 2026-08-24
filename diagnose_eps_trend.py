import yfinance as yf
from universe import STOCK_UNIVERSES

# Sample a handful of tickers across different baskets, not the whole universe
sample_tickers = ["AAPL", "MSFT", "NVDA", "JNJ", "XOM", "NEE", "PLTR", "SOFI", "F", "KO"]

for ticker in sample_tickers:
    print(f"\n=== {ticker} ===")
    try:
        asset = yf.Ticker(ticker)
        eps_trend = asset.eps_trend
        print(eps_trend)
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")