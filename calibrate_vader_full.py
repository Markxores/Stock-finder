import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from universe import STOCK_UNIVERSES

all_tickers = set()
for basket in STOCK_UNIVERSES.values():
    for ticker in basket.values():
        all_tickers.add(ticker)

all_tickers = sorted(all_tickers)
print(f"Found {len(all_tickers)} unique tickers.\n")

vader = SentimentIntensityAnalyzer()
per_ticker_records = []
all_headline_scores = []

for i, ticker in enumerate(all_tickers):
    print(f"[{i+1}/{len(all_tickers)}] {ticker}...")
    try:
        asset = yf.Ticker(ticker)
        news_items = asset.news

        if not news_items:
            per_ticker_records.append({"Ticker": ticker, "Headline_Count": 0, "Avg_Compound": None})
            continue

        seen = set()
        compounds = []

        for item in news_items[:15]:
            content = item.get('content', {})
            title = content.get('title', '')
            summary = content.get('summary', '')
            text = f"{title}. {summary}".strip('. ')
            text_lower = text.lower()

            if not text_lower or text_lower in seen:
                continue
            seen.add(text_lower)

            compound = vader.polarity_scores(text)['compound']
            compounds.append(compound)
            all_headline_scores.append({"Ticker": ticker, "Compound": compound, "Text": text[:100]})

        if compounds:
            avg_compound = sum(compounds) / len(compounds)
            per_ticker_records.append({
                "Ticker": ticker,
                "Headline_Count": len(compounds),
                "Avg_Compound": avg_compound
            })
        else:
            per_ticker_records.append({"Ticker": ticker, "Headline_Count": 0, "Avg_Compound": None})

    except Exception:
        per_ticker_records.append({"Ticker": ticker, "Headline_Count": 0, "Avg_Compound": None})

ticker_df = pd.DataFrame(per_ticker_records)
headline_df = pd.DataFrame(all_headline_scores)

ticker_df.to_csv("vader_per_ticker_calibration.csv", index=False)
headline_df.to_csv("vader_per_headline_calibration.csv", index=False)

print(f"\n✅ Saved {len(ticker_df)} ticker-level rows and {len(headline_df)} headline-level rows.\n")

# --- Headline count / density check ---
valid_counts = ticker_df['Headline_Count']
print("--- HEADLINE COUNT PER TICKER ---")
print(f"min={valid_counts.min()}  max={valid_counts.max()}  median={valid_counts.median()}  "
      f"mean={valid_counts.mean():.1f}")
print(f"Tickers with 0 headlines: {(valid_counts == 0).sum()} / {len(ticker_df)}")
print(f"Tickers with < 5 headlines: {(valid_counts < 5).sum()} / {len(ticker_df)}")

# --- Ticker-level average compound distribution (this is what actually gets scored) ---
valid_avg = ticker_df['Avg_Compound'].dropna()
print("\n--- TICKER-LEVEL AVG COMPOUND DISTRIBUTION (this is the real scoring input) ---")
print(f"n={len(valid_avg)}  min={valid_avg.min():.3f}  max={valid_avg.max():.3f}  "
      f"mean={valid_avg.mean():.3f}  median={valid_avg.median():.3f}  std={valid_avg.std():.3f}")

# --- Headline-level distribution, for reference ---
valid_headlines = headline_df['Compound']
print("\n--- INDIVIDUAL HEADLINE COMPOUND DISTRIBUTION (raw, pre-averaging) ---")
print(f"n={len(valid_headlines)}  min={valid_headlines.min():.3f}  max={valid_headlines.max():.3f}  "
      f"mean={valid_headlines.mean():.3f}  median={valid_headlines.median():.3f}  std={valid_headlines.std():.3f}")

# --- Clip-rate check using the TICKER-LEVEL average (the actual scoring basis) ---
print("\n--- CLIP-RATE CHECK ON TICKER-LEVEL AVERAGE (candidate multipliers) ---")
for mult in [50, 75, 100, 125, 150]:
    scaled = (valid_avg * mult).clip(-100, 100)
    pct_clip = (scaled.abs() >= 99.5).mean() * 100
    print(f"  ×{mult}: median_scaled={valid_avg.median()*mult:.1f}  std_scaled={valid_avg.std()*mult:.1f}  "
          f"pct_clipped={pct_clip:.1f}%")