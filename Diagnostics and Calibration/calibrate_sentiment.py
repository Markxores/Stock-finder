import yfinance as yf
import pandas as pd
import pysentiment2 as ps
from universe import STOCK_UNIVERSES

all_tickers = set()
for basket in STOCK_UNIVERSES.values():
    for ticker in basket.values():
        all_tickers.add(ticker)

all_tickers = sorted(all_tickers)
print(f"Found {len(all_tickers)} unique tickers.\n")

lm = ps.LM()
MIN_OI_LEG = 30
MIN_VOL_LEG = 20
results = []

for i, ticker in enumerate(all_tickers):
    print(f"[{i+1}/{len(all_tickers)}] {ticker}...")
    row = {"Ticker": ticker}
    asset = yf.Ticker(ticker)

    # --- Options PCR: front 3 expirations, summed, per-leg liquidity checks ---
    try:
        expirations = asset.options
        if expirations:
            total_call_vol, total_put_vol = 0, 0
            total_call_oi, total_put_oi = 0, 0

            for exp in expirations[:3]:
                try:
                    chain = asset.option_chain(exp)
                    total_call_vol += chain.calls['volume'].fillna(0).sum()
                    total_put_vol += chain.puts['volume'].fillna(0).sum()
                    total_call_oi += chain.calls['openInterest'].fillna(0).sum()
                    total_put_oi += chain.puts['openInterest'].fillna(0).sum()
                except Exception:
                    continue

            if total_call_oi > 0 or total_call_vol > 0:
                total_vol = total_call_vol + total_put_vol

                oi_pcr = None
                if total_call_oi >= MIN_OI_LEG and total_put_oi >= MIN_OI_LEG:
                    oi_pcr = total_put_oi / total_call_oi

                vol_pcr = None
                if total_call_vol >= MIN_VOL_LEG and total_put_vol >= MIN_VOL_LEG:
                    vol_pcr = total_put_vol / total_call_vol

                if total_vol < 500:
                    blended_pcr = oi_pcr
                elif vol_pcr is not None and oi_pcr is not None:
                    blended_pcr = (vol_pcr * 0.50) + (oi_pcr * 0.50)
                elif oi_pcr is not None:
                    blended_pcr = oi_pcr
                elif vol_pcr is not None:
                    blended_pcr = vol_pcr
                else:
                    blended_pcr = None

                row["Blended_PCR"] = blended_pcr
                row["Total_Call_OI"] = total_call_oi
                row["Total_Put_OI"] = total_put_oi
                row["Total_Call_Vol"] = total_call_vol
                row["Total_Put_Vol"] = total_put_vol
                row["Total_Vol"] = total_vol
                row["Used_Blend"] = (total_vol >= 500 and vol_pcr is not None and oi_pcr is not None)
            else:
                row["Blended_PCR"] = None
        else:
            row["Blended_PCR"] = None
    except Exception:
        row["Blended_PCR"] = None

    # --- News polarity (unchanged from last fix) ---
    try:
        news_items = asset.news
        if news_items:
            total_polarity = 0
            count = 0
            matched = 0
            seen = set()

            for item in news_items[:15]:
                content = item.get('content', {})
                title = content.get('title', '')
                summary = content.get('summary', '')
                text = f"{title}. {summary}".strip('. ')
                text_lower = text.lower()

                if not text_lower:
                    continue
                if text_lower in seen:
                    continue
                seen.add(text_lower)

                tokens = lm.tokenize(text)
                score_dict = lm.get_score(tokens)
                polarity = score_dict.get('Polarity', 0.0)

                if score_dict.get('Positive', 0) > 0 or score_dict.get('Negative', 0) > 0:
                    matched += 1

                total_polarity += polarity
                count += 1

            if count > 0:
                row["Avg_Polarity"] = total_polarity / count
            else:
                row["Avg_Polarity"] = None
        else:
            row["Avg_Polarity"] = None
    except Exception:
        row["Avg_Polarity"] = None

    results.append(row)

df = pd.DataFrame(results)
df.to_csv("sentiment_calibration_raw_v2.csv", index=False)
print(f"\n✅ Saved {len(df)} rows.\n")

# --- PCR summary ---
valid_pcr = df['Blended_PCR'].dropna()
print("--- PCR SUMMARY (3-expiration aggregated, per-leg filtered) ---")
print(f"n={len(valid_pcr)}  min={valid_pcr.min():.3f}  max={valid_pcr.max():.3f}  "
      f"mean={valid_pcr.mean():.3f}  median={valid_pcr.median():.3f}  std={valid_pcr.std():.3f}")

# --- How often was the blend actually used vs OI-only fallback? ---
if 'Used_Blend' in df.columns:
    blend_rate = df['Used_Blend'].dropna().mean() * 100
    print(f"\nUsed 50/50 blend (total_vol >= 500 AND both legs valid): {blend_rate:.1f}% of readings")

# --- Total volume distribution, to sanity-check the 500 threshold ---
if 'Total_Vol' in df.columns:
    valid_totvol = df['Total_Vol'].dropna()
    print(f"\nTotal_Vol (summed across 3 expirations): "
          f"min={valid_totvol.min():.0f}  median={valid_totvol.median():.0f}  "
          f"mean={valid_totvol.mean():.0f}  max={valid_totvol.max():.0f}")
    pct_above_500 = (valid_totvol >= 500).mean() * 100
    print(f"pct of readings with Total_Vol >= 500: {pct_above_500:.1f}%")

# --- News polarity summary (unchanged) ---
valid_pol = df['Avg_Polarity'].dropna()
print("\n--- NEWS POLARITY SUMMARY ---")
if len(valid_pol) > 0:
    print(f"n={len(valid_pol)}  min={valid_pol.min():.4f}  max={valid_pol.max():.4f}  "
          f"mean={valid_pol.mean():.4f}  median={valid_pol.median():.4f}  std={valid_pol.std():.4f}")