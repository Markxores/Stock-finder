import yfinance as yf
import pandas as pd
from universe import STOCK_UNIVERSES

all_tickers = set()
for basket in STOCK_UNIVERSES.values():
    for ticker in basket.values():
        all_tickers.add(ticker)

all_tickers = sorted(all_tickers)
print(f"Found {len(all_tickers)} unique tickers.\n")

results = []

for i, ticker in enumerate(all_tickers):
    print(f"[{i+1}/{len(all_tickers)}] {ticker}...")
    try:
        asset = yf.Ticker(ticker)
        info = asset.info

        revenue_growth = info.get('revenueGrowth')
        earnings_growth = info.get('earningsGrowth')
        net_margin = info.get('profitMargins')
        roe = info.get('returnOnEquity')

        results.append({
            "Ticker": ticker,
            "Revenue_Growth": revenue_growth,
            "Earnings_Growth": earnings_growth,
            "Net_Margin": net_margin,
            "ROE": roe,
        })
    except Exception:
        results.append({"Ticker": ticker, "Revenue_Growth": None, "Earnings_Growth": None,
                         "Net_Margin": None, "ROE": None})

df = pd.DataFrame(results)
df.to_csv("growth_profitability_calibration.csv", index=False)
print(f"\n✅ Saved {len(df)} rows.\n")


def check_metric(name, series, pre_clip_low, pre_clip_high, test_constants):
    valid = series.dropna()
    clipped_raw = valid.clip(pre_clip_low, pre_clip_high)

    print(f"\n--- {name} ---")
    print(f"n={len(valid)}  raw_min={valid.min():.2f}  raw_max={valid.max():.2f}  "
          f"raw_median={valid.median():.3f}  raw_std={valid.std():.3f}")
    print(f"After pre-clip [{pre_clip_low}, {pre_clip_high}]: "
          f"pct_hit_preclip_bound={((valid < pre_clip_low) | (valid > pre_clip_high)).mean()*100:.1f}%")

    for c in test_constants:
        scaled = (clipped_raw * c).clip(-100, 100)
        pct_clipped_final = (scaled.abs() >= 99.5).mean() * 100
        print(f"  ×{c}: median_scaled={clipped_raw.median()*c:.1f}  "
              f"std_scaled={clipped_raw.std()*c:.1f}  pct_at_100_ceiling={pct_clipped_final:.1f}%")


check_metric("Earnings Growth (pre-clip -1.0 to 1.5)", df['Earnings_Growth'],
             -1.0, 1.5, [40, 50, 65, 80])

check_metric("ROE (pre-clip -0.5 to 1.0)", df['ROE'],
             -0.5, 1.0, [70, 85, 100, 120])