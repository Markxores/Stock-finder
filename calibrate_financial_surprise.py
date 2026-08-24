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
    row = {"Ticker": ticker}
    asset = yf.Ticker(ticker)

    # --- Earnings surprise ---
    try:
        earnings_dates = asset.earnings_dates
        if earnings_dates is not None and not earnings_dates.empty:
            reported = earnings_dates.dropna(subset=['Reported EPS'])
            if not reported.empty:
                reported = reported.sort_index(ascending=False)
                latest = reported.iloc[0]
                eps_estimate = latest.get('EPS Estimate')
                eps_reported = latest.get('Reported EPS')
                surprise_pct = latest.get('Surprise(%)')

                if surprise_pct is None or pd.isna(surprise_pct):
                    if eps_estimate is not None and eps_estimate != 0 and eps_reported is not None:
                        surprise_pct = ((eps_reported - eps_estimate) / abs(eps_estimate)) * 100
                    else:
                        surprise_pct = None

                row["Surprise_Pct"] = surprise_pct
            else:
                row["Surprise_Pct"] = None
        else:
            row["Surprise_Pct"] = None
    except Exception:
        row["Surprise_Pct"] = None

    # --- EPS estimate revision (30-day, current quarter) ---
    try:
        eps_trend = asset.eps_trend
        if eps_trend is not None and not eps_trend.empty and '0q' in eps_trend.index:
            trend_row = eps_trend.loc['0q']
            current_est = trend_row.get('current')
            past_est = trend_row.get('30daysAgo')

            if (current_est is not None and past_est is not None and
                    not pd.isna(current_est) and not pd.isna(past_est) and past_est != 0):
                revision_pct = ((current_est - past_est) / abs(past_est)) * 100
                row["Revision_Pct"] = revision_pct
            else:
                row["Revision_Pct"] = None
        else:
            row["Revision_Pct"] = None
    except Exception:
        row["Revision_Pct"] = None

    results.append(row)

df = pd.DataFrame(results)
df.to_csv("financial_surprise_calibration.csv", index=False)
print(f"\n✅ Saved {len(df)} rows.\n")


def check_metric(name, series, pre_clip_low, pre_clip_high, test_constants):
    valid = series.dropna()
    clipped_raw = valid.clip(pre_clip_low, pre_clip_high)

    print(f"\n--- {name} ---")
    print(f"n={len(valid)}  raw_min={valid.min():.2f}  raw_max={valid.max():.2f}  "
          f"raw_median={valid.median():.2f}  raw_std={valid.std():.2f}")
    print(f"pct_hit_preclip_bound={((valid < pre_clip_low) | (valid > pre_clip_high)).mean()*100:.1f}%")

    for c in test_constants:
        scaled = (clipped_raw * c).clip(-100, 100)
        pct_clipped = (scaled.abs() >= 99.5).mean() * 100
        print(f"  ×{c}: median_scaled={clipped_raw.median()*c:.1f}  "
              f"std_scaled={clipped_raw.std()*c:.1f}  pct_at_100_ceiling={pct_clipped:.1f}%")


check_metric("Earnings Surprise (pre-clip -100 to 150)", df['Surprise_Pct'],
             -100, 150, [0.3, 0.4, 0.6, 0.8, 1.0])

check_metric("EPS Estimate Revision (pre-clip -100 to 100)", df['Revision_Pct'],
             -100, 100, [0.5, 1.0, 1.5, 2.0, 3.0])