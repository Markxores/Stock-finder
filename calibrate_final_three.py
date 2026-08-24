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

        net_margin = info.get('profitMargins')
        debt_to_equity = info.get('debtToEquity')
        current_ratio = info.get('currentRatio')

        results.append({
            "Ticker": ticker,
            "Net_Margin": net_margin,
            "Debt_To_Equity": debt_to_equity,
            "Current_Ratio": current_ratio,
        })
    except Exception:
        results.append({"Ticker": ticker, "Net_Margin": None,
                         "Debt_To_Equity": None, "Current_Ratio": None})

df = pd.DataFrame(results)
df.to_csv("final_three_calibration.csv", index=False)
print(f"\n✅ Saved {len(df)} rows.\n")


def check_simple_metric(name, series, test_constants):
    """For metrics with no pre-clip step (Net_Margin)"""
    valid = series.dropna()
    print(f"\n--- {name} (no pre-clip) ---")
    print(f"n={len(valid)}  raw_min={valid.min():.3f}  raw_max={valid.max():.3f}  "
          f"raw_median={valid.median():.3f}  raw_std={valid.std():.3f}")
    for c in test_constants:
        scaled = (valid * c).clip(-100, 100)
        pct_clipped = (scaled.abs() >= 99.5).mean() * 100
        print(f"  ×{c}: median_scaled={valid.median()*c:.1f}  "
              f"std_scaled={valid.std()*c:.1f}  pct_at_100_ceiling={pct_clipped:.1f}%")


def check_preclip_metric(name, series, pre_clip_low, pre_clip_high, test_constants):
    """For metrics WITH a pre-clip step (Debt_To_Equity)"""
    valid = series.dropna()
    clipped_raw = valid.clip(pre_clip_low, pre_clip_high)
    print(f"\n--- {name} (pre-clip [{pre_clip_low}, {pre_clip_high}]) ---")
    print(f"n={len(valid)}  raw_min={valid.min():.2f}  raw_max={valid.max():.2f}  "
          f"raw_median={valid.median():.2f}  raw_std={valid.std():.2f}")
    print(f"pct_hit_preclip_bound={((valid < pre_clip_low) | (valid > pre_clip_high)).mean()*100:.1f}%")
    for c in test_constants:
        transformed = (100 - clipped_raw) * c
        scaled = transformed.clip(-100, 100)
        pct_clipped = (scaled.abs() >= 99.5).mean() * 100
        print(f"  ×{c}: median_scaled={((100 - clipped_raw.median())*c):.1f}  "
              f"std_scaled={(clipped_raw*c).std():.1f}  pct_at_100_ceiling={pct_clipped:.1f}%")


def check_offset_metric(name, series, offset, test_constants):
    """For metrics scored as (value - offset) * constant (Current_Ratio)"""
    valid = series.dropna()
    print(f"\n--- {name} (offset={offset}, no pre-clip) ---")
    print(f"n={len(valid)}  raw_min={valid.min():.2f}  raw_max={valid.max():.2f}  "
          f"raw_median={valid.median():.2f}  raw_std={valid.std():.2f}")
    for c in test_constants:
        scaled = ((valid - offset) * c).clip(-100, 100)
        pct_clipped = (scaled.abs() >= 99.5).mean() * 100
        print(f"  ×{c}: median_scaled={(valid.median()-offset)*c:.1f}  "
              f"std_scaled={(valid*c).std():.1f}  pct_at_100_ceiling={pct_clipped:.1f}%")


check_simple_metric("Net_Margin", df['Net_Margin'], [150, 200, 300, 400])
check_preclip_metric("Debt_To_Equity", df['Debt_To_Equity'], 0, 400, [0.3, 0.5, 0.8, 1.0])
check_offset_metric("Current_Ratio", df['Current_Ratio'], 1.5, [20, 30, 40, 50])