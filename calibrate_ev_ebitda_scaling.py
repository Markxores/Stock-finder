import yfinance as yf
import pandas as pd
from universe import STOCK_UNIVERSES
from sector_map import TICKER_TRUE_SECTOR
from fundamentals import SECTOR_EV_EBITDA_NEUTRAL

# --- Deduplicate every ticker across all baskets ---
all_tickers = set()
for basket in STOCK_UNIVERSES.values():
    for ticker in basket.values():
        all_tickers.add(ticker)

all_tickers = sorted(all_tickers)
print(f"Found {len(all_tickers)} unique tickers across the universe.\n")

results = []

for i, ticker in enumerate(all_tickers):
    print(f"[{i+1}/{len(all_tickers)}] {ticker}...")
    try:
        asset = yf.Ticker(ticker)
        info = asset.info

        ev_ebitda = info.get('enterpriseToEbitda')
        sector = TICKER_TRUE_SECTOR.get(ticker)
        sector_median = SECTOR_EV_EBITDA_NEUTRAL.get(sector) if sector else None

        # Apply the exact same sanity guard used in fundamentals.py
        if (sector_median is None or ev_ebitda is None or 
            ev_ebitda <= 0 or ev_ebitda > 100):
            results.append({
                "Ticker": ticker, "Sector": sector, "EV_EBITDA": ev_ebitda,
                "Sector_Median": sector_median, "Gap_Pct": None, "Status": "Excluded"
            })
            continue

        gap_pct = ((sector_median - ev_ebitda) / sector_median) * 100
        results.append({
            "Ticker": ticker, "Sector": sector, "EV_EBITDA": ev_ebitda,
            "Sector_Median": sector_median, "Gap_Pct": gap_pct, "Status": "OK"
        })

    except Exception:
        results.append({"Ticker": ticker, "Sector": None, "EV_EBITDA": None,
                         "Sector_Median": None, "Gap_Pct": None, "Status": "Error"})

df = pd.DataFrame(results)
df.to_csv("ev_ebitda_gap_calibration.csv", index=False)

valid = df[df['Status'] == 'OK']['Gap_Pct']
print(f"\n✅ Saved {len(df)} rows. {len(valid)} valid (non-excluded) readings.\n")

print("--- GAP_PCT SUMMARY STATS ---")
print(f"n={len(valid)}")
print(f"min={valid.min():.2f}  max={valid.max():.2f}")
print(f"mean={valid.mean():.2f}  median={valid.median():.2f}  std={valid.std():.2f}")

print("\n--- DISTRIBUTION CHECK (how many would clip at various constants) ---")
for test_const in [1.0, 1.5, 2.0, 2.5, 3.0]:
    scaled = (valid * test_const).clip(-100, 100)
    pct_clipped = ((scaled.abs() >= 99.5).sum() / len(scaled)) * 100
    print(f"  ×{test_const}: median_scaled={valid.median()*test_const:.1f}  "
          f"std_scaled={valid.std()*test_const:.1f}  pct_clipped={pct_clipped:.1f}%")