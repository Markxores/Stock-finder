import yfinance as yf
import pandas as pd
from universe import STOCK_UNIVERSES
from sector_map import TICKER_TRUE_SECTOR

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
        sector = TICKER_TRUE_SECTOR.get(ticker, "UNMAPPED")

        results.append({
            "Ticker": ticker,
            "Sector": sector,
            "EV_EBITDA": ev_ebitda
        })

    except Exception as e:
        results.append({"Ticker": ticker, "Sector": "ERROR", "EV_EBITDA": None})

df = pd.DataFrame(results)
df.to_csv("ev_ebitda_calibration_raw.csv", index=False)
print(f"\n✅ Saved raw data to ev_ebitda_calibration_raw.csv\n")

# --- Compute median EV/EBITDA per sector ---
valid_df = df[df['EV_EBITDA'].notna() & (df['Sector'] != "UNMAPPED") & (df['Sector'] != "ERROR")]

print("--- MEDIAN EV/EBITDA BY SECTOR ---\n")
sector_medians = {}

for sector in sorted(valid_df['Sector'].unique()):
    sector_data = valid_df[valid_df['Sector'] == sector]['EV_EBITDA']
    median_val = sector_data.median()
    n = len(sector_data)
    sector_medians[sector] = round(median_val, 2)
    print(f"{sector}: median={median_val:.2f}  (n={n})")
    print(f"   values: {sorted(sector_data.round(1).tolist())}")

# --- Output as a ready-to-paste Python dictionary ---
print("\n\n--- READY-TO-PASTE DICTIONARY ---\n")
print("SECTOR_EV_EBITDA_NEUTRAL = {")
for sector, median_val in sector_medians.items():
    print(f'    "{sector}": {median_val},')
print("}")