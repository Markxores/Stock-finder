import yfinance as yf
import pandas as pd
from universe import STOCK_UNIVERSES

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

        if not info or len(info) < 5:
            results.append({"Ticker": ticker, "Status": "No Data"})
            continue

        trailing_pe = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        price_to_sales = info.get('priceToSalesTrailing12Months')
        revenue_growth = info.get('revenueGrowth')
        earnings_growth = info.get('earningsGrowth')
        net_margin = info.get('profitMargins')
        roe = info.get('returnOnEquity')
        debt_to_equity = info.get('debtToEquity')
        current_ratio = info.get('currentRatio')

        pe_gap_pct = None
        ps_gap_pct = None
        valuation_path = None

        if trailing_pe is not None and trailing_pe > 0:
            valuation_path = "PE"
            if forward_pe is not None and forward_pe > 0:
                pe_gap_pct = ((trailing_pe - forward_pe) / trailing_pe) * 100
        else:
            valuation_path = "PS_FALLBACK"
            if price_to_sales is not None and price_to_sales > 0:
                try:
                    df = asset.history(period="1y")
                    if not df.empty and len(df) >= 100:
                        current_price = df['Close'].iloc[-1]
                        implied_rev_per_share = current_price / price_to_sales
                        if implied_rev_per_share > 0:
                            approx_ps_series = df['Close'] / implied_rev_per_share
                            avg_ps = approx_ps_series.mean()
                            if avg_ps > 0:
                                ps_gap_pct = ((avg_ps - price_to_sales) / avg_ps) * 100
                except Exception:
                    pass

        results.append({
            "Ticker": ticker,
            "Status": "OK",
            "Valuation_Path": valuation_path,
            "Trailing_PE": trailing_pe,
            "Forward_PE": forward_pe,
            "PE_Gap_Pct": pe_gap_pct,
            "Price_To_Sales": price_to_sales,
            "PS_Gap_Pct": ps_gap_pct,
            "Revenue_Growth": revenue_growth,
            "Earnings_Growth": earnings_growth,
            "Net_Margin": net_margin,
            "ROE": roe,
            "Debt_To_Equity": debt_to_equity,
            "Current_Ratio": current_ratio,
        })

    except Exception as e:
        results.append({"Ticker": ticker, "Status": f"Error: {str(e)}"})

df = pd.DataFrame(results)
df.to_csv("fundamentals_calibration_data.csv", index=False)

print(f"\n✅ Saved {len(df)} rows to fundamentals_calibration_data.csv")

# --- Quick summary stats for the metrics that matter for scaling ---
print("\n--- SUMMARY STATS (for calibration) ---")
numeric_cols = ["PE_Gap_Pct", "PS_Gap_Pct", "Revenue_Growth", "Earnings_Growth",
                 "Net_Margin", "ROE", "Debt_To_Equity", "Current_Ratio"]

for col in numeric_cols:
    if col in df.columns:
        valid = df[col].dropna()
        if len(valid) > 0:
            print(f"\n{col} (n={len(valid)}):")
            print(f"  min={valid.min():.2f}  max={valid.max():.2f}  mean={valid.mean():.2f}  "
                  f"median={valid.median():.2f}  std={valid.std():.2f}")