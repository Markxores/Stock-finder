import pandas as pd

df = pd.read_csv("sentiment_calibration_raw_v2.csv")
valid_pcr = df['Blended_PCR'].dropna()

print(f"n={len(valid_pcr)}")
print(f"\n--- PERCENTILE DISTRIBUTION ---")
for p in [5, 10, 15, 25, 50, 75, 85, 90, 95]:
    print(f"  {p}th percentile: {valid_pcr.quantile(p/100):.3f}")

print(f"\nmedian={valid_pcr.median():.3f}  std={valid_pcr.std():.3f}")