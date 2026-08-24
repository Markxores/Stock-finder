import pandas as pd

df = pd.read_csv("sentiment_calibration_raw_v2.csv")
valid = df['Total_Vol'].dropna()

pct_above_1000 = (valid >= 1000).mean() * 100
print(f"pct of readings with Total_Vol >= 1000: {pct_above_1000:.1f}%")