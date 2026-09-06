import pandas as pd

df = pd.read_csv("vader_per_ticker_calibration.csv")
valid_avg = df['Avg_Compound'].dropna()

BASELINE = 0.365
de_meaned = valid_avg - BASELINE

for mult in [350, 400, 450]:
    scaled = (de_meaned * mult).clip(-100, 100)
    pct_clip = (scaled.abs() >= 99.5).mean() * 100
    print(f"  ×{mult}: median_scaled={de_meaned.median()*mult:.1f}  std_scaled={de_meaned.std()*mult:.1f}  "
          f"pct_clipped={pct_clip:.1f}%")