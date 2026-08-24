import pandas as pd

df = pd.read_csv("financial_surprise_calibration.csv")


def check_metric(name, series, pre_clip_low, pre_clip_high, test_constants):
    valid = series.dropna()
    clipped_raw = valid.clip(pre_clip_low, pre_clip_high)

    print(f"\n--- {name} ---")
    print(f"n={len(valid)}  raw_median={valid.median():.2f}  raw_std={valid.std():.2f}")

    for c in test_constants:
        scaled = (clipped_raw * c).clip(-100, 100)
        pct_clipped = (scaled.abs() >= 99.5).mean() * 100
        print(f"  ×{c}: median_scaled={clipped_raw.median()*c:.1f}  "
              f"std_scaled={clipped_raw.std()*c:.1f}  pct_at_100_ceiling={pct_clipped:.1f}%")


check_metric("Earnings Surprise (pre-clip -100 to 150)", df['Surprise_Pct'],
             -100, 150, [1.0, 1.2, 1.5, 1.8])

check_metric("EPS Estimate Revision (pre-clip -100 to 100)", df['Revision_Pct'],
             -100, 100, [3.0, 3.5, 4.0, 4.5])