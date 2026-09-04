import pandas as pd

df = pd.read_csv("sentiment_calibration_raw_v2.csv")
valid_pcr = df['Blended_PCR'].dropna()

LOWER_BOUND = 0.6
UPPER_BOUND = 0.9

below = valid_pcr[valid_pcr < LOWER_BOUND]
above = valid_pcr[valid_pcr > UPPER_BOUND]
inside = valid_pcr[(valid_pcr >= LOWER_BOUND) & (valid_pcr <= UPPER_BOUND)]

print(f"Total: {len(valid_pcr)}")
print(f"Inside dead zone (0 score): {len(inside)} ({len(inside)/len(valid_pcr)*100:.1f}%)")
print(f"Below (bearish contrarian): {len(below)} ({len(below)/len(valid_pcr)*100:.1f}%)")
print(f"Above (bullish contrarian): {len(above)} ({len(above)/len(valid_pcr)*100:.1f}%)")

below_distances = LOWER_BOUND - below
above_distances = above - UPPER_BOUND

print(f"\n--- BELOW-ZONE DISTANCES (greed side) ---")
print(f"min={below_distances.min():.3f}  max={below_distances.max():.3f}  "
      f"median={below_distances.median():.3f}  95th_pct={below_distances.quantile(0.95):.3f}")

print(f"\n--- ABOVE-ZONE DISTANCES (fear side) ---")
print(f"min={above_distances.min():.3f}  max={above_distances.max():.3f}  "
      f"median={above_distances.median():.3f}  95th_pct={above_distances.quantile(0.95):.3f}")

print(f"\n--- CLIP-RATE CHECK ACROSS CANDIDATE SCALES ---")
for low_scale in [100, 150, 200, 250]:
    scaled = (below_distances * low_scale).clip(0, 100)
    pct_clip = (scaled >= 99.5).mean() * 100
    print(f"  low_side_scale={low_scale}: median_score={-(below_distances.median()*low_scale):.1f}  pct_clipped={pct_clip:.1f}%")

for high_scale in [60, 90, 120, 150]:
    scaled = (above_distances * high_scale).clip(0, 100)
    pct_clip = (scaled >= 99.5).mean() * 100
    print(f"  high_side_scale={high_scale}: median_score={(above_distances.median()*high_scale):.1f}  pct_clipped={pct_clip:.1f}%")