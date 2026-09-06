import pandas as pd

df = pd.read_csv("sentiment_model_comparison_raw.csv")
print(f"Total headlines: {len(df)}\n")

print("--- LM POLARITY DISTRIBUTION ---")
print(f"min={df['lm_polarity'].min():.3f}  max={df['lm_polarity'].max():.3f}  "
      f"mean={df['lm_polarity'].mean():.3f}  median={df['lm_polarity'].median():.3f}  "
      f"std={df['lm_polarity'].std():.3f}")

print("\n--- VADER COMPOUND DISTRIBUTION ---")
print(f"min={df['vader_compound'].min():.3f}  max={df['vader_compound'].max():.3f}  "
      f"mean={df['vader_compound'].mean():.3f}  median={df['vader_compound'].median():.3f}  "
      f"std={df['vader_compound'].std():.3f}")

# Specifically check the extremes on each side
print("\n--- MOST BEARISH (LM) — top 10 most negative LM readings ---")
most_bearish_lm = df.nsmallest(10, 'lm_polarity')[['ticker', 'text', 'lm_polarity', 'vader_compound']]
for _, row in most_bearish_lm.iterrows():
    print(f"  LM={row['lm_polarity']:.3f}  VADER={row['vader_compound']:.3f}  | {row['text'][:100]}")

print("\n--- MOST BEARISH (VADER) — top 10 most negative VADER readings ---")
most_bearish_vader = df.nsmallest(10, 'vader_compound')[['ticker', 'text', 'lm_polarity', 'vader_compound']]
for _, row in most_bearish_vader.iterrows():
    print(f"  LM={row['lm_polarity']:.3f}  VADER={row['vader_compound']:.3f}  | {row['text'][:100]}")

print("\n--- BIGGEST DISAGREEMENTS — where LM and VADER point opposite directions most strongly ---")
df['disagreement'] = (df['lm_polarity'] - df['vader_compound']).abs()
biggest_gaps = df.nlargest(15, 'disagreement')[['ticker', 'text', 'lm_polarity', 'vader_compound']]
for _, row in biggest_gaps.iterrows():
    print(f"  LM={row['lm_polarity']:.3f}  VADER={row['vader_compound']:.3f}  | {row['text'][:100]}")

# Check specifically: does LM show more extreme negative readings than VADER, proportionally?
lm_extreme_neg = (df['lm_polarity'] <= -0.5).sum()
vader_extreme_neg = (df['vader_compound'] <= -0.5).sum()
print(f"\n--- EXTREME BEARISH COUNT (<= -0.5) ---")
print(f"LM: {lm_extreme_neg} ({lm_extreme_neg/len(df)*100:.1f}%)")
print(f"VADER: {vader_extreme_neg} ({vader_extreme_neg/len(df)*100:.1f}%)")