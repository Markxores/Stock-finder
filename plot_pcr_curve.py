import math


def score_pcr_contrarian_at_extremes(blended_pcr, neutral_baseline=0.75,
                                       extreme_high=1.360, extreme_low=0.415,
                                       confirmation_scale=25.0, contrarian_scale=100.0,
                                       steepness=8):
    dev_pcr = blended_pcr - neutral_baseline

    # MIDDLE ZONE (confirmation): low PCR -> bullish -> POSITIVE
    #                              high PCR -> bearish -> NEGATIVE
    confirmation_score = -dev_pcr * confirmation_scale

    # EXTREME ZONE (contrarian): flips the sign entirely
    #   low PCR extreme  -> bearish -> NEGATIVE
    #   high PCR extreme -> bullish -> POSITIVE
    if blended_pcr >= neutral_baseline:
        distance_into_extreme = (blended_pcr - extreme_high)
        contrarian_weight = 1 / (1 + math.exp(-distance_into_extreme * steepness))
        contrarian_score = contrarian_weight * contrarian_scale  # positive = bullish
    else:
        distance_into_extreme = (extreme_low - blended_pcr)
        contrarian_weight = 1 / (1 + math.exp(-distance_into_extreme * steepness))
        contrarian_score = -(contrarian_weight * contrarian_scale)  # negative = bearish

    # As contrarian_weight rises toward 1 near/past the extreme threshold, the
    # confirmation component is proportionally faded out, so the two don't stack
    final_score = (confirmation_score * (1 - contrarian_weight)) + contrarian_score

    return max(-100, min(100, final_score)), contrarian_weight


test_values = [0.1, 0.25, 0.35, 0.41, 0.5, 0.6, 0.75, 0.9, 1.1, 1.36, 1.5, 1.7, 2.0, 3.0]

for steepness_test in [8, 15, 20, 30]:
    print(f"\n=== steepness={steepness_test} ===")
    print(f"{'PCR':>6} | {'Score':>8} | {'Contrarian Wt':>14}")
    print("-" * 40)
    for pcr in test_values:
        score, weight = score_pcr_contrarian_at_extremes(pcr, steepness=steepness_test)
        print(f"{pcr:>6.2f} | {score:>8.1f} | {weight:>14.2f}")