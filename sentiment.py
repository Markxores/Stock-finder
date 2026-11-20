import pandas as pd
import streamlit as st
from yf_session import DataFetchError, get_ticker, with_retry

# ============================================================
# THIS FILE NOW PRODUCES TWO SEPARATE COLUMNS
#   1. Options          (put/call ratio, contrarian)
#   2. Catalyst Action  (analyst upgrades and downgrades)
#
# News sentiment has been removed completely.
#
# Every function here returns a small "result" like this:
#     {"score": number or None, "issue": None / "missing" / "failed", "details": {...}}
#
#   score = a real number (including a REAL zero)  -> the app shows the number
#   score = None                                    -> the app shows a dash "–"
#   issue = "failed"  -> the download broke (worth a re-scan)
#   issue = "missing" -> the data genuinely does not exist for this stock
#
# Scores here are still on the old -100 to +100 scale. The app.py file
# scales them down (to -2 to +2) in one place.
# ============================================================


# ============================================================
# 1. OPTIONS (Put/Call Ratio)
# — Dead-zone contrarian scoring, calibrated against real PCR
#   distribution across the stock universe:
#     Dead zone (0.6-0.9): a REAL zero, no signal
#     Below 0.6 (greed): contrarian -> BEARISH, scales with distance
#     Above 0.9 (fear): contrarian -> BULLISH, scales with distance
#   Separately calibrated per side since real PCR is right-skewed.
# ============================================================
PCR_LOWER_BOUND = 0.6
PCR_UPPER_BOUND = 0.9


def _score_pcr_dead_zone_contrarian(blended_pcr, lower_bound=PCR_LOWER_BOUND, upper_bound=PCR_UPPER_BOUND,
                                      low_side_scale=200.0, high_side_scale=90.0):
    if lower_bound <= blended_pcr <= upper_bound:
        return 0.0
    elif blended_pcr > upper_bound:
        distance = blended_pcr - upper_bound
        return max(-100.0, min(100.0, distance * high_side_scale))
    else:
        distance = lower_bound - blended_pcr
        return max(-100.0, min(100.0, -(distance * low_side_scale)))


@with_retry(max_retries=2, delay_seconds=2)
def _fetch_expirations(asset):
    return tuple(asset.options or ())


@with_retry(max_retries=2, delay_seconds=2)
def _fetch_chain_totals(asset, expiration):
    chain = asset.option_chain(expiration)
    return (
        float(chain.calls['volume'].fillna(0).sum()),
        float(chain.puts['volume'].fillna(0).sum()),
        float(chain.calls['openInterest'].fillna(0).sum()),
        float(chain.puts['openInterest'].fillna(0).sum()),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _get_options_totals(ticker_symbol):
    """
    Returns (call_volume, put_volume, call_open_interest, put_open_interest)
    added up over the first 3 expiry dates, or None if the stock has no options.
    If ANY download fails, it raises DataFetchError, so a half-finished
    total is never saved.
    """
    asset = get_ticker(ticker_symbol)
    expirations = _fetch_expirations(asset)
    if not expirations:
        return None

    totals = [0.0, 0.0, 0.0, 0.0]
    for exp in expirations[:3]:
        chain_totals = _fetch_chain_totals(asset, exp)
        for i, value in enumerate(chain_totals):
            totals[i] += value

    return tuple(totals)


def _blend_pcr(totals, min_oi_leg=30, min_vol_leg=20):
    """Turns the raw options totals into one put/call ratio, or None if there is too little activity."""
    total_call_vol, total_put_vol, total_call_oi, total_put_oi = totals

    if total_call_oi == 0 and total_call_vol == 0:
        return None

    total_vol = total_call_vol + total_put_vol

    oi_pcr = None
    if total_call_oi >= min_oi_leg and total_put_oi >= min_oi_leg:
        oi_pcr = total_put_oi / total_call_oi

    vol_pcr = None
    if total_call_vol >= min_vol_leg and total_put_vol >= min_vol_leg:
        vol_pcr = total_put_vol / total_call_vol

    if total_vol < 1000:
        return oi_pcr
    elif vol_pcr is not None and oi_pcr is not None:
        return (vol_pcr * 0.50) + (oi_pcr * 0.50)
    elif oi_pcr is not None:
        return oi_pcr
    elif vol_pcr is not None:
        return vol_pcr
    return None


def calculate_options_column(ticker_symbol):
    try:
        totals = _get_options_totals(ticker_symbol)
    except DataFetchError as e:
        return {"score": None, "issue": "failed",
                "details": {"Options (Put/Call)": f"⚠️ Fetch failed. {e}"}}

    # No options at all for this stock = MISSING (not a neutral zero)
    if totals is None:
        return {"score": None, "issue": "missing",
                "details": {"Options (Put/Call)": "No options exist for this stock"}}

    blended_pcr = _blend_pcr(totals)
    if blended_pcr is None:
        return {"score": None, "issue": "missing",
                "details": {"Options (Put/Call)": "Too little options activity to read"}}

    score = round(_score_pcr_dead_zone_contrarian(blended_pcr), 1)

    if blended_pcr > PCR_UPPER_BOUND:
        zone = "Fear zone (contrarian bullish)"
    elif blended_pcr < PCR_LOWER_BOUND:
        zone = "Greed zone (contrarian bearish)"
    else:
        zone = "Dead zone (neutral, real zero)"

    return {"score": score, "issue": None,
            "details": {
                "Put/Call Ratio (blended)": f"{blended_pcr:.2f}",
                "Zone": zone,
            }}


# ============================================================
# 2. CATALYST ACTION (analyst upgrades / downgrades, last 14 days)
#   - "No coverage" and "no recent actions" are REAL zeros.
#   - Only a failed download counts as missing.
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
@with_retry(max_retries=2, delay_seconds=2)
def _get_analyst_actions(ticker_symbol):
    return get_ticker(ticker_symbol).upgrades_downgrades


def calculate_catalyst_column(ticker_symbol, days_back=14):
    try:
        recos = _get_analyst_actions(ticker_symbol)
    except DataFetchError as e:
        return {"score": None, "issue": "failed",
                "details": {"Catalyst Action": f"⚠️ Fetch failed. {e}"}}

    if recos is None or recos.empty:
        return {"score": 0.0, "issue": None,
                "details": {"Catalyst Action": "No analyst coverage (counts as 0)"}}

    recos = recos.copy()
    cutoff_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days_back)

    if recos.index.tz is None:
        recos.index = recos.index.tz_localize('UTC')

    recent_actions = recos[recos.index >= cutoff_date]

    if recent_actions.empty:
        return {"score": 0.0, "issue": None,
                "details": {"Catalyst Action": f"No analyst actions in the last {days_back} days (counts as 0)"}}

    score = 0
    action_details = []

    for index, row in recent_actions.iterrows():
        action = str(row.get('Action', '')).lower()
        to_grade = str(row.get('ToGrade', '')).lower()

        if action == 'up' or 'buy' in to_grade or 'outperform' in to_grade or 'overweight' in to_grade:
            score += 50
            action_details.append(f"Upgrade to {to_grade.title()}")
        elif action == 'down' or 'sell' in to_grade or 'underperform' in to_grade or 'underweight' in to_grade:
            score -= 50
            action_details.append(f"Downgrade to {to_grade.title()}")

    final_score = float(max(-100, min(100, score)))
    summary = " | ".join(action_details) if action_details else "Neutral actions only (counts as 0)"

    return {"score": final_score, "issue": None,
            "details": {"Catalyst Action": summary}}
