import yfinance as yf
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from yf_session import get_yf_session, with_retry

# ============================================================
# 1. RETAIL & MEDIA SENTIMENT (VADER)
# — Switched from Loughran-McDonald after diagnostic testing showed
#   LM producing frequent, unjustified extreme (-1.000) readings on
#   short financial headlines due to small-sample degeneracy in its
#   polarity formula (a single dictionary word match can pin the
#   score to the absolute floor/ceiling regardless of real meaning).
#   VADER was more stable and better matched real-world reading of
#   headline sentiment in side-by-side testing.
#
#   VADER itself reads structurally positive on financial text in
#   general (calibrated median ticker-level compound was +0.365
#   across 261 tickers) — this is de-meaned against that baseline
#   before scaling, so the score reflects deviation from "typical"
#   financial news tone, not an absolute positivity reading.
# ============================================================
VADER_NEWS_BASELINE = 0.365  # calibrated median ticker-level compound score across the universe
VADER_NEWS_MULTIPLIER = 400.0  # calibrated

@with_retry(max_retries=2, delay_seconds=2)
def calculate_news_sentiment(ticker_symbol, limit=15):
    try:
        session = get_yf_session()
        asset = yf.Ticker(ticker_symbol, session=session)
        news_items = asset.news

        if not news_items:
            return None

        vader = SentimentIntensityAnalyzer()
        total_compound = 0.0
        count = 0
        seen_headlines = set()

        for item in news_items[:limit]:
            content = item.get('content', {})
            title = content.get('title', '')
            summary = content.get('summary', '')
            text = f"{title}. {summary}".strip('. ')
            text_lower = text.lower()

            if not text_lower:
                continue
            if text_lower in seen_headlines:
                continue
            seen_headlines.add(text_lower)

            compound = vader.polarity_scores(text)['compound']
            total_compound += compound
            count += 1

        if count > 0:
            avg_compound = total_compound / count
            relative_compound = avg_compound - VADER_NEWS_BASELINE
            scaled_score = relative_compound * VADER_NEWS_MULTIPLIER

            return max(-100, min(100, round(scaled_score, 1)))

        return None
    except Exception:
        return None


# ============================================================
# 2. SMART MONEY FLOW (Options PCR & OI)
# — Dead-zone contrarian scoring, calibrated against real PCR
#   distribution across the stock universe:
#     Dead zone (0.6-0.9): flat ZERO, no signal
#     Below 0.6 (greed): contrarian -> BEARISH, scales with distance
#     Above 0.9 (fear): contrarian -> BULLISH, scales with distance
#   Separately calibrated per side since real PCR is right-skewed
#   (fear-side deviations run much larger than greed-side deviations).
# ============================================================
def _score_pcr_dead_zone_contrarian(blended_pcr, lower_bound=0.6, upper_bound=0.9,
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
def calculate_options_flow(ticker_symbol, min_oi_leg=30, min_vol_leg=20):
    try:
        session = get_yf_session()
        asset = yf.Ticker(ticker_symbol, session=session)
        expirations = asset.options
        if not expirations:
            return None

        total_call_vol, total_put_vol = 0, 0
        total_call_oi, total_put_oi = 0, 0

        for exp in expirations[:3]:
            try:
                chain = asset.option_chain(exp)
                total_call_vol += chain.calls['volume'].fillna(0).sum()
                total_put_vol += chain.puts['volume'].fillna(0).sum()
                total_call_oi += chain.calls['openInterest'].fillna(0).sum()
                total_put_oi += chain.puts['openInterest'].fillna(0).sum()
            except Exception:
                continue

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
            blended_pcr = oi_pcr
        elif vol_pcr is not None and oi_pcr is not None:
            blended_pcr = (vol_pcr * 0.50) + (oi_pcr * 0.50)
        elif oi_pcr is not None:
            blended_pcr = oi_pcr
        elif vol_pcr is not None:
            blended_pcr = vol_pcr
        else:
            blended_pcr = None

        if blended_pcr is None:
            return None

        raw_score = _score_pcr_dead_zone_contrarian(blended_pcr)

        return round(raw_score, 1)

    except Exception:
        return None


# ============================================================
# 3. ANALYST CATALYST (Event Hype)
# ============================================================
@with_retry(max_retries=2, delay_seconds=2)
def calculate_analyst_catalyst(ticker_symbol, days_back=14):
    try:
        session = get_yf_session()
        asset = yf.Ticker(ticker_symbol, session=session)
        recos = asset.upgrades_downgrades

        if recos is None or recos.empty:
            return 0, "No Coverage"

        cutoff_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days_back)

        if recos.index.tz is None:
            recos.index = recos.index.tz_localize('UTC')

        recent_actions = recos[recos.index >= cutoff_date]

        if recent_actions.empty:
            return 0, "No Recent Actions"

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

        final_catalyst_score = max(-100, min(100, score))
        summary = " | ".join(action_details) if action_details else "Neutral Actions Only"

        return final_catalyst_score, summary

    except Exception as e:
        return 0, f"API Error: {str(e)}"


# ============================================================
# 4. MASTER SENTIMENT ENGINE
# ============================================================
@st.cache_data(ttl=3600)
def calculate_sentiment_score(ticker_symbol):
    """
    Blends Options Flow, News NLP, and Analyst Catalysts.
    Dynamically redistributes weights around whichever pillars
    actually have usable data.
    """
    options_score = calculate_options_flow(ticker_symbol)
    news_score = calculate_news_sentiment(ticker_symbol)
    catalyst_score, catalyst_details = calculate_analyst_catalyst(ticker_symbol)

    has_catalyst = catalyst_details not in ["No Coverage", "No Recent Actions", "Neutral Actions Only"] \
                   and "API Error" not in catalyst_details

    base_weights = {"options": 0.50, "news": 0.30, "catalyst": 0.20}

    active_weights = {}
    if options_score is not None:
        active_weights["options"] = base_weights["options"]
    if news_score is not None:
        active_weights["news"] = base_weights["news"]
    if has_catalyst:
        active_weights["catalyst"] = base_weights["catalyst"]

    if not active_weights:
        details = {
            "Smart Money (Options Flow)": "No Data",
            "News NLP (VADER)": "No Data",
            "Analyst Catalyst": "No Recent Actions",
        }
        return 0, details

    total_active_weight = sum(active_weights.values())
    master_score = 0
    if "options" in active_weights:
        master_score += options_score * (active_weights["options"] / total_active_weight)
    if "news" in active_weights:
        master_score += news_score * (active_weights["news"] / total_active_weight)
    if "catalyst" in active_weights:
        master_score += catalyst_score * (active_weights["catalyst"] / total_active_weight)

    details = {
        "Smart Money (Options Flow)": options_score if options_score is not None else "No Data",
        "News NLP (VADER)": news_score if news_score is not None else "No Data",
        "Analyst Catalyst": f"{catalyst_score} ({catalyst_details})" if has_catalyst else "No Recent Actions",
    }

    return max(-100, min(100, round(master_score, 1))), details