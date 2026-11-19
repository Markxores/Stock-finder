import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from yf_session import DataFetchError, get_ticker, with_retry

# ============================================================
# HOW FAILURES WORK IN THIS FILE
#   - Each Yahoo download (news, options, analyst actions) is cached on
#     its own. If Yahoo fails, it raises DataFetchError and Streamlit
#     does NOT save the failure.
#   - "No news", "no options" and "no analyst actions" are accepted as
#     real answers, because some stocks genuinely have none.
#   - The scoring maths is NOT cached. It is instant.
# ============================================================

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


@st.cache_data(ttl=3600, show_spinner=False)
@with_retry(max_retries=2, delay_seconds=2)
def _get_news_items(ticker_symbol):
    return list(get_ticker(ticker_symbol).news or [])


def calculate_news_sentiment(ticker_symbol, limit=15):
    news_items = _get_news_items(ticker_symbol)  # raises DataFetchError if Yahoo fails

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


def calculate_options_flow(ticker_symbol, min_oi_leg=30, min_vol_leg=20):
    totals = _get_options_totals(ticker_symbol)  # raises DataFetchError if Yahoo fails
    if totals is None:
        return None

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


# ============================================================
# 3. ANALYST CATALYST (Event Hype)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
@with_retry(max_retries=2, delay_seconds=2)
def _get_analyst_actions(ticker_symbol):
    return get_ticker(ticker_symbol).upgrades_downgrades


def calculate_analyst_catalyst(ticker_symbol, days_back=14):
    recos = _get_analyst_actions(ticker_symbol)  # raises DataFetchError if Yahoo fails

    if recos is None or recos.empty:
        return 0, "No Coverage"

    recos = recos.copy()
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


# ============================================================
# 4. MASTER SENTIMENT ENGINE
# (Not cached itself: it only does quick maths on cached downloads.)
#
# Blends Options Flow, News NLP, and Analyst Catalysts.
# Dynamically redistributes weights around whichever pillars
# actually have usable data. A source that FAILED to download is
# treated like a source with no data (its weight is shared out),
# and a warning note is added. If ALL three failed, it raises
# DataFetchError so the app can show the sentiment as failed.
# ============================================================
def calculate_sentiment_score(ticker_symbol):
    failed_sources = []

    try:
        options_score = calculate_options_flow(ticker_symbol)
        options_failed = False
    except DataFetchError:
        options_score = None
        options_failed = True
        failed_sources.append("Options flow")

    try:
        news_score = calculate_news_sentiment(ticker_symbol)
        news_failed = False
    except DataFetchError:
        news_score = None
        news_failed = True
        failed_sources.append("News")

    try:
        catalyst_score, catalyst_details = calculate_analyst_catalyst(ticker_symbol)
        catalyst_failed = False
    except DataFetchError:
        catalyst_score, catalyst_details = 0, "Fetch failed"
        catalyst_failed = True
        failed_sources.append("Analyst actions")

    if len(failed_sources) == 3:
        raise DataFetchError("All sentiment sources failed (options, news, analyst actions)")

    has_catalyst = (not catalyst_failed) and catalyst_details not in ["No Coverage", "No Recent Actions", "Neutral Actions Only"]

    base_weights = {"options": 0.50, "news": 0.30, "catalyst": 0.20}

    active_weights = {}
    if options_score is not None:
        active_weights["options"] = base_weights["options"]
    if news_score is not None:
        active_weights["news"] = base_weights["news"]
    if has_catalyst:
        active_weights["catalyst"] = base_weights["catalyst"]

    if options_failed:
        options_text = "⚠️ Fetch failed"
    else:
        options_text = options_score if options_score is not None else "No Data"

    if news_failed:
        news_text = "⚠️ Fetch failed"
    else:
        news_text = news_score if news_score is not None else "No Data"

    if catalyst_failed:
        catalyst_text = "⚠️ Fetch failed"
    elif has_catalyst:
        catalyst_text = f"{catalyst_score} ({catalyst_details})"
    else:
        catalyst_text = "No Recent Actions"

    details = {
        "Smart Money (Options Flow)": options_text,
        "News NLP (VADER)": news_text,
        "Analyst Catalyst": catalyst_text,
    }
    if failed_sources:
        details["⚠️ Data Notes"] = ", ".join(failed_sources) + " unavailable (score uses the remaining sources)"

    if not active_weights:
        return 0, details

    total_active_weight = sum(active_weights.values())
    master_score = 0
    if "options" in active_weights:
        master_score += options_score * (active_weights["options"] / total_active_weight)
    if "news" in active_weights:
        master_score += news_score * (active_weights["news"] / total_active_weight)
    if "catalyst" in active_weights:
        master_score += catalyst_score * (active_weights["catalyst"] / total_active_weight)

    return max(-100, min(100, round(master_score, 1))), details
