import streamlit as st
import pandas as pd
from sector_map import TICKER_TRUE_SECTOR
from yf_session import DataFetchError, get_ticker, get_daily_history, with_retry

SECTOR_EV_EBITDA_NEUTRAL = {
    "⛏️ Materials (XLB)": 11.59,
    "🏢 Real Estate (XLRE)": 20.01,
    "🏥 Health Care (XLV)": 16.26,
    "🏦 Financials (XLF)": None,
    "🏭 Industrials (XLI)": 16.46,
    "💡 Utilities (XLU)": 13.01,
    "💻 Information Technology (XLK)": 30.83,
    "📡 Communication Services (XLC)": 10.82,
    "🛍️ Consumer Discretionary (XLY)": 18.57,
    "🛒 Consumer Staples (XLP)": 13.64,
    "🛢️ Energy (XLE)": 8.68,
}


def _clip(value, low=-100, high=100):
    return max(low, min(high, value))


# ============================================================
# HOW FAILURES WORK IN THIS FILE
#   - The three downloads below (company info, earnings dates, EPS
#     estimates) are cached on their own. If Yahoo fails, they raise
#     DataFetchError, and Streamlit does NOT save a failure.
#   - An empty answer counts as a failure for these, because every
#     company in your universe has company info, earnings dates and
#     analyst estimates.
#   - The scoring maths further down is NOT cached. It is instant.
# ============================================================
@st.cache_data(ttl=86400, show_spinner=False)
@with_retry(max_retries=2, delay_seconds=2)
def _get_company_info(ticker_symbol):
    info = get_ticker(ticker_symbol).info
    if not info or len(info) < 5:
        raise ValueError(f"Yahoo returned no company data for {ticker_symbol}")
    return info


@st.cache_data(ttl=43200, show_spinner=False)  # 12 hours — earnings dates rarely change once scheduled
@with_retry(max_retries=2, delay_seconds=2)
def _get_earnings_dates(ticker_symbol):
    earnings_dates = get_ticker(ticker_symbol).earnings_dates
    if earnings_dates is None or earnings_dates.empty:
        raise ValueError(f"No earnings dates returned for {ticker_symbol}")
    return earnings_dates


@st.cache_data(ttl=86400, show_spinner=False)
@with_retry(max_retries=2, delay_seconds=2)
def _get_eps_trend(ticker_symbol):
    eps_trend = get_ticker(ticker_symbol).eps_trend
    if eps_trend is None or eps_trend.empty:
        raise ValueError(f"No EPS estimate data returned for {ticker_symbol}")
    return eps_trend


# ============================================================
# SCORING HELPERS (pure maths on already-downloaded data)
# ============================================================
def _get_ps_vs_own_history_score(ticker_symbol, current_ps):
    # May raise DataFetchError (price history download failed) — the caller handles it.
    df = get_daily_history(ticker_symbol)
    if len(df) < 100:
        return 0, "P/S: Insufficient price history for own-average comparison"

    current_price = df['Close'].iloc[-1]
    if current_price <= 0 or current_ps <= 0:
        return 0, "P/S: Invalid price/ratio data"

    implied_revenue_per_share = current_price / current_ps
    if implied_revenue_per_share <= 0:
        return 0, "P/S: Could not derive implied revenue per share"

    approx_ps_series = df['Close'] / implied_revenue_per_share
    avg_ps = approx_ps_series.mean()

    if avg_ps <= 0:
        return 0, "P/S: Invalid average"

    ps_gap_pct = ((avg_ps - current_ps) / avg_ps) * 100
    score = _clip(ps_gap_pct * 2.5)

    direction = "cheaper" if ps_gap_pct > 0 else "more expensive"
    label = f"P/S: {current_ps:.1f} vs own 1Y avg: {avg_ps:.1f} ({direction} than own history)"
    return score, label


def _get_ev_ebitda_score(ticker_symbol, ev_ebitda):
    sector = TICKER_TRUE_SECTOR.get(ticker_symbol)
    if sector is None:
        return 0, "N/A (no sector mapped)"

    sector_median = SECTOR_EV_EBITDA_NEUTRAL.get(sector)
    if sector_median is None:
        return 0, f"N/A (EV/EBITDA not meaningful for {sector})"

    if ev_ebitda is None or ev_ebitda <= 0 or ev_ebitda > 100:
        return 0, "No Data (EV/EBITDA not meaningful — near-zero/negative EBITDA)"

    ev_ebitda_gap_pct = ((sector_median - ev_ebitda) / sector_median) * 100
    score = _clip(ev_ebitda_gap_pct * 1.0)

    direction = "cheaper" if ev_ebitda_gap_pct > 0 else "more expensive"
    label = f"EV/EBITDA: {ev_ebitda:.1f} vs sector median: {sector_median:.1f} ({direction} than sector)"
    return score, label


def _score_earnings_surprise(earnings_dates):
    reported = earnings_dates.dropna(subset=['Reported EPS'])
    if reported.empty:
        return 0, "No Data"

    reported = reported.sort_index(ascending=False)
    latest = reported.iloc[0]

    eps_estimate = latest.get('EPS Estimate')
    eps_reported = latest.get('Reported EPS')
    surprise_pct = latest.get('Surprise(%)')

    if surprise_pct is None or pd.isna(surprise_pct):
        if eps_estimate is not None and eps_estimate != 0 and eps_reported is not None:
            surprise_pct = ((eps_reported - eps_estimate) / abs(eps_estimate)) * 100
        else:
            return 0, "No Data"

    clipped_surprise = max(-100, min(150, surprise_pct))
    score = _clip(clipped_surprise * 1.8)

    direction = "beat" if surprise_pct > 0 else ("missed" if surprise_pct < 0 else "met")
    label = f"EPS {direction} estimate by {surprise_pct:.1f}% (Est: {eps_estimate}, Actual: {eps_reported})"
    return score, label


def _score_eps_revision(eps_trend):
    if '0q' not in eps_trend.index:
        return 0, "No Data"

    row = eps_trend.loc['0q']
    current_est = row.get('current')
    past_est = row.get('30daysAgo')

    if current_est is None or past_est is None or pd.isna(current_est) or pd.isna(past_est):
        return 0, "No Data"

    if past_est == 0:
        return 0, "No Data (zero baseline estimate)"

    revision_pct = ((current_est - past_est) / abs(past_est)) * 100

    clipped_revision = max(-100, min(100, revision_pct))
    score = _clip(clipped_revision * 3.5)

    direction = "raised" if revision_pct > 0 else ("cut" if revision_pct < 0 else "unchanged")
    label = f"Current-Q EPS estimate {direction} {abs(revision_pct):.1f}% over 30 days (was: {past_est:.2f}, now: {current_est:.2f})"
    return score, label


# ============================================================
# EARNINGS COUNTDOWN (informational only — not part of the score)
# Uses the same cached earnings dates as the fundamentals score, so it
# costs no extra Yahoo request. Raises DataFetchError if unavailable.
# ============================================================
def get_earnings_countdown(ticker_symbol):
    earnings_dates = _get_earnings_dates(ticker_symbol)

    now = pd.Timestamp.now(tz=earnings_dates.index.tz) if earnings_dates.index.tz else pd.Timestamp.now()
    future = earnings_dates[earnings_dates.index >= now]

    if future.empty:
        return None

    future = future.sort_index(ascending=True)
    next_date = future.index[0]
    days_until = (next_date.date() - now.date()).days

    return {"date": next_date, "days_until": days_until}


# ============================================================
# MASTER FUNDAMENTAL SCORE
# (Not cached itself: it only does quick maths on cached downloads.)
#
# What happens when data is missing:
#   - Company info fails        -> raises DataFetchError (whole score fails)
#   - Price history (P/S check) -> valuation uses the EV/EBITDA part only, with a note
#   - Earnings surprise/EPS     -> that part is left out and the remaining parts
#                                  share the weight, with a note
# ============================================================
def calculate_fundamental_score(ticker_symbol):
    info = _get_company_info(ticker_symbol)  # raises DataFetchError if Yahoo fails

    details = {}
    notes = []

    trailing_pe = info.get('trailingPE')
    forward_pe = info.get('forwardPE')
    price_to_sales = info.get('priceToSalesTrailing12Months')
    ev_ebitda_raw = info.get('enterpriseToEbitda')

    pe_ps_score = 0
    if trailing_pe is not None and trailing_pe > 0:
        if forward_pe is not None and forward_pe > 0:
            pe_gap_pct = ((trailing_pe - forward_pe) / trailing_pe) * 100
            pe_ps_score = _clip(pe_gap_pct * 1.6)
            details["Valuation (P/E)"] = (
                f"Trailing P/E: {trailing_pe:.1f} vs Forward P/E: {forward_pe:.1f} "
                f"({'growth expected' if pe_gap_pct > 0 else 'contraction expected'})"
            )
        else:
            pe_ps_score = 0
            details["Valuation (P/E)"] = f"Trailing P/E: {trailing_pe:.1f} (forward estimate unavailable)"
    else:
        if price_to_sales is not None and price_to_sales > 0:
            try:
                pe_ps_score, ps_label = _get_ps_vs_own_history_score(ticker_symbol, price_to_sales)
                details["Valuation (P/E)"] = ps_label
            except DataFetchError:
                pe_ps_score = None
                details["Valuation (P/E)"] = "⚠️ P/S check failed (price history unavailable)"
                notes.append("P/S price history unavailable (valuation uses EV/EBITDA only)")
        else:
            pe_ps_score = 0
            details["Valuation (P/E)"] = "No Data"

    ev_ebitda_score, ev_ebitda_label = _get_ev_ebitda_score(ticker_symbol, ev_ebitda_raw)
    details["Valuation (EV/EBITDA)"] = ev_ebitda_label

    if pe_ps_score is None:
        valuation_score = ev_ebitda_score
    else:
        valuation_score = (pe_ps_score + ev_ebitda_score) / 2

    revenue_growth = info.get('revenueGrowth')
    earnings_growth = info.get('earningsGrowth')

    rev_score = _clip(revenue_growth * 150) if revenue_growth is not None else 0

    if earnings_growth is not None:
        clipped_earnings_growth = max(-1.0, min(1.5, earnings_growth))
        earn_score = _clip(clipped_earnings_growth * 80)
    else:
        earn_score = 0

    growth_score = (rev_score + earn_score) / 2

    details["Revenue Growth (YoY)"] = f"{revenue_growth*100:.1f}%" if revenue_growth is not None else "No Data"
    details["Earnings Growth (YoY)"] = f"{earnings_growth*100:.1f}%" if earnings_growth is not None else "No Data"

    net_margin = info.get('profitMargins')
    roe = info.get('returnOnEquity')

    margin_score = _clip(net_margin * 200) if net_margin is not None else 0

    if roe is not None:
        clipped_roe = max(-0.5, min(1.0, roe))
        roe_score = _clip(clipped_roe * 110)
    else:
        roe_score = 0

    profitability_score = (margin_score + roe_score) / 2

    details["Net Margin"] = f"{net_margin*100:.1f}%" if net_margin is not None else "No Data"
    details["ROE"] = f"{roe*100:.1f}%" if roe is not None else "No Data"

    debt_to_equity = info.get('debtToEquity')
    current_ratio = info.get('currentRatio')

    if debt_to_equity is not None:
        clipped_de = max(0, min(400, debt_to_equity))
        de_score = _clip((100 - clipped_de) * 0.5)
    else:
        de_score = 0

    cr_score = _clip((current_ratio - 1.5) * 40) if current_ratio is not None else 0

    health_score = (de_score + cr_score) / 2

    details["Debt/Equity"] = f"{debt_to_equity:.1f}" if debt_to_equity is not None else "No Data"
    details["Current Ratio"] = f"{current_ratio:.2f}" if current_ratio is not None else "No Data"

    # --- Earnings surprise & EPS revisions (each may fail on its own) ---
    surprise_score = None
    try:
        surprise_score, surprise_label = _score_earnings_surprise(_get_earnings_dates(ticker_symbol))
    except DataFetchError:
        surprise_label = "⚠️ Fetch failed (left out of the score)"
        notes.append("Earnings surprise unavailable")

    revision_score = None
    try:
        revision_score, revision_label = _score_eps_revision(_get_eps_trend(ticker_symbol))
    except DataFetchError:
        revision_label = "⚠️ Fetch failed (left out of the score)"
        notes.append("EPS revision unavailable")

    details["Earnings Surprise"] = surprise_label
    details["EPS Estimate Revision (30d)"] = revision_label

    surprise_parts = [s for s in (surprise_score, revision_score) if s is not None]
    financial_surprise_score = sum(surprise_parts) / len(surprise_parts) if surprise_parts else None

    # --- Combine the five pillars (equal weight, shared among whichever are available) ---
    pillar_scores = [valuation_score, growth_score, profitability_score, health_score]
    if financial_surprise_score is not None:
        pillar_scores.append(financial_surprise_score)

    final_score = sum(pillar_scores) / len(pillar_scores)

    if notes:
        details["⚠️ Data Notes"] = "; ".join(notes)

    return _clip(final_score), details
