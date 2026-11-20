import streamlit as st
import pandas as pd
from sector_map import TICKER_TRUE_SECTOR
from yf_session import DataFetchError, get_ticker, get_daily_history, with_retry

# ============================================================
# THIS FILE NOW PRODUCES SIX SEPARATE COLUMNS
#   Valuation, Growth, Profitability, Financial Health,
#   Earnings Surprise, EPS Revision
#
# Every column returns a small "result" like this:
#     {"score": number or None, "issue": None / "missing" / "failed", "details": {...}}
#
#   score = a real number (including a REAL zero) -> the app shows the number
#   score = None                                   -> the app shows a dash "–"
#   issue = "failed"  -> a download broke (worth a re-scan)
#   issue = "missing" -> the data genuinely does not exist for this stock
#
# Columns with two sub-parts (Valuation, Growth, Profitability, Financial
# Health): each sub-part is scored from -100 to +100 on its own, then the
# two are ADDED (so the column can reach -200 to +200). A sub-part with no
# data simply adds nothing. If BOTH sub-parts have no data, the column is
# missing (dash).
#
# app.py scales these raw scores down to -2 to +2 in one place.
# ============================================================

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


def _pts(score):
    """Text for a score inside the detail view. A real zero shows as 0.0, missing shows as a dash."""
    if score is None:
        return "–"
    value = round(score, 1) + 0.0
    return "0.0" if value == 0 else f"{value:+.1f}"


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
# SUB-PART HELPERS
# Each returns a "part": (label, score, text, issue)
#   score None + issue None      -> deliberately not applicable (adds nothing, not a problem)
#   score None + issue "missing" -> data genuinely not available (adds nothing)
#   score None + issue "failed"  -> a download broke (adds nothing)
# ============================================================
def _build_column(parts):
    """Adds up the sub-parts. Missing sub-parts add nothing. If nothing was scored, the column is missing."""
    scores = [score for _, score, _, _ in parts if score is not None]
    issues = [issue for _, _, _, issue in parts if issue]

    details = {}
    for label, score, text, _ in parts:
        details[label] = text if score is None else f"{text}  →  {_pts(score)} pts"

    if not scores:
        return {"score": None, "issue": "failed" if "failed" in issues else "missing", "details": details}

    if "failed" in issues:
        issue = "failed"
    elif issues:
        issue = "missing"
    else:
        issue = None
    return {"score": sum(scores), "issue": issue, "details": details}


def _get_ps_vs_own_history_score(ticker_symbol, current_ps):
    # May raise DataFetchError (price history download failed) — the caller handles it.
    df = get_daily_history(ticker_symbol)
    if len(df) < 100:
        return None, "P/S: Insufficient price history for own-average comparison"

    current_price = df['Close'].iloc[-1]
    if current_price <= 0 or current_ps <= 0:
        return None, "P/S: Invalid price/ratio data"

    implied_revenue_per_share = current_price / current_ps
    if implied_revenue_per_share <= 0:
        return None, "P/S: Could not derive implied revenue per share"

    approx_ps_series = df['Close'] / implied_revenue_per_share
    avg_ps = approx_ps_series.mean()

    if avg_ps <= 0:
        return None, "P/S: Invalid average"

    ps_gap_pct = ((avg_ps - current_ps) / avg_ps) * 100
    score = _clip(ps_gap_pct * 2.5)

    direction = "cheaper" if ps_gap_pct > 0 else "more expensive"
    label = f"P/S: {current_ps:.1f} vs own 1Y avg: {avg_ps:.1f} ({direction} than own history)"
    return score, label


def _pe_part(ticker_symbol, info):
    trailing_pe = info.get('trailingPE')
    forward_pe = info.get('forwardPE')
    price_to_sales = info.get('priceToSalesTrailing12Months')
    label = "Valuation: P/E"

    if trailing_pe is not None and trailing_pe > 0:
        if forward_pe is not None and forward_pe > 0:
            pe_gap_pct = ((trailing_pe - forward_pe) / trailing_pe) * 100
            score = _clip(pe_gap_pct * 1.6)
            text = (f"Trailing P/E: {trailing_pe:.1f} vs Forward P/E: {forward_pe:.1f} "
                    f"({'growth expected' if pe_gap_pct > 0 else 'contraction expected'})")
            return (label, score, text, None)
        return (label, None, f"Trailing P/E: {trailing_pe:.1f} (forward estimate unavailable)", "missing")

    # No usable P/E: fall back to price-to-sales vs the stock's own 1-year average
    if price_to_sales is not None and price_to_sales > 0:
        try:
            score, text = _get_ps_vs_own_history_score(ticker_symbol, price_to_sales)
        except DataFetchError:
            return (label, None, "⚠️ P/S check failed (price history unavailable)", "failed")
        if score is None:
            return (label, None, text, "missing")
        return (label, score, text, None)

    return (label, None, "No P/E or P/S data", "missing")


def _ev_ebitda_part(ticker_symbol, ev_ebitda):
    label = "Valuation: EV/EBITDA"
    sector = TICKER_TRUE_SECTOR.get(ticker_symbol)
    if sector is None:
        return (label, None, "No sector mapped for this ticker (add it to sector_map.py)", "missing")

    sector_median = SECTOR_EV_EBITDA_NEUTRAL.get(sector)
    if sector_median is None:
        # Deliberately not used for this sector (e.g. banks) — not a data problem
        return (label, None, f"Not used for {sector} (EV/EBITDA is not meaningful here)", None)

    if ev_ebitda is None or ev_ebitda <= 0 or ev_ebitda > 100:
        return (label, None, "EV/EBITDA not meaningful (missing, negative or extreme)", "missing")

    gap_pct = ((sector_median - ev_ebitda) / sector_median) * 100
    score = _clip(gap_pct * 1.0)

    direction = "cheaper" if gap_pct > 0 else "more expensive"
    text = f"EV/EBITDA: {ev_ebitda:.1f} vs sector median: {sector_median:.1f} ({direction} than sector)"
    return (label, score, text, None)


def _growth_parts(info):
    revenue_growth = info.get('revenueGrowth')
    earnings_growth = info.get('earningsGrowth')

    if revenue_growth is not None:
        rev_part = ("Growth: Revenue (YoY)", _clip(revenue_growth * 150), f"{revenue_growth*100:.1f}%", None)
    else:
        rev_part = ("Growth: Revenue (YoY)", None, "No Data", "missing")

    if earnings_growth is not None:
        clipped = max(-1.0, min(1.5, earnings_growth))
        earn_part = ("Growth: Earnings (YoY)", _clip(clipped * 80), f"{earnings_growth*100:.1f}%", None)
    else:
        earn_part = ("Growth: Earnings (YoY)", None, "No Data", "missing")

    return [rev_part, earn_part]


def _profitability_parts(info):
    net_margin = info.get('profitMargins')
    roe = info.get('returnOnEquity')

    if net_margin is not None:
        margin_part = ("Profitability: Net Margin", _clip(net_margin * 200), f"{net_margin*100:.1f}%", None)
    else:
        margin_part = ("Profitability: Net Margin", None, "No Data", "missing")

    if roe is not None:
        clipped = max(-0.5, min(1.0, roe))
        roe_part = ("Profitability: ROE", _clip(clipped * 110), f"{roe*100:.1f}%", None)
    else:
        roe_part = ("Profitability: ROE", None, "No Data", "missing")

    return [margin_part, roe_part]


def _health_parts(info):
    debt_to_equity = info.get('debtToEquity')
    current_ratio = info.get('currentRatio')

    if debt_to_equity is not None:
        clipped_de = max(0, min(400, debt_to_equity))
        de_part = ("Health: Debt/Equity", _clip((100 - clipped_de) * 0.5), f"{debt_to_equity:.1f}", None)
    else:
        de_part = ("Health: Debt/Equity", None, "No Data", "missing")

    if current_ratio is not None:
        cr_part = ("Health: Current Ratio", _clip((current_ratio - 1.5) * 40), f"{current_ratio:.2f}", None)
    else:
        cr_part = ("Health: Current Ratio", None, "No Data", "missing")

    return [de_part, cr_part]


# ============================================================
# EARNINGS SURPRISE and EPS REVISION (one number each)
# ============================================================
def _score_earnings_surprise(earnings_dates):
    reported = earnings_dates.dropna(subset=['Reported EPS'])
    if reported.empty:
        return None, "No Data"

    reported = reported.sort_index(ascending=False)
    latest = reported.iloc[0]

    eps_estimate = latest.get('EPS Estimate')
    eps_reported = latest.get('Reported EPS')
    surprise_pct = latest.get('Surprise(%)')

    if surprise_pct is None or pd.isna(surprise_pct):
        if eps_estimate is not None and not pd.isna(eps_estimate) and eps_estimate != 0 and eps_reported is not None:
            surprise_pct = ((eps_reported - eps_estimate) / abs(eps_estimate)) * 100
        else:
            return None, "No Data"

    clipped_surprise = max(-100, min(150, surprise_pct))
    score = _clip(clipped_surprise * 1.8)

    direction = "beat" if surprise_pct > 0 else ("missed" if surprise_pct < 0 else "met")
    label = f"EPS {direction} estimate by {surprise_pct:.1f}% (Est: {eps_estimate}, Actual: {eps_reported})"
    return score, label


def _score_eps_revision(eps_trend):
    if '0q' not in eps_trend.index:
        return None, "No Data"

    row = eps_trend.loc['0q']
    current_est = row.get('current')
    past_est = row.get('30daysAgo')

    if current_est is None or past_est is None or pd.isna(current_est) or pd.isna(past_est):
        return None, "No Data"

    if past_est == 0:
        return None, "No Data (zero baseline estimate)"

    revision_pct = ((current_est - past_est) / abs(past_est)) * 100

    clipped_revision = max(-100, min(100, revision_pct))
    score = _clip(clipped_revision * 3.5)

    direction = "raised" if revision_pct > 0 else ("cut" if revision_pct < 0 else "unchanged")
    label = f"Current-Q EPS estimate {direction} {abs(revision_pct):.1f}% over 30 days (was: {past_est:.2f}, now: {current_est:.2f})"
    return score, label


def _single_value_column(label, fetch_function, score_function, ticker_symbol):
    """Shared by Earnings Surprise and EPS Revision: one download, one number."""
    try:
        data = fetch_function(ticker_symbol)
    except DataFetchError as e:
        return {"score": None, "issue": "failed", "details": {label: f"⚠️ Fetch failed. {e}"}}

    score, text = score_function(data)
    if score is None:
        return {"score": None, "issue": "missing", "details": {label: text}}
    return {"score": score, "issue": None, "details": {label: f"{text}  →  {_pts(score)} pts"}}


# ============================================================
# EARNINGS COUNTDOWN (informational only — not part of the score)
# Uses the same cached earnings dates as the Earnings Surprise column, so it
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
# MASTER FUNDAMENTAL FUNCTION
# Returns a dictionary with the six columns. (Not cached itself: it only
# does quick maths on cached downloads.)
#
# If the company info download fails, the first four columns fail, but
# Earnings Surprise and EPS Revision still try on their own downloads.
# ============================================================
INFO_COLUMNS = ["Valuation", "Growth", "Profitability", "Financial Health"]


def calculate_fundamental_columns(ticker_symbol):
    results = {}

    try:
        info = _get_company_info(ticker_symbol)
    except DataFetchError as e:
        info = None
        for name in INFO_COLUMNS:
            results[name] = {"score": None, "issue": "failed",
                             "details": {"Company data": f"⚠️ Fetch failed. {e}"}}

    if info is not None:
        results["Valuation"] = _build_column([
            _pe_part(ticker_symbol, info),
            _ev_ebitda_part(ticker_symbol, info.get('enterpriseToEbitda')),
        ])
        results["Growth"] = _build_column(_growth_parts(info))
        results["Profitability"] = _build_column(_profitability_parts(info))
        results["Financial Health"] = _build_column(_health_parts(info))

    results["Earnings Surprise"] = _single_value_column(
        "Earnings Surprise", _get_earnings_dates, _score_earnings_surprise, ticker_symbol)
    results["EPS Revision"] = _single_value_column(
        "EPS Estimate Revision (30d)", _get_eps_trend, _score_eps_revision, ticker_symbol)

    return results
