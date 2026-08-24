import streamlit as st
import pandas as pd
import yfinance as yf
from sector_map import TICKER_TRUE_SECTOR

# ============================================================
# SECTOR EV/EBITDA NEUTRAL POINTS (calibrated from real universe data)
# Financials intentionally excluded — EV/EBITDA is not a standard
# valuation tool for banks/insurers (debt is operational raw material,
# not a capital-structure choice), confirmed by that sector's own
# calibration data showing a much noisier, less coherent distribution
# than every other sector.
# ============================================================
SECTOR_EV_EBITDA_NEUTRAL = {
    "⛏️ Materials (XLB)": 11.59,
    "🏢 Real Estate (XLRE)": 20.01,
    "🏥 Health Care (XLV)": 16.26,
    "🏦 Financials (XLF)": None,  # excluded — see note above
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


def _get_ps_vs_own_history_score(ticker_symbol, current_ps):
    """
    Approximates a 1-year trailing average P/S using only price history,
    by assuming revenue-per-share stayed roughly constant over the window.
    Less accurate for fast-revenue-growth companies, but avoids needing
    fragile historical revenue data.
    """
    try:
        asset = yf.Ticker(ticker_symbol)
        df = asset.history(period="1y")
        if df.empty or len(df) < 100:
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
        score = _clip(ps_gap_pct * 2.5)  # calibrated

        direction = "cheaper" if ps_gap_pct > 0 else "more expensive"
        label = f"P/S: {current_ps:.1f} vs own 1Y avg: {avg_ps:.1f} ({direction} than own history)"
        return score, label

    except Exception:
        return 0, "P/S: History Fetch Failed"


def _get_ev_ebitda_score(ticker_symbol, ev_ebitda):
    """
    Scores EV/EBITDA relative to the stock's true sector median.
    Includes a sanity guard: near-zero/negative-EBITDA companies produce
    mathematically nonsensical ratios (e.g. -48009 or 3232), which are
    excluded here rather than scored, since the metric is meaningless
    for them regardless of the raw number's magnitude.
    """
    sector = TICKER_TRUE_SECTOR.get(ticker_symbol)
    if sector is None:
        return 0, "N/A (no sector mapped)"

    sector_median = SECTOR_EV_EBITDA_NEUTRAL.get(sector)
    if sector_median is None:
        return 0, f"N/A (EV/EBITDA not meaningful for {sector})"

    # Sanity guard: exclude nonsensical ratios from near-zero/negative EBITDA
    if ev_ebitda is None or ev_ebitda <= 0 or ev_ebitda > 100:
        return 0, "No Data (EV/EBITDA not meaningful — near-zero/negative EBITDA)"

    ev_ebitda_gap_pct = ((sector_median - ev_ebitda) / sector_median) * 100
    score = _clip(ev_ebitda_gap_pct * 1.0)  # calibrated

    direction = "cheaper" if ev_ebitda_gap_pct > 0 else "more expensive"
    label = f"EV/EBITDA: {ev_ebitda:.1f} vs sector median: {sector_median:.1f} ({direction} than sector)"
    return score, label


def _get_earnings_surprise_score(ticker_symbol):
    """
    Compares the most recent quarter's actual reported EPS against what
    analysts expected going into the report — a backward-looking
    result-surprise signal.
    """
    try:
        asset = yf.Ticker(ticker_symbol)
        earnings_dates = asset.earnings_dates

        if earnings_dates is None or earnings_dates.empty:
            return 0, "No Data"

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
        score = _clip(clipped_surprise * 1.8)  # Calibrated

        direction = "beat" if surprise_pct > 0 else ("missed" if surprise_pct < 0 else "met")
        label = f"EPS {direction} estimate by {surprise_pct:.1f}% (Est: {eps_estimate}, Actual: {eps_reported})"
        return score, label

    except Exception:
        return 0, "No Data"


def _get_eps_revision_score(ticker_symbol):
    """
    Measures whether analysts have been revising the CURRENT quarter's
    EPS estimate up or down over the past 30 days — a forward-looking
    expectation-surprise signal, distinct from the backward-looking
    earnings surprise metric above.
    """
    try:
        asset = yf.Ticker(ticker_symbol)
        eps_trend = asset.eps_trend

        if eps_trend is None or eps_trend.empty or '0q' not in eps_trend.index:
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
        score = _clip(clipped_revision * 3.5)  # Calibrated

        direction = "raised" if revision_pct > 0 else ("cut" if revision_pct < 0 else "unchanged")
        label = f"Current-Q EPS estimate {direction} {abs(revision_pct):.1f}% over 30 days (was: {past_est:.2f}, now: {current_est:.2f})"
        return score, label

    except Exception:
        return 0, "No Data"


@st.cache_data(ttl=21600)  # 6 hour cache — fundamentals change slowly
def calculate_fundamental_score(ticker_symbol):
    try:
        asset = yf.Ticker(ticker_symbol)
        info = asset.info

        if not info or len(info) < 5:
            return 0, {"⚠️ STATUS": "No Fundamental Data Available"}

        details = {}

        # ============================================================
        # 1. VALUATION (20%) — two components, blended 50/50:
        #    a) P/E-vs-forward (or P/S-vs-own-history fallback)
        #    b) EV/EBITDA vs sector median
        # ============================================================
        trailing_pe = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        price_to_sales = info.get('priceToSalesTrailing12Months')
        ev_ebitda_raw = info.get('enterpriseToEbitda')

        pe_ps_score = 0
        if trailing_pe is not None and trailing_pe > 0:
            if forward_pe is not None and forward_pe > 0:
                pe_gap_pct = ((trailing_pe - forward_pe) / trailing_pe) * 100
                pe_ps_score = _clip(pe_gap_pct * 1.6)  # calibrated
                details["Valuation (P/E)"] = (
                    f"Trailing P/E: {trailing_pe:.1f} vs Forward P/E: {forward_pe:.1f} "
                    f"({'growth expected' if pe_gap_pct > 0 else 'contraction expected'})"
                )
            else:
                pe_ps_score = 0
                details["Valuation (P/E)"] = f"Trailing P/E: {trailing_pe:.1f} (forward estimate unavailable)"
        else:
            if price_to_sales is not None and price_to_sales > 0:
                pe_ps_score, ps_label = _get_ps_vs_own_history_score(ticker_symbol, price_to_sales)
                details["Valuation (P/E)"] = ps_label
            else:
                pe_ps_score = 0
                details["Valuation (P/E)"] = "No Data"

        ev_ebitda_score, ev_ebitda_label = _get_ev_ebitda_score(ticker_symbol, ev_ebitda_raw)
        details["Valuation (EV/EBITDA)"] = ev_ebitda_label

        valuation_score = (pe_ps_score + ev_ebitda_score) / 2

        # ============================================================
        # 2. GROWTH (20%)
        # ============================================================
        revenue_growth = info.get('revenueGrowth')
        earnings_growth = info.get('earningsGrowth')

        rev_score = _clip(revenue_growth * 150) if revenue_growth is not None else 0  # calibrated

        if earnings_growth is not None:
            clipped_earnings_growth = max(-1.0, min(1.5, earnings_growth))
            earn_score = _clip(clipped_earnings_growth * 80)  # calibrated
        else:
            earn_score = 0

        growth_score = (rev_score + earn_score) / 2

        details["Revenue Growth (YoY)"] = f"{revenue_growth*100:.1f}%" if revenue_growth is not None else "No Data"
        details["Earnings Growth (YoY)"] = f"{earnings_growth*100:.1f}%" if earnings_growth is not None else "No Data"

        # ============================================================
        # 3. PROFITABILITY & QUALITY (20%)
        # ============================================================
        net_margin = info.get('profitMargins')
        roe = info.get('returnOnEquity')

        margin_score = _clip(net_margin * 200) if net_margin is not None else 0  # calibrated

        if roe is not None:
            clipped_roe = max(-0.5, min(1.0, roe))
            roe_score = _clip(clipped_roe * 110)  # calibrated
        else:
            roe_score = 0

        profitability_score = (margin_score + roe_score) / 2

        details["Net Margin"] = f"{net_margin*100:.1f}%" if net_margin is not None else "No Data"
        details["ROE"] = f"{roe*100:.1f}%" if roe is not None else "No Data"

        # ============================================================
        # 4. FINANCIAL HEALTH / LEVERAGE (20%)
        # ============================================================
        debt_to_equity = info.get('debtToEquity')
        current_ratio = info.get('currentRatio')

        if debt_to_equity is not None:
            clipped_de = max(0, min(400, debt_to_equity))
            de_score = _clip((100 - clipped_de) * 0.5)
        else:
            de_score = 0

        cr_score = _clip((current_ratio - 1.5) * 40) if current_ratio is not None else 0  # calibrated

        health_score = (de_score + cr_score) / 2

        details["Debt/Equity"] = f"{debt_to_equity:.1f}" if debt_to_equity is not None else "No Data"
        details["Current Ratio"] = f"{current_ratio:.2f}" if current_ratio is not None else "No Data"

        # ============================================================
        # 5. FINANCIAL SURPRISE (20%) — two components:
        #    a) Earnings surprise (backward-looking: last report vs estimate)
        #    b) EPS estimate revision (forward-looking: 30-day estimate trend)
        # ============================================================
        surprise_score, surprise_label = _get_earnings_surprise_score(ticker_symbol)
        revision_score, revision_label = _get_eps_revision_score(ticker_symbol)

        financial_surprise_score = (surprise_score + revision_score) / 2

        details["Earnings Surprise"] = surprise_label
        details["EPS Estimate Revision (30d)"] = revision_label

        # ============================================================
        # FINAL BLEND (20% each, five categories)
        # ============================================================
        final_score = (
            (valuation_score * 0.20) +
            (growth_score * 0.20) +
            (profitability_score * 0.20) +
            (health_score * 0.20) +
            (financial_surprise_score * 0.20)
        )

        return _clip(final_score), details

    except Exception as e:
        return 0, {"⚠️ STATUS": f"Fundamentals API Failure: {str(e)}"}