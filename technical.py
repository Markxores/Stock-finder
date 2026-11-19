import streamlit as st
import pandas as pd
from sector_map import TICKER_TRUE_SECTOR
from yf_session import DataFetchError, get_ticker, get_daily_history, with_retry

# ============================================================
# SECTOR ETF MAPPING
# Curated baskets (Mega-Cap, Momentum, Dividend, etc.) intentionally
# have no entry here — they'll fall through to "no adjustment"
# ============================================================
SECTOR_ETF_MAPPING = {
    "💻 Information Technology (XLK)": "XLK",
    "🏦 Financials (XLF)": "XLF",
    "🏥 Health Care (XLV)": "XLV",
    "🛍️ Consumer Discretionary (XLY)": "XLY",
    "📡 Communication Services (XLC)": "XLC",
    "🏭 Industrials (XLI)": "XLI",
    "🛒 Consumer Staples (XLP)": "XLP",
    "🛢️ Energy (XLE)": "XLE",
    "⛏️ Materials (XLB)": "XLB",
    "🏢 Real Estate (XLRE)": "XLRE",
    "💡 Utilities (XLU)": "XLU",
}

# ============================================================
# HOW FAILURES WORK IN THIS FILE
#   - If Yahoo fails (after retrying), the function raises DataFetchError.
#     Streamlit never saves a function that raised an error, so a failed
#     lookup is NOT remembered. The next scan tries again.
#   - The caching sits on the raw downloads, not on the final score, so a
#     half-failed score can never be saved either.
# ============================================================


# ============================================================
# DAILY TREND SCORE
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def calculate_daily_trend_score(ticker_symbol):
    df = get_daily_history(ticker_symbol)  # raises DataFetchError if Yahoo fails
    if len(df) < 200:
        return 0, {"⚠️ STATUS": "Insufficient History"}

    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = ema_12 - ema_26
    df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']

    current = df.iloc[-1]
    close = current['Close']
    ema_20 = current['EMA_20']
    sma_50 = current['SMA_50']
    sma_200 = current['SMA_200']
    rsi_14 = current['RSI_14']
    macd_hist = current['MACDh_12_26_9']

    score = 0
    score += 10 if close > ema_20 else -10
    score += 10 if close > sma_50 else -10
    score += 10 if close > sma_200 else -10

    if ema_20 > sma_50 and sma_50 > sma_200: score += 20
    elif ema_20 < sma_50 and sma_50 < sma_200: score -= 20

    if rsi_14 > 70: score -= 25
    elif rsi_14 < 30: score += 25
    elif rsi_14 > 50: score += 15
    else: score -= 15

    if macd_hist > 0: score += 25
    else: score -= 25

    details = {
        "Close Price": round(close, 4),
        "EMA 20": round(ema_20, 4),
        "SMA 50": round(sma_50, 4),
        "SMA 200": round(sma_200, 4),
        "RSI (14)": round(rsi_14, 2),
        "MACD Hist": round(macd_hist, 4)
    }
    return max(-100, min(100, score)), details


# ============================================================
# 4H DATA + INDICATORS
# ============================================================
@st.cache_data(ttl=1800, show_spinner=False)
@with_retry(max_retries=2, delay_seconds=2)
def _download_1h_history(ticker_symbol):
    df = get_ticker(ticker_symbol).history(period="60d", interval="1h")
    if df is None or df.empty:
        raise ValueError(f"No hourly price history returned for {ticker_symbol}")
    return df


def get_4h_indicators(ticker_symbol):
    """
    Returns the 4H table, or None if there are too few candles to use.
    Raises DataFetchError if the hourly download itself failed.
    """
    df_1h = _download_1h_history(ticker_symbol)

    df = df_1h.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    if len(df) < 60:
        return None

    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss))

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = macd - signal

    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()

    return df.dropna()


# ============================================================
# 4H MOMENTUM SCORE
# ============================================================
def calculate_4h_momentum_score(df4h, daily_direction):
    if df4h is None or len(df4h) < 5 or daily_direction == 0:
        return 0

    c = df4h.iloc[-1]
    prev = df4h.iloc[-2]
    score = 0

    if daily_direction > 0:
        if c['Close'] > c['EMA_20'] and c['EMA_20'] > prev['EMA_20']:
            score += 30
        if c['RSI_14'] > 55 and c['RSI_14'] > prev['RSI_14']:
            score += 30
        if c['MACD_hist'] > 0 and c['MACD_hist'] > prev['MACD_hist']:
            score += 40
    else:
        if c['Close'] < c['EMA_20'] and c['EMA_20'] < prev['EMA_20']:
            score += 30
        if c['RSI_14'] < 45 and c['RSI_14'] < prev['RSI_14']:
            score += 30
        if c['MACD_hist'] < 0 and c['MACD_hist'] < prev['MACD_hist']:
            score += 40
        score = -score

    return max(-100, min(100, score))


# ============================================================
# 4H PULLBACK SCORE
# ============================================================
def calculate_4h_pullback_score(df4h, daily_direction):
    if df4h is None or len(df4h) < 5 or daily_direction == 0:
        return 0

    c = df4h.iloc[-1]
    prev = df4h.iloc[-2]
    atr = c['ATR_14']
    if atr == 0 or pd.isna(atr):
        return 0

    dist_from_ema20_atr = (c['Close'] - c['EMA_20']) / atr
    score = 0

    if daily_direction > 0:
        if -1.5 <= dist_from_ema20_atr <= 0.5:
            score += 30
        if 30 <= c['RSI_14'] <= 50:
            score += 30
        if c['RSI_14'] > prev['RSI_14']:
            score += 20
        if c['MACD_hist'] > prev['MACD_hist']:
            score += 20
    else:
        if -0.5 <= dist_from_ema20_atr <= 1.5:
            score += 30
        if 50 <= c['RSI_14'] <= 70:
            score += 30
        if c['RSI_14'] < prev['RSI_14']:
            score += 20
        if c['MACD_hist'] < prev['MACD_hist']:
            score += 20
        score = -score

    return max(-100, min(100, score))


# ============================================================
# SECTOR BENCHMARK SCORE
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def calculate_sector_score(sector_etf_ticker):
    df = get_daily_history(sector_etf_ticker)  # raises DataFetchError if Yahoo fails
    if len(df) < 200:
        return 0, {"⚠️ STATUS": "Insufficient Sector History"}

    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    current = df.iloc[-1]
    close = current['Close']
    ema_20 = current['EMA_20']
    sma_50 = current['SMA_50']
    sma_200 = current['SMA_200']

    score = 0
    score += 15 if close > ema_20 else -15
    score += 15 if close > sma_50 else -15
    score += 15 if close > sma_200 else -15

    if ema_20 > sma_50 and sma_50 > sma_200: score += 25
    elif ema_20 < sma_50 and sma_50 < sma_200: score -= 25

    details = {
        "Sector ETF": sector_etf_ticker,
        "Sector Close": round(close, 2),
        "Sector EMA 20": round(ema_20, 2),
        "Sector SMA 50": round(sma_50, 2),
        "Sector SMA 200": round(sma_200, 2),
    }
    return max(-100, min(100, score)), details


# ============================================================
# MASTER COMBINED TECHNICAL ENGINE
# (Not cached itself: it only does quick maths on cached downloads.)
#
# What happens when data is missing:
#   - Daily prices fail  -> raises DataFetchError (the whole technical score fails)
#   - 4H prices fail     -> the score uses the daily trend only, with a warning note
#   - Sector ETF fails   -> no sector adjustment is applied, with a warning note
# ============================================================
def calculate_technical_score(ticker_symbol, category_name=None):
    notes = []

    daily_score, daily_details = calculate_daily_trend_score(ticker_symbol)
    daily_direction = 1 if daily_score > 0 else (-1 if daily_score < 0 else 0)

    four_h_available = True
    try:
        df4h = get_4h_indicators(ticker_symbol)
    except DataFetchError:
        df4h = None
        four_h_available = False
        notes.append("4H data unavailable (score uses the daily trend only)")

    momentum_score = calculate_4h_momentum_score(df4h, daily_direction)
    pullback_score = calculate_4h_pullback_score(df4h, daily_direction)

    if not four_h_available:
        four_h_score = 0
        active_mode = "Unavailable"
    elif abs(momentum_score) >= abs(pullback_score):
        four_h_score = momentum_score
        active_mode = "Momentum" if momentum_score != 0 else "Neutral"
    else:
        four_h_score = pullback_score
        active_mode = "Pullback"

    if four_h_available:
        core_score = (daily_score * 0.7) + (four_h_score * 0.3)
    else:
        core_score = daily_score

    sector_category = TICKER_TRUE_SECTOR.get(ticker_symbol)
    sector_etf = SECTOR_ETF_MAPPING.get(sector_category) if sector_category else None

    sector_details = {}
    sector_modifier = 1.0
    sector_label = "N/A (no sector ETF mapped)"

    if sector_etf:
        try:
            sector_score, sector_details = calculate_sector_score(sector_etf)
            sector_modifier = 1 + (sector_score / 100) * 0.15
            sector_label = f"{sector_modifier:.3f}"
        except DataFetchError:
            sector_details = {}
            sector_modifier = 1.0
            sector_label = "N/A (sector data unavailable)"
            notes.append("Sector ETF data unavailable (no sector adjustment applied)")

    final_score = max(-100, min(100, core_score * sector_modifier))

    details = {
        **daily_details,
        "4H Trading Mode": active_mode,
        "4H Momentum Score": round(momentum_score, 1),
        "4H Pullback Score": round(pullback_score, 1),
        "4H Trigger Component": round(four_h_score, 1),
        "Sector Modifier": sector_label,
    }
    if sector_details:
        details.update(sector_details)
    if notes:
        details["⚠️ Data Notes"] = "; ".join(notes)

    return max(-100, min(100, final_score)), details
