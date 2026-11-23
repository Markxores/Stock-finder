import streamlit as st
import pandas as pd
from sector_map import TICKER_TRUE_SECTOR
from yf_session import DataFetchError, get_ticker, get_daily_history, with_retry

# ============================================================
# HOW THIS FILE WORKS NOW (two separate jobs)
#
#   JOB 1 - TREND SCORE (feeds the Master Score)
#     Daily chart only. Moving averages only. No RSI, no MACD.
#     Range: -100 to +100. Then nudged by the sector (see below).
#
#   JOB 2 - REGIME (a label for your eyes only, NOT in the score)
#     Calculated twice: once on the daily chart, once on the 4H chart.
#     Each one says: Momentum / Pullback / Overextended / Neutral.
#     The daily regime uses RSI + MACD + ATR (the parts that were
#     taken OUT of the score). The 4H regime uses the same logic.
#
#   Both regimes use the TREND SCORE to know which direction the
#   stock is trending (up = +1, down = -1).
# ============================================================


# ============================================================
# SETTINGS YOU CAN TUNE
# These are the numbers we will optimize later. Change them here
# and nothing else in the file needs to change.
# ============================================================

# --- Trend score: the moving average lengths (in days) ---
MA_FAST = 20      # fast exponential moving average (EMA)
MA_MID = 50       # middle simple moving average (SMA)
MA_SLOW = 200     # slow simple moving average (SMA)

# --- Trend score: how many points each check is worth ---
# Add them up: 20 + 20 + 20 + 40 = 100, so the score runs -100 to +100.
# (Each check gives +points if true, -points if false.)
POINTS_PRICE_VS_FAST = 20
POINTS_PRICE_VS_MID = 20
POINTS_PRICE_VS_SLOW = 20
POINTS_PERFECT_ORDER = 40    # +40 if FAST > MID > SLOW, -40 if FAST < MID < SLOW, else 0

# --- Sector nudge ---
# 0.15 means a perfect sector can change the score by up to 15% of the
# sector's strength (sector score runs -70 to +70, so about +/-10.5% at most).
SECTOR_INFLUENCE = 0.15

# --- Regime settings (used on BOTH the daily and the 4H chart) ---
RSI_STRETCHED_HIGH = 70   # RSI above this = stretched upward
RSI_STRETCHED_LOW = 30    # RSI below this = stretched downward

# A timeframe that is moving hard AGAINST the trend (for example a sharp
# drop inside an uptrend) reads Neutral. This is how strong that move must
# be (0 to 100) before it counts. Lower = more things read Neutral.
AGAINST_TREND_NEUTRAL_AT = 60


# ============================================================
# THE FINAL REGIME LABEL (how the daily and 4H regimes combine)
#
# Rules, in this order (the first one that applies wins):
#   1. No trend direction (trend score of exactly 0)  -> Neutral
#   2. Daily regime is Overextended                   -> Overextended
#                                                       (the 4H is ignored)
#   3. Daily and 4H are both Momentum                 -> Momentum
#   4. One is Pullback and the other is Momentum
#      or Pullback                                    -> Pullback
#   5. Anything else (a Neutral is involved)          -> the DAILY regime decides
#
# The 4H regime never says "Overextended". When the 4H is stretched in
# the direction of the trend it counts as Momentum.
# ============================================================

# The text shown on screen. Change the right-hand side to rename a label.
# The key is (trend direction, regime type).
FINAL_LABELS = {
    (1, "Momentum"): "Upward Momentum",
    (1, "Pullback"): "Bullish Pullback",
    (1, "Overextended"): "Overextended (Up)",
    (-1, "Momentum"): "Downward Momentum",
    (-1, "Pullback"): "Bearish Pullback",
    (-1, "Overextended"): "Overextended (Down)",
}
NEUTRAL_LABEL = "Neutral"


def get_final_regime(direction, daily_regime, h4_regime):
    """
    Combines the daily and 4H regimes into ONE regime type:
    "Momentum", "Pullback", "Overextended" or "Neutral".
    """
    if direction == 0:
        return "Neutral"
    if daily_regime == "Overextended":
        return "Overextended"
    if daily_regime == "Momentum" and h4_regime == "Momentum":
        return "Momentum"
    if "Pullback" in (daily_regime, h4_regime) and h4_regime in ("Momentum", "Pullback") and daily_regime in ("Momentum", "Pullback"):
        return "Pullback"
    # A Neutral is involved (or the 4H is unavailable): the daily regime decides
    return daily_regime


def get_combined_regime_label(direction, daily_regime, h4_regime):
    """Returns (regime_type, text_for_the_table)."""
    regime_type = get_final_regime(direction, daily_regime, h4_regime)
    text = FINAL_LABELS.get((direction, regime_type), NEUTRAL_LABEL)

    # If the 4H could not be downloaded, the daily regime decided alone. Say so.
    if h4_regime == "Unavailable" and regime_type != "Overextended" and direction != 0:
        text += " (no 4H data)"
    return regime_type, text


# ============================================================
# SECTOR ETF MAPPING
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
# HOW FAILURES WORK IN THIS FILE (unchanged)
#   - If Yahoo fails (after retrying), the function raises DataFetchError.
#     Streamlit never saves a function that raised an error, so a failed
#     lookup is NOT remembered. The next scan tries again.
#   - If a stock has fewer days of history than the slow moving average
#     needs, the trend score is None (MISSING, shown as a dash).
# ============================================================


# ============================================================
# JOB 1: DAILY TREND SCORE (moving averages only)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def calculate_trend_score(ticker_symbol):
    df = get_daily_history(ticker_symbol)  # raises DataFetchError if Yahoo fails
    if len(df) < MA_SLOW:
        return None, {"⚠️ STATUS": "Insufficient History"}

    ema_fast = df['Close'].ewm(span=MA_FAST, adjust=False).mean().iloc[-1]
    sma_mid = df['Close'].rolling(window=MA_MID).mean().iloc[-1]
    sma_slow = df['Close'].rolling(window=MA_SLOW).mean().iloc[-1]
    close = df['Close'].iloc[-1]

    score = 0
    score += POINTS_PRICE_VS_FAST if close > ema_fast else -POINTS_PRICE_VS_FAST
    score += POINTS_PRICE_VS_MID if close > sma_mid else -POINTS_PRICE_VS_MID
    score += POINTS_PRICE_VS_SLOW if close > sma_slow else -POINTS_PRICE_VS_SLOW

    if ema_fast > sma_mid and sma_mid > sma_slow:
        score += POINTS_PERFECT_ORDER
    elif ema_fast < sma_mid and sma_mid < sma_slow:
        score -= POINTS_PERFECT_ORDER

    details = {
        "Close Price": round(close, 4),
        f"EMA {MA_FAST}": round(ema_fast, 4),
        f"SMA {MA_MID}": round(sma_mid, 4),
        f"SMA {MA_SLOW}": round(sma_slow, 4),
    }
    return max(-100, min(100, score)), details


# ============================================================
# JOB 2: REGIME (shared by the daily chart and the 4H chart)
# ============================================================
def _add_regime_indicators(df):
    """
    Adds the indicators the regime needs: a 20-period EMA, RSI(14),
    MACD histogram and ATR(14). Works on any timeframe.
    (The regime's EMA 20 is separate from the trend score's moving
    averages, so optimizing the trend score will not change the regime.)
    """
    df = df.copy()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

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

    return df.dropna(subset=['EMA_20', 'RSI_14', 'MACD_hist', 'ATR_14'])


def _momentum_strength(c, prev, direction):
    """0 to 100: how strongly the chart is moving WITH the trend right now."""
    s = 0
    if direction > 0:
        if c['Close'] > c['EMA_20'] and c['EMA_20'] > prev['EMA_20']:
            s += 30
        if c['RSI_14'] > 55 and c['RSI_14'] > prev['RSI_14']:
            s += 30
        if c['MACD_hist'] > 0 and c['MACD_hist'] > prev['MACD_hist']:
            s += 40
    else:
        if c['Close'] < c['EMA_20'] and c['EMA_20'] < prev['EMA_20']:
            s += 30
        if c['RSI_14'] < 45 and c['RSI_14'] < prev['RSI_14']:
            s += 30
        if c['MACD_hist'] < 0 and c['MACD_hist'] < prev['MACD_hist']:
            s += 40
    return s


def _pullback_strength(c, prev, direction):
    """0 to 100: how much the chart looks like a dip (or bounce) that is turning back WITH the trend."""
    atr = c['ATR_14']
    if atr == 0 or pd.isna(atr):
        return 0

    dist_from_ema20_atr = (c['Close'] - c['EMA_20']) / atr
    s = 0
    if direction > 0:
        if -1.5 <= dist_from_ema20_atr <= 0.5:
            s += 30
        if 30 <= c['RSI_14'] <= 50:
            s += 30
        if c['RSI_14'] > prev['RSI_14']:
            s += 20
        if c['MACD_hist'] > prev['MACD_hist']:
            s += 20
    else:
        if -0.5 <= dist_from_ema20_atr <= 1.5:
            s += 30
        if 50 <= c['RSI_14'] <= 70:
            s += 30
        if c['RSI_14'] < prev['RSI_14']:
            s += 20
        if c['MACD_hist'] < prev['MACD_hist']:
            s += 20
    return s


def calculate_regime(df, direction, is_daily):
    """
    Works out the regime for ONE timeframe.
      df        = price table that already has the regime indicators
      direction = +1 (uptrend), -1 (downtrend) or 0 (no trend)
      is_daily  = True for the daily chart, False for the 4H chart
    Returns (regime_name, momentum_strength, pullback_strength, last_row)

    Order of checks (the first one that applies wins):
      1. RSI stretched IN the direction of the trend (above 70 in an uptrend,
         below 30 in a downtrend):
            daily -> "Overextended"
            4H    -> "Momentum"  (the 4H never says Overextended)
      2. Moving AGAINST the trend -> "Neutral", in two cases:
            - RSI is stretched the other way (below 30 in an uptrend,
              above 70 in a downtrend), always Neutral
            - a strong counter-move (e.g. a sharp drop inside an uptrend)
              that scores AGAINST_TREND_NEUTRAL_AT or more, UNLESS the
              pullback score is higher than that counter-move score
              (a tie stays Neutral)
      3. Whichever of Momentum / Pullback is stronger (a tie goes to Momentum)
      4. If both are zero -> "Neutral"
    """
    if df is None or len(df) < 5 or direction == 0:
        return "Neutral", 0, 0, None

    c = df.iloc[-1]
    prev = df.iloc[-2]
    rsi = c['RSI_14']

    momentum = _momentum_strength(c, prev, direction)
    pullback = _pullback_strength(c, prev, direction)
    # Same momentum check, but looking the OPPOSITE way (against the trend)
    against_trend = _momentum_strength(c, prev, -direction)

    # 1. Stretched in the direction of the trend
    stretched_with_trend = (direction > 0 and rsi > RSI_STRETCHED_HIGH) or \
                           (direction < 0 and rsi < RSI_STRETCHED_LOW)
    if stretched_with_trend:
        return ("Overextended" if is_daily else "Momentum"), momentum, pullback, c

    # 2. Moving hard against the trend
    stretched_against_trend = (direction > 0 and rsi < RSI_STRETCHED_LOW) or \
                              (direction < 0 and rsi > RSI_STRETCHED_HIGH)
    if stretched_against_trend:
        return "Neutral", momentum, pullback, c
    # A strong move against the trend reads Neutral, UNLESS the pullback
    # score is higher than it (then the pullback takes precedence).
    if against_trend >= AGAINST_TREND_NEUTRAL_AT and pullback <= against_trend:
        return "Neutral", momentum, pullback, c

    # 3 and 4. Momentum, Pullback or Neutral
    if momentum == 0 and pullback == 0:
        return "Neutral", momentum, pullback, c
    if momentum >= pullback:
        return "Momentum", momentum, pullback, c
    return "Pullback", momentum, pullback, c


def get_daily_regime(ticker_symbol, direction):
    """Daily regime. Uses the same cached daily download as the trend score (no extra Yahoo request)."""
    df = _add_regime_indicators(get_daily_history(ticker_symbol))
    return calculate_regime(df, direction, is_daily=True)


# ============================================================
# 4H DATA (only used for the 4H regime)
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
    Returns the 4H table with regime indicators, or None if there are too few candles.
    Raises DataFetchError if the hourly download itself failed.
    """
    df_1h = _download_1h_history(ticker_symbol)

    df = df_1h.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    if len(df) < 60:
        return None

    return _add_regime_indicators(df)


# ============================================================
# SECTOR BENCHMARK SCORE (unchanged)
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
# MASTER TECHNICAL FUNCTION
# (Not cached itself: it only does quick maths on cached downloads.)
#
# What happens when data is missing:
#   - Daily prices fail        -> raises DataFetchError (the whole technical score fails)
#   - Under MA_SLOW days       -> returns None (MISSING, shown as a dash)
#   - 4H prices fail           -> the score is NOT affected (it is daily only now).
#                                 Only the 4H regime is "Unavailable", with a warning note.
#   - Sector ETF fails         -> no sector adjustment is applied, with a warning note
# ============================================================
def calculate_technical_score(ticker_symbol):
    notes = []

    # --- JOB 1: trend score from the moving averages ---
    trend_score, trend_details = calculate_trend_score(ticker_symbol)

    if trend_score is None:
        return None, trend_details

    direction = 1 if trend_score > 0 else (-1 if trend_score < 0 else 0)

    # --- JOB 2a: daily regime ---
    daily_regime, daily_mom, daily_pull, daily_row = get_daily_regime(ticker_symbol, direction)

    # --- JOB 2b: 4H regime ---
    try:
        df4h = get_4h_indicators(ticker_symbol)
    except DataFetchError:
        df4h = None
        notes.append("4H data unavailable (4H regime not shown; the score is not affected)")

    if df4h is None:
        h4_regime, h4_mom, h4_pull, h4_row = "Unavailable", 0, 0, None
    else:
        h4_regime, h4_mom, h4_pull, h4_row = calculate_regime(df4h, direction, is_daily=False)

    # --- Sector nudge (FIXED) ---
    # "direction" flips the sign for a bearish stock, so:
    #   strong sector + bullish stock -> score gets bigger (stronger bullish)
    #   strong sector + bearish stock -> score gets smaller (weaker bearish)
    #   weak sector   + bullish stock -> score gets smaller
    #   weak sector   + bearish stock -> score gets bigger (stronger bearish)
    sector_category = TICKER_TRUE_SECTOR.get(ticker_symbol)
    sector_etf = SECTOR_ETF_MAPPING.get(sector_category) if sector_category else None

    sector_details = {}
    sector_modifier = 1.0
    sector_label = "N/A (no sector ETF mapped)"

    if sector_etf:
        try:
            sector_score, sector_details = calculate_sector_score(sector_etf)
            sector_modifier = 1 + direction * (sector_score / 100) * SECTOR_INFLUENCE
            sector_label = f"{sector_modifier:.3f}"
        except DataFetchError:
            sector_details = {}
            sector_modifier = 1.0
            sector_label = "N/A (sector data unavailable)"
            notes.append("Sector ETF data unavailable (no sector adjustment applied)")

    final_score = max(-100, min(100, trend_score * sector_modifier))

    # --- Final regime label ---
    regime_type, regime_text = get_combined_regime_label(direction, daily_regime, h4_regime)

    # --- Details shown in the Deep Dive ---
    details = {
        **trend_details,
        "Combined Regime": regime_text,
        "Regime Type": regime_type,
        "Daily Regime": daily_regime,
        "4H Regime": h4_regime,
        "Trend Score (before sector)": round(trend_score, 1),
        "Sector Modifier": sector_label,
        "Daily Momentum Strength": daily_mom,
        "Daily Pullback Strength": daily_pull,
        "4H Momentum Strength": h4_mom,
        "4H Pullback Strength": h4_pull,
    }
    if daily_row is not None:
        details["Daily RSI (14)"] = round(daily_row['RSI_14'], 2)
        details["Daily MACD Hist"] = round(daily_row['MACD_hist'], 4)
    if h4_row is not None:
        details["4H RSI (14)"] = round(h4_row['RSI_14'], 2)
        details["4H MACD Hist"] = round(h4_row['MACD_hist'], 4)
    if sector_details:
        details.update(sector_details)
    if notes:
        details["⚠️ Data Notes"] = "; ".join(notes)

    return final_score, details
