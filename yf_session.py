import functools
import random
import threading
import time

import streamlit as st
import yfinance as yf

# ============================================================
# SETTINGS YOU CAN TUNE
# If Yahoo still blocks you too often, make the pauses longer.
# If scans feel too slow and you are NOT being blocked, make them shorter.
# ============================================================
MIN_GAP_SECONDS = 0.6           # shortest pause between ANY two Yahoo requests
JITTER_SECONDS = 0.3            # small random extra pause, so requests look less robotic
COOLDOWN_BASE_SECONDS = 15      # first pause after Yahoo says "too many requests"
COOLDOWN_MAX_SECONDS = 120      # the longest pause we will ever use
MAX_WAIT_SECONDS = 60           # if the pause would be longer than this, we stop and report failure
STRIKE_MEMORY_SECONDS = 600     # after 10 quiet minutes, forget earlier "too many requests" warnings


class DataFetchError(Exception):
    """Raised when data could not be fetched from Yahoo, even after retrying."""
    pass


# ============================================================
# SHARED TRAFFIC CONTROL
# Every Yahoo request in the app goes through here, so all of them
# share ONE pace. This is what stops requests arriving in bursts.
# ============================================================
_lock = threading.Lock()
_state = {
    "next_slot": 0.0,       # earliest time the next request may start
    "blocked_until": 0.0,   # Yahoo told us to slow down until this time
    "strikes": 0,           # how many "too many requests" answers in a row
    "last_strike": 0.0,
}


def _is_rate_limit_error(exc):
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "ratelimit" in name
        or "too many requests" in text
        or "rate limit" in text
        or "rate limited" in text
        or "429" in text
    )


def _wait_for_turn():
    """Waits until it is safe to send the next request. Fails fast if Yahoo has blocked us for a long time."""
    with _lock:
        now = time.monotonic()
        remaining_block = _state["blocked_until"] - now
        if remaining_block > MAX_WAIT_SECONDS:
            raise DataFetchError(
                f"Yahoo is limiting requests. Try again in about {int(remaining_block)} seconds."
            )
        start_at = max(now, _state["next_slot"], _state["blocked_until"])
        _state["next_slot"] = start_at + MIN_GAP_SECONDS + random.uniform(0, JITTER_SECONDS)

    delay = start_at - time.monotonic()
    if delay > 0:
        time.sleep(delay)


def _record_success():
    with _lock:
        _state["strikes"] = 0


def _record_rate_limit():
    """Yahoo said 'too many requests': pause everything, and pause longer each time it repeats."""
    with _lock:
        now = time.monotonic()
        if now - _state["last_strike"] > STRIKE_MEMORY_SECONDS:
            _state["strikes"] = 0
        _state["strikes"] += 1
        _state["last_strike"] = now
        wait = min(COOLDOWN_BASE_SECONDS * (2 ** (_state["strikes"] - 1)), COOLDOWN_MAX_SECONDS)
        _state["blocked_until"] = now + wait


def seconds_until_unblocked():
    """How many seconds are left before we will send requests to Yahoo again (0 = not paused)."""
    with _lock:
        return max(0, int(_state["blocked_until"] - time.monotonic()))


# ============================================================
# RETRY WRAPPER
# - Waits for its turn before every attempt (so retries are paced too)
# - Retries real errors, waiting 2s then 4s (genuinely doubling)
# - If Yahoo says "too many requests", everything pauses (see above)
# - After the last attempt fails, raises DataFetchError. Streamlit does
#   NOT remember a function that ended in an error, so failures are
#   never saved, and the next scan simply tries again.
# ============================================================
def with_retry(max_retries=2, delay_seconds=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                _wait_for_turn()
                try:
                    result = func(*args, **kwargs)
                    _record_success()
                    return result
                except Exception as e:
                    last_exception = e
                    if _is_rate_limit_error(e):
                        _record_rate_limit()
                    if attempt < max_retries:
                        time.sleep(delay_seconds * (2 ** attempt))
            message = f"{type(last_exception).__name__}: {last_exception}"
            raise DataFetchError(message[:300]) from last_exception
        return wrapper
    return decorator


# ============================================================
# YAHOO ACCESS
# ============================================================
def get_ticker(symbol):
    """
    Creates a yfinance Ticker.

    We deliberately do NOT pass our own session or cookies any more.
    Current yfinance versions handle the cookie handshake and browser
    disguise themselves, and they REJECT a hand-made `requests` session
    with the error "stop setting session, let YF handle".
    """
    return yf.Ticker(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
@with_retry(max_retries=2, delay_seconds=2)
def get_daily_history(ticker_symbol):
    """
    One year of daily prices. Shared by the technical score, the sector
    score and the P/S check, so each stock's price history is downloaded
    once per hour instead of two or three times.
    An empty answer counts as a failure, because a real, active stock
    always has price history.
    """
    df = get_ticker(ticker_symbol).history(period="1y")
    if df is None or df.empty:
        raise ValueError(f"No daily price history returned for {ticker_symbol}")
    return df
