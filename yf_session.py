import streamlit as st
import requests
import time
import functools

# ============================================================
# SHARED SPOOFED SESSION
# Reduces the odds of Yahoo's servers flagging traffic as bot-like.
# ============================================================
@st.cache_resource
def get_yf_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    })
    return session


# ============================================================
# RETRY DECORATOR
# Wraps a data-fetching function: if it raises an exception
# (e.g. rate limit, timeout), wait briefly and try again, up to
# `max_retries` times, before finally giving up and re-raising.
# ============================================================
def with_retry(max_retries=2, delay_seconds=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay_seconds)
                    continue
            raise last_exception
        return wrapper
    return decorator