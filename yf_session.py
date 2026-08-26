import streamlit as st
import requests
import time
import functools

# ============================================================
# SHARED SPOOFED SESSION WITH COOKIE HANDSHAKE
# ============================================================
@st.cache_resource
def get_yf_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive"
    })
    
    # Pre-emptively hit Yahoo Finance to secure valid cookies and prevent 401/403 blocks
    try:
        session.get("https://finance.yahoo.com", timeout=5)
    except Exception:
        pass
        
    return session


# ============================================================
# RETRY DECORATOR WITH EXPONENTIAL BACKOFF
# ============================================================
def with_retry(max_retries=3, delay_seconds=1.5):
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
                        # Exponential backoff: wait longer on each subsequent failure
                        time.sleep(delay_seconds * (attempt + 1))
                    continue
            raise last_exception
        return wrapper
    return decorator