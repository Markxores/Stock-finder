import streamlit as st
import pandas as pd
import time
import json
import os
import requests

from universe import STOCK_UNIVERSES
from technical import calculate_technical_score
from fundamentals import calculate_fundamental_score, get_earnings_countdown
from sentiment import calculate_sentiment_score

# ============================================================
# WATCHLIST PERSISTENCE (GitHub Gist, with local file fallback)
#
# How it works:
#   - If GITHUB_TOKEN and GIST_ID exist in Streamlit Secrets, the
#     watchlist is stored in your private Gist (works on every device).
#   - If they don't exist (e.g. testing on your own computer), the app
#     falls back to a local watchlist.json file.
#   - If the Gist exists but can't be reached, the app goes into an
#     "error" state and BLOCKS saving, so a network hiccup can never
#     overwrite your real watchlist with an empty one.
# ============================================================
WATCHLIST_FILE = "watchlist.json"      # local backup / fallback file
GIST_FILENAME = "watchlist.json"       # the file name inside your Gist
GIST_API_URL = "https://api.github.com/gists/"


def _gist_settings():
    """Reads the token and Gist ID from Streamlit Secrets. Returns (None, None) if missing."""
    try:
        token = st.secrets["GITHUB_TOKEN"]
        gist_id = st.secrets["GIST_ID"]
        if token and gist_id:
            return token, gist_id
    except Exception:
        pass
    return None, None


def _gist_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def _load_local_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            return {}
    return {}


def _save_local_watchlist(data):
    try:
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception:
        return False


def load_watchlist():
    """
    Returns (watchlist_dict, status, message).
    status is one of:
      "gist"  -> loaded from your Gist (normal)
      "local" -> Gist not set up, using the local file
      "error" -> Gist is set up but could not be read (saving is blocked)
    """
    token, gist_id = _gist_settings()

    if token is None:
        return _load_local_watchlist(), "local", "Gist not set up - using local file."

    last_error = "Unknown error"
    for attempt in range(3):
        try:
            response = requests.get(
                GIST_API_URL + gist_id,
                headers=_gist_headers(token),
                timeout=10,
            )
            response.raise_for_status()
            files = response.json().get("files", {})

            if GIST_FILENAME not in files:
                return {}, "error", f"The Gist has no file called '{GIST_FILENAME}'. Check the file name."

            content = (files[GIST_FILENAME].get("content") or "").strip() or "{}"
            data = json.loads(content)

            if not isinstance(data, dict):
                return {}, "error", "The Gist file is not a valid watchlist (expected {...})."

            return data, "gist", "Watchlist synced with GitHub Gist."

        except Exception as e:
            last_error = str(e)
            if attempt < 2:
                time.sleep(1.5)

    return {}, "error", f"Could not reach your Gist: {last_error}"


def save_watchlist(data):
    """
    Saves the watchlist. Returns (success, message).
    Refuses to save if the Gist is set up but we are not sure we can reach it.
    """
    token, gist_id = _gist_settings()

    # No Gist set up -> local file only
    if token is None:
        if _save_local_watchlist(data):
            return True, "Saved to local file."
        return False, "Could not write the local watchlist file."

    try:
        response = requests.patch(
            GIST_API_URL + gist_id,
            headers=_gist_headers(token),
            json={"files": {GIST_FILENAME: {"content": json.dumps(data, indent=4)}}},
            timeout=10,
        )
        response.raise_for_status()
    except Exception as e:
        return False, f"Could not save to your Gist: {e}"

    # Keep a local backup copy too (it's fine if this part fails)
    _save_local_watchlist(data)
    return True, "Saved to Gist."


def update_watchlist(change_function):
    """
    Safe way to change the watchlist:
      1. Fetch the LATEST list from the Gist (so a stale phone/laptop tab
         can't overwrite changes made on another device)
      2. Apply the change
      3. Save it
      4. Only update the app's memory if the save worked
    Returns (success, message).
    """
    latest, status, message = load_watchlist()
    if status == "error":
        st.session_state.watchlist_status = "error"
        st.session_state.watchlist_message = message
        return False, message

    change_function(latest)
    ok, save_message = save_watchlist(latest)

    if ok:
        st.session_state.custom_watchlist = latest
    return ok, save_message


# ============================================================
# PAGE SETUP & STATE INITIALIZATION
# ============================================================
st.set_page_config(
    page_title="Stock Finder",
    page_icon="📈",
    layout="wide"
)

if "custom_watchlist" not in st.session_state:
    _wl, _status, _message = load_watchlist()
    st.session_state.custom_watchlist = _wl
    st.session_state.watchlist_status = _status
    st.session_state.watchlist_message = _message

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚙️ Stock Finder Engine")
    
    # Inject Custom Watchlist into the dropdown
    universe_options = ["⭐ Custom Watchlist"] + list(STOCK_UNIVERSES.keys())
    
    selected_category = st.selectbox(
        "Select Universe / Sector",
        universe_options
    )
    
    # Route the engine to the correct dictionary
    if selected_category == "⭐ Custom Watchlist":
        tickers_in_category = st.session_state.custom_watchlist
    else:
        tickers_in_category = STOCK_UNIVERSES[selected_category]

    st.info(f"Target universe contains **{len(tickers_in_category)}** equities.")

    # --- Watchlist storage status ---
    if st.session_state.watchlist_status == "gist":
        st.caption("☁️ Watchlist synced with GitHub Gist")
    elif st.session_state.watchlist_status == "local":
        st.caption("💾 Watchlist stored in a local file (Gist not set up)")
    else:
        st.error(f"⚠️ Watchlist unavailable. {st.session_state.watchlist_message}")
        if st.button("🔄 Retry connection", use_container_width=True):
            _wl, _status, _message = load_watchlist()
            st.session_state.custom_watchlist = _wl
            st.session_state.watchlist_status = _status
            st.session_state.watchlist_message = _message
            st.session_state.last_scanned_category = None  # force a fresh scan
            st.rerun()

    st.markdown("---")
    st.caption("**Setup Icon Legend**")
    st.caption("🚀 Bullish Momentum &nbsp;&nbsp; 🎯 Bullish Pullback")
    st.caption("⚡ Bearish Momentum &nbsp;&nbsp; 🎯 Bearish Pullback")
    st.caption("➖ No Clear 4H Setup")

st.title(f"📊 Stock Finder: {selected_category}")
st.divider()

# ============================================================
# HELPER: SETUP ICON
# ============================================================
def get_setup_icon(tech_score, tech_details):
    """
    Combines the technical score's direction with the 4H Trading Mode.
    Unified Pullback Icon: 🎯 used for both bull and bear pullbacks.
    """
    mode = tech_details.get("4H Trading Mode", "Neutral")

    if mode == "Pullback":
        return "🎯"
    elif tech_score > 0 and mode == "Momentum":
        return "🚀"
    elif tech_score < 0 and mode == "Momentum":
        return "⚡"
    else:
        return "➖"

# ============================================================
# LIVE DATA SCANNER (The Loop)
# ============================================================
if "last_scanned_category" not in st.session_state or st.session_state.last_scanned_category != selected_category:

    # --- PREVENT EMPTY SCAN CRASH ---
    if len(tickers_in_category) == 0:
        if selected_category == "⭐ Custom Watchlist" and st.session_state.watchlist_status == "error":
            st.warning("Your watchlist could not be loaded, so it is not shown. Use 'Retry connection' in the sidebar.")
        else:
            st.warning("Your Custom Watchlist is empty. Select a standard sector to scan and add some stocks!")
        st.session_state.scanned_data = []
        st.session_state.breakdown_data = {}
        st.session_state.last_scanned_category = selected_category
    else:
        scanned_data = []
        breakdown_data = {}
        total_stocks = len(tickers_in_category)
        my_bar = st.progress(0, text="Scanning live markets...")

        for i, (name, ticker) in enumerate(tickers_in_category.items()):

            tech_score, tech_details = calculate_technical_score(ticker, category_name=selected_category)
            fund_score, fund_details = calculate_fundamental_score(ticker)
            sent_score, sent_details = calculate_sentiment_score(ticker)

            master_score = (tech_score * 0.33) + (fund_score * 0.33) + (sent_score * 0.34)

            if master_score >= 50: bias_label = "🔥 Very Bullish"
            elif master_score >= 15: bias_label = "📈 Bullish"
            elif master_score > -15: bias_label = "⚖️ Neutral"
            elif master_score > -50: bias_label = "📉 Bearish"
            else: bias_label = "❄️ Very Bearish"

            setup_icon = get_setup_icon(tech_score, tech_details)

            scanned_data.append({
                "Setup": setup_icon,
                "Stock": name,
                "Master Score": round(master_score, 1),
                "Bias Status": bias_label,
                "Technicals (33%)": int(tech_score),
                "Fundamentals (33%)": int(fund_score),
                "Sentiment (34%)": int(sent_score),
            })

            breakdown_data[name] = {
                "Ticker": ticker,
                "Technicals": tech_details,
                "Fundamentals": fund_details,
                "Sentiment": sent_details,
            }

            my_bar.progress((i + 1) / total_stocks)
            # 0.8s rate-limit buffer to prevent Yahoo WAF blocks
            time.sleep(0.8)

        my_bar.empty()

        st.session_state.scanned_data = scanned_data
        st.session_state.breakdown_data = breakdown_data
        st.session_state.last_scanned_category = selected_category

# ============================================================
# DATAFRAME RENDERING
# ============================================================
df = pd.DataFrame(st.session_state.scanned_data)
if not df.empty:
    df = df.sort_values(by="Master Score", ascending=False).reset_index(drop=True)
breakdown_data = st.session_state.breakdown_data

def color_scores(val):
    if isinstance(val, (int, float)):
        if val > 0:
            return 'color: #00FF00; font-weight: bold;'
        elif val < 0:
            return 'color: #FF4136; font-weight: bold;'
        else:
            return 'color: gray;'
    return ''

score_cols = ["Master Score", "Technicals (33%)", "Fundamentals (33%)", "Sentiment (34%)"]

if not df.empty:
    styled_df = (
        df.style
        .map(color_scores, subset=score_cols)
        .format("{:.1f}", subset=["Master Score"])
    )

    event = st.dataframe(
        styled_df,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row"
    )
    
    selected_rows = event.selection.rows
else:
    selected_rows = []

# ============================================================
# THE DRILL-DOWN BREAKDOWN UI
# ============================================================
if selected_rows:
    selected_idx = selected_rows[0]
    selected_stock = df.iloc[selected_idx]["Stock"]
    details = breakdown_data[selected_stock]
    selected_ticker = details["Ticker"]

    st.divider()
    
    # --- WATCHLIST TOGGLE UI ---
    col_title, col_button = st.columns([3, 1])
    with col_title:
        st.subheader(f"🔍 Deep Dive: {selected_stock}")
        
    with col_button:
        is_saved = selected_stock in st.session_state.custom_watchlist
        # Buttons are switched off if the watchlist could not be loaded,
        # so we can never overwrite your real list with an empty one.
        watchlist_blocked = st.session_state.watchlist_status == "error"
        
        if is_saved:
            if st.button("❌ Remove from Watchlist", use_container_width=True, disabled=watchlist_blocked):
                def _remove(wl):
                    wl.pop(selected_stock, None)

                ok, msg = update_watchlist(_remove)
                if ok:
                    # Force a re-scan if they remove a stock while actively viewing the watchlist
                    if selected_category == "⭐ Custom Watchlist":
                        st.session_state.last_scanned_category = None
                    st.rerun()
                else:
                    st.error(f"❌ Not removed. {msg}")
        else:
            if st.button("⭐ Add to Watchlist", type="primary", use_container_width=True, disabled=watchlist_blocked):
                def _add(wl):
                    wl[selected_stock] = selected_ticker

                ok, msg = update_watchlist(_add)
                if ok:
                    st.rerun()
                else:
                    st.error(f"❌ Not added. {msg}")

    # --- Earnings countdown ---
    countdown = get_earnings_countdown(selected_ticker)
    if countdown and 0 <= countdown["days_until"] <= 7:
        date_str = countdown["date"].strftime("%b %d, %Y")
        if countdown["days_until"] == 0:
            st.warning(f"📅 **Earnings report today** ({date_str}) — expect elevated volatility.")
        else:
            st.warning(f"📅 **Earnings in {countdown['days_until']} day(s)** ({date_str}) — expect elevated volatility.")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📈 Technical Analysis", expanded=True):
            t_cols = st.columns(3)
            idx = 0
            for key, val in details["Technicals"].items():
                t_cols[idx % 3].metric(label=key, value=val)
                idx += 1

        with st.expander("🏦 Fundamentals"):
            f_cols = st.columns(2)
            idx = 0
            for key, val in details["Fundamentals"].items():
                f_cols[idx % 2].metric(label=key, value=val)
                idx += 1

    with col2:
        with st.expander("🧠 Sentiment", expanded=True):
            s_cols = st.columns(2)
            idx = 0
            for key, val in details["Sentiment"].items():
                s_cols[idx % 2].metric(label=key, value=val)
                idx += 1
