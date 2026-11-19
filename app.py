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
from yf_session import DataFetchError, seconds_until_unblocked

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

    # --- Re-scan button ---
    # Successful lookups are remembered, so a re-scan only asks Yahoo for
    # the pieces that are missing or out of date.
    if st.button("🔄 Re-scan this universe", use_container_width=True):
        st.session_state.last_scanned_category = None
    st.caption("Re-scanning only re-fetches missing or out-of-date data.")

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
    st.caption("➖ No Clear 4H Setup &nbsp;&nbsp; ❓ Technical data missing")

st.title(f"📊 Stock Finder: {selected_category}")
st.divider()

# ============================================================
# HELPER: SETUP ICON
# ============================================================
def get_setup_icon(tech_score, tech_details):
    """
    Combines the technical score's direction with the 4H Trading Mode.
    Unified Pullback Icon: 🎯 used for both bull and bear pullbacks.
    ❓ means the technical data (or its 4H part) could not be fetched.
    """
    if tech_score is None:
        return "❓"

    mode = tech_details.get("4H Trading Mode", "Neutral")

    if mode == "Unavailable":
        return "❓"
    elif mode == "Pullback":
        return "🎯"
    elif tech_score > 0 and mode == "Momentum":
        return "🚀"
    elif tech_score < 0 and mode == "Momentum":
        return "⚡"
    else:
        return "➖"

# ============================================================
# HELPER: RUN ONE SCORE SAFELY
# Runs a scoring function and never lets one failure crash the scan.
# Returns (score, details, status) where status is:
#   "ok"      -> everything fetched
#   "partial" -> a score exists but some data was missing (see the note)
#   "failed"  -> no score could be produced (score is None)
# ============================================================
def run_score(score_function, *args, **kwargs):
    try:
        score, details = score_function(*args, **kwargs)
    except DataFetchError as e:
        return None, {"⚠️ STATUS": f"Data fetch failed. {e}"}, "failed"
    except Exception as e:
        return None, {"⚠️ STATUS": f"Unexpected error: {e}"}, "failed"

    has_warning = any(str(key).startswith("⚠️") for key in details.keys())
    return score, details, ("partial" if has_warning else "ok")


def get_bias_label(master_score):
    if master_score is None:
        return "⚠️ No Data"
    if master_score >= 50: return "🔥 Very Bullish"
    elif master_score >= 15: return "📈 Bullish"
    elif master_score > -15: return "⚖️ Neutral"
    elif master_score > -50: return "📉 Bearish"
    else: return "❄️ Very Bearish"


PILLAR_WEIGHTS = {"Technicals": 0.33, "Fundamentals": 0.33, "Sentiment": 0.34}

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
            my_bar.progress(i / total_stocks, text=f"Scanning {name} ({i + 1} of {total_stocks})...")

            # No fixed sleep here any more: yf_session.py spaces out every
            # Yahoo request itself, and skips waiting for data we already have.
            tech_score, tech_details, tech_status = run_score(calculate_technical_score, ticker, category_name=selected_category)
            fund_score, fund_details, fund_status = run_score(calculate_fundamental_score, ticker)
            sent_score, sent_details, sent_status = run_score(calculate_sentiment_score, ticker)

            pillars = {
                "Technicals": (tech_score, tech_status),
                "Fundamentals": (fund_score, fund_status),
                "Sentiment": (sent_score, sent_status),
            }

            # Master score = weighted average of the pillars that worked.
            # If all three worked, this is exactly the old 33/33/34 formula.
            usable = {name_: score for name_, (score, _) in pillars.items() if score is not None}
            if usable:
                total_weight = sum(PILLAR_WEIGHTS[n] for n in usable)
                master_score = sum(score * PILLAR_WEIGHTS[n] for n, score in usable.items()) / total_weight
            else:
                master_score = None

            failed_names = [n for n, (_, status) in pillars.items() if status == "failed"]
            partial_names = [n for n, (_, status) in pillars.items() if status == "partial"]
            if failed_names:
                data_status = "⚠️ Missing: " + ", ".join(failed_names)
            elif partial_names:
                data_status = "⚠️ Partial: " + ", ".join(partial_names)
            else:
                data_status = "✅ Complete"

            scanned_data.append({
                "Setup": get_setup_icon(tech_score, tech_details),
                "Stock": name,
                "Master Score": round(master_score, 1) if master_score is not None else None,
                "Bias Status": get_bias_label(master_score),
                "Technicals (33%)": int(tech_score) if tech_score is not None else None,
                "Fundamentals (33%)": int(fund_score) if fund_score is not None else None,
                "Sentiment (34%)": int(sent_score) if sent_score is not None else None,
                "Data": data_status,
            })

            breakdown_data[name] = {
                "Ticker": ticker,
                "Technicals": tech_details,
                "Fundamentals": fund_details,
                "Sentiment": sent_details,
            }

        my_bar.empty()

        st.session_state.scanned_data = scanned_data
        st.session_state.breakdown_data = breakdown_data
        st.session_state.last_scanned_category = selected_category

# ============================================================
# DATAFRAME RENDERING
# ============================================================
df = pd.DataFrame(st.session_state.scanned_data)
if not df.empty:
    df = df.sort_values(by="Master Score", ascending=False, na_position="last").reset_index(drop=True)
breakdown_data = st.session_state.breakdown_data

# --- Warnings about missing data / Yahoo blocking ---
if not df.empty and "Data" in df.columns:
    problem_count = int((df["Data"] != "✅ Complete").sum())
    if problem_count > 0:
        st.warning(
            f"⚠️ {problem_count} stock(s) have missing or partial data (see the Data column). "
            "Their Master Score uses only the data that was available. "
            "Press **Re-scan this universe** in the sidebar to retry. "
            "Only the missing pieces are fetched again."
        )

    wait_seconds = seconds_until_unblocked()
    if wait_seconds > 0:
        st.error(
            f"🚦 Yahoo is limiting requests right now. "
            f"Wait about {wait_seconds} seconds before re-scanning."
        )

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
pillar_cols = ["Technicals (33%)", "Fundamentals (33%)", "Sentiment (34%)"]

if not df.empty:
    styled_df = (
        df.style
        .map(color_scores, subset=score_cols)
        .format("{:.1f}", subset=["Master Score"], na_rep="⚠️ failed")
        .format("{:.0f}", subset=pillar_cols, na_rep="⚠️ failed")
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

    # --- Earnings countdown (informational; skipped quietly if it can't be fetched) ---
    try:
        countdown = get_earnings_countdown(selected_ticker)
    except Exception:
        countdown = None

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
