import streamlit as st
import pandas as pd
import time
import json
import os
import requests

from universe import STOCK_UNIVERSES
from technical import calculate_technical_score
from fundamentals import calculate_fundamental_columns, get_earnings_countdown
from sentiment import calculate_options_column, calculate_catalyst_column
from yf_session import DataFetchError, seconds_until_unblocked

# ============================================================
# ⚙️ SCORING SETTINGS — THE EASY-TO-EDIT PART
#
# Every column is first scored on its own "raw" scale, then scaled down
# to a smaller range. To change how much a column can add or subtract,
# change its "scaled_max" number. Nothing else needs to change.
#
#   raw_max    = the biggest raw score the column can produce (do not change)
#   scaled_max = the most points the column can add or subtract in the final sum
#
# Master Score = simple SUM of all the scaled column scores.
# Columns with missing data show a dash "–" and add nothing to the sum.
# ============================================================
SCORE_COLUMNS = {
    "Technicals":        {"raw_max": 100, "scaled_max": 4},
    "Options":           {"raw_max": 100, "scaled_max": 2},
    "Catalyst Action":   {"raw_max": 100, "scaled_max": 2},
    "Valuation":         {"raw_max": 200, "scaled_max": 2},   # P/E part + EV/EBITDA part
    "Growth":            {"raw_max": 200, "scaled_max": 2},   # revenue part + earnings part
    "Profitability":     {"raw_max": 200, "scaled_max": 2},   # net margin part + ROE part
    "Financial Health":  {"raw_max": 200, "scaled_max": 2},   # debt/equity part + current ratio part
    "Earnings Surprise": {"raw_max": 100, "scaled_max": 2},
    "EPS Revision":      {"raw_max": 100, "scaled_max": 2},
}

# Bias labels, based on the Master Score (the sum of all columns).
# The Master Score can run from about -20 to +20.
VERY_BULLISH_AT = 10     # Master Score of +10 or more
BULLISH_AT = 6           # Master Score of +6 or more
# Below -6 is Bearish, and -10 or lower is Very Bearish (mirror image).

FUNDAMENTAL_COLUMNS = ["Valuation", "Growth", "Profitability", "Financial Health",
                       "Earnings Surprise", "EPS Revision"]
MARKET_COLUMNS = ["Options", "Catalyst Action"]


def column_label(name):
    """The header shown in the table, e.g. 'Technicals (±4)'."""
    return f"{name} (±{SCORE_COLUMNS[name]['scaled_max']})"


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
    st.markdown("---")
    st.caption("**Reading the score columns**")
    st.caption("**0.0** = a real score of zero (neutral)")
    st.caption("**–** = data is missing (adds nothing to the Master Score)")

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
# HELPERS: RUN EACH COLUMN SAFELY
# Every column ends up as a small "result":
#     {"score": raw number or None, "issue": None/"missing"/"failed", "details": {...}}
# One failure can never crash the scan.
# ============================================================
def failed_result(message):
    return {"score": None, "issue": "failed", "details": {"⚠️ STATUS": message}}


def run_technical(ticker):
    try:
        score, details = calculate_technical_score(ticker, category_name=selected_category)
    except DataFetchError as e:
        return failed_result(f"Data fetch failed. {e}")
    except Exception as e:
        return failed_result(f"Unexpected error: {e}")

    if score is None:
        issue = "missing"                                  # e.g. not enough price history
    elif "⚠️ Data Notes" in details:
        issue = "failed"                                   # a 4H or sector download broke
    elif any(str(key).startswith("⚠️") for key in details.keys()):
        issue = "missing"
    else:
        issue = None
    return {"score": score, "issue": issue, "details": details}


def run_single(function, ticker):
    try:
        return function(ticker)
    except Exception as e:
        return failed_result(f"Unexpected error: {e}")


def run_fundamentals(ticker):
    try:
        return calculate_fundamental_columns(ticker)
    except Exception as e:
        return {name: failed_result(f"Unexpected error: {e}") for name in FUNDAMENTAL_COLUMNS}


def add_scaled_score(column_name, result):
    """Turns the raw score into the scaled score (one decimal). A missing score stays None."""
    raw = result["score"]
    if raw is None:
        result["scaled"] = None
        return

    settings = SCORE_COLUMNS[column_name]
    cap = settings["scaled_max"]
    scaled = raw / settings["raw_max"] * cap
    scaled = max(-cap, min(cap, scaled))
    result["scaled"] = round(scaled, 1) + 0.0    # "+ 0.0" stops a "-0.0" from appearing


def describe_data_status(columns):
    failed, missing = [], []
    for name, result in columns.items():
        tag = name + (" (partial)" if result["score"] is not None else "")
        if result["issue"] == "failed":
            failed.append(tag)
        elif result["issue"] == "missing":
            missing.append(tag)

    parts = []
    if failed:
        parts.append("⚠️ Fetch failed: " + ", ".join(failed))
    if missing:
        parts.append("– No data: " + ", ".join(missing))
    return " | ".join(parts) if parts else "✅ Complete"


def get_bias_label(master_score):
    if master_score is None:
        return "⚠️ No Data"
    if master_score >= VERY_BULLISH_AT: return "🔥 Very Bullish"
    elif master_score >= BULLISH_AT: return "📈 Bullish"
    elif master_score > -BULLISH_AT: return "⚖️ Neutral"
    elif master_score > -VERY_BULLISH_AT: return "📉 Bearish"
    else: return "❄️ Very Bearish"

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

            # No fixed sleep here: yf_session.py spaces out every Yahoo request
            # itself, and skips waiting for data we already have.
            columns = {}
            columns["Technicals"] = run_technical(ticker)
            columns["Options"] = run_single(calculate_options_column, ticker)
            columns["Catalyst Action"] = run_single(calculate_catalyst_column, ticker)
            columns.update(run_fundamentals(ticker))

            # Scale every column, then add them up.
            for column_name in SCORE_COLUMNS:
                add_scaled_score(column_name, columns[column_name])

            # Master Score = simple SUM of the scaled columns.
            # A missing column adds nothing (but shows a dash in the table).
            scaled_values = [columns[c]["scaled"] for c in SCORE_COLUMNS if columns[c]["scaled"] is not None]
            master_score = round(sum(scaled_values), 1) + 0.0 if scaled_values else None

            row = {
                "Setup": get_setup_icon(columns["Technicals"]["score"], columns["Technicals"]["details"]),
                "Stock": name,
                "Master Score": master_score,
                "Bias Status": get_bias_label(master_score),
            }
            for column_name in SCORE_COLUMNS:
                row[column_label(column_name)] = columns[column_name]["scaled"]
            row["Data"] = describe_data_status(columns)
            scanned_data.append(row)

            breakdown_data[name] = {
                "Ticker": ticker,
                "Columns": columns,
            }

        my_bar.empty()

        st.session_state.scanned_data = scanned_data
        st.session_state.breakdown_data = breakdown_data
        st.session_state.last_scanned_category = selected_category

# ============================================================
# DATAFRAME RENDERING
# ============================================================
df = pd.DataFrame(st.session_state.scanned_data)

score_cols = ["Master Score"] + [column_label(c) for c in SCORE_COLUMNS]
column_score_cols = [column_label(c) for c in SCORE_COLUMNS]

if not df.empty:
    # Make sure every score column is a proper number column (missing = blank/NaN)
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(by="Master Score", ascending=False, na_position="last").reset_index(drop=True)
breakdown_data = st.session_state.breakdown_data

# --- Warnings about failed downloads / Yahoo blocking ---
if not df.empty and "Data" in df.columns:
    failed_count = int(df["Data"].str.contains("⚠️ Fetch failed", regex=False).sum())
    missing_count = int(df["Data"].str.contains("– No data", regex=False).sum())

    if failed_count > 0:
        st.warning(
            f"⚠️ {failed_count} stock(s) had a download fail (see the Data column). "
            "Press **Re-scan this universe** in the sidebar to retry. "
            "Only the missing pieces are fetched again."
        )
    if missing_count > 0:
        st.caption(
            "ℹ️ A dash (–) means that data does not exist for that stock (or could not be read). "
            "It adds nothing to the Master Score. A **0.0** is a real score of zero."
        )

    wait_seconds = seconds_until_unblocked()
    if wait_seconds > 0:
        st.error(
            f"🚦 Yahoo is limiting requests right now. "
            f"Wait about {wait_seconds} seconds before re-scanning."
        )

def color_scores(val):
    if isinstance(val, (int, float)) and not pd.isna(val):
        if val > 0:
            return 'color: #00FF00; font-weight: bold;'
        elif val < 0:
            return 'color: #FF4136; font-weight: bold;'
        else:
            return 'color: gray;'
    return ''

if not df.empty:
    styled_df = (
        df.style
        .map(color_scores, subset=score_cols)
        .format("{:.1f}", subset=score_cols, na_rep="–")
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
def show_column_details(column_name, result, per_row=2):
    """Shows one column's heading (with its scaled score) and its detail numbers."""
    scaled = result.get("scaled")
    score_text = "–" if scaled is None else f"{scaled:.1f}"
    max_points = SCORE_COLUMNS[column_name]["scaled_max"]
    st.markdown(f"**{column_name}: {score_text}** (range ±{max_points})")

    cols = st.columns(per_row)
    for idx, (key, val) in enumerate(result["details"].items()):
        cols[idx % per_row].metric(label=key, value=str(val))


if selected_rows:
    selected_idx = selected_rows[0]
    selected_stock = df.iloc[selected_idx]["Stock"]
    details = breakdown_data[selected_stock]
    selected_ticker = details["Ticker"]
    columns = details["Columns"]

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
            tech_result = columns["Technicals"]
            scaled = tech_result.get("scaled")
            st.markdown(f"**Technicals: {'–' if scaled is None else f'{scaled:.1f}'}** (range ±{SCORE_COLUMNS['Technicals']['scaled_max']})")
            t_cols = st.columns(3)
            for idx, (key, val) in enumerate(tech_result["details"].items()):
                t_cols[idx % 3].metric(label=key, value=val)

        with st.expander("🏦 Fundamentals"):
            for column_name in FUNDAMENTAL_COLUMNS:
                show_column_details(column_name, columns[column_name])
                st.markdown("---")

    with col2:
        with st.expander("🧠 Options & Analyst Catalyst", expanded=True):
            for column_name in MARKET_COLUMNS:
                show_column_details(column_name, columns[column_name])
                st.markdown("---")
