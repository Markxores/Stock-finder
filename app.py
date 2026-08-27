import streamlit as st
import pandas as pd
import time
import json
import os

from universe import STOCK_UNIVERSES
from technical import calculate_technical_score
from fundamentals import calculate_fundamental_score, get_earnings_countdown
from sentiment import calculate_sentiment_score

# ============================================================
# WATCHLIST PERSISTENCE (JSON)
# ============================================================
WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_watchlist(data):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ============================================================
# PAGE SETUP & STATE INITIALIZATION
# ============================================================
st.set_page_config(
    page_title="Stock Finder",
    page_icon="📈",
    layout="wide"
)

if "custom_watchlist" not in st.session_state:
    st.session_state.custom_watchlist = load_watchlist()

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
        
        if is_saved:
            if st.button("❌ Remove from Watchlist", use_container_width=True):
                del st.session_state.custom_watchlist[selected_stock]
                save_watchlist(st.session_state.custom_watchlist)
                # Force a re-scan if they remove a stock while actively viewing the watchlist
                if selected_category == "⭐ Custom Watchlist":
                    st.session_state.last_scanned_category = None 
                st.rerun()
        else:
            if st.button("⭐ Add to Watchlist", type="primary", use_container_width=True):
                st.session_state.custom_watchlist[selected_stock] = selected_ticker
                save_watchlist(st.session_state.custom_watchlist)
                st.rerun()

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