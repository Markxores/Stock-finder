import streamlit as st
import pandas as pd
import time

from universe import STOCK_UNIVERSES
from technical import calculate_technical_score
from fundamentals import calculate_fundamental_score, get_earnings_countdown
from sentiment import calculate_sentiment_score

st.set_page_config(
    page_title="Stock Finder",
    page_icon="📈",
    layout="wide"
)

with st.sidebar:
    st.title("⚙️ Stock Finder Engine")
    selected_category = st.selectbox(
        "Select Universe / Sector",
        list(STOCK_UNIVERSES.keys())
    )
    st.info(f"Target universe contains **{len(STOCK_UNIVERSES[selected_category])}** equities.")

    st.markdown("---")
    st.caption("**Setup Icon Legend**")
    st.caption("🚀 Bullish Momentum &nbsp;&nbsp; 🎯 Bullish Pullback")
    st.caption("⚡ Bearish Momentum &nbsp;&nbsp; 🪤 Bearish Pullback")
    st.caption("➖ No Clear 4H Setup")

st.title(f"📊 Stock Finder: {selected_category}")
st.divider()


def get_setup_icon(tech_score, tech_details):
    """
    Combines the technical score's direction with the 4H Trading Mode
    (already computed inside technical.py) to produce a single icon
    showing, at a glance, whether a stock is in a momentum or pullback
    setup — no click required.
    """
    mode = tech_details.get("4H Trading Mode", "Neutral")

    if tech_score > 0 and mode == "Momentum":
        return "🚀"
    elif tech_score > 0 and mode == "Pullback":
        return "🎯"
    elif tech_score < 0 and mode == "Momentum":
        return "⚡"
    elif tech_score < 0 and mode == "Pullback":
        return "🎯"
    else:
        return "➖"


if "last_scanned_category" not in st.session_state or st.session_state.last_scanned_category != selected_category:

    scanned_data = []
    breakdown_data = {}
    tickers_in_category = STOCK_UNIVERSES[selected_category]
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
        time.sleep(0.8)

    my_bar.empty()

    st.session_state.scanned_data = scanned_data
    st.session_state.breakdown_data = breakdown_data
    st.session_state.last_scanned_category = selected_category

df = pd.DataFrame(st.session_state.scanned_data).sort_values(by="Master Score", ascending=False).reset_index(drop=True)
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

if selected_rows:
    selected_idx = selected_rows[0]
    selected_stock = df.iloc[selected_idx]["Stock"]
    details = breakdown_data[selected_stock]
    selected_ticker = details["Ticker"]

    st.divider()
    st.subheader(f"🔍 Deep Dive: {selected_stock}")

    # --- Earnings countdown: informational only, does NOT affect scoring ---
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