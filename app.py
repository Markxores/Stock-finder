import streamlit as st
import pandas as pd
import yfinance as yf

# 1. IMPORTING THE NECESSARY SCRIPTS
from universe import STOCK_UNIVERSES
from technical import calculate_technical_score
from fundamentals import calculate_fundamental_score
from sentiment import calculate_sentiment_score


# 2. PAGE SETUP
st.set_page_config(
    page_title="Stock Finder",
    page_icon="📈",
    layout="wide"
)

# 3. SIDEBAR CONTROLS
with st.sidebar:
    st.title("⚙️ Stock Finder Engine")
    selected_category = st.selectbox(
        "Select Universe / Sector",
        list(STOCK_UNIVERSES.keys())
    )
    st.info(f"Target universe contains **{len(STOCK_UNIVERSES[selected_category])}** equities.")

# 4. MAIN DASHBOARD VIEW
st.title(f"📊 Stock Finder: {selected_category}")
st.divider()

# 5. LIVE DATA SCANNER (The Loop)
if "last_scanned_category" not in st.session_state or st.session_state.last_scanned_category != selected_category:

    scanned_data = []
    breakdown_data = {}
    tickers_in_category = STOCK_UNIVERSES[selected_category]
    total_stocks = len(tickers_in_category)
    my_bar = st.progress(0, text="Scanning live markets...")

    for i, (name, ticker) in enumerate(tickers_in_category.items()):

        # --- 1. CALLING ENGINES (Unpacking Tuples) ---
        tech_score, tech_details = calculate_technical_score(ticker, category_name=selected_category)
        fund_score, fund_details = calculate_fundamental_score(ticker)
        sent_score, sent_details = calculate_sentiment_score(ticker)

        # --- 2. MASTER WEIGHTING MATH ---
        # Technicals (33%), Fundamentals (33%), Sentiment (34%)
        # No seasonality engine for this build.
        master_score = (tech_score * 0.34) + (fund_score * 0.33) + (sent_score * 0.33)

        # --- 3. BIAS LABELING ---
        if master_score >= 50: bias_label = "🔥 Very Bullish"
        elif master_score >= 15: bias_label = "📈 Bullish"
        elif master_score > -15: bias_label = "⚖️ Neutral"
        elif master_score > -50: bias_label = "📉 Bearish"
        else: bias_label = "❄️ Very Bearish"

        # --- 4. ADD TO TABLE & SAVE BREAKDOWNS ---
        scanned_data.append({
            "Stock": name,
            "Master Score": round(master_score, 1),
            "Bias Status": bias_label,
            "Technicals (33%)": int(tech_score),
            "Fundamentals (33%)": int(fund_score),
            "Sentiment (34%)": int(sent_score),
        })

        breakdown_data[name] = {
            "Technicals": tech_details,
            "Fundamentals": fund_details,
            "Sentiment": sent_details,
        }

        my_bar.progress((i + 1) / total_stocks)

    my_bar.empty()

    st.session_state.scanned_data = scanned_data
    st.session_state.breakdown_data = breakdown_data
    st.session_state.last_scanned_category = selected_category

# Retrieve from session state for display
df = pd.DataFrame(st.session_state.scanned_data).sort_values(by="Master Score", ascending=False).reset_index(drop=True)
breakdown_data = st.session_state.breakdown_data

# --- 6. THE COLOR FORMATTING ENGINE ---
def color_scores(val):
    """Colors positive numbers green and negative numbers red."""
    if isinstance(val, (int, float)):
        if val > 0:
            return 'color: #00FF00; font-weight: bold;'
        elif val < 0:
            return 'color: #FF4136; font-weight: bold;'
        else:
            return 'color: gray;'
    return ''

score_cols = [
    "Master Score",
    "Technicals (33%)",
    "Fundamentals (33%)",
    "Sentiment (34%)",
]

styled_df = (
    df.style
    .map(color_scores, subset=score_cols)
    .format("{:.1f}", subset=["Master Score"])
)

# Display the interactive dataframe with row-selection enabled
event = st.dataframe(
    styled_df,
    width="stretch",
    on_select="rerun",
    selection_mode="single-row"
)

# --- 7. THE DRILL-DOWN BREAKDOWN UI ---
selected_rows = event.selection.rows

if selected_rows:
    selected_idx = selected_rows[0]
    selected_stock = df.iloc[selected_idx]["Stock"]

    details = breakdown_data[selected_stock]

    st.divider()
    st.subheader(f"🔍 Deep Dive: {selected_stock}")

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