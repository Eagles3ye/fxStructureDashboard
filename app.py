import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# --- Page Configuration ---
st.set_page_config(
    page_title="ICT Structure & Liquidity Engine",
    page_icon="📈",
    layout="wide",
)

# --- Sidebar Inputs ---
st.sidebar.header("Engine Settings")

asset_options = {
    "Gold (GC=F)": "GC=F",
    "Silver (SI=F)": "SI=F",
    "USD/CAD": "USDCAD=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "S&P 500 ETF (SPY)": "SPY",
}

selected_asset_name = st.sidebar.selectbox(
    "Select Asset", list(asset_options.keys()), index=0
)
ticker = asset_options[selected_asset_name]

timeframe = st.sidebar.selectbox(
    "Dashboard Timeframe Context", ["1M", "1W", "1D", "4h", "1h", "15m"], index=3
)
chart_style = st.sidebar.radio(
    "Chart Visual Type", ["Candlestick", "Line Chart"], index=0
)
major_len = st.sidebar.slider("Major Swing Length", 2, 10, 3)
LOOKBACK = 1

BULL_COLOR = "#26a69a"
BEAR_COLOR = "#ef5350"
EQ_COLOR = "#ffeb3b"
DR_HIGH_COLOR = "#ab47bc"
DR_LOW_COLOR = "#42a5f5"


# --- Safe Data Fetching ---
@st.cache_data(ttl=60)
def load_data(symbol, tf):
  try:
    df = yf.download(symbol, period="60d", interval="60m", progress=False, threads=False)
    if df.empty:
      df = yf.download(symbol, period="1y", interval="1d", progress=False, threads=False)
    if df.empty:
      return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df
  except Exception:
    return pd.DataFrame()


df = load_data(ticker, timeframe)

if df.empty:
  st.warning(f"⚠️ Could not load data for {selected_asset_name}. Please try another asset.")
  st.stop()

# --- UI Header ---
st.title("📈 ICT Structure & Liquidity Engine")
st.markdown(f"**Asset:** `{selected_asset_name}` | **Timeframe:** `{timeframe.upper()}`")

# --- Basic Chart Rendering ---
fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
    )
)
fig.update_layout(
    template="plotly_dark",
    height=600,
    xaxis_rangeslider_visible=False,
    dragmode="pan",
)
st.plotly_chart(df.empty is False, use_container_width=True) # Safe placeholder render
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
