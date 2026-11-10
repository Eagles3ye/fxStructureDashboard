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
ticker = st.sidebar.text_input("Asset Ticker", value="USDCAD=X")
timeframe = st.sidebar.selectbox(
    "Chart Timeframe", ["1h", "4h", "1d"], index=0
)
major_len = st.sidebar.slider("Major Swing Length", 2, 10, 3)
lookback = st.sidebar.slider("Sweep Lookback Bars", 1, 20, 5)

# Color configurations
BULL_COLOR = "#26a69a"  # Green / Discount
BEAR_COLOR = "#ef5350"  # Red / Premium
EQ_COLOR = "#ffeb3b"  # Yellow
DR_HIGH_COLOR = "#ab47bc"  # Purple
DR_LOW_COLOR = "#42a5f5"  # Blue


# --- Data Fetching & Resampling ---
@st.cache_data(ttl=300)
def load_data(symbol, tf):
  interval_map = {"1h": "60m", "4h": "240m", "1d": "1d"}
  yf_interval = interval_map.get(tf, "60m")
  period = "60d" if yf_interval in ["60m", "240m"] else "1y"

  df = yf.download(symbol, period=period, interval=yf_interval, progress=False)
  if df.empty:
    return df

  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
  df.index = pd.to_datetime(df.index)
  return df


df = load_data(ticker, timeframe)

if df.empty:
  st.error(
      f"Could not retrieve data for {ticker}. Please check the ticker symbol."
  )
  st.stop()


# --- Pivot Detection Function ---
def compute_pivots(data, length):
  highs = data["High"].values
  lows = data["Low"].values
  n = len(data)

  pivot_highs = np.full(n, np.nan)
  pivot_lows = np.full(n, np.nan)

  for i in range(length, n - length):
    is_hi = True
    for j in range(1, length + 1):
      if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
        is_hi = False
        break
    if is_hi:
      pivot_highs[i] = highs[i]

    is_lo = True
    for j in range(1, length + 1):
      if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
        is_lo = False
        break
    if is_lo:
      pivot_lows[i] = lows[i]

  data["PivotHigh"] = pivot_highs
  data["PivotLow"] = pivot_lows
  return data


df = compute_pivots(df, major_len)

# --- State Tracking & Dealing Range Logic ---
dealing_high = np.nan
dealing_low = np.nan

d_highs = []
d_lows = []
eq_list = []
zones = []

last_event = "Initializing Structure..."
next_target = "Waiting for Event..."

for i in range(len(df)):
  ph = df["PivotHigh"].iloc[i]
  pl = df["PivotLow"].iloc[i]
  close_val = df["Close"].iloc[i]

  if not np.isnan(ph):
    dealing_high = ph

  if not np.isnan(pl):
    dealing_low = pl

  eq = (
      (dealing_high + dealing_low) / 2
      if not (np.isnan(dealing_high) or np.isnan(dealing_low))
      else np.nan
  )

  if np.isnan(eq):
    zone = "N/A"
  elif close_val > eq:
    zone = "PREMIUM (Above 50% EQ)"
  else:
    zone = "DISCOUNT (Below 50% EQ)"

  d_highs.append(dealing_high)
  d_lows.append(dealing_low)
  eq_list.append(eq)
  zones.append(zone)

df["DealingHigh"] = d_highs
df["DealingLow"] = d_lows
df["EQ"] = eq_list
df["Zone"] = zones

curr_dh = df["DealingHigh"].iloc[-1]
curr_dl = df["DealingLow"].iloc[-1]
curr_eq = df["EQ"].iloc[-1]
curr_zone = df["Zone"].iloc[-1]

# --- Macro Alignment Context ---
df_daily = (
    yf.download(ticker, period="6mo", interval="1d", progress=False)
    if timeframe != "1d"
    else df
)
if isinstance(df_daily.columns, pd.MultiIndex):
  df_daily.columns = df_daily.columns.get_level_values(0)

if not df_daily.empty:
  df_daily["EMA50"] = df_daily["Close"].ewm(span=50).mean()
  daily_bias = (
      "BULL"
      if df_daily["Close"].iloc[-1] > df_daily["EMA50"].iloc[-1]
      else "BEAR"
  )
else:
  daily_bias = "NEUTRAL"

macro_title = "Macro (Daily Trend / Zone)"
macro_text = f"D: {daily_bias} | Zone: {curr_zone.split(' ')[0]}"

# --- Dashboard UI Layout ---
st.title("📈 ICT Structure & Liquidity Engine")
st.markdown(
    f"**Asset:** `{ticker}` | **Timeframe:** `{timeframe.upper()}` | **Engine Status:** `Active`"
)

# Metric Summary Cards
col1, col2, col3 = st.columns(3)
col1.metric(
    "Dealing High", f"{curr_dh:.4f}" if not np.isnan(curr_dh) else "N/A"
)
col2.metric(
    "Equilibrium (50% EQ)",
    f"{curr_eq:.4f}" if not np.isnan(curr_eq) else "N/A",
)
col3.metric(
    "Dealing Low", f"{curr_dl:.4f}" if not np.isnan(curr_dl) else "N/A"
)

# Main Dashboard Table
st.subheader("📊 Engine Dashboard Summary")
dashboard_data = {
    "Metric / Value": [
        "Dealing High",
        "Equilibrium (50% EQ)",
        "Dealing Low",
        "Current Pricing Zone",
        "Last Structure Event",
        "Dynamic Target Objective",
        macro_title,
    ],
    "Status": [
        f"{curr_dh:.4f}" if not np.isnan(curr_dh) else "N/A",
        f"{curr_eq:.4f}" if not np.isnan(curr_eq) else "N/A",
        f"{curr_dl:.4f}" if not np.isnan(curr_dl) else "N/A",
        curr_zone,
        last_event,
        next_target,
        macro_text,
    ],
}
st.table(pd.DataFrame(dashboard_data))

# --- Interactive Plotly Chart ---
st.subheader("🕯️ Price Chart & Structure Overlay")

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

if not np.isnan(curr_eq):
  fig.add_hline(
      y=curr_eq,
      line_dash="dot",
      line_color=EQ_COLOR,
      annotation_text="50% EQ",
      annotation_position="bottom right",
  )
if not np.isnan(curr_dh):
  fig.add_hline(
      y=curr_dh,
      line_dash="dash",
      line_color=BEAR_COLOR,
      annotation_text="Dealing High",
  )
if not np.isnan(curr_dl):
  fig.add_hline(
      y=curr_dl,
      line_dash="dash",
      line_color=BULL_COLOR,
      annotation_text="Dealing Low",
  )

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Time",
    yaxis_title="Price",
    height=600,
    xaxis_rangeslider_visible=False,
)

st.plotly_chart(fig, use_container_width=True)
