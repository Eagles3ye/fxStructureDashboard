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

# --- Pivot Detection ---
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

dealing_high, dealing_low = np.nan, np.nan
d_highs, d_lows, eq_list, zones, event_list = [], [], [], [], []
chart_lines = []

last_maj_a, last_maj_a_idx = np.nan, 0
last_maj_v, last_maj_v_idx = np.nan, 0
maj_a_broken, maj_v_broken = False, False

current_event = "Initializing Structure..."
current_target = "Waiting for Event..."

for i in range(len(df)):
  ph = df["PivotHigh"].iloc[i]
  pl = df["PivotLow"].iloc[i]
  close_val = df["Close"].iloc[i]
  high_val = df["High"].iloc[i]
  low_val = df["Low"].iloc[i]
  curr_time = df.index[i]

  if not np.isnan(ph):
    last_maj_a = ph
    last_maj_a_idx = i
    dealing_high = ph
    maj_a_broken = False

  if not np.isnan(pl):
    last_maj_v = pl
    last_maj_v_idx = i
    dealing_low = pl
    maj_v_broken = False

  is_sweep_high = False
  is_sweep_low = False
  if i >= LOOKBACK:
    prev_high = df["High"].iloc[i - LOOKBACK]
    prev_low = df["Low"].iloc[i - LOOKBACK]
    is_sweep_high = (high_val > prev_high) and (close_val < prev_high)
    is_sweep_low = (low_val < prev_low) and (close_val > prev_low)

  if is_sweep_high:
    current_event = "LIQUIDITY SWEEP HIGH (Wick)"
    chart_lines.append({
        "x0": df.index[i - LOOKBACK],
        "x1": curr_time,
        "y": df["High"].iloc[i - LOOKBACK],
        "color": BEAR_COLOR,
        "width": 1,
        "dash": "dot",
    })
  elif is_sweep_low:
    current_event = "LIQUIDITY SWEEP LOW (Wick)"
    chart_lines.append({
        "x0": df.index[i - LOOKBACK],
        "x1": curr_time,
        "y": df["Low"].iloc[i - LOOKBACK],
        "color": BULL_COLOR,
        "width": 1,
        "dash": "dot",
    })

  eq = (
      (dealing_high + dealing_low) / 2
      if not (np.isnan(dealing_high) or np.isnan(dealing_low))
      else np.nan
  )
  zone = (
      "N/A"
      if np.isnan(eq)
      else ("PREMIUM (Above 50% EQ)" if close_val > eq else "DISCOUNT (Below 50% EQ)")
  )

  d_highs.append(dealing_high)
  d_lows.append(dealing_low)
  eq_list.append(eq)
  zones.append(zone)
  event_list.append(current_event)

df["DealingHigh"] = d_highs
df["DealingLow"] = d_lows
df["EQ"] = eq_list
df["Zone"] = zones
df["Event"] = event_list

curr_dh = df["DealingHigh"].iloc[-1]
curr_dl = df["DealingLow"].iloc[-1]
curr_eq = df["EQ"].iloc[-1]
curr_zone = df["Zone"].iloc[-1]

# --- UI Header ---
st.title("📈 ICT Structure & Liquidity Engine")
st.markdown(f"**Asset:** `{selected_asset_name}` | **Timeframe:** `{timeframe.upper()}`")

# Metric Summary Cards
col1, col2, col3 = st.columns(3)
col1.metric("Dealing High", f"{curr_dh:.4f}" if not np.isnan(curr_dh) else "N/A")
col2.metric("Equilibrium (50% EQ)", f"{curr_eq:.4f}" if not np.isnan(curr_eq) else "N/A")
col3.metric("Dealing Low", f"{curr_dl:.4f}" if not np.isnan(curr_dl) else "N/A")

# --- Interactive Plotly Chart ---
fig = go.Figure()

if chart_style == "Candlestick":
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
else:
  fig.add_trace(
      go.Scatter(
          x=df.index,
          y=df["Close"],
          mode="lines",
          name="Close Price",
          line=dict(color="#29b6f6", width=2),
      )
  )

fig.add_trace(
    go.Scatter(
        x=df.index[df["PivotHigh"].notna()],
        y=df["PivotHigh"].dropna(),
        mode="markers",
        name="Major Pivot High",
        marker=dict(size=9, color=BEAR_COLOR, symbol="circle"),
    )
)
fig.add_trace(
    go.Scatter(
        x=df.index[df["PivotLow"].notna()],
        y=df["PivotLow"].dropna(),
        mode="markers",
        name="Major Pivot Low",
        marker=dict(size=9, color=BULL_COLOR, symbol="circle"),
    )
)

for line in chart_lines:
  fig.add_shape(
      type="line",
      x0=line["x0"],
      y0=line["y"],
      x1=line["x1"],
      y1=line["y"],
      line=dict(color=line["color"], width=line["width"], dash=line["dash"]),
  )

if not np.isnan(curr_eq):
  fig.add_hline(
      y=curr_eq,
      line_dash="dot",
      line_color=EQ_COLOR,
      annotation_text=f"50% EQ: {curr_eq:.4f}",
      annotation_position="bottom right",
  )

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Time",
    yaxis_title="Price",
    height=600,
    xaxis_rangeslider_visible=False,
    dragmode="pan",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

fig.update_xaxes(fixedrange=False)
fig.update_yaxes(fixedrange=False, autorange=True)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToAdd": ["zoom2d", "pan2d", "autoScale2d", "resetScale2d"],
    },
)
