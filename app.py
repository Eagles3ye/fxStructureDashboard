import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Optional: Try importing yfinance for live data fetching
try:
  import yfinance as yf

  YFINANCE_AVAILABLE = True
except ImportError:
  YFINANCE_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="ICT Structure & Liquidity Sweep Engine",
    page_icon="📈",
    layout="wide",
)

st.title("🛡️ 6if Structure & Liquidity Sweep Engine [v6] - Streamlit Mirror")
st.markdown(
    "Mirroring TradingView Pine Script v6 ICT concepts: Major Pivots, BOS,"
    " Liquidity Sweeps, Equilibrium (EQ), and Dealing Ranges."
)

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Settings & Inputs")

symbol = st.sidebar.text_input("Ticker Symbol", value="AAPL")
interval = st.sidebar.selectbox(
    "Timeframe", ["1d", "1h", "15m"], index=0
)  # Simplified for python mapping

major_len = st.sidebar.number_input(
    "Major Swing Length", min_value=2, max_value=50, value=3, step=1
)
sweep_bars = st.sidebar.number_input(
    "CRT Sweep Look Bar Window", min_value=1, max_value=50, value=10, step=1
)

st.sidebar.subheader("Chart Visual Toggles")
show_dots = st.sidebar.checkbox("Show All Major Pivot Dots", value=True)
show_lines = st.sidebar.checkbox("Show BOS / Sweep Lines", value=True)
show_eq = st.sidebar.checkbox("Show Equilibrium (EQ) Line", value=True)

st.sidebar.subheader("Color Settings")
bull_color = st.sidebar.color_picker(
    "Bullish / Discount Color", value="#00FF00"
)
bear_color = st.sidebar.color_picker(
    "Bearish / Premium Color", value="#FF0000"
)
eq_color = st.sidebar.color_picker("Equilibrium (EQ) Color", value="#FFFF00")


# --- Data Loading Function ---
@st.cache_data
def load_data(ticker, period="6mo", interval="1d"):
  if YFINANCE_AVAILABLE:
    try:
      df = yf.download(ticker, period=period, interval=interval, progress=False)
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
      df = df.dropna()
      return df
    except Exception as e:
      st.warning(
          f"Could not fetch data from yfinance ({e}). Generating synthetic"
          " data..."
      )

  # Fallback Synthetic Data Generator if yfinance fails or is missing
  np.random.seed(42)
  dates = pd.date_range(end=pd.Timestamp.today(), periods=200, freq="B")
  close = 150 + np.cumsum(np.random.randn(200) * 2)
  high = close + np.random.rand(200) * 3
  low = close - np.random.rand(200) * 3
  open_p = close + (np.random.rand(200) - 0.5) * 2
  df = pd.DataFrame(
      {
          "Open": open_p,
          "High": high,
          "Low": low,
          "Close": close,
          "Volume": np.random.randint(1000, 10000, size=200),
      },
      index=dates,
  )
  return df


df = load_data(symbol, interval=interval)

if df.empty:
  st.error("No data available for the selected symbol/timeframe.")
  st.stop()


# --- Pivot Calculation Logic (Python equivalent of ta.pivothigh / pivotlow) ---
def calculate_pivots(df, length):
  highs = df["High"].values
  lows = df["Low"].values
  n = len(df)

  pivot_highs = [np.nan] * n
  pivot_lows = [np.nan] * n

  for i in range(length, n - length):
    # Check for Pivot High
    is_ph = True
    for j in range(1, length + 1):
      if highs[i] < highs[i - j] or highs[i] < highs[i + j]:
        is_ph = False
        break
    if is_ph:
      pivot_highs[i] = highs[i]

    # Check for Pivot Low
    is_pl = True
    for j in range(1, length + 1):
      if lows[i] > lows[i - j] or lows[i] > lows[i + j]:
        is_pl = False
        break
    if is_pl:
      pivot_lows[i] = lows[i]

  df["PivotHigh"] = pivot_highs
  df["PivotLow"] = pivot_lows
  return df


df = calculate_pivots(df, major_len)

# --- State Tracking & BOS / Sweep Engine ---
last_swing_high = np.nan
last_swing_low = np.nan
swing_h_break = False
swing_l_break = False

last_event = "Initializing Structure..."
event_col = "gray"
next_target = "Waiting for Event..."

events_log = []
lines_to_plot = []

for i in range(len(df)):
  ph = df["PivotHigh"].iloc[i]
  pl = df["PivotLow"].iloc[i]
  h = df["High"].iloc[i]
  l = df["Low"].iloc[i]
  c = df["Close"].iloc[i]
  dt = df.index[i]

  if not np.isnan(ph):
    last_swing_high = ph
    swing_h_break = False
    last_swing_h_time = dt

  if not np.isnan(pl):
    last_swing_low = pl
    swing_l_break = False
    last_swing_l_time = dt

  # Check High Break / Sweep
  if not np.isnan(last_swing_high) and not swing_h_break and h > last_swing_high:
    if c > last_swing_high:
      swing_h_break = True
      last_event = "BOS Bullish (Body Close)"
      event_col = bull_color
      next_target = "Target: Next External High / FVG"
      if show_lines:
        lines_to_plot.append({
            "type": "bull_bos",
            "x0": last_swing_h_time,
            "x1": dt,
            "y0": last_swing_high,
            "y1": last_swing_high,
        })
    else:
      last_event = "LIQUIDITY SWEEP HIGH (Wick)"
      event_col = bear_color
      next_target = "Target: EQ (50%) -> Range Low"
      if show_lines:
        lines_to_plot.append({
            "type": "bear_sweep",
            "x0": last_swing_h_time,
            "x1": dt,
            "y0": last_swing_high,
            "y1": last_swing_high,
        })

  # Check Low Break / Sweep
  if not np.isnan(last_swing_low) and not swing_l_break and l < last_swing_low:
    if c < last_swing_low:
      swing_l_break = True
      last_event = "BOS Bearish (Body Close)"
      event_col = bear_color
      next_target = "Target: Next External Low / FVG"
      if show_lines:
        lines_to_plot.append({
            "type": "bear_bos",
            "x0": last_swing_l_time,
            "x1": dt,
            "y0": last_swing_low,
            "y1": last_swing_low,
        })
    else:
      last_event = "LIQUIDITY SWEEP LOW (Wick)"
      event_col = bull_color
      next_target = "Target: EQ (50%) -> Range High"
      if show_lines:
        lines_to_plot.append({
            "type": "bull_sweep",
            "x0": last_swing_l_time,
            "x1": dt,
            "y0": last_swing_low,
            "y1": last_swing_low,
        })

# Dealing Range & EQ
dealing_high = last_swing_high
dealing_low = last_swing_low
eq = (
    (dealing_high + dealing_low) / 2
    if not np.isnan(dealing_high) and not np.isnan(dealing_low)
    else np.nan
)
current_close = df["Close"].iloc[-1]
active_zone = (
    "N/A"
    if np.isnan(eq)
    else ("PREMIUM" if current_close > eq else "DISCOUNT")
)
zone_col = (
    "gray"
    if np.isnan(eq)
    else (bear_color if current_close > eq else bull_color)
)

# --- Dashboard Metrics (Matching Pine Script Table) ---
st.subheader("📊 ICT Dashboard Engine")

col1, col2, col3 = st.columns(3)
with col1:
  st.metric(
      label="Dealing High",
      value=f"{dealing_high:.2f}" if not np.isnan(dealing_high) else "N/A",
  )
  st.metric(
      label="Current Pricing Zone",
      value=active_zone,
      delta="Premium / Discount",
  )
with col2:
  st.metric(
      label="Equilibrium (50% EQ)",
      value=f"{eq:.2f}" if not np.isnan(eq) else "N/A",
  )
  st.metric(label="Last Structure Event", value=last_event)
with col3:
  st.metric(
      label="Dealing Low",
      value=f"{dealing_low:.2f}" if not np.isnan(dealing_low) else "N/A",
  )
  st.metric(label="Dynamic Target Objective", value=next_target)

st.markdown("---")

# --- Plotly Chart Visualization ---
st.subheader("📈 Chart Visuals & Structure Mirror")

fig = go.Figure()

# Candlestick chart
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

# Show Major Pivot Dots
if show_dots:
  ph_filtered = df[df["PivotHigh"].notna()]
  pl_filtered = df[df["PivotLow"].notna()]

  fig.add_trace(
      go.Scatter(
          x=ph_filtered.index,
          y=ph_filtered["PivotHigh"],
          mode="markers",
          marker=dict(size=10, color=bear_color),
          name="Major High Pivot",
      )
  )

  fig.add_trace(
      go.Scatter(
          x=pl_filtered.index,
          y=pl_filtered["PivotLow"],
          mode="markers",
          marker=dict(size=10, color=bull_color),
          name="Major Low Pivot",
      )
  )

# Plot Equilibrium Line
if show_eq and not np.isnan(eq):
  fig.add_shape(
      type="line",
      x0=df.index[0],
      x1=df.index[-1],
      y0=eq,
      y1=eq,
      line=dict(color=eq_color, width=1.5, dash="dot"),
  )
  fig.add_annotation(
      x=df.index[-1],
      y=eq,
      text="EQ (50%)",
      showarrow=False,
      font=dict(color=eq_color),
      xanchor="left",
  )

# Plot BOS / Sweep Lines
if show_lines:
  for l_info in lines_to_plot:
    line_color = (
        bull_color
        if "bull" in l_info["type"]
        else (bear_color if "bear" in l_info["type"] else "gray")
    )
    line_dash = "dash" if "sweep" in l_info["type"] else "solid"
    fig.add_shape(
        type="line",
        x0=l_info["x0"],
        x1=l_info["x1"],
        y0=l_info["y0"],
        y1=l_info["y1"],
        line=dict(color=line_color, width=1.5, dash=line_dash),
    )

fig.update_layout(
    title=f"{symbol} - ICT Structure & Liquidity Engine",
    xaxis_title="Time",
    yaxis_title="Price",
    template="plotly_dark",
    height=650,
    xaxis_rangeslider_visible=False,
)

st.plotly_chart(fig, use_container_width=True)
