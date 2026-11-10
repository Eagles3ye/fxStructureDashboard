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
st.sidebar.markdown("🟢 **Real-Time Stream Active** (1s Tick Engine)")

# Predefined Asset Dictionary
asset_options = {
    "USD/CAD": "USDCAD=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "Gold (GC=F)": "GC=F",
    "Silver (SI=F)": "SI=F",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "S&P 500 ETF (SPY)": "SPY",
}

selected_asset_name = st.sidebar.selectbox(
    "Select Asset", list(asset_options.keys()), index=0
)
ticker = asset_options[selected_asset_name]

# Timeframe Selection
timeframe = st.sidebar.selectbox(
    "Dashboard Timeframe Context", ["1M", "1W", "1D", "4h", "1h", "15m"], index=2
)

# Visual Chart Style Toggle
chart_style = st.sidebar.radio(
    "Chart Visual Type", ["Candlestick", "Line Chart"], index=0
)

major_len = st.sidebar.slider("Major Swing Length", 2, 10, 3)
lookback = st.sidebar.slider("Sweep Lookback Bars", 1, 20, 1)

# Color configurations
BULL_COLOR = "#26a69a"  # Green / Discount / Bullish
BEAR_COLOR = "#ef5350"  # Red / Premium / Bearish
EQ_COLOR = "#ffeb3b"  # Yellow
DR_HIGH_COLOR = "#ab47bc"  # Purple (Dealing High)
DR_LOW_COLOR = "#42a5f5"  # Blue (Dealing Low)


# --- Data Fetching & Multi-Timeframe Resampling ---
@st.cache_data(ttl=5)
def load_and_resample_data(symbol, tf):
  base_period = "max" if tf in ["1W", "1M"] else "2y"

  df = yf.download(
      symbol, period=base_period, interval="60m", progress=False, threads=False
  )
  if df.empty:
    df = yf.download(
        symbol, period="max", interval="1d", progress=False, threads=False
    )

  if df.empty:
    return df

  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
  df.index = pd.to_datetime(df.index)

  rule_map = {
      "15m": "15min",
      "1h": "1h",
      "4h": "4h",
      "1D": "1D",
      "1W": "1W",
      "1M": "1ME",
  }

  rule = rule_map.get(tf, "1h")
  if rule == "15min":
    df_15 = yf.download(
        symbol, period="60d", interval="15m", progress=False, threads=False
    )
    if not df_15.empty:
      if isinstance(df_15.columns, pd.MultiIndex):
        df_15.columns = df_15.columns.get_level_values(0)
      return df_15[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df

  if rule != "1h":
    df_resampled = (
        df.resample(rule)
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )
    return df_resampled

  return df


df = load_and_resample_data(ticker, timeframe)

if df.empty:
  st.error(
      f"Could not retrieve data for {selected_asset_name}. Please check the"
      " ticker configuration."
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

# --- Dealing Range, Structure & Structural Trend Engine ---
dealing_high = np.nan
dealing_low = np.nan

d_highs = []
d_lows = []
eq_list = []
zones = []
event_list = []
trend_list = []

chart_lines = []

last_maj_a = np.nan
last_maj_a_idx = 0
last_maj_v = np.nan
last_maj_v_idx = 0

maj_a_broken = False
maj_v_broken = False

current_event = "Initializing Structure..."
current_target = "Waiting for Event..."
current_structural_trend = "BULLISH"

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

  if not np.isnan(last_maj_a) and not maj_a_broken and high_val > last_maj_a:
    start_time = df.index[last_maj_a_idx]
    if close_val > last_maj_a:
      maj_a_broken = True
      current_event = "BOS Bullish (Body Close)"
      current_target = "Target: Next External High / FVG"
      current_structural_trend = "BULLISH"
      chart_lines.append({
          "x0": start_time,
          "x1": curr_time,
          "y": last_maj_a,
          "color": BULL_COLOR,
          "width": 2,
          "dash": "solid",
      })
    else:
      current_event = "LIQUIDITY SWEEP HIGH (Wick)"
      current_target = "Target: EQ (50%) -> Range Low"
      chart_lines.append({
          "x0": start_time,
          "x1": curr_time,
          "y": last_maj_a,
          "color": BEAR_COLOR,
          "width": 1,
          "dash": "dot",
      })

  if not np.isnan(last_maj_v) and not maj_v_broken and low_val < last_maj_v:
    start_time = df.index[last_maj_v_idx]
    if close_val < last_maj_v:
      maj_v_broken = True
      current_event = "BOS Bearish (Body Close)"
      current_target = "Target: Next External Low / FVG"
      chart_lines.append({
          "x0": start_time,
          "x1": curr_time,
          "y": last_maj_v,
          "color": BEAR_COLOR,
          "width": 2,
          "dash": "solid",
      })
    else:
      current_event = "LIQUIDITY SWEEP LOW (Wick)"
      current_target = "Target: EQ (50%) -> Range High"
      chart_lines.append({
          "x0": start_time,
          "x1": curr_time,
          "y": last_maj_v,
          "color": BULL_COLOR,
          "width": 1,
          "dash": "dot",
      })

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
  event_list.append(current_event)
  trend_list.append(current_structural_trend)

df["DealingHigh"] = d_highs
df["DealingLow"] = d_lows
df["EQ"] = eq_list
df["Zone"] = zones
df["Event"] = event_list
df["StructuralTrend"] = trend_list

curr_dh = df["DealingHigh"].iloc[-1]
curr_dl = df["DealingLow"].iloc[-1]
curr_eq = df["EQ"].iloc[-1]
curr_zone = df["Zone"].iloc[-1]
curr_event = df["Event"].iloc[-1]
active_structural_trend = df["StructuralTrend"].iloc[-1]
latest_price = df["Close"].iloc[-1]


# --- Structure-Driven Conditional CRT & Sweep Engine ---
def compute_structural_crt_sweep(data, lb_bars, trend):
  if len(data) < lb_bars + 1:
    return "H: N/A | L: N/A", "Sweep: N/A", None, None

  ref_slice = data.iloc[-(lb_bars + 1) : -1]
  if ref_slice.empty:
    ref_slice = data.iloc[-2:-1]

  crt_h = ref_slice["High"].max()
  crt_l = ref_slice["Low"].min()

  curr_bar = data.iloc[-1]
  curr_h = curr_bar["High"]
  curr_l = curr_bar["Low"]

  sweep_line_level = None
  sweep_line_color = None

  if trend == "BULLISH":
    swept = curr_l < crt_l
    sweep_str = "Low Swept (Bullish Setup)" if swept else "Waiting for Low Sweep"
    if swept:
      sweep_line_level = crt_l
      sweep_line_color = BULL_COLOR
  else:
    swept = curr_h > crt_h
    sweep_str = (
        "High Swept (Bearish Setup)" if swept else "Waiting for High Sweep"
    )
    if swept:
      sweep_line_level = crt_h
      sweep_line_color = BEAR_COLOR

  crt_val_str = f"H: {crt_h:.4f} | L: {crt_l:.4f}"
  sweep_status_str = f"Sweep: {sweep_str}"

  return crt_val_str, sweep_status_str, sweep_line_level, sweep_line_color


(
    active_crt_val,
    active_sweep_status,
    crt_sweep_level,
    crt_sweep_color,
) = compute_structural_crt_sweep(df, lookback, active_structural_trend)

# --- Macro Alignment Context ---
macro_title = f"Structural Trend Bias ({timeframe.upper()})"
macro_text = f"Trend: {active_structural_trend} | Zone: {curr_zone.split(' ')[0]}"

# --- Dashboard UI Layout (Using Streamlit Native Fragment for 1-Second Live Updates) ---
st.title("📈 ICT Structure & Liquidity Engine (Live 1s)")
st.markdown(
    f"**Asset:** `{selected_asset_name}` | **Timeframe:** `{timeframe.upper()}`"
    f" | **Lookback:** `{lookback}` | **Structure Trend:**"
    f" `{active_structural_trend}`"
)


@st.fragment(run_every=1.0)  # <--- Updates every 1 second without page flicker
def live_ticker_dashboard():
  # Fetch latest live tick value
  live_df = yf.download(
      ticker, period="1d", interval="1m", progress=False, threads=False
  )
  live_price = latest_price
  if not live_df.empty:
    if isinstance(live_df.columns, pd.MultiIndex):
      live_df.columns = live_df.columns.get_level_values(0)
    live_price = live_df["Close"].iloc[-1]

  # Metric Summary Cards
  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Live Price", f"{live_price:.4f}")
  col2.metric(
      "Dealing High", f"{curr_dh:.4f}" if not np.isnan(curr_dh) else "N/A"
  )
  col3.metric(
      "Equilibrium (50% EQ)",
      f"{curr_eq:.4f}" if not np.isnan(curr_eq) else "N/A",
  )
  col4.metric(
      "Dealing Low", f"{curr_dl:.4f}" if not np.isnan(curr_dl) else "N/A"
  )


live_ticker_dashboard()

# Main Dashboard Table
st.subheader(f"📊 Engine Dashboard Summary ({timeframe.upper()})")
dashboard_data = {
    "Metric / Value": [
        "Live Market Price",
        "Dealing High",
        "Equilibrium (50% EQ)",
        "Dealing Low",
        f"Current Pricing Zone ({timeframe.upper()})",
        f"Last Structure Event ({timeframe.upper()})",
        f"CRT Value (Lookback: {lookback})",
        f"Conditional Sweep ({active_structural_trend} Trend Filter)",
        "Dynamic Target Objective",
        macro_title,
    ],
    "Status": [
        f"{latest_price:.4f}",
        f"{curr_dh:.4f}" if not np.isnan(curr_dh) else "N/A",
        f"{curr_eq:.4f}" if not np.isnan(curr_eq) else "N/A",
        f"{curr_dl:.4f}" if not np.isnan(curr_dl) else "N/A",
        curr_zone,
        curr_event,
        active_crt_val,
        active_sweep_status,
        current_target,
        macro_text,
    ],
}
st.table(pd.DataFrame(dashboard_data))

# --- Interactive Plotly Chart ---
st.subheader(
    f"🕯️ Structure & Liquidity Chart ({selected_asset_name} -"
    f" {timeframe.upper()})"
)

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

# Major Pivot Dots
fig.add_trace(
    go.Scatter(
        x=df.index[df["PivotHigh"].notna()],
        y=df["PivotHigh"].dropna(),
        mode="markers",
        name="Major Pivot High (A)",
        marker=dict(size=10, color=BEAR_COLOR, symbol="circle"),
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index[df["PivotLow"].notna()],
        y=df["PivotLow"].dropna(),
        mode="markers",
        name="Major Pivot Low (V)",
        marker=dict(size=10, color=BULL_COLOR, symbol="circle"),
    )
)

# BOS & Sweep Lines
for line in chart_lines:
  fig.add_shape(
      type="line",
      x0=line["x0"],
      y0=line["y"],
      x1=line["x1"],
      y1=line["y"],
      line=dict(
          color=line["color"], width=line["width"], dash=line["dash"]
      ),
  )

# Plot Active Conditional CRT Sweep Line if triggered
if crt_sweep_level is not None:
  fig.add_hline(
      y=crt_sweep_level,
      line_dash="dot",
      line_color=crt_sweep_color,
      annotation_text=f"CRT Sweep Level: {crt_sweep_level:.4f}",
      annotation_position="top left",
  )

# Lines
if not np.isnan(curr_eq):
  fig.add_hline(
      y=curr_eq,
      line_dash="dot",
      line_color=EQ_COLOR,
      annotation_text=f"50% EQ: {curr_eq:.4f}",
      annotation_position="bottom right",
  )

if not np.isnan(curr_dh):
  fig.add_hline(
      y=curr_dh,
      line_dash="dash",
      line_color=DR_HIGH_COLOR,
      annotation_text=f"Dealing High: {curr_dh:.4f}",
  )

if not np.isnan(curr_dl):
  fig.add_hline(
      y=curr_dl,
      line_dash="dash",
      line_color=BULL_COLOR,
      annotation_text=f"Dealing Low: {curr_dl:.4f}",
  )

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Time",
    yaxis_title="Price",
    height=650,
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
