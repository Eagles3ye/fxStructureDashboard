import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

# --- Page Configuration ---
st.set_page_config(
    page_title="ICT Structure & Liquidity Engine",
    page_icon="📈",
    layout="wide",
)

# --- Live Auto-Refresh Setup (Refreshes every 5 seconds for live updates) ---
refresh_interval = (
    5000  # 5000ms = 5 seconds (optimized to bypass Yahoo Finance rate limits)
)
count = st_autorefresh(
    interval=refresh_interval, limit=None, key="live_data_refresh"
)

# --- Sidebar Inputs ---
st.sidebar.header("Engine Settings")
st.sidebar.markdown(f"🟢 **Live Sync Active** (Refresh count: {count})")

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
lookback = st.sidebar.slider("Sweep Lookback Bars", 1, 20, 5)

# Color configurations
BULL_COLOR = "#26a69a"  # Green / Discount / Bullish
BEAR_COLOR = "#ef5350"  # Red / Premium / Bearish
EQ_COLOR = "#ffeb3b"  # Yellow
DR_HIGH_COLOR = "#ab47bc"  # Purple (Dealing High)
DR_LOW_COLOR = "#42a5f5"  # Blue (Dealing Low)


# --- Data Fetching & Multi-Timeframe Resampling ---
@st.cache_data(ttl=10)  # Short cache TTL to allow fresh live data pulls
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

# --- Dealing Range, Structure & Line Event Logic ---
dealing_high = np.nan
dealing_low = np.nan

d_highs = []
d_lows = []
eq_list = []
zones = []
event_list = []

chart_lines = []

last_maj_a = np.nan
last_maj_a_idx = 0
last_maj_v = np.nan
last_maj_v_idx = 0

maj_a_broken = False
maj_v_broken = False

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

  if not np.isnan(last_maj_a) and not maj_a_broken and high_val > last_maj_a:
    start_time = df.index[last_maj_a_idx]
    if close_val > last_maj_a:
      maj_a_broken = True
      current_event = "BOS Bullish (Body Close)"
      current_target = "Target: Next External High / FVG"
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

df["DealingHigh"] = d_highs
df["DealingLow"] = d_lows
df["EQ"] = eq_list
df["Zone"] = zones
df["Event"] = event_list

curr_dh = df["DealingHigh"].iloc[-1]
curr_dl = df["DealingLow"].iloc[-1]
curr_eq = df["EQ"].iloc[-1]
curr_zone = df["Zone"].iloc[-1]
curr_event = df["Event"].iloc[-1]


# --- Macro & CRT (Candle Range Theory) Calculations ---
def get_macro_and_crt_status(sym, length=3):
  results = {}
  tf_configs = {"Monthly": "1mo", "Weekly": "1wk", "Daily": "1d"}

  for tf_name, interval_code in tf_configs.items():
    temp_df = yf.download(
        sym, period="1y", interval=interval_code, progress=False, threads=False
    )
    if not temp_df.empty:
      if isinstance(temp_df.columns, pd.MultiIndex):
        temp_df.columns = temp_df.columns.get_level_values(0)

      # Calculate CRT (Using previous completed candle range as reference CRT)
      if len(temp_df) > 2:
        prev_candle = temp_df.iloc[-2]
        curr_candle = temp_df.iloc[-1]

        crt_high = prev_candle["High"]
        crt_low = prev_candle["Low"]
        crt_mid = (crt_high + crt_low) / 2

        # Sweep Status Check High or Low
        sweep_status = "Neutral"
        if curr_candle["High"] > crt_high and curr_candle["Low"] < crt_low:
          sweep_status = "Both High & Low Swept"
        elif curr_candle["High"] > crt_high:
          sweep_status = "Sweep High"
        elif curr_candle["Low"] < crt_low:
          sweep_status = "Sweep Low"

        results[tf_name] = {
            "crt_val": f"H: {crt_high:.4f} | L: {crt_low:.4f}",
            "sweep_status": sweep_status,
        }
      else:
        results[tf_name] = {"crt_val": "N/A", "sweep_status": "N/A"}
    else:
      results[tf_name] = {"crt_val": "N/A", "sweep_status": "N/A"}

  # H4 Zone Calculation for Daily Macro
  df_h4 = yf.download(
      sym, period="59d", interval="60m", progress=False, threads=False
  )
  h4_txt = "H4: NA"
  if not df_h4.empty:
    if isinstance(df_h4.columns, pd.MultiIndex):
      df_h4.columns = df_h4.columns.get_level_values(0)
    df_h4 = (
        df_h4.resample("4h")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )
    if len(df_h4) > length * 2:
      h4_hi = df_h4["High"].max()
      h4_lo = df_h4["Low"].min()
      h4_eq = (h4_hi + h4_lo) / 2
      h4_close = df_h4["Close"].iloc[-1]
      zone_str = (
          "PREM" if h4_close > h4_eq else "DISC" if h4_close < h4_eq else "EQ"
      )
      h4_txt = f"H4: {h4_eq:.4f} ({zone_str})"

  # Trend Bias (Daily EMA50)
  df_d = yf.download(
      sym, period="1y", interval="1d", progress=False, threads=False
  )
  d_bias = "NEUTRAL"
  if not df_d.empty:
    if isinstance(df_d.columns, pd.MultiIndex):
      df_d.columns = df_d.columns.get_level_values(0)
    df_d["EMA50"] = df_d["Close"].ewm(span=50).mean()
    d_bias = (
        "BULL" if df_d["Close"].iloc[-1] > df_d["EMA50"].iloc[-1] else "BEAR"
    )

  results["Daily_Bias"] = d_bias
  results["H4_Zone"] = h4_txt
  return results


macro_data = get_macro_and_crt_status(ticker, major_len)

if timeframe in ["1D", "D"]:
  macro_title = "Macro (Daily Trend / H4 Zone)"
  macro_text = f"D: {macro_data['Daily_Bias']} | {macro_data['H4_Zone']}"
elif timeframe in ["1W", "W"]:
  macro_title = "Macro (Weekly / Daily Bias)"
  macro_text = (
      f"W: {macro_data['Weekly']['sweep_status']} | D:"
      f" {macro_data['Daily_Bias']}"
  )
elif timeframe in ["1M", "M"]:
  macro_title = "Macro (Monthly / Weekly Bias)"
  macro_text = (
      f"M: {macro_data['Monthly']['sweep_status']} | W:"
      f" {macro_data['Weekly']['sweep_status']}"
  )
else:
  macro_title = f"Macro Alignment ({timeframe.upper()})"
  macro_text = (
      f"Trend Bias: {macro_data['Daily_Bias']} | Zone:"
      f" {curr_zone.split(' ')[0]}"
  )

# --- Dashboard UI Layout ---
st.title("📈 ICT Structure & Liquidity Engine (Live)")
st.markdown(
    f"**Asset:** `{selected_asset_name}` | **Timeframe:** `{timeframe.upper()}`"
    " | **Status:** `Live Real-time Feed Active`"
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

# Main Dashboard Table (Including CRT values & Sweep Status checks across Monthly, Weekly, Daily)
st.subheader(f"📊 Engine Dashboard Summary ({timeframe.upper()})")
dashboard_data = {
    "Metric / Value": [
        "Dealing High",
        "Equilibrium (50% EQ)",
        "Dealing Low",
        f"Current Pricing Zone ({timeframe.upper()})",
        f"Last Structure Event ({timeframe.upper()})",
        "Dynamic Target Objective",
        "Monthly CRT Value",
        "Monthly CRT Sweep Status",
        "Weekly CRT Value",
        "Weekly CRT Sweep Status",
        "Daily CRT Value",
        "Daily CRT Sweep Status",
        macro_title,
    ],
    "Status": [
        f"{curr_dh:.4f}" if not np.isnan(curr_dh) else "N/A",
        f"{curr_eq:.4f}" if not np.isnan(curr_eq) else "N/A",
        f"{curr_dl:.4f}" if not np.isnan(curr_dl) else "N/A",
        curr_zone,
        curr_event,
        current_target,
        macro_data["Monthly"]["crt_val"],
        macro_data["Monthly"]["sweep_status"],
        macro_data["Weekly"]["crt_val"],
        macro_data["Weekly"]["sweep_status"],
        macro_data["Daily"]["crt_val"],
        macro_data["Daily"]["sweep_status"],
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
      line_color=DR_LOW_COLOR,
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
