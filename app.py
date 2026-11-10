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

# Auto-refresh the dashboard every 60 seconds to pull fresh candle data
st_autorefresh(interval=60000, key="datarefresh")

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


# --- Safe Data Fetching with Fallback ---
@st.cache_data(ttl=60)
def load_and_resample_data(symbol, tf):
  try:
    base_period = "max" if tf in ["1W", "1M"] else "1y"
    df = yf.download(
        symbol, period=base_period, interval="60m", progress=False, threads=False
    )
    if df.empty:
      df = yf.download(
          symbol, period="max", interval="1d", progress=False, threads=False
      )

    if df.empty:
      return pd.DataFrame()

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
          symbol,
          period="30d",
          interval="15m",
          progress=False,
          threads=False,
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
  except Exception as e:
    return pd.DataFrame()


df = load_and_resample_data(ticker, timeframe)

if df.empty or len(df) < major_len * 2:
  st.warning(
      f"⚠️ Unable to fetch live data for {selected_asset_name} right now. Yahoo"
      " Finance may be rate-limiting requests. Please wait a moment or switch"
      " asset."
  )
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

dealing_high = np.nan
dealing_low = np.nan

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
    current_target = "Target: EQ (50%) -> Range Low"
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
    current_target = "Target: EQ (50%) -> Range High"
    chart_lines.append({
        "x0": df.index[i - LOOKBACK],
        "x1": curr_time,
        "y": df["Low"].iloc[i - LOOKBACK],
        "color": BULL_COLOR,
        "width": 1,
        "dash": "dot",
    })

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
curr_event = df["Event"].iloc[-1]


@st.cache_data(ttl=60)
def scan_h4_setup(sym, length=3):
  try:
    df_h4 = yf.download(
        sym, period="30d", interval="60m", progress=False, threads=False
    )
    if df_h4.empty:
      return "NO DATA", "N/A", "N/A", "N/A"
    if isinstance(df_h4.columns, pd.MultiIndex):
      df_h4.columns = df_h4.columns.get_level_values(0)

    df_h4 = (
        df_h4.resample("4h")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )
    if len(df_h4) < length * 2:
      return "INSUFFICIENT DATA", "N/A", "N/A", "N/A"

    df_h4["EMA50"] = df_h4["Close"].ewm(span=50).mean()
    h4_trend = (
        "BEARISH"
        if df_h4["Close"].iloc[-1] < df_h4["EMA50"].iloc[-1]
        else "BULLISH"
    )

    highs, lows, closes = (
        df_h4["High"].values,
        df_h4["Low"].values,
        df_h4["Close"].values,
    )
    active_hi, active_lo = np.nan, np.nan
    for i in range(length, len(df_h4) - length):
      if all(highs[i] > highs[i - j] for j in range(1, length + 1)) and all(
          highs[i] >= highs[i + j] for j in range(1, length + 1)
      ):
        active_hi = highs[i]
      if all(lows[i] < lows[i - j] for j in range(1, length + 1)) and all(
          lows[i] <= lows[i + j] for j in range(1, length + 1)
      ):
        active_lo = lows[i]

    if np.isnan(active_hi) or np.isnan(active_lo):
      active_hi, active_lo = highs.max(), lows.min()

    h4_eq = (active_hi + active_lo) / 2
    h4_zone = "PREMIUM" if closes[-1] > h4_eq else "DISCOUNT"

    h4_sweep_high = (highs[-1] > highs[-2]) and (closes[-1] < highs[-2])
    h4_sweep_low = (lows[-1] < lows[-2]) and (closes[-1] > lows[-2])
    sweep_status = (
        "HIGH SWEEP"
        if h4_sweep_high
        else ("LOW SWEEP" if h4_sweep_low else "NO SWEEP")
    )

    return h4_trend, h4_zone, h4_eq, sweep_status
  except Exception:
    return "ERROR", "N/A", "N/A", "N/A"


h4_trend, h4_zone, h4_eq_val, h4_sweep = scan_h4_setup(ticker, major_len)

# --- UI Dashboard ---
st.title("📈 ICT Structure & Liquidity Engine")
st.markdown(
    f"**Asset:** `{selected_asset_name}` | **Timeframe Context:**"
    f" `{timeframe.upper()}` (Auto-Refreshes every 60s)"
)

st.subheader("🚨 H4 CRT & Setup Scanner (Lookback = 1)")
col_a, col_b, col_c = st.columns(3)
col_a.metric("H4 Trend Bias", h4_trend)
col_b.metric("H4 Pricing Zone", h4_zone)
col_c.metric("H4 Sweep Status (LB=1)", h4_sweep)

if h4_trend == "BULLISH" and h4_zone == "DISCOUNT" and h4_sweep == "LOW SWEEP":
  st.success(
      "🔥 **BULLISH CRT SETUP ACTIVE ON H4!** Trend is Bullish | Price is in"
      " Discount | Low Sweep confirmed."
  )
elif h4_trend == "BEARISH" and h4_zone == "PREMIUM" and h4_sweep == "HIGH SWEEP":
  st.success(
      "🔥 **BEARISH CRT SETUP ACTIVE ON H4!** Trend is Bearish | Price is in"
      " Premium | High Sweep confirmed."
  )
elif h4_sweep != "NO SWEEP":
  st.warning(
      f"⚠️ **{h4_sweep} detected in {h4_zone} zone**, but trend ({h4_trend}) does"
      " not align for entry."
  )
else:
  st.info(
      "ℹ️ Waiting for alignment: (Bullish + Discount + Low Sweep) OR (Bearish +"
      " Premium + High Sweep)."
  )

# --- Plotly Chart ---
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
      line=dict(
          color=line["color"], width=line["width"], dash=line["dash"]
      ),
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
