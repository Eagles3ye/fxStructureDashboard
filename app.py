import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

# --- Page Configuration ---
st.set_page_config(
    page_title="ICT Structure & Liquidity Engine",
    page_icon="📈",
    layout="wide",
)

# --- Sidebar Inputs ---
st.sidebar.header("Engine Settings")

# Predefined Asset Dictionary with TradingView & YFinance compatible symbols
asset_options = {
    "USD/CAD": {"tv": "FX:USDCAD", "yf": "USDCAD=X"},
    "EUR/USD": {"tv": "FX:EURUSD", "yf": "EURUSD=X"},
    "GBP/USD": {"tv": "FX:GBPUSD", "yf": "GBPUSD=X"},
    "USD/JPY": {"tv": "FX:USDJPY", "yf": "USDJPY=X"},
    "Gold (XAU/USD)": {"tv": "OANDA:XAUUSD", "yf": "GC=F"},
    "Silver (XAG/USD)": {"tv": "OANDA:XAGUSD", "yf": "SI=F"},
    "Bitcoin (BTC/USD)": {"tv": "BINANCE:BTCUSDT", "yf": "BTC-USD"},
    "S&P 500 ETF (SPY)": {"tv": "AMEX:SPY", "yf": "SPY"},
}

selected_asset_name = st.sidebar.selectbox(
    "Select Asset", list(asset_options.keys()), index=0
)
asset_info = asset_options[selected_asset_name]
ticker = asset_info["yf"]
tv_symbol = asset_info["tv"]

# Timeframe Selection
timeframe = st.sidebar.selectbox(
    "Dashboard Timeframe Context", ["1M", "1W", "1D", "4h", "1h", "15m"], index=2
)

major_len = st.sidebar.slider("Major Swing Length", 2, 10, 3)
lookback = st.sidebar.slider("Sweep Lookback Bars", 1, 20, 5)

# Color configurations
BULL_COLOR = "#26a69a"  # Green / Discount
BEAR_COLOR = "#ef5350"  # Red / Premium
EQ_COLOR = "#ffeb3b"  # Yellow


# --- Data Fetching & Multi-Timeframe Resampling ---
@st.cache_data(ttl=300)
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

# --- Dealing Range & Structure Event Logic ---
dealing_high = np.nan
dealing_low = np.nan

d_highs = []
d_lows = []
eq_list = []
zones = []

last_maj_a = np.nan
last_maj_v = np.nan
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

  if not np.isnan(ph):
    last_maj_a = ph
    dealing_high = ph
    maj_a_broken = False

  if not np.isnan(pl):
    last_maj_v = pl
    dealing_low = pl
    maj_v_broken = False

  if not np.isnan(last_maj_a) and not maj_a_broken and high_val > last_maj_a:
    if close_val > last_maj_a:
      maj_a_broken = True
      current_event = "BOS Bullish (Body Close)"
      current_target = "Target: Next External High / FVG"
    else:
      current_event = "LIQUIDITY SWEEP HIGH (Wick)"
      current_target = "Target: EQ (50%) -> Range Low"

  if not np.isnan(last_maj_v) and not maj_v_broken and low_val < last_maj_v:
    if close_val < last_maj_v:
      maj_v_broken = True
      current_event = "BOS Bearish (Body Close)"
      current_target = "Target: Next External Low / FVG"
    else:
      current_event = "LIQUIDITY SWEEP LOW (Wick)"
      current_target = "Target: EQ (50%) -> Range High"

  eq = (
      (dealing_high + dealing_low) / 2
      if not (np.isnan(dealing_high) and np.isnan(dealing_low))
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


# --- Macro Context ---
def get_bias_and_zone(sym, interval_rule):
  temp_df = yf.download(
      sym, period="1y", interval=interval_rule, progress=False, threads=False
  )
  if not temp_df.empty:
    if isinstance(temp_df.columns, pd.MultiIndex):
      temp_df.columns = temp_df.columns.get_level_values(0)
    temp_df["EMA50"] = temp_df["Close"].ewm(span=50).mean()
    bias = (
        "BULL"
        if temp_df["Close"].iloc[-1] > temp_df["EMA50"].iloc[-1]
        else "BEAR"
    )
    return bias
  return "NEUTRAL"


monthly_bias = get_bias_and_zone(ticker, "1mo")
weekly_bias = get_bias_and_zone(ticker, "1wk")
daily_bias = get_bias_and_zone(ticker, "1d")

macro_title = f"Macro Alignment ({timeframe.upper()})"
macro_text = f"M: {monthly_bias} | W: {weekly_bias} | D: {daily_bias}"

# --- Dashboard UI Layout ---
st.title("📈 ICT Structure & Liquidity Engine")
st.markdown(
    f"**Asset:** `{selected_asset_name}` | **Timeframe:** `{timeframe.upper()}`"
    " | **Engine Status:** `Active`"
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
st.subheader(f"📊 Engine Dashboard Summary ({timeframe.upper()})")
dashboard_data = {
    "Metric / Value": [
        "Dealing High",
        "Equilibrium (50% EQ)",
        "Dealing Low",
        f"Current Pricing Zone ({timeframe.upper()})",
        f"Last Structure Event ({timeframe.upper()})",
        "Dynamic Target Objective",
        macro_title,
    ],
    "Status": [
        f"{curr_dh:.4f}" if not np.isnan(curr_dh) else "N/A",
        f"{curr_eq:.4f}" if not np.isnan(curr_eq) else "N/A",
        f"{curr_dl:.4f}" if not np.isnan(curr_dl) else "N/A",
        curr_zone,
        current_event,
        current_target,
        macro_text,
    ],
}
st.table(pd.DataFrame(dashboard_data))

# --- Native Embedded TradingView Advanced Chart Widget ---
st.subheader(
    f"🕯️ Live TradingView Chart ({selected_asset_name} - {timeframe.upper()})"
)

# Map our timeframe selection to TradingView interval format
tv_interval_map = {
    "1M": "M",
    "1W": "W",
    "1D": "D",
    "4h": "240",
    "1h": "60",
    "15m": "15",
}
tv_interval = tv_interval_map.get(timeframe, "D")

# HTML/JS widget container for TradingView Advanced Real-Time Chart
tv_widget_html = f"""
<div class="tradingview-widget-container" style="height:600px;width:100%">
  <div id="tradingview_widget" style="height:100%;width:100%"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget(
  {{
    "autosize": true,
    "symbol": "{tv_symbol}",
    "interval": "{tv_interval}",
    "timezone": "Etc/UTC",
    "theme": "dark",
    "style": "1",
    "locale": "en",
    "toolbar_bg": "#f1f3f6",
    "enable_publishing": false,
    "hide_side_toolbar": false,
    "allow_symbol_change": true,
    "details": true,
    "hotlist": true,
    "calendar": true,
    "container_id": "tradingview_widget"
  }}
  );
  </script>
</div>
"""

# Render the interactive TradingView widget inside Streamlit
components.html(tv_widget_html, height=620)
