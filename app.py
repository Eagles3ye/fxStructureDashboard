import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
  import yfinance as yf

  YFINANCE_AVAILABLE = True
except ImportError:
  YFINANCE_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="ICT Structure & Liquidity Sweep Engine [v6]",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ 6if Structure & Liquidity Sweep Engine [v6] - Full Mirror")
st.markdown(
    "Accurately mirrors TradingView Pine Script v6: Major Pivots, BOS/Sweeps,"
    " CRT Lookback Windows, Bounded EQ Line, and Multi-Timeframe Macro Table."
)

# --- Sidebar Inputs & Controls ---
st.sidebar.header("⚙️ Configuration & Tickers")

# Ticker Selection Dropdown + Custom Input
ticker_options = [
    "AAPL",
    "MSFT",
    "BTC-USD",
    "EURUSD=X",
    "GC=F",
    "TSLA",
    "ETH-USD",
    "Custom...",
]
selected_ticker_option = st.sidebar.selectbox("Select Asset", ticker_options)

if selected_ticker_option == "Custom...":
  symbol = st.sidebar.text_input(
      "Enter Custom Ticker (Yahoo Finance format)", value="NVDA"
  ).upper()
else:
  symbol = selected_ticker_option

interval = st.sidebar.selectbox(
    "Chart Timeframe", ["1d", "1h", "15m", "4h"], index=0
)
period_map = {"1d": "1y", "1h": "60d", "15m": "30d", "4h": "60d"}
selected_period = period_map.get(interval, "6mo")

st.sidebar.subheader("Pivot Settings")
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
@st.cache_data(ttl=3600)
def load_data(ticker, period, interval):
  if YFINANCE_AVAILABLE:
    try:
      yf_interval = "1h" if interval == "4h" else interval
      df = yf.download(
          ticker, period=period, interval=yf_interval, progress=False
      )
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
      df = df.dropna()
      if interval == "4h" and not df.empty:
        df = (
            df.resample("4h")
            .agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            })
            .dropna()
        )
      return df
    except Exception as e:
      st.warning(
          f"Could not fetch data from yfinance ({e}). Falling back to synthetic"
          " data..."
      )

  # Synthetic Data Fallback
  np.random.seed(42)
  dates = pd.date_range(end=pd.Timestamp.today(), periods=300, freq="B")
  close = 150 + np.cumsum(np.random.randn(300) * 2)
  high = close + np.random.rand(300) * 3
  low = close - np.random.rand(300) * 3
  open_p = close + (np.random.rand(300) - 0.5) * 2
  df = pd.DataFrame(
      {
          "Open": open_p,
          "High": high,
          "Low": low,
          "Close": close,
          "Volume": np.random.randint(1000, 10000, size=300),
      },
      index=dates,
  )
  return df


df = load_data(symbol, selected_period, interval)

if df.empty:
  st.error("No data retrieved for the selected asset. Please check the symbol.")
  st.stop()


# --- Helper Functions: Pivot Engine & Bias Calculation ---
def calculate_pivots(data, length):
  highs = data["High"].values
  lows = data["Low"].values
  n = len(data)
  ph = [np.nan] * n
  pl = [np.nan] * n
  for i in range(length, n - length):
    if all(highs[i] >= highs[i - j] for j in range(1, length + 1)) and all(
        highs[i] >= highs[i + j] for j in range(1, length + 1)
    ):
      ph[i] = highs[i]
    if all(lows[i] <= lows[i - j] for j in range(1, length + 1)) and all(
        lows[i] <= lows[i + j] for j in range(1, length + 1)
    ):
      pl[i] = lows[i]
  data["PivotHigh"] = ph
  data["PivotLow"] = pl
  return data


def get_bias_state(data, length=3):
  if len(data) < (length * 2 + 1):
    return 0, "Waiting"
  temp_df = calculate_pivots(data.copy(), length)
  hi, lo, b, ev = np.nan, np.nan, 0, "Waiting"
  hi_broken, lo_broken = False, False

  for i in range(len(temp_df)):
    ph = temp_df["PivotHigh"].iloc[i]
    pl = temp_df["PivotLow"].iloc[i]
    h = temp_df["High"].iloc[i]
    l = temp_df["Low"].iloc[i]
    c = temp_df["Close"].iloc[i]

    if not np.isnan(ph):
      hi = ph
      hi_broken = False
    if not np.isnan(pl):
      lo = pl
      lo_broken = False

    if not np.isnan(hi) and not hi_broken and h > hi:
      if c > hi:
        hi_broken = True
        b = 1
        ev = "BOS Bullish"
      else:
        ev = "Sweep High"
    if not np.isnan(lo) and not lo_broken and l < lo:
      if c < lo:
        lo_broken = True
        b = -1
        ev = "BOS Bearish"
      else:
        ev = "Sweep Low"
  return b, ev


def get_htf_bias(df_base):
  try:
    df_m = (
        df_base.resample("ME")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        })
        .dropna()
    )
    df_w = (
        df_base.resample("W")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        })
        .dropna()
    )
    df_d = (
        df_base.resample("D")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        })
        .dropna()
    )
    df_4h = (
        df_base.resample("4h")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        })
        .dropna()
    )
    df_1h = (
        df_base.resample("1h")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        })
        .dropna()
    )

    m_b, m_e = get_bias_state(df_m, 3) if len(df_m) > 7 else (0, "N/A")
    w_b, w_e = get_bias_state(df_w, 3) if len(df_w) > 7 else (0, "N/A")
    d_b, d_e = get_bias_state(df_d, 3) if len(df_d) > 7 else (0, "N/A")
    h4_b, h4_e = get_bias_state(df_4h, 3) if len(df_4h) > 7 else (0, "N/A")
    h1_b, h1_e = get_bias_state(df_1h, 3) if len(df_1h) > 7 else (0, "N/A")

    return {
        "Monthly": (m_b, m_e),
        "Weekly": (w_b, w_e),
        "Daily": (d_b, d_e),
        "H4": (h4_b, h4_e),
        "H1": (h1_b, h1_e),
    }
  except Exception:
    return {
        "Monthly": (0, "N/A"),
        "Weekly": (0, "N/A"),
        "Daily": (0, "N/A"),
        "H4": (0, "N/A"),
        "H1": (0, "N/A"),
    }


htf_data = get_htf_bias(df)

# --- Main Chart Pivot & Structure Processing ---
df = calculate_pivots(df, major_len)

last_swing_high, last_swing_low = np.nan, np.nan
swing_h_break, swing_l_break = False, False
last_event = "Initializing Structure..."
next_target = "Waiting for Event..."
lines_to_plot = []
crt_sweeps = []

for i in range(len(df)):
  ph = df["PivotHigh"].iloc[i]
  pl = df["PivotLow"].iloc[i]
  h = df["High"].iloc[i]
  l = df["Low"].iloc[i]
  c = df["Close"].iloc[i]
  o = df["Open"].iloc[i]
  dt = df.index[i]

  if not np.isnan(ph):
    last_swing_high = ph
    swing_h_break = False
    last_swing_h_time = dt
  if not np.isnan(pl):
    last_swing_low = pl
    swing_l_break = False
    last_swing_l_time = dt

  # BOS / Sweep Engine
  if not np.isnan(last_swing_high) and not swing_h_break and h > last_swing_high:
    if c > last_swing_high:
      swing_h_break = True
      last_event = "BOS Bullish (Body Close)"
      next_target = "Target: Next External High / FVG"
      if show_lines:
        lines_to_plot.append((
            last_swing_h_time,
            dt,
            last_swing_high,
            last_swing_high,
            bull_color,
            False,
        ))  # Fixed: False instead of false
    else:
      last_event = "LIQUIDITY SWEEP HIGH (Wick)"
      next_target = "Target: EQ (50%) -> Range Low"
      if show_lines:
        lines_to_plot.append((
            last_swing_h_time,
            dt,
            last_swing_high,
            last_swing_high,
            bear_color,
            True,
        ))

  if not np.isnan(last_swing_low) and not swing_l_break and l < last_swing_low:
    if c < last_swing_low:
      swing_l_break = True
      last_event = "BOS Bearish (Body Close)"
      next_target = "Target: Next External Low / FVG"
      if show_lines:
        lines_to_plot.append((
            last_swing_l_time,
            dt,
            last_swing_low,
            last_swing_low,
            bear_color,
            False,
        ))  # Fixed: False instead of false
    else:
      last_event = "LIQUIDITY SWEEP LOW (Wick)"
      next_target = "Target: EQ (50%) -> Range High"
      if show_lines:
        lines_to_plot.append((
            last_swing_l_time,
            dt,
            last_swing_low,
            last_swing_low,
            bull_color,
            True,
        ))

  # CRT Sweeps
  if i >= len(df) - sweep_bars and i > 0:
    crh = df["High"].iloc[i - 1]
    crl = df["Low"].iloc[i - 1]
    bull_bar = c > o
    bear_sweep = c < crh if bull_bar else o < crh
    sweep_high = h > crh and bear_sweep

    bull_sweep_cond = o > crl if bull_bar else c > crl
    sweep_low = l < crl and bull_sweep_cond

    if show_lines:
      if sweep_high:
        crt_sweeps.append((df.index[i - 1], dt, crh, crh, "#4B4B4B"))
      if sweep_low:
        crt_sweeps.append((df.index[i - 1], dt, crl, crl, "#4B4B4B"))

# Dealing Range & EQ calculations
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

# --- Dashboard View & Metrics ---
st.subheader(f"📊 ICT Structure Engine Dashboard ({symbol})")

col1, col2, col3 = st.columns(3)
with col1:
  st.metric(
      "Dealing High",
      f"{dealing_high:.2f}" if not np.isnan(dealing_high) else "N/A",
  )
  st.metric("Pricing Zone", active_zone)
with col2:
  st.metric("Equilibrium (50% EQ)", f"{eq:.2f}" if not np.isnan(eq) else "N/A")
  st.metric("Last Structure Event", last_event)
with col3:
  st.metric(
      "Dealing Low",
      f"{dealing_low:.2f}" if not np.isnan(dealing_low) else "N/A",
  )
  st.metric("Dynamic Target Objective", next_target)

# Macro Alignment Table Display
st.markdown("### 🌐 Multi-Timeframe Macro Alignment Matrix")
macro_cols = st.columns(5)
tf_names = ["Monthly", "Weekly", "Daily", "H4", "H1"]
for idx, tf in enumerate(tf_names):
  b_val, e_str = htf_data[tf]
  color_badge = (
      "🟢 Bullish" if b_val == 1 else ("🔴 Bearish" if b_val == -1 else "⚪ Neutral")
  )
  with macro_cols[idx]:
    st.markdown(
        f"**{tf}**\n\nStatus: `{e_str}`\n\nBias: **{color_badge}**",
        unsafe_allow_html=True,
    )

st.markdown("---")

# --- Plotly Chart Visuals ---
st.subheader("📈 Interactive Price & Structure Chart")

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

if show_dots:
  ph_df = df[df["PivotHigh"].notna()]
  pl_df = df[df["PivotLow"].notna()]
  fig.add_trace(
      go.Scatter(
          x=ph_df.index,
          y=ph_df["PivotHigh"],
          mode="markers",
          marker=dict(size=11, color=bear_color),
          name="Major Swing High",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=pl_df.index,
          y=pl_df["PivotLow"],
          mode="markers",
          marker=dict(size=11, color=bull_color),
          name="Major Swing Low",
      )
  )

if show_eq and not np.isnan(eq):
  eq_start_idx = max(0, len(df) - 20)
  fig.add_shape(
      type="line",
      x0=df.index[eq_start_idx],
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

if show_lines:
  for x0, x1, y0, y1, col, is_dashed in lines_to_plot:
    fig.add_shape(
        type="line",
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        line=dict(
            color=col, width=1.5, dash="dash" if is_dashed else "solid"
        ),
    )
  for x0, x1, y0, y1, col in crt_sweeps:
    fig.add_shape(
        type="line",
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        line=dict(color=col, width=1, dash="dot"),
    )

fig.update_layout(
    title=f"{symbol} - Structure & Liquidity Engine",
    xaxis_title="Timeline",
    yaxis_title="Price",
    template="plotly_dark",
    height=650,
    xaxis_rangeslider_visible=False,
)

st.plotly_chart(fig, use_container_width=True)
