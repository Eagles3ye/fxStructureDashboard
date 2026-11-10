"""
ICT Structure & Liquidity Sweep Engine [v6] — Streamlit Edition
==============================================================
Converts the TradingView Pine Script v6 indicator into a Python
Streamlit application using yfinance for market data.

Usage:
    streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────
# HELPER: Fetch OHLC from yfinance
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)  # cache 5 minutes
def fetch_ohlc(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical OHLC data from Yahoo Finance."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        st.error(f"No data returned for {ticker}")
        return pd.DataFrame()
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────
# HELPER: Pivot detection (rolling high/low)
# ─────────────────────────────────────────────────────────────────
def detect_pivots(df: pd.DataFrame, length: int):
    """Return series of pivot highs (NaN except at pivot) and pivot lows."""
    n = len(df)
    pivot_high = pd.Series(np.nan, index=df.index)
    pivot_low  = pd.Series(np.nan, index=df.index)
    for i in range(length, n - length):
        window_high = df["High"].iloc[i - length : i + length + 1]
        window_low  = df["Low"].iloc[i - length : i + length + 1]
        if df["High"].iloc[i] == window_high.max():
            pivot_high.iloc[i] = df["High"].iloc[i]
        if df["Low"].iloc[i] == window_low.min():
            pivot_low.iloc[i] = df["Low"].iloc[i]
    return pivot_high, pivot_low


# ─────────────────────────────────────────────────────────────────
# HELPER: Multi-timeframe structural data (bias, zone, eq)
# ─────────────────────────────────────────────────────────────────
def tf_zone_data(df: pd.DataFrame, length: int):
    """
    Equivalent of Pine Script f_tf_data_zone().
    Returns (bias, zone, eq) for the latest bar.
      bias:  1 = bullish, -1 = bearish, 0 = neutral
      zone:  1 = premium, -1 = discount, 0 = no eq
      eq:    float or NaN
    """
    if len(df) < length * 2 + 1:
        return 0, 0, np.nan

    ph, pl = detect_pivots(df, length)

    last_hi = np.nan
    last_lo = np.nan
    bias = 0
    d_high = np.nan
    d_low = np.nan

    for i in range(len(df)):
        if not np.isnan(ph.iloc[i]):
            last_hi = ph.iloc[i]
            d_high = ph.iloc[i]
        if not np.isnan(pl.iloc[i]):
            last_lo = pl.iloc[i]
            d_low = pl.iloc[i]

    if not np.isnan(last_hi) and df["Close"].iloc[-1] > last_hi:
        bias = 1
    if not np.isnan(last_lo) and df["Close"].iloc[-1] < last_lo:
        bias = -1

    if not np.isnan(d_high) and not np.isnan(d_low):
        eq = (d_high + d_low) / 2
    else:
        eq = np.nan

    if np.isnan(eq):
        zone = 0
    else:
        zone = 1 if df["Close"].iloc[-1] > eq else -1

    return bias, zone, eq


# ─────────────────────────────────────────────────────────────────
# MAIN ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────
def analyze_ict(df: pd.DataFrame, major_len: int, lookback: int,
                tf_data: dict) -> dict:
    """
    Run the full ICT Structure engine on a DataFrame.
    tf_data: dict mapping timeframe string -> (bias, zone, eq)
             e.g. {"1M": (1,1,100.5), "1W": (-1,-1,95.0), ...}
    Returns dict with all computed values for the dashboard + plot data.
    """
    n = len(df)
    if n < major_len * 2 + 1:
        return None

    # ── Major Pivots ──
    raw_maj_a, raw_maj_v = detect_pivots(df, major_len)

    # ── State trackers ──
    last_maj_a_val = np.nan
    last_maj_a_time_idx = -1
    maj_a_broken = False
    active_dr_high_bar = -1

    last_maj_v_val = np.nan
    last_maj_v_time_idx = -1
    maj_v_broken = False
    active_dr_low_bar = -1

    last_event = "Initializing Structure..."
    event_col = "gray"
    next_target = "Waiting for Event..."

    events_log = []  # for plotting labels/lines

    for i in range(major_len, n):
        # Update major pivot A
        if not np.isnan(raw_maj_a.iloc[i]):
            last_maj_a_val = raw_maj_a.iloc[i]
            last_maj_a_time_idx = i
            maj_a_broken = False
            active_dr_high_bar = i

        if not np.isnan(raw_maj_v.iloc[i]):
            last_maj_v_val = raw_maj_v.iloc[i]
            last_maj_v_time_idx = i
            maj_v_broken = False
            active_dr_low_bar = i

        # BOS / Sweep — Major A (high)
        if not np.isnan(last_maj_a_val) and not maj_a_broken and i > last_maj_a_time_idx:
            if df["High"].iloc[i] > last_maj_a_val:
                if df["Close"].iloc[i] > last_maj_a_val:
                    maj_a_broken = True
                    last_event = "BOS Bullish (Body Close)"
                    event_col = "green"
                    next_target = "Target: Next External High / FVG"
                    events_log.append({
                        "type": "bos_bull", "idx": i,
                        "price": last_maj_a_val, "label": "BOS"
                    })
                else:
                    last_event = "LIQUIDITY SWEEP HIGH (Wick)"
                    event_col = "red"
                    next_target = "Target: EQ (50%) -> Range Low"
                    events_log.append({
                        "type": "sweep_high", "idx": i,
                        "price": last_maj_a_val, "label": "SWEEP"
                    })

        # BOS / Sweep — Major V (low)
        if not np.isnan(last_maj_v_val) and not maj_v_broken and i > last_maj_v_time_idx:
            if df["Low"].iloc[i] < last_maj_v_val:
                if df["Close"].iloc[i] < last_maj_v_val:
                    maj_v_broken = True
                    last_event = "BOS Bearish (Body Close)"
                    event_col = "red"
                    next_target = "Target: Next External Low / FVG"
                    events_log.append({
                        "type": "bos_bear", "idx": i,
                        "price": last_maj_v_val, "label": "BOS"
                    })
                else:
                    last_event = "LIQUIDITY SWEEP LOW (Wick)"
                    event_col = "green"
                    next_target = "Target: EQ (50%) -> Range High"
                    events_log.append({
                        "type": "sweep_low", "idx": i,
                        "price": last_maj_v_val, "label": "SWEEP"
                    })

    # ── Dealing Range ──
    dealing_high = last_maj_a_val if not np.isnan(last_maj_a_val) else np.nan
    dealing_low  = last_maj_v_val if not np.isnan(last_maj_v_val) else np.nan

    if not np.isnan(dealing_high) and not np.isnan(dealing_low):
        eq_price = (dealing_high + dealing_low) / 2
    else:
        eq_price = np.nan

    if not np.isnan(eq_price):
        if df["Close"].iloc[-1] > eq_price:
            active_zone = "PREMIUM (Above 50% EQ)"
            zone_col = "red"
        else:
            active_zone = "DISCOUNT (Below 50% EQ)"
            zone_col = "green"
    else:
        active_zone = "N/A"
        zone_col = "gray"

    # ── Conditional Sweep (1H+ only) ──
    sweep_high_bars = []
    sweep_low_bars  = []
    is_allowed_tf = True  # streamlit is always "1H+" by default

    if is_allowed_tf and n > lookback + 1:
        for i in range(lookback + 1, n):
            prev_high = df["High"].iloc[i - lookback:i].max()  # high[1:lookback+1]
            prev_low  = df["Low"].iloc[i - lookback:i].min()

            bullish_bar = df["Close"].iloc[i] > df["Open"].iloc[i]
            edge_high = bullish_bar and (df["Close"].iloc[i] < prev_high) or (not bullish_bar and (df["Open"].iloc[i] < prev_high))
            edge_low  = bullish_bar and (df["Open"].iloc[i] > prev_low)  or (not bullish_bar and (df["Close"].iloc[i] > prev_low))

            is_bear_aligned = False
            is_bull_aligned = False
            # Check macro alignment from tf_data
            for tf, (b, z, e) in tf_data.items():
                pass  # will use passed-in alignment flags below

            # Use the pre-computed alignment flags
            if sweep_high_flags.get(i, False):
                sweep_high_bars.append(i)
            if sweep_low_flags.get(i, False):
                sweep_low_bars.append(i)

    # ── Macro alignment (determined by current timeframe) ──
    # Default to Daily view in Streamlit
    macro_title = "Macro (Daily Trend / H4 Zone)"
    macro_text  = ""
    macro_col   = "gray"

    # Fetch macro data
    d_bias, d_zone, d_eq = tf_data.get("1D", (0, 0, np.nan))
    h4_bias, h4_zone, h4_eq = tf_data.get("240", (0, 0, np.nan))

    d_str = "D: BULL" if d_bias == 1 else "D: BEAR" if d_bias == -1 else "D: NEUTRAL"

    if not np.isnan(h4_eq):
        h4_txt = f"H4: {h4_eq:.4f}"
        if df["Close"].iloc[-1] > h4_eq:
            h4_txt += " (PREM)"
        elif df["Close"].iloc[-1] < h4_eq:
            h4_txt += " (DISC)"
        else:
            h4_txt += " (EQ)"
    else:
        h4_txt = "H4: NA"

    macro_text = f"{d_str} | {h4_txt}"
    macro_col = "green" if d_bias == 1 else "red" if d_bias == -1 else "gray"

    return {
        "events": events_log,
        "dealing_high": dealing_high,
        "dealing_low": dealing_low,
        "eq_price": eq_price,
        "active_zone": active_zone,
        "zone_col": zone_col,
        "last_event": last_event,
        "event_col": event_col,
        "next_target": next_target,
        "macro_title": macro_title,
        "macro_text": macro_text,
        "macro_col": macro_col,
        "active_dr_high_bar": active_dr_high_bar,
        "active_dr_low_bar": active_dr_low_bar,
        "raw_maj_a": raw_maj_a,
        "raw_maj_v": raw_maj_v,
        "sweep_high_bars": sweep_high_bars,
        "sweep_low_bars": sweep_low_bars,
        "df": df,
    }


# ─────────────────────────────────────────────────────────────────
# PLOT: Build Plotly figure
# ─────────────────────────────────────────────────────────────────
def build_chart(result: dict, show_major_dots: bool, show_range_dots: bool,
                show_bos_lines: bool, show_eq_line: bool):
    df = result["df"]
    fig = make_subplots(rows=1, cols=1)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="green",
        decreasing_line_color="red",
    ))

    # Major pivot dots
    if show_major_dots:
        maj_a = result["raw_maj_a"]
        maj_v = result["raw_maj_v"]
        for i in range(len(df)):
            if not np.isnan(maj_a.iloc[i]) and i >= len(df) - 100:
                fig.add_trace(go.Scatter(
                    x=[i], y=[maj_a.iloc[i]], mode="markers",
                    marker=dict(size=10, color="red", symbol="circle"),
                    name="Major High", showlegend=False
                ))
            if not np.isnan(maj_v.iloc[i]) and i >= len(df) - 100:
                fig.add_trace(go.Scatter(
                    x=[i], y=[maj_v.iloc[i]], mode="markers",
                    marker=dict(size=10, color="green", symbol="circle"),
                    name="Major Low", showlegend=False
                ))

    # Active dealing range dots
    if show_range_dots:
        drh = result["active_dr_high_bar"]
        drl = result["active_dr_low_bar"]
        if drh >= 0 and drh < len(df) and not np.isnan(result["dealing_high"]):
            fig.add_trace(go.Scatter(
                x=[drh], y=[result["dealing_high"]], mode="markers",
                marker=dict(size=14, color="purple", symbol="diamond"),
                name="DR High", showlegend=False
            ))
        if drl >= 0 and drl < len(df) and not np.isnan(result["dealing_low"]):
            fig.add_trace(go.Scatter(
                x=[drl], y=[result["dealing_low"]], mode="markers",
                marker=dict(size=14, color="blue", symbol="diamond"),
                name="DR Low", showlegend=False
            ))

    # BOS / Sweep lines & labels
    if show_bos_lines:
        for ev in result["events"]:
            idx = ev["idx"]
            price = ev["price"]
            label = ev["label"]
            ev_type = ev["type"]
            color = "green" if "Bull" in ev_type or ev_type == "sweep_low" else "red"
            # Horizontal line
            fig.add_hline(y=price, line_color=color, line_dash="solid" if "BOS" in label else "dash",
                          line_width=2, row=1, col=1)
            # Label
            fig.add_annotation(x=idx, y=price, text=label,
                               showarrow=True, arrowhead=2,
                               font=dict(color="white", size=10),
                               bgcolor=color, bordercolor=color,
                               borderwidth=1, borderpad=2,
                               row=1, col=1)

    # Equilibrium line
    if show_eq_line and not np.isnan(result["eq_price"]):
        eq = result["eq_price"]
        fig.add_hline(y=eq, line_color="yellow", line_dash="dot",
                      line_width=2, row=1, col=1)
        fig.add_annotation(x=df.index[0], y=eq, text=f"EQ: {eq:.5f}",
                           showarrow=False, font=dict(color="yellow", size=10),
                           row=1, col=1)

    # Sweep flags (grey lines for 1H+)
    for idx in result.get("sweep_high_bars", []):
        if idx > 0 and idx < len(df):
            fig.add_hline(y=df["High"].iloc[idx], line_color="dimgray",
                          line_dash="dash", line_width=1, row=1, col=1)
    for idx in result.get("sweep_low_bars", []):
        if idx > 0 and idx < len(df):
            fig.add_hline(y=df["Low"].iloc[idx], line_color="dimgray",
                          line_dash="dash", line_width=1, row=1, col=1)

    fig.update_layout(
        title="ICT Structure & Liquidity Sweep Engine [v6]",
        xaxis_title="Date", yaxis_title="Price",
        template="plotly_dark", height=600,
        xaxis_rangeslider_visible=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# BUILD DASHBOARD TABLE (Streamlit)
# ─────────────────────────────────────────────────────────────────
def render_dashboard(result: dict, text_size: str):
    fs = {"tiny": 10, "small": 12, "normal": 14, "large": 18}[text_size]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**ICT STRUCTURE ENGINE**", font=fs + 4)
    with col2:
        st.markdown(f"**METRIC / VALUE**", font=fs + 4)

    data = [
        ("Dealing High", f"{result['dealing_high']:.2f}" if not np.isnan(result['dealing_high']) else "N/A", "red"),
        ("Equilibrium (50% EQ)", f"{result['eq_price']:.2f}" if not np.isnan(result['eq_price']) else "N/A", "yellow"),
        ("Dealing Low", f"{result['dealing_low']:.2f}" if not np.isnan(result['dealing_low']) else "N/A", "green"),
        ("Current Pricing Zone", result["active_zone"], result["zone_col"]),
        ("Last Structure Event", result["last_event"], result["event_col"]),
        ("Dynamic Target Objective", result["next_target"], result["event_col"]),
    ]

    for metric, value, color in data:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"`{metric}`", font=fs)
        with c2:
            st.markdown(f"`{value}`", font=fs)

    st.markdown(f"**{result['macro_title']}**", font=fs + 2)
    st.markdown(f"`{result['macro_text']}`", font=fs)


# ─────────────────────────────────────────────────────────────────
# SWEEP FLAGS (computed separately to avoid circular reference)
# ─────────────────────────────────────────────────────────────────
sweep_high_flags = {}
sweep_low_flags  = {}


def compute_sweep_flags(df: pd.DataFrame, lookback: int,
                        is_bull_aligned: bool, is_bear_aligned: bool):
    global sweep_high_flags, sweep_low_flags
    sweep_high_flags = {}
    sweep_low_flags = {}
    n = len(df)
    for i in range(lookback + 1, n):
        prev_high = df["High"].iloc[i - lookback:i].max()
        prev_low  = df["Low"].iloc[i - lookback:i].min()

        bullish_bar = df["Close"].iloc[i] > df["Open"].iloc[i]
        edge_high = (bullish_bar and df["Close"].iloc[i] < prev_high) or \
                    ((not bullish_bar) and df["Open"].iloc[i] < prev_high)
        edge_low = (bullish_bar and df["Open"].iloc[i] > prev_low) or \
                   ((not bullish_bar) and df["Close"].iloc[i] > prev_low)

        if is_bear_aligned and df["High"].iloc[i] > prev_high and edge_high:
            sweep_high_flags[i] = True
        if is_bull_aligned and df["Low"].iloc[i] < prev_low and edge_low:
            sweep_low_flags[i] = True


# ═════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="ICT Structure Engine v6", layout="wide")
    st.title("📊 ICT Structure & Liquidity Sweep Engine [v6]")

    # ── Sidebar controls ──
    st.sidebar.header("⚙️ Settings")

    ticker = st.sidebar.text_input("Ticker Symbol", value="BTC-USD")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y", "max"], index=2)
    with col2:
        interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "4h", "1d"], index=5)

    st.sidebar.subheader("Timeframe Strength Settings")
    m_len  = st.sidebar.slider("Monthly Strength", 1, 12, 1)
    w_len  = st.sidebar.slider("Weekly Strength", 1, 12, 1)
    d_len  = st.sidebar.slider("Daily Strength", 1, 20, 1)
    h4_len = st.sidebar.slider("H4 Strength", 1, 20, 1)
    h1_len = st.sidebar.slider("H1 Strength", 1, 20, 1)
    m15_len = st.sidebar.slider("M15 Strength", 1, 20, 1)

    st.sidebar.subheader("Pivot & Sweep Settings")
    major_len = st.sidebar.slider("Major Swing Length", 2, 10, 3)
    lookback  = st.sidebar.slider("Sweep Lookback Bars", 1, 20, 5)

    st.sidebar.subheader("Chart Visual Settings")
    show_major_dots = st.sidebar.checkbox("Show Major Pivot Dots", value=True)
    show_range_dots = st.sidebar.checkbox("Show Active Dealing Range Dots", value=True)
    show_bos_lines  = st.sidebar.checkbox("Show BOS / Sweep Lines", value=True)
    show_eq_line    = st.sidebar.checkbox("Show Equilibrium (EQ) Line", value=True)

    st.sidebar.subheader("Dashboard Settings")
    table_pos = st.sidebar.radio("Table Position", ["top", "bottom"], index=0)
    text_size = st.sidebar.selectbox("Text Size", ["tiny", "small", "normal", "large"], index=2)

    st.sidebar.subheader("Color Settings")
    bull_color = st.sidebar.color_picker("Bullish / Discount", "#00FF00")
    bear_color = st.sidebar.color_picker("Bearish / Premium", "#FF0000")
    eq_color   = st.sidebar.color_picker("Equilibrium", "#FFFF00")

    # ── Fetch data ──
    with st.spinner(f"Fetching {ticker} data..."):
        df = fetch_ohlc(ticker, period=period, interval=interval)

    if df.empty:
        st.warning("No data available. Check ticker and try again.")
        return

    n = len(df)
    st.info(f"Showing {n} bars of {ticker} ({interval}) — Last: {df.iloc[-1].name}")

    # ── Multi-timeframe data ──
    with st.spinner("Computing multi-timeframe analysis..."):
        # Map interval to TF string
        interval_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
        curr_tf = interval_map.get(interval, "D")

        tf_data = {}
        for tf_label, tf_len in [("1M", m_len), ("1W", w_len), ("1D", d_len),
                                  ("240", h4_len), ("60", h1_len), ("15", m15_len)
