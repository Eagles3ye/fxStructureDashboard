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
import time

# ─────────────────────────────────────────────────────────────────
# HELPER: Fetch OHLC from yfinance
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_ohlc(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical OHLC data from Yahoo Finance."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return pd.DataFrame()
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
    pivot_low = pd.Series(np.nan, index=df.index)
    for i in range(length, n - length):
        window_high = df["High"].iloc[i - length : i + length + 1]
        window_low = df["Low"].iloc[i - length : i + length + 1]
        if df["High"].iloc[i] == window_high.max():
            pivot_high.iloc[i] = df["High"].iloc[i]
        if df["Low"].iloc[i] == window_low.min():
            pivot_low.iloc[i] = df["Low"].iloc[i]
    return pivot_high, pivot_low


# ─────────────────────────────────────────────────────────────────
# CORE ENGINE: Unified Active Dealing Range & Structure State
# ─────────────────────────────────────────────────────────────────
def compute_ict_structure(df: pd.DataFrame, length: int):
    """
    Computes active dealing range, equilibrium, bias, and the 
    chronologically latest structural event (BOS or Sweep).
    """
    if len(df) < length * 2 + 1:
        return None

    raw_maj_a, raw_maj_v = detect_pivots(df, length)

    last_maj_a_val = np.nan
    last_maj_a_idx = -1
    last_maj_v_val = np.nan
    last_maj_v_idx = -1

    # Find latest valid pivot points
    for i in range(length, len(df)):
        if not np.isnan(raw_maj_a.iloc[i]):
            last_maj_a_val = raw_maj_a.iloc[i]
            last_maj_a_idx = i
        if not np.isnan(raw_maj_v.iloc[i]):
            last_maj_v_val = raw_maj_v.iloc[i]
            last_maj_v_idx = i

    last_event = "Range Bounded / Waiting"
    next_target = "Waiting for Event..."
    events_log = []

    active_high_broken = False
    active_low_broken = False

    scan_start = max(length, min(last_maj_a_idx if last_maj_a_idx != -1 else 0, last_maj_v_idx if last_maj_v_idx != -1 else 0))

    for i in range(scan_start, len(df)):
        # Evaluate High Breaks / Sweeps
        if not np.isnan(last_maj_a_val) and i > last_maj_a_idx:
            if df["High"].iloc[i] > last_maj_a_val:
                if df["Close"].iloc[i] > last_maj_a_val:
                    active_high_broken = True
                    active_low_broken = False
                    last_event = "BOS Bullish (Body Close)"
                    next_target = "Target: Next External High / FVG"
                    events_log.append({
                        "type": "bos_bull", "idx": i,
                        "price": last_maj_a_val, "label": "BOS",
                        "line_x1": last_maj_a_idx, "line_x2": i,
                        "line_y1": last_maj_a_val, "line_y2": last_maj_a_val
                    })
                else:
                    last_event = "LIQUIDITY SWEEP HIGH (Wick)"
                    next_target = "Target: EQ (50%) -> Range Low"
                    events_log.append({
                        "type": "sweep_high", "idx": i,
                        "price": last_maj_a_val, "label": "SWEEP",
                        "line_x1": last_maj_a_idx, "line_x2": i,
                        "line_y1": last_maj_a_val, "line_y2": last_maj_a_val
                    })

        # Evaluate Low Breaks / Sweeps
        if not np.isnan(last_maj_v_val) and i > last_maj_v_idx:
            if df["Low"].iloc[i] < last_maj_v_val:
                if df["Close"].iloc[i] < last_maj_v_val:
                    active_low_broken = True
                    active_high_broken = False
                    last_event = "BOS Bearish (Body Close)"
                    next_target = "Target: Next External Low / FVG"
                    events_log.append({
                        "type": "bos_bear", "idx": i,
                        "price": last_maj_v_val, "label": "BOS",
                        "line_x1": last_maj_v_idx, "line_x2": i,
                        "line_y1": last_maj_v_val, "line_y2": last_maj_v_val
                    })
                else:
                    last_event = "LIQUIDITY SWEEP LOW (Wick)"
                    next_target = "Target: EQ (50%) -> Range High"
                    events_log.append({
                        "type": "sweep_low", "idx": i,
                        "price": last_maj_v_val, "label": "SWEEP",
                        "line_x1": last_maj_v_idx, "line_x2": i,
                        "line_y1": last_maj_v_val, "line_y2": last_maj_v_val
                    })

    dealing_high = last_maj_a_val if not np.isnan(last_maj_a_val) else np.nan
    dealing_low = last_maj_v_val if not np.isnan(last_maj_v_val) else np.nan
    eq_price = (dealing_high + dealing_low) / 2 if (not np.isnan(dealing_high) and not np.isnan(dealing_low)) else np.nan

    bias = 0
    if not np.isnan(eq_price):
        if df["Close"].iloc[-1] > eq_price:
            bias = 1
            active_zone = "PREMIUM (Above 50% EQ)"
        else:
            bias = -1
            active_zone = "DISCOUNT (Below 50% EQ)"
    else:
        active_zone = "N/A"

    if "BOS Bullish" in last_event:
        bias = 1
    elif "BOS Bearish" in last_event:
        bias = -1

    return {
        "events": events_log,
        "dealing_high": dealing_high,
        "dealing_low": dealing_low,
        "eq_price": eq_price,
        "active_zone": active_zone,
        "zone": 1 if active_zone.startswith("PREMIUM") else (-1 if active_zone.startswith("DISCOUNT") else 0),
        "bias": bias,
        "last_event": last_event,
        "next_target": next_target,
        "active_dr_high_bar": last_maj_a_idx,
        "active_dr_low_bar": last_maj_v_idx,
        "raw_maj_a": raw_maj_a,
        "raw_maj_v": raw_maj_v,
        "df": df,
    }


# ─────────────────────────────────────────────────────────────────
# CONDITIONAL SWEEP FLAGS
# ─────────────────────────────────────────────────────────────────
def compute_sweep_flags(df: pd.DataFrame, lookback: int,
                        is_bull_aligned: bool, is_bear_aligned: bool):
    """Compute sweep flags for the current timeframe."""
    sweep_high_flags = {}
    sweep_low_flags = {}
    n = len(df)
    for i in range(lookback + 1, n):
        prev_high = df["High"].iloc[i - lookback:i].max()
        prev_low = df["Low"].iloc[i - lookback:i].min()

        bullish_bar = df["Close"].iloc[i] > df["Open"].iloc[i]
        edge_high = (bullish_bar and df["Close"].iloc[i] < prev_high) or \
                    ((not bullish_bar) and df["Open"].iloc[i] < prev_high)
        edge_low = (bullish_bar and df["Open"].iloc[i] > prev_low) or \
                   ((not bullish_bar) and df["Close"].iloc[i] > prev_low)

        if is_bear_aligned and df["High"].iloc[i] > prev_high and edge_high:
            sweep_high_flags[i] = True
        if is_bull_aligned and df["Low"].iloc[i] < prev_low and edge_low:
            sweep_low_flags[i] = True
    return sweep_high_flags, sweep_low_flags


# ─────────────────────────────────────────────────────────────────
# PLOT: Plotly figure
# ─────────────────────────────────────────────────────────────────
def build_chart(result: dict, show_major_dots: bool, show_range_dots: bool,
                show_bos_lines: bool, show_eq_line: bool):
    df = result["df"]
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="green",
        decreasing_line_color="red",
    ))

    if show_major_dots:
        maj_a = result["raw_maj_a"]
        maj_v = result["raw_maj_v"]
        start = max(0, len(df) - 100)
        for i in range(start, len(df)):
            if not np.isnan(maj_a.iloc[i]):
                fig.add_trace(go.Scatter(
                    x=[df.index[i]], y=[maj_a.iloc[i]], mode="markers",
                    marker=dict(size=10, color="red", symbol="circle"),
                    name="Major High", showlegend=False
                ))
            if not np.isnan(maj_v.iloc[i]):
                fig.add_trace(go.Scatter(
                    x=[df.index[i]], y=[maj_v.iloc[i]], mode="markers",
                    marker=dict(size=10, color="green", symbol="circle"),
                    name="Major Low", showlegend=False
                ))

    if show_range_dots:
        drh = result["active_dr_high_bar"]
        drl = result["active_dr_low_bar"]
        if drh >= 0 and drh < len(df) and not np.isnan(result["dealing_high"]):
            fig.add_trace(go.Scatter(
                x=[df.index[drh]], y=[result["dealing_high"]], mode="markers",
                marker=dict(size=14, color="purple", symbol="diamond"),
                name="DR High", showlegend=False
            ))
        if drl >= 0 and drl < len(df) and not np.isnan(result["dealing_low"]):
            fig.add_trace(go.Scatter(
                x=[df.index[drl]], y=[result["dealing_low"]], mode="markers",
                marker=dict(size=14, color="blue", symbol="diamond"),
                name="DR Low", showlegend=False
            ))

    if show_bos_lines:
        for ev in result["events"]:
            x1 = df.index[ev["line_x1"]]
            x2 = df.index[ev["line_x2"]]
            y = ev["line_y1"]
            label = ev["label"]
            color = "green" if ev["type"] in ("bos_bull", "sweep_low") else "red"
            dash = "solid" if "BOS" in label else "dash"

            fig.add_trace(go.Scatter(
                x=[x1, x2], y=[y, y], mode="lines",
                line=dict(color=color, width=2, dash=dash),
                showlegend=False, hoverinfo="skip"
            ))

            fig.add_annotation(
                x=x2, y=y, text=label,
                showarrow=False,
                xanchor="left", yanchor="bottom",
                font=dict(color="white", size=10),
                bgcolor=color, bordercolor=color,
                borderwidth=1, borderpad=2,
            )

    if show_eq_line and not np.isnan(result["eq_price"]):
        eq = result["eq_price"]
        drh = result["active_dr_high_bar"]
        drl = result["active_dr_low_bar"]

        if drh >= 0 and drl >= 0:
            x_start = df.index[min(drh, drl)]
            x_end = df.index[max(drh, drl)]
        else:
            x_start = df.index[0]
            x_end = df.index[-1]

        fig.add_trace(go.Scatter(
            x=[x_start, x_end], y=[eq, eq], mode="lines",
            line=dict(color="yellow", width=2, dash="dot"),
            name="EQ", showlegend=False
        ))
        fig.add_annotation(
            x=x_start, y=eq, text=f"EQ: {eq:.5f}",
            showarrow=False, xanchor="left",
            font=dict(color="yellow", size=10),
        )

    fig.update_layout(
        title="ICT Structure Engine [v6]",
        xaxis_title="Date", yaxis_title="Price",
        template="plotly_dark", height=650,
        xaxis_rangeslider_visible=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# BUILD DASHBOARD
# ─────────────────────────────────────────────────────────────────
def render_dashboard(result: dict, tf_results: dict, text_size: str):
    fs = {"tiny": 10, "small": 12, "normal": 14, "large": 18}[text_size]

    st.markdown(
        f"<div style='font-size:{fs+4}px; font-weight:bold; margin-bottom:4px;'>"
        f"ICT STRUCTURE ENGINE &nbsp;|&nbsp; METRIC / VALUE</div>",
        unsafe_allow_html=True
    )

    data = [
        ("Dealing High", f"{result['dealing_high']:.2f}" if not np.isnan(result['dealing_high']) else "N/A"),
        ("Equilibrium (50% EQ)", f"{result['eq_price']:.2f}" if not np.isnan(result['eq_price']) else "N/A"),
        ("Dealing Low", f"{result['dealing_low']:.2f}" if not np.isnan(result['dealing_low']) else "N/A"),
        ("Current Pricing Zone", result["active_zone"]),
        ("Last Structure Event (Chart)", result["last_event"]),
        ("Dynamic Target Objective", result["next_target"]),
    ]

    for metric, value in data:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"<div style='font-size:{fs}px; font-family:monospace;'>{metric}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div style='font-size:{fs}px; font-family:monospace;'>{value}</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:{fs+2}px; font-weight:bold; margin-top:12px;'>Multi-Timeframe Structure & Bias Alignment</div>", unsafe_allow_html=True)
    
    table_html = "<table style='width:100%; font-family:monospace; font-size:{}px; border-collapse:collapse;'>".format(fs)
    table_html += "<tr style='border-bottom: 1px solid #444; text-align:left;'><th>TF</th><th>Bias</th><th>Zone</th><th>Last Structural Event</th></tr>"
    
    for tf_label in ["1M", "1W", "1D", "240"]:
        res = tf_results.get(tf_label, {})
        bias_val = res.get("bias", 0)
        bias_str = "BULLISH 🟢" if bias_val == 1 else ("BEARISH 🔴" if bias_val == -1 else "NEUTRAL ⚪")
        
        zone_val = res.get("zone", 0)
        zone_str = "PREMIUM" if zone_val == 1 else ("DISCOUNT" if zone_val == -1 else "EQ/N/A")
        
        event_str = res.get("last_event", "N/A")
        
        table_html += f"<tr style='border-bottom: 1px solid #222;'><td><b>{tf_label}</b></td><td>{bias_str}</td><td>{zone_str}</td><td>{event_str}</td></tr>"
    
    table_html += "</table>"
    st.markdown(table_html, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="ICT Structure Engine v6", layout="wide")
    st.title("📊 ICT Structure & Liquidity Sweep Engine [v6]")

    st.sidebar.header("⚙️ Settings")

    predefined_tickers = [
        "BTC-USD", "ETH-USD", "SOL-USD", 
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", 
        "GC=F", "CL=F", "AAPL", "NVDA", "TSLA", "MSFT", "CUSTOM"
    ]
    
    selected_ticker_option = st.sidebar.selectbox("Select Ticker Symbol", predefined_tickers, index=0)
    ticker = st.sidebar.text_input("Enter Custom Ticker", value="BTC-USD") if selected_ticker_option == "CUSTOM" else selected_ticker_option

    col1, col2 = st.sidebar.columns(2)
    with col1:
        period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y", "max"], index=2)
    with col2:
        interval_options = {
            "1 Minute": "1m", "5 Minutes": "5m", "15 Minutes": "15m", 
            "1 Hour": "1h", "4 Hours": "4h", "Daily": "1d", 
            "Weekly": "1wk", "Monthly": "1mo"
        }
        selected_interval_label = st.sidebar.selectbox("Interval", list(interval_options.keys()), index=5)
        interval = interval_options[selected_interval_label]

    st.sidebar.subheader("Live Updates")
    auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)
    refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 15, 300, 30)

    st.sidebar.subheader("Timeframe Strength Settings")
    m_len = st.sidebar.slider("Monthly Strength", 1, 12, 1)
    w_len = st.sidebar.slider("Weekly Strength", 1, 12, 1)
    d_len = st.sidebar.slider("Daily Strength", 1, 20, 1)
    h4_len = st.sidebar.slider("H4 Strength", 1, 20, 1)

    st.sidebar.subheader("Pivot & Sweep Settings")
    major_len = st.sidebar.slider("Major Swing Length", 2, 10, 3)
    lookback = st.sidebar.slider("Sweep Lookback Bars", 1, 20, 5)

    st.sidebar.subheader("Chart Visual Settings")
    show_major_dots = st.sidebar.checkbox("Show Major Pivot Dots", value=True)
    show_range_dots = st.sidebar.checkbox("Show Active Dealing Range Dots", value=True)
    show_bos_lines = st.sidebar.checkbox("Show BOS / Sweep Lines", value=True)
    show_eq_line = st.sidebar.checkbox("Show Equilibrium (EQ) Line", value=True)

    st.sidebar.subheader("Dashboard Settings")
    table_pos = st.sidebar.radio("Table Position", ["top", "bottom"], index=0)
    text_size = st.sidebar.selectbox("Text Size", ["tiny", "small", "normal", "large"], index=2)

    with st.spinner(f"Fetching {ticker} data ({selected_interval_label})..."):
        df = fetch_ohlc(ticker, period=period, interval=interval)

    if df.empty:
        st.warning("No data available. Check ticker symbol and try again.")
        return

    n = len(df)
    st.info(f"Showing {n} bars of {ticker} ({selected_interval_label}) — Last close: {df['Close'].iloc[-1]:.5f}")

    # Compute multi-timeframe analysis using the unified core engine
    with st.spinner("Computing multi-timeframe analysis..."):
        tf_results = {}
        tf_configs = [
            ("1M", "1mo", m_len),
            ("1W", "1wk", w_len),
            ("1D", "1d", d_len),
            ("240", "4h", h4_len),
        ]

        for tf_label, tf_interval, tf_len in tf_configs:
            try:
                tf_period = "max" if tf_label in ["1M", "1W", "1D"] else "2y"
                tdf = fetch_ohlc(ticker, period=tf_period, interval=tf_interval)
                res = compute_ict_structure(tdf, tf_len)
                tf_results[tf_label] = res if res else {"bias": 0, "zone": 0, "eq": np.nan, "last_event": "Insufficient Data"}
            except Exception:
                tf_results[tf_label] = {"bias": 0, "zone": 0, "eq": np.nan, "last_event": "Error"}

    m_bias = tf_results.get("1M", {}).get("bias", 0)
    w_bias = tf_results.get("1W", {}).get("bias", 0)
    d_bias = tf_results.get("1D", {}).get("bias", 0)
    h4_zone = tf_results.get("240", {}).get("zone", 0)

    macroWeeklyH4Bull = (w_bias == 1 and h4_zone == -1)
    macroWeeklyH4Bear = (w_bias == -1 and h4_zone == 1)

    is_bull_aligned = macroWeeklyH4Bull
    is_bear_aligned = macroWeeklyH4Bear

    sweep_high_flags, sweep_low_flags = compute_sweep_flags(df, lookback, is_bull_aligned, is_bear_aligned)

    result = compute_ict_structure(df, major_len)

    if result is None:
        st.warning("Not enough data for the selected major swing length. Try a smaller length or more data.")
        return

    if table_pos == "top":
        render_dashboard(result, tf_results, text_size)
        st.plotly_chart(build_chart(result, show_major_dots, show_range_dots, show_bos_lines, show_eq_line), use_container_width=True)
    else:
        st.plotly_chart(build_chart(result, show_major_dots, show_range_dots, show_bos_lines, show_eq_line), use_container_width=True)
        render_dashboard(result, tf_results, text_size)

    with st.expander("📋 Raw Data (OHLC)"):
        st.dataframe(df.tail(50).style.format({
            "Open": "{:.5f}", "High": "{:.5f}",
            "Low": "{:.5f}", "Close": "{:.5f}"
        }))

    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
